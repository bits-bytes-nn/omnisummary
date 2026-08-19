from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from collectors.web_search import WebSearchCollector, fetch_pinned_items
from shared.config import TrendSearch, WebSearchCollectorConfig
from shared.constants import SourceType


def _collector(**kwargs) -> WebSearchCollector:
    cfg = WebSearchCollectorConfig(**kwargs)
    cfg.reference_time = datetime(2026, 6, 3, tzinfo=UTC)
    cfg.lookback_hours = 24
    # no llm_factory -> _llm stays None, no Tavily client call in _parse_results
    return WebSearchCollector(cfg, llm_factory=None)


def _result(score, *, days_old=0, title="X", url="https://example.com/a"):
    pub = datetime(2026, 6, 3, tzinfo=UTC).timestamp() - days_old * 86400
    return {
        "url": url,
        "title": title,
        "content": "body",
        "published_date": datetime.fromtimestamp(pub, tz=UTC).isoformat(),
        "score": score,
    }


class TestParseResults:
    def test_filters_low_relevance(self):
        c = _collector(min_search_score=0.3)
        resp = {"results": [_result(0.02, title="off-topic"), _result(0.8, title="relevant")]}
        items = c._parse_results(resp, trend_name="t")
        titles = [i.title for i in items]
        assert "relevant" in titles
        assert "off-topic" not in titles

    def test_filters_stale_by_date(self):
        c = _collector(min_search_score=0.0)
        resp = {"results": [_result(0.9, days_old=10, title="old")]}
        assert c._parse_results(resp, trend_name="t") == []

    def test_filters_results_published_after_the_reference_time(self):
        # Only the parked path closed the upper end, so a `--date` backfill's Tavily results were
        # filtered against a cutoff alone and today's live hits were ingested as that day's.
        c = _collector(min_search_score=0.0)
        resp = {"results": [_result(0.9, days_old=-2, title="tomorrow")]}
        assert c._parse_results(resp, trend_name="t") == []

    def test_skips_missing_date(self):
        c = _collector(min_search_score=0.0)
        resp = {"results": [{"url": "u", "title": "no-date", "content": "x", "score": 0.9}]}
        assert c._parse_results(resp, trend_name="t") == []

    def test_keeps_relevant_recent(self):
        c = _collector(min_search_score=0.3)
        resp = {"results": [_result(0.7, days_old=0, title="good")]}
        items = c._parse_results(resp, trend_name="t")
        assert len(items) == 1
        assert items[0].source_type == SourceType.WEB
        assert items[0].metadata["search_score"] == 0.7

    def test_missing_score_not_filtered(self):
        # if Tavily omits score, don't drop the item on relevance grounds
        c = _collector(min_search_score=0.3)
        r = _result(0.9)
        del r["score"]
        items = c._parse_results({"results": [r]}, trend_name="t")
        assert len(items) == 1

    def test_date_only_published_date_is_kept(self):
        # Tavily often returns a date-only 'published_date' ('2026-06-03'), which parses to a naive
        # datetime; comparing it to the tz-aware cutoff used to raise TypeError and silently drop
        # the result. It must now be treated as UTC and kept when recent.
        c = _collector(min_search_score=0.0)
        r = {
            "url": "https://example.com/d",
            "title": "date-only",
            "content": "x",
            "score": 0.9,
            "published_date": "2026-06-03",
        }
        items = c._parse_results({"results": [r]}, trend_name="t")
        assert len(items) == 1
        assert items[0].published_at.tzinfo is not None


def _search_collector(*, llm_factory=None, **kwargs) -> WebSearchCollector:
    """A collector configured with real trend searches, for exercising collect()."""
    cfg = WebSearchCollectorConfig(
        trend_searches=[
            TrendSearch(name="frontier", queries=["q1", "q2"]),
            TrendSearch(name="infra", queries=["q3"], domains=["arxiv.org"], topic="general"),
        ],
        retry_backoff_sec=0,
        **kwargs,
    )
    cfg.reference_time = datetime(2026, 6, 3, tzinfo=UTC)
    cfg.lookback_hours = 24
    return WebSearchCollector(cfg, llm_factory=llm_factory)


class TestCollect:
    @pytest.mark.asyncio
    async def test_skips_without_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "")
        c = _collector(min_search_score=0.3)
        assert await c.collect() == []

    @pytest.mark.asyncio
    async def test_unreadable_secret_store_fails_the_source_instead_of_emptying_it(self, monkeypatch):
        # A denied/throttled SSM read used to return "" — the same answer as "no key configured" —
        # so the digest silently lost its whole web-search source with one warning line. It must
        # raise, which reports the collector FAILED and alerts.
        from shared.utils import SecretUnavailableError

        def _boom(*args, **kwargs):
            assert kwargs.get("strict") is True
            raise SecretUnavailableError("AccessDenied")

        monkeypatch.setattr("collectors.web_search.resolve_secret", _boom)
        c = _collector(min_search_score=0.3)
        with pytest.raises(SecretUnavailableError):
            await c.collect()

    @pytest.mark.asyncio
    async def test_disabled_collector_short_circuits(self):
        c = _search_collector(enabled=False)
        with patch("collectors.web_search.resolve_secret") as resolve:
            assert await c.collect() == []
        resolve.assert_not_called()  # a disabled source must not even resolve its key

    @pytest.mark.asyncio
    async def test_no_queries_configured_short_circuits(self, monkeypatch):
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        c = _collector(min_search_score=0.3)  # no trend_searches
        assert await c.collect() == []

    @pytest.mark.asyncio
    async def test_searches_every_query_and_dedups_urls(self, monkeypatch):
        # One query per trend query (3 total); the same URL returned twice collapses to one item.
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        c = _search_collector(min_search_score=0.3)
        client = MagicMock()
        client.search = AsyncMock(
            side_effect=[
                {"results": [_result(0.9, url="https://a.example/1", title="A")]},
                {"results": [_result(0.9, url="https://a.example/1", title="A again")]},
                {"results": [_result(0.9, url="https://b.example/2", title="B")]},
            ]
        )
        c._client_instance = client
        items = await c.collect()
        assert client.search.await_count == 3
        assert {i.url for i in items} == {"https://a.example/1", "https://b.example/2"}
        # The per-trend include_domains/topic reach Tavily as configured.
        last = client.search.await_args_list[-1].kwargs
        assert last["include_domains"] == ["arxiv.org"] and last["topic"] == "general"
        # The window Tavily is asked about is the run's own, anchored to reference_time.
        assert (last["start_date"], last["end_date"]) == ("2026-06-02", "2026-06-03")

    @pytest.mark.asyncio
    async def test_the_requested_window_widens_with_the_configured_lookback(self, monkeypatch):
        # `days = lookback_hours // 24` truncated the shipped 30-hour window to ONE day for the
        # largest source, so widening lookback_hours changed nothing upstream.
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        c = _search_collector(min_search_score=0.3)
        c.config.lookback_hours = 30
        client = MagicMock()
        client.search = AsyncMock(return_value={"results": []})
        c._client_instance = client
        await c.collect()
        kwargs = client.search.await_args_list[0].kwargs
        assert (kwargs["start_date"], kwargs["end_date"]) == ("2026-06-01", "2026-06-03")

    @pytest.mark.asyncio
    async def test_search_fan_out_is_bounded_by_config(self, monkeypatch):
        # Unbounded fan-out threw every configured query at Tavily at once and self-throttled.
        import asyncio

        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        c = _search_collector(min_search_score=0.3, max_concurrency=1)
        state = {"active": 0, "peak": 0}

        async def _search(**kwargs):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return {"results": []}

        client = MagicMock()
        client.search = AsyncMock(side_effect=_search)
        c._client_instance = client
        await c.collect()
        assert client.search.await_count == 3
        assert state["peak"] == 1, f"fan-out exceeded max_concurrency: peak={state['peak']}"

    @pytest.mark.asyncio
    async def test_all_queries_failing_raises(self, monkeypatch):
        # A total Tavily outage must surface as FAILED, not a silently empty web source.
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        c = _search_collector(min_search_score=0.3, max_retries=1)
        client = MagicMock()
        client.search = AsyncMock(side_effect=RuntimeError("tavily down"))
        c._client_instance = client
        with pytest.raises(RuntimeError):
            await c.collect()

    @pytest.mark.asyncio
    async def test_permanent_tavily_rejection_is_not_retried(self, monkeypatch):
        # A revoked key / exhausted quota is a VERDICT: retrying it burned three attempts per query
        # before health could report DEGRADED. One attempt each, then the source fails.
        from tavily.errors import InvalidAPIKeyError

        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        c = _search_collector(min_search_score=0.3, max_retries=3)
        client = MagicMock()
        client.search = AsyncMock(side_effect=InvalidAPIKeyError("revoked key"))
        c._client_instance = client
        with pytest.raises(Exception, match="revoked key"):
            await c.collect()
        assert client.search.await_count == 3  # 3 queries x 1 attempt, not 3 x 3

    @pytest.mark.asyncio
    async def test_transient_search_failure_is_still_retried(self, monkeypatch):
        # The narrower predicate must not lose the retry that matters: a hung request still gets
        # another attempt, and the query survives.
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        c = _search_collector(min_search_score=0.3, max_retries=2)
        client = MagicMock()
        client.search = AsyncMock(
            side_effect=[
                TimeoutError("hung"),
                {"results": [_result(0.9, url="https://ok.example/1", title="OK")]},
                {"results": []},
                {"results": []},
            ]
        )
        c._client_instance = client
        items = await c.collect()
        assert [i.url for i in items] == ["https://ok.example/1"]
        assert client.search.await_count == 4  # the timed-out query was retried

    @pytest.mark.asyncio
    async def test_partial_query_failure_keeps_the_rest(self, monkeypatch):
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        c = _search_collector(min_search_score=0.3, max_retries=1)
        client = MagicMock()
        client.search = AsyncMock(
            side_effect=[
                RuntimeError("one query failed"),
                {"results": [_result(0.9, url="https://ok.example/1", title="OK")]},
                {"results": []},
            ]
        )
        c._client_instance = client
        items = await c.collect()
        assert [i.url for i in items] == ["https://ok.example/1"]

    @pytest.mark.asyncio
    async def test_refinement_adds_llm_generated_queries(self, monkeypatch):
        # With a refine LLM configured, collect() runs a second phase using the LLM's queries and
        # merges the results (deduped against the broad phase).
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content='["refined query"]'))
        c = _search_collector(min_search_score=0.3, max_refine_queries=2, llm_factory=factory)
        client = MagicMock()
        client.search = AsyncMock(
            side_effect=[
                {"results": [_result(0.9, url="https://a.example/1", title="A")]},
                {"results": []},
                {"results": []},
                {"results": [_result(0.9, url="https://refined.example/9", title="R")]},
            ]
        )
        c._client_instance = client
        items = await c.collect()
        assert client.search.await_count == 4  # 3 broad + 1 refined
        assert {i.url for i in items} == {"https://a.example/1", "https://refined.example/9"}

    @pytest.mark.asyncio
    async def test_refinement_skipped_when_broad_phase_empty(self, monkeypatch):
        # Nothing to refine FROM → no extra LLM call and no extra searches.
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content='["never used"]'))
        c = _search_collector(min_search_score=0.3, llm_factory=factory)
        client = MagicMock()
        client.search = AsyncMock(return_value={"results": []})
        c._client_instance = client
        assert await c.collect() == []
        assert client.search.await_count == 3  # broad only

    @pytest.mark.asyncio
    async def test_refinement_llm_failure_keeps_broad_results(self, monkeypatch):
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "key")
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("bedrock down")))
        c = _search_collector(min_search_score=0.3, llm_factory=factory)
        client = MagicMock()
        client.search = AsyncMock(
            side_effect=[
                {"results": [_result(0.9, url="https://a.example/1", title="A")]},
                {"results": []},
                {"results": []},
            ]
        )
        c._client_instance = client
        items = await c.collect()
        assert [i.url for i in items] == ["https://a.example/1"]  # refinement is non-fatal


class TestFetchPinnedItems:
    @pytest.mark.asyncio
    async def test_returns_collected_items_marked_pinned(self):
        client = MagicMock()
        client.extract = AsyncMock(
            return_value={
                "results": [
                    {"url": "https://darioamodei.com/post/policy-on-the-ai-exponential", "raw_content": "body text"}
                ]
            }
        )
        with patch("collectors.web_search.resolve_secret", return_value="key"):
            with patch("collectors.web_search.AsyncTavilyClient", return_value=client):
                items = await fetch_pinned_items(["https://darioamodei.com/post/policy-on-the-ai-exponential"])
        assert len(items) == 1
        assert items[0].metadata["pinned"] is True
        assert items[0].text == "body text"
        # no extractor title → fall back to the URL slug
        assert "policy on the ai exponential" in items[0].title

    @pytest.mark.asyncio
    async def test_youtube_pin_uses_data_api_not_tavily(self):
        # YouTube URLs route to the YouTube Data API, never Tavily (whose extractor only sees a
        # video page's metadata, never its content). Title + description come from the API; the
        # transcript is best-effort and may be empty (datacenter IP block), leaving the description.
        api_resp = MagicMock(status_code=200)
        api_resp.json.return_value = {
            "items": [
                {
                    "snippet": {
                        "title": "The data black hole at the center of AI",
                        "description": "It is easy to forget how much data these models train on.",
                        "channelTitle": "Dwarkesh Patel",
                    }
                }
            ]
        }
        http_client = AsyncMock()
        http_client.get = AsyncMock(return_value=api_resp)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=http_client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        tavily = MagicMock()
        tavily.extract = AsyncMock()
        with patch("collectors.web_search.resolve_secret", return_value="key"):
            with patch("collectors.web_search.httpx.AsyncClient", return_value=ctx):
                with patch("collectors.web_search.fetch_youtube_transcript", return_value=""):
                    with patch("collectors.web_search.AsyncTavilyClient", return_value=tavily):
                        items = await fetch_pinned_items(["https://www.youtube.com/watch?v=4pG3SJQPAwk"])
        assert len(items) == 1
        assert items[0].title == "The data black hole at the center of AI"
        assert items[0].author == "Dwarkesh Patel"
        assert items[0].source_type == SourceType.YOUTUBE
        assert items[0].metadata["pinned"] is True
        assert items[0].text == "It is easy to forget how much data these models train on."
        tavily.extract.assert_not_called()  # YouTube never touches Tavily

    @pytest.mark.asyncio
    async def test_youtube_pin_prefers_transcript_over_description(self):
        api_resp = MagicMock(status_code=200)
        api_resp.json.return_value = {
            "items": [{"snippet": {"title": "T", "description": "short desc", "channelTitle": "Ch"}}]
        }
        http_client = AsyncMock()
        http_client.get = AsyncMock(return_value=api_resp)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=http_client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("collectors.web_search.resolve_secret", return_value="key"):
            with patch("collectors.web_search.httpx.AsyncClient", return_value=ctx):
                with patch("collectors.web_search.fetch_youtube_transcript", return_value="full transcript"):
                    items = await fetch_pinned_items(["https://youtu.be/4pG3SJQPAwk"])
        assert items[0].text == "full transcript"  # transcript wins when present

    @pytest.mark.asyncio
    async def test_mixed_pins_route_youtube_and_tavily_separately(self):
        api_resp = MagicMock(status_code=200)
        api_resp.json.return_value = {"items": [{"snippet": {"title": "Vid", "description": "d", "channelTitle": "C"}}]}
        http_client = AsyncMock()
        http_client.get = AsyncMock(return_value=api_resp)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=http_client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        tavily = MagicMock()
        tavily.extract = AsyncMock(return_value={"results": [{"url": "https://example.com/post", "raw_content": "b"}]})
        with patch("collectors.web_search.resolve_secret", return_value="key"):
            with patch("collectors.web_search.httpx.AsyncClient", return_value=ctx):
                with patch("collectors.web_search.fetch_youtube_transcript", return_value=""):
                    with patch("collectors.web_search.AsyncTavilyClient", return_value=tavily):
                        items = await fetch_pinned_items(
                            ["https://www.youtube.com/watch?v=4pG3SJQPAwk", "https://example.com/post"]
                        )
        urls = {it.url for it in items}
        assert urls == {"https://www.youtube.com/watch?v=4pG3SJQPAwk", "https://example.com/post"}
        tavily.extract.assert_awaited_once()
        assert tavily.extract.await_args.kwargs["urls"] == ["https://example.com/post"]  # YouTube excluded

    @pytest.mark.asyncio
    async def test_youtube_pin_non_200_is_dropped_and_surfaced(self):
        # A YouTube pin whose Data API lookup fails (e.g. 404) must be dropped, not crash, and
        # the missing-pin warning must fire so a silently-lost pin is visible.
        api_resp = MagicMock(status_code=404)
        api_resp.json.return_value = {}
        http_client = AsyncMock()
        http_client.get = AsyncMock(return_value=api_resp)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=http_client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch("collectors.web_search.resolve_secret", return_value="key"):
            with patch("collectors.web_search.httpx.AsyncClient", return_value=ctx):
                with patch("collectors.web_search.logger") as mock_logger:
                    items = await fetch_pinned_items(["https://www.youtube.com/watch?v=4pG3SJQPAwk"])
        assert items == []
        warned = " ".join(str(c.args) for c in mock_logger.warning.call_args_list)
        assert "could not be fetched" in warned

    @pytest.mark.asyncio
    async def test_youtube_pin_empty_items_is_dropped(self):
        api_resp = MagicMock(status_code=200)
        api_resp.json.return_value = {"items": []}
        http_client = AsyncMock()
        http_client.get = AsyncMock(return_value=api_resp)
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=http_client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch("collectors.web_search.resolve_secret", return_value="key"):
            with patch("collectors.web_search.httpx.AsyncClient", return_value=ctx):
                items = await fetch_pinned_items(["https://www.youtube.com/watch?v=4pG3SJQPAwk"])
        assert items == []

    @pytest.mark.asyncio
    async def test_empty_urls_short_circuits(self):
        assert await fetch_pinned_items([]) == []
        assert await fetch_pinned_items(["", "   "]) == []

    @pytest.mark.asyncio
    async def test_no_api_key_returns_empty(self):
        with patch("collectors.web_search.resolve_secret", return_value=""):
            assert await fetch_pinned_items(["https://x.com/a"]) == []

    @pytest.mark.asyncio
    async def test_extract_failure_returns_empty(self):
        client = MagicMock()
        client.extract = AsyncMock(side_effect=RuntimeError("boom"))
        with patch("collectors.web_search.resolve_secret", return_value="key"):
            with patch("collectors.web_search.AsyncTavilyClient", return_value=client):
                assert await fetch_pinned_items(["https://x.com/a"]) == []


class TestTitleFromUrl:
    def test_article_slug(self):
        from collectors.web_search import _title_from_url

        assert (
            _title_from_url("https://darioamodei.com/post/policy-on-the-ai-exponential")
            == "policy on the ai exponential"
        )

    def test_youtube_falls_back_to_host(self):
        from collectors.web_search import _title_from_url

        assert _title_from_url("https://www.youtube.com/watch?v=haK1KoQWm18") == "youtube.com"

    def test_x_status_id_falls_back_to_host(self):
        from collectors.web_search import _title_from_url

        assert _title_from_url("https://x.com/karpathy/status/1944435413395685866") == "x.com"


class TestYoutubeVideoId:
    def test_extracts_id_from_url_forms(self):
        from collectors.web_search import _youtube_video_id

        assert _youtube_video_id("https://www.youtube.com/watch?v=4pG3SJQPAwk") == "4pG3SJQPAwk"
        assert _youtube_video_id("https://youtu.be/4pG3SJQPAwk") == "4pG3SJQPAwk"
        assert _youtube_video_id("https://m.youtube.com/watch?v=4pG3SJQPAwk&t=10s") == "4pG3SJQPAwk"
        assert _youtube_video_id("https://www.youtube.com/shorts/4pG3SJQPAwk") == "4pG3SJQPAwk"

    def test_returns_empty_for_non_youtube(self):
        from collectors.web_search import _youtube_video_id

        assert _youtube_video_id("https://example.com/watch?v=4pG3SJQPAwk") == ""
        assert _youtube_video_id("https://www.youtube.com/@DwarkeshPatel") == ""
