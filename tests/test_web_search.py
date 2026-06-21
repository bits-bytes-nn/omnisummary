from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collectors.web_search import WebSearchCollector, fetch_pinned_items
from shared.config import WebSearchCollectorConfig
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


class TestCollect:
    @pytest.mark.asyncio
    async def test_skips_without_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        monkeypatch.setattr("collectors.web_search.resolve_secret", lambda *a, **k: "")
        c = _collector(min_search_score=0.3)
        assert await c.collect() == []


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
