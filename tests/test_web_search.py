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
    async def test_prefers_extractor_title_for_youtube(self):
        # YouTube /watch?v=ID has no useful slug; the extractor's title must win.
        client = MagicMock()
        client.extract = AsyncMock(
            return_value={
                "results": [
                    {
                        "url": "https://www.youtube.com/watch?v=haK1KoQWm18",
                        "title": "Claude Fable 5 - Full Breakdown",
                        "raw_content": "transcript text",
                    }
                ]
            }
        )
        with patch("collectors.web_search.resolve_secret", return_value="key"):
            with patch("collectors.web_search.AsyncTavilyClient", return_value=client):
                items = await fetch_pinned_items(["https://www.youtube.com/watch?v=haK1KoQWm18"])
        assert items[0].title == "Claude Fable 5 - Full Breakdown"

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
