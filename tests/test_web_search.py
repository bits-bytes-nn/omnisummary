from datetime import UTC, datetime

from collectors.web_search import WebSearchCollector
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
