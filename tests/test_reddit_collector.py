from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from collectors.reddit import RedditCollector
from shared.config import RedditCollectorConfig
from shared.constants import SourceType


def _config(**kwargs) -> RedditCollectorConfig:
    base = {"subreddits": ["LocalLLaMA"], "sort": "hot", "limit": 5}
    base.update(kwargs)
    cfg = RedditCollectorConfig(**base)
    cfg.reference_time = datetime(2026, 6, 2, tzinfo=UTC)
    cfg.lookback_hours = 24
    return cfg


def _feed(entries):
    class FakeFeed:
        bozo = False
        bozo_exception = None

        def __init__(self, entries):
            self.entries = entries

    return FakeFeed(entries)


class _Entry(dict):
    """Mimics feedparser's FeedParserDict (attribute + key access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _entry(title="Test Post", link="https://www.reddit.com/r/LocalLLaMA/comments/abc123/test/", **kw):
    e = _Entry(
        {
            "title": title,
            "link": link,
            "id": "t3_abc123",
            "summary": "post body",
            "author": "alice",
            "published_parsed": (2026, 6, 2, 0, 0, 0, 0, 0, 0),
        }
    )
    e.update(kw)
    return e


class TestRedditCollect:
    @pytest.mark.asyncio
    async def test_no_subreddits_returns_empty(self):
        collector = RedditCollector(_config(subreddits=[]))
        assert await collector.collect() == []

    @pytest.mark.asyncio
    async def test_collects_via_rss(self):
        collector = RedditCollector(_config())
        with patch("collectors.reddit.feedparser.parse", return_value=_feed([_entry()])):
            items = await collector.collect()
        assert len(items) == 1
        item = items[0]
        assert item.source_type == SourceType.REDDIT
        assert item.metadata["subreddit"] == "LocalLLaMA"
        assert item.item_id == "abc123"
        assert "score" not in item.metadata

    @pytest.mark.asyncio
    async def test_filters_old_posts(self):
        old = _entry(published_parsed=(2026, 5, 1, 0, 0, 0, 0, 0, 0))
        collector = RedditCollector(_config())
        with patch("collectors.reddit.feedparser.parse", return_value=_feed([old])):
            items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    async def test_bozo_feed_returns_empty(self):
        class BadFeed:
            bozo = True
            bozo_exception = Exception("parse error")
            entries: list = []

        collector = RedditCollector(_config())
        with patch("collectors.reddit.feedparser.parse", return_value=BadFeed()):
            items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    async def test_uses_proxied_rss_url(self):
        collector = RedditCollector(_config(sort="top"))
        with patch("collectors.reddit.feedparser.parse", return_value=_feed([])) as mock_parse:
            with patch("collectors.reddit.get_proxied_url", side_effect=lambda u: u) as mock_proxy:
                await collector.collect()
        called_url = mock_proxy.call_args.args[0]
        assert "/r/LocalLLaMA/top/.rss" in called_url
        assert "limit=5" in called_url
        assert mock_parse.called


class TestExtractPostId:
    def test_from_comments_link(self):
        link = "https://www.reddit.com/r/LocalLLaMA/comments/xyz789/title/"
        assert RedditCollector._extract_post_id("t3_xyz789", link) == "xyz789"

    def test_from_entry_id_fallback(self):
        assert RedditCollector._extract_post_id("t3_abc", "https://example.com/no-match") == "abc"

    def test_generated_when_no_id(self):
        out = RedditCollector._extract_post_id("", "https://example.com/x")
        assert out and len(out) == 16
