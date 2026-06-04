from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from collectors.rss import RSSCollector
from shared.config import RSSCollectorConfig
from shared.constants import SourceType


def _config(feeds=None, **kwargs) -> RSSCollectorConfig:
    cfg = RSSCollectorConfig(feeds=feeds or ["https://example.com/feed"], **kwargs)
    cfg.reference_time = datetime(2026, 6, 2, tzinfo=UTC)
    cfg.lookback_hours = 24
    return cfg


class _Feed(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _feed(entries, *, bozo=False, status=200, title="Example"):
    return _Feed(
        entries=entries,
        bozo=bozo,
        bozo_exception=Exception("x") if bozo else None,
        status=status,
        feed=_Feed(title=title),
    )


def _entry():
    return _Feed(
        title="Post",
        link="https://example.com/p/1",
        id="p1",
        summary="body",
        author="alice",
        published_parsed=(2026, 6, 2, 0, 0, 0, 0, 0, 0),
    )


class TestRSSCollect:
    @pytest.mark.asyncio
    async def test_collects_entries(self):
        c = RSSCollector(_config())
        with patch("collectors.rss.feedparser.parse", return_value=_feed([_entry()])):
            items = await c.collect()
        assert len(items) == 1
        assert items[0].source_type == SourceType.RSS
        assert items[0].metadata["feed_url"] == "https://example.com/feed"

    @pytest.mark.asyncio
    async def test_http_error_returns_empty_and_logs(self):
        c = RSSCollector(_config())
        with patch("collectors.rss.feedparser.parse", return_value=_feed([], status=404)):
            with patch("collectors.rss.logger") as log:
                items = await c.collect()
        assert items == []
        assert log.warning.called  # dead feed must be logged, not silently swallowed

    @pytest.mark.asyncio
    async def test_bozo_without_entries_returns_empty_and_logs(self):
        c = RSSCollector(_config())
        with patch("collectors.rss.feedparser.parse", return_value=_feed([], bozo=True, status=200)):
            with patch("collectors.rss.logger") as log:
                items = await c.collect()
        assert items == []
        assert log.warning.called

    @pytest.mark.asyncio
    async def test_bozo_with_entries_still_parses(self):
        # feedparser sets bozo on minor XML issues but still returns entries
        c = RSSCollector(_config())
        with patch("collectors.rss.feedparser.parse", return_value=_feed([_entry()], bozo=True)):
            items = await c.collect()
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_raises_when_all_feeds_fail(self):
        # A total outage (every feed errors) must surface as FAILED, not a silent empty result.
        c = RSSCollector(_config(feeds=["https://a.example/feed", "https://b.example/feed"]))
        with patch("collectors.rss.feedparser.parse", side_effect=OSError("network down")):
            with pytest.raises(RuntimeError):
                await c.collect()
