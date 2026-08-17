import threading
import time
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


def _parse_by_url(mapping):
    """feedparser stand-in keyed by URL — the feeds are parsed in worker threads, so a plain
    side_effect LIST would be consumed in nondeterministic order."""

    def _parse(url):
        return mapping[url]

    return _parse


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
    async def test_dead_feed_among_healthy_ones_is_skipped_and_logged(self):
        # Partial tolerance: one dead feed loses only its own items, and is logged (not silent).
        c = RSSCollector(_config(feeds=["https://dead.example/feed", "https://ok.example/feed"]))
        parse = _parse_by_url(
            {
                "https://dead.example/feed": _feed([], status=404),
                "https://ok.example/feed": _feed([_entry()]),
            }
        )
        with patch("collectors.rss.feedparser.parse", side_effect=parse):
            with patch("collectors.base.logger") as log:
                items = await c.collect()
        assert len(items) == 1
        assert log.warning.called  # dead feed must be logged, not silently swallowed

    @pytest.mark.asyncio
    async def test_all_feeds_http_error_raises(self):
        # Total outage (every feed 4xx/5xx) must be FAILED, not a silently empty digest.
        c = RSSCollector(_config(feeds=["https://a.example/feed", "https://b.example/feed"]))
        with patch("collectors.rss.feedparser.parse", return_value=_feed([], status=503)):
            with pytest.raises(RuntimeError, match="Failed RSS feed"):
                await c.collect()

    @pytest.mark.asyncio
    async def test_all_feeds_unparseable_raises(self):
        c = RSSCollector(_config(feeds=["https://a.example/feed", "https://b.example/feed"]))
        with patch("collectors.rss.feedparser.parse", return_value=_feed([], bozo=True, status=200)):
            with pytest.raises(RuntimeError, match="Failed RSS feed"):
                await c.collect()

    @pytest.mark.asyncio
    async def test_all_feeds_timing_out_raises(self):
        # A hung feed counts as a failure, so an all-hung run reports FAILED too.
        c = RSSCollector(_config(feeds=["https://a.example/feed", "https://b.example/feed"]))

        async def _timeout(awaitable, timeout):
            awaitable.close()  # avoid an un-awaited to_thread coroutine warning
            raise TimeoutError

        with patch("collectors.rss.asyncio.wait_for", side_effect=_timeout):
            with pytest.raises(RuntimeError, match="timed out"):
                await c.collect()

    @pytest.mark.asyncio
    async def test_one_hung_feed_among_many_is_skipped(self):
        c = RSSCollector(_config(feeds=["https://hang.example/feed", "https://ok.example/feed"]))
        calls: list[str] = []

        async def _timeout_first(awaitable, timeout):
            # wait_for is called once per feed, in feed order.
            if len(calls) == 0:
                calls.append("hang")
                awaitable.close()
                raise TimeoutError
            calls.append("ok")
            return await awaitable

        with patch("collectors.rss.feedparser.parse", return_value=_feed([_entry()])):
            with patch("collectors.rss.asyncio.wait_for", side_effect=_timeout_first):
                items = await c.collect()
        assert len(items) == 1  # the healthy feed still delivered

    @pytest.mark.asyncio
    async def test_bozo_with_entries_still_parses(self):
        # feedparser sets bozo on minor XML issues but still returns entries
        c = RSSCollector(_config())
        with patch("collectors.rss.feedparser.parse", return_value=_feed([_entry()], bozo=True)):
            items = await c.collect()
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_fan_out_is_bounded_by_config(self):
        # Every feed's feedparser.parse parks a worker thread; unbounded fan-out let a feed's
        # timeout expire while its parse was still QUEUED, turning a healthy feed into a FAILURE.
        feeds = [f"https://f{n}.example/feed" for n in range(8)]
        c = RSSCollector(_config(feeds=feeds, max_concurrency=2))
        lock = threading.Lock()
        in_flight = 0
        peak = 0

        def _parse(url):
            nonlocal in_flight, peak
            with lock:
                in_flight += 1
                peak = max(peak, in_flight)
            time.sleep(0.02)
            with lock:
                in_flight -= 1
            return _feed([_entry()])

        with patch("collectors.rss.feedparser.parse", side_effect=_parse):
            items = await c.collect()
        assert len(items) == 8  # every feed still collected
        assert peak <= 2, f"fan-out exceeded max_concurrency: peak={peak}"

    @pytest.mark.asyncio
    async def test_raises_when_all_feeds_fail(self):
        # A total outage (every feed errors) must surface as FAILED, not a silent empty result.
        c = RSSCollector(_config(feeds=["https://a.example/feed", "https://b.example/feed"]))
        with patch("collectors.rss.feedparser.parse", side_effect=OSError("network down")):
            with pytest.raises(RuntimeError):
                await c.collect()
