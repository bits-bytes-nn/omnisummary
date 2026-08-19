import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from collectors.base import TransientStatusError
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


def _feed(entries, *, title="Example"):
    return _Feed(entries=entries, bozo=False, feed=_Feed(title=title))


def _entry():
    return _Feed(
        title="Post",
        link="https://example.com/p/1",
        id="p1",
        summary="body",
        author="alice",
        published_parsed=(2026, 6, 2, 0, 0, 0, 0, 0, 0),
    )


def _fetch_by_url(mapping, *, record=None):
    """Stand-in for collectors.base.fetch_feed keyed by URL: an exception value is raised (the
    collector's retry/classification chain still runs on it), anything else is returned."""

    async def _fetch(url, **kwargs):
        if record is not None:
            record.append(url)
        outcome = mapping[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return _fetch


class TestRSSCollect:
    @pytest.mark.asyncio
    async def test_collects_entries(self):
        c = RSSCollector(_config())
        with patch("collectors.base.fetch_feed", return_value=_feed([_entry()])):
            items = await c.collect()
        assert len(items) == 1
        assert items[0].source_type == SourceType.RSS
        assert items[0].metadata["feed_url"] == "https://example.com/feed"
        assert items[0].metadata["feed_title"] == "Example"

    @pytest.mark.asyncio
    async def test_dead_feed_among_healthy_ones_is_skipped_and_logged(self):
        # Partial tolerance: one dead feed loses only its own items, and is logged (not silent).
        c = RSSCollector(_config(feeds=["https://dead.example/feed", "https://ok.example/feed"]))
        fetch = _fetch_by_url(
            {
                "https://dead.example/feed": RuntimeError("RSS feed returned HTTP 404"),
                "https://ok.example/feed": _feed([_entry()]),
            }
        )
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            with patch("collectors.base.logger") as log:
                items = await c.collect()
        assert len(items) == 1
        assert log.warning.called  # dead feed must be logged, not silently swallowed

    @pytest.mark.asyncio
    async def test_all_feeds_http_error_raises(self):
        # Total outage (every feed 4xx/5xx) must be FAILED, not a silently empty digest.
        c = RSSCollector(_config(feeds=["https://a.example/feed", "https://b.example/feed"]))
        with patch("collectors.base.fetch_feed", side_effect=RuntimeError("RSS feed returned HTTP 404")):
            with pytest.raises(RuntimeError, match="returned HTTP 404"):
                await c.collect()

    @pytest.mark.asyncio
    async def test_all_feeds_timing_out_raises(self):
        # A hung feed counts as a failure, so an all-hung run reports FAILED too.
        c = RSSCollector(
            _config(feeds=["https://a.example/feed", "https://b.example/feed"], max_retries=2, retry_backoff_sec=0)
        )
        with patch("collectors.base.fetch_feed", side_effect=TransientStatusError("timed out after 30s")):
            with pytest.raises(RuntimeError, match="timed out"):
                await c.collect()

    @pytest.mark.asyncio
    async def test_one_hung_feed_among_many_is_skipped(self):
        # One feed that never answers (every attempt times out) loses only its own items.
        c = RSSCollector(
            _config(
                feeds=["https://hang.example/feed", "https://ok.example/feed"],
                request_timeout=1,
                max_retries=1,
                retry_backoff_sec=0,
            )
        )
        fetch = _fetch_by_url(
            {
                "https://hang.example/feed": TransientStatusError("RSS feed timed out after 1s"),
                "https://ok.example/feed": _feed([_entry()]),
            }
        )
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            items = await c.collect()
        assert len(items) == 1  # the healthy feed still delivered

    @pytest.mark.asyncio
    async def test_transient_status_is_retried_then_succeeds(self):
        # A single 503 used to lose that feed's whole day of items. It is retried instead, using
        # the shared classification in collectors.base: 429/5xx retry and 403/404 do not.
        c = RSSCollector(_config(max_retries=3, retry_backoff_sec=0))
        outcomes = [TransientStatusError("returned HTTP 503"), _feed([_entry()])]

        async def _fetch(url, **kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("collectors.base.fetch_feed", side_effect=_fetch):
            items = await c.collect()
        assert len(items) == 1
        assert outcomes == []  # both the failed attempt and the retry ran

    @pytest.mark.asyncio
    async def test_transient_status_exhausted_still_fails_the_source(self):
        # Retries must not paper over a real outage: an exhausted chain still raises, so an
        # all-feeds-down run reports FAILED rather than a silent empty digest.
        c = RSSCollector(
            _config(feeds=["https://a.example/feed", "https://b.example/feed"], max_retries=2, retry_backoff_sec=0)
        )
        with patch("collectors.base.fetch_feed", side_effect=TransientStatusError("returned HTTP 503")) as fetch:
            with pytest.raises(RuntimeError, match="returned HTTP 503"):
                await c.collect()
        assert fetch.call_count == 4  # 2 feeds x 2 attempts

    @pytest.mark.asyncio
    async def test_permanent_status_is_not_retried(self):
        c = RSSCollector(_config(max_retries=3, retry_backoff_sec=0))
        with patch("collectors.base.fetch_feed", side_effect=RuntimeError("returned HTTP 403")) as fetch:
            with pytest.raises(RuntimeError, match="returned HTTP 403"):
                await c.collect()
        assert fetch.call_count == 1  # a verdict, not a blip

    @pytest.mark.asyncio
    async def test_every_attempt_carries_the_configured_timeout(self):
        # The retry wraps the whole fetch, so every attempt gets its own request_timeout instead of
        # sharing one budget; a feed that answers on the second try still delivers.
        c = RSSCollector(_config(max_retries=3, retry_backoff_sec=0, request_timeout=11))
        timeouts: list[float] = []
        outcomes = [TransientStatusError("returned HTTP 503"), _feed([_entry()])]

        async def _fetch(url, *, description, timeout):
            timeouts.append(timeout)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("collectors.base.fetch_feed", side_effect=_fetch):
            items = await c.collect()
        assert len(items) == 1
        assert timeouts == [11, 11]

    @pytest.mark.asyncio
    async def test_fan_out_is_bounded_by_config(self):
        # Unbounded fan-out let a feed's timeout expire while its fetch was still QUEUED behind
        # other feeds, turning a healthy feed into a bogus FAILURE.
        feeds = [f"https://f{n}.example/feed" for n in range(8)]
        c = RSSCollector(_config(feeds=feeds, max_concurrency=2))
        in_flight = 0
        peak = 0

        async def _fetch(url, **kwargs):
            nonlocal in_flight, peak
            in_flight += 1
            peak = max(peak, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _feed([_entry()])

        with patch("collectors.base.fetch_feed", side_effect=_fetch):
            items = await c.collect()
        assert len(items) == 8  # every feed still collected
        assert peak <= 2, f"fan-out exceeded max_concurrency: peak={peak}"

    @pytest.mark.asyncio
    async def test_raises_when_all_feeds_fail(self):
        # A total outage (every feed errors) must surface as FAILED, not a silent empty result.
        c = RSSCollector(_config(feeds=["https://a.example/feed", "https://b.example/feed"]))
        with patch("collectors.base.fetch_feed", side_effect=OSError("network down")):
            with pytest.raises(RuntimeError):
                await c.collect()

    @pytest.mark.asyncio
    async def test_mostly_empty_feeds_report_degraded_when_configured(self):
        # Every feed answers 200 with no entries but one: no failure rate trips, and the source used
        # to report a clean OK.
        feeds = [f"https://f{n}.example/feed" for n in range(4)]
        c = RSSCollector(_config(feeds=feeds, empty_rate_threshold=50.0))
        mapping = {url: _feed([]) for url in feeds}
        mapping[feeds[0]] = _feed([_entry()])
        with patch("collectors.base.fetch_feed", side_effect=_fetch_by_url(mapping)):
            items = await c.collect()
        assert len(items) == 1
        assert "3/4 feeds returned nothing" in c.degraded_detail
