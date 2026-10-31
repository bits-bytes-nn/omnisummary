from unittest.mock import patch

import pytest

from adapters.collectors.base import ParkedItems, ParkOutcome
from domain.models import CollectedItem, SourceStatus
from shared import Config
from shared.constants import SourceType


def _item(url: str) -> CollectedItem:
    return CollectedItem(source_type=SourceType.RSS, title="t", url=url)


def _sources(tasks, labels, collectors) -> list:
    """(name, collector) pairs whose collect() answers the given coroutine."""
    for task, collector in zip(tasks, collectors, strict=True):
        collector.collect = lambda task=task: task
    return list(zip(labels, collectors, strict=True))


class _Collector:
    """Stand-in for a built collector: run_collectors_with_health only reads park_status and
    degraded_detail off it."""

    def __init__(self, park_status: ParkedItems | None = None, degraded_detail: str = "") -> None:
        self.park_status = park_status
        self.degraded_detail = degraded_detail


@pytest.mark.asyncio
async def test_health_report_classifies_sources():
    from application import digest_pipeline as runner

    async def ok():
        return [_item("http://a.com"), _item("http://b.com")]

    async def empty():
        return []

    async def boom():
        raise RuntimeError("403 blocked")

    tasks = [ok(), empty(), boom()]
    labels = ["rss", "reddit", "youtube"]
    collectors = [_Collector(), _Collector(), _Collector()]
    items, report = await runner.run_collectors_with_health(Config(), _sources(tasks, labels, collectors))

    assert len(items) == 2
    by_name = {s.name: s for s in report.sources}
    assert by_name["rss"].status == SourceStatus.OK
    assert by_name["rss"].item_count == 2
    assert by_name["reddit"].status == SourceStatus.EMPTY
    assert by_name["youtube"].status == SourceStatus.FAILED
    assert "403 blocked" in by_name["youtube"].detail
    assert report.failed_sources == ["youtube"]


@pytest.mark.asyncio
async def test_no_active_collectors_returns_empty_report():
    from application import digest_pipeline as runner

    items, report = await runner.run_collectors_with_health(Config(), _sources([], [], []))

    assert items == []
    assert report.sources == []
    assert report.failed_sources == []


@pytest.mark.asyncio
async def test_stale_park_file_reports_stale_not_ok():
    # A source served from a too-old park file produced items, so the old code called it OK and a
    # dead local cron stayed invisible for days. It must read STALE (and not as a FAILURE).
    from application import digest_pipeline as runner

    async def parked_items():
        return [_item("http://a.com")]

    park = ParkedItems(outcome=ParkOutcome.STALE, detail="park file is 72.0h old (>36h)")
    items, report = await runner.run_collectors_with_health(
        Config(), _sources([parked_items()], ["youtube"], [_Collector(park)])
    )

    assert len(items) == 1
    source = report.sources[0]
    assert source.status == SourceStatus.STALE
    assert source.item_count == 1
    assert "72.0h old" in source.detail
    assert report.failed_sources == []
    assert report.stale_sources == ["youtube"]


@pytest.mark.asyncio
async def test_unreadable_park_file_reports_stale_after_live_fallback():
    # An unreadable park file falls through to live collection; the live items are kept, but the
    # source is still flagged STALE so the broken park read surfaces.
    from application import digest_pipeline as runner

    async def live_items():
        return [_item("http://a.com")]

    park = ParkedItems(outcome=ParkOutcome.ERROR, detail="could not read park file: AccessDenied")
    items, report = await runner.run_collectors_with_health(
        Config(), _sources([live_items()], ["rsshub"], [_Collector(park)])
    )

    assert len(items) == 1
    assert report.sources[0].status == SourceStatus.STALE
    assert report.failed_sources == []


@pytest.mark.asyncio
async def test_fresh_park_file_is_ok():
    from application import digest_pipeline as runner

    async def parked_items():
        return [_item("http://a.com")]

    park = ParkedItems(outcome=ParkOutcome.FRESH)
    _items, report = await runner.run_collectors_with_health(
        Config(), _sources([parked_items()], ["youtube"], [_Collector(park)])
    )

    assert report.sources[0].status == SourceStatus.OK
    assert report.stale_sources == []


@pytest.mark.asyncio
async def test_partially_collected_source_reports_degraded_not_ok():
    # Items arrived, on time, but from a fraction of the source's feeds. That used to read as a
    # healthy OK, which is how X could shrink from 40 accounts to 3 without anyone noticing.
    from application import digest_pipeline as runner

    async def some_items():
        return [_item("http://a.com")]

    collector = _Collector(degraded_detail="30/40 account feeds failed (>50%)")
    items, report = await runner.run_collectors_with_health(Config(), _sources([some_items()], ["rsshub"], [collector]))

    # Reporting only — the items still reach the aggregator untouched.
    assert len(items) == 1
    source = report.sources[0]
    assert source.status == SourceStatus.DEGRADED
    assert source.item_count == 1
    assert "30/40" in source.detail
    assert report.failed_sources == []
    assert report.degraded_sources == ["rsshub"]
    assert report.stale_sources == []


@pytest.mark.asyncio
async def test_reddit_subreddits_missing_from_the_feed_report_degraded_when_armed():
    # All subreddits share ONE request, so a subreddit that vanished (banned, renamed) shows up as
    # an input with no entries — judged by the configured empty-rate verdict.
    from adapters.collectors.reddit import RedditCollector
    from shared.config import RedditCollectorConfig

    class _Feed(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as e:
                raise AttributeError(name) from e

    entries = [
        {"title": sub, "link": f"https://www.reddit.com/r/{sub}/comments/{sub}1/t/", "id": f"t3_{sub}1"}
        for sub in ("a", "b")
    ]
    config = RedditCollectorConfig(
        subreddits=["a", "b", "c", "d", "e", "f"], retry_backoff_sec=0, empty_rate_threshold=50.0
    )
    collector = RedditCollector(config)

    with patch("adapters.collectors.base.fetch_feed", return_value=_Feed(entries=entries, bozo=False)):
        items = await collector.collect()

    # Reporting only: the two healthy subreddits' items still reach the aggregator.
    assert len(items) == 2
    assert "4/6 subreddits returned nothing" in collector.degraded_detail


@pytest.mark.asyncio
async def test_reddit_all_subreddits_answering_is_not_degraded():
    from adapters.collectors.reddit import RedditCollector
    from shared.config import RedditCollectorConfig

    class _Feed(dict):
        def __getattr__(self, name):
            try:
                return self[name]
            except KeyError as e:
                raise AttributeError(name) from e

    collector = RedditCollector(RedditCollectorConfig(subreddits=["a", "b"], retry_backoff_sec=0))
    with patch("adapters.collectors.base.fetch_feed", return_value=_Feed(entries=[], bozo=False)):
        assert await collector.collect() == []

    # A quiet day (every feed answered, nothing new) must NOT be reported as degraded.
    assert collector.degraded_detail == ""


@pytest.mark.asyncio
async def test_stale_wins_over_degraded():
    # A stale park file is the more actionable finding (the sync itself has stopped), so it keeps
    # the report slot rather than being masked by the degradation flag.
    from application import digest_pipeline as runner

    async def parked_items():
        return [_item("http://a.com")]

    park = ParkedItems(outcome=ParkOutcome.STALE, detail="park file is 72.0h old (>36h)")
    collector = _Collector(park, degraded_detail="30/40 account feeds failed")
    _items, report = await runner.run_collectors_with_health(
        Config(), _sources([parked_items()], ["rsshub"], [collector])
    )

    assert report.sources[0].status == SourceStatus.STALE


@pytest.mark.asyncio
async def test_a_collector_over_its_wall_clock_budget_is_reported_failed():
    # Only YouTube bounded itself, and only per channel: the per-input timeouts leave the SOURCE
    # unbounded, and with the shipped RSSHub fan-out the worst case exceeds the digest Lambda's own
    # budget — so one wedged source killed the run instead of failing alone. The timeout must land as
    # FAILED, never as a silent EMPTY (which reads as a legitimately quiet day).
    import asyncio

    from application import digest_pipeline as runner

    async def wedged():
        await asyncio.sleep(60)
        return []

    with pytest.raises(RuntimeError, match="budget"):
        await runner._collect_within_budget(wedged(), "rsshub", budget_sec=0.01)


@pytest.mark.asyncio
async def test_the_budget_comes_from_config_and_zero_leaves_collectors_unbounded():
    from application import digest_pipeline as runner

    async def fine():
        return [_item("http://a.com")]

    config = Config()
    config.collectors.collector_budget_sec = 0
    with patch.object(runner, "_collect_within_budget", wraps=runner._collect_within_budget) as bounded:
        items, report = await runner.run_collectors_with_health(config, _sources([fine()], ["rss"], [_Collector()]))

    assert bounded.call_args.args[2] == 0
    assert report.sources[0].status == SourceStatus.OK
    assert len(items) == 1
