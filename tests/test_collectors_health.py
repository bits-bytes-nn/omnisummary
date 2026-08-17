from unittest.mock import patch

import pytest

from collectors.base import ParkedItems, ParkOutcome
from shared.constants import SourceType
from shared.models import CollectedItem, SourceStatus


def _item(url: str) -> CollectedItem:
    return CollectedItem(source_type=SourceType.RSS, title="t", url=url)


class _Collector:
    """Stand-in for a built collector: run_collectors_with_health only reads park_status off it."""

    def __init__(self, park_status: ParkedItems | None = None) -> None:
        self.park_status = park_status


@pytest.mark.asyncio
async def test_health_report_classifies_sources():
    import main

    async def ok():
        return [_item("http://a.com"), _item("http://b.com")]

    async def empty():
        return []

    async def boom():
        raise RuntimeError("403 blocked")

    tasks = [ok(), empty(), boom()]
    labels = ["rss", "reddit", "youtube"]
    collectors = [_Collector(), _Collector(), _Collector()]
    with patch.object(main, "_build_collector_tasks", return_value=(tasks, labels, collectors)):
        items, report = await main.run_collectors_with_health(config=None, llm_factory=None)

    assert len(items) == 2
    by_name = {s.name: s for s in report.sources}
    assert by_name["rss"].status == SourceStatus.OK
    assert by_name["rss"].item_count == 2
    assert by_name["reddit"].status == SourceStatus.EMPTY
    assert by_name["youtube"].status == SourceStatus.FAILED
    assert "403 blocked" in by_name["youtube"].detail
    assert report.has_failures is True


@pytest.mark.asyncio
async def test_no_active_collectors_returns_empty_report():
    import main

    with patch.object(main, "_build_collector_tasks", return_value=([], [], [])):
        items, report = await main.run_collectors_with_health(config=None, llm_factory=None)

    assert items == []
    assert report.sources == []
    assert report.has_failures is False


@pytest.mark.asyncio
async def test_stale_park_file_reports_stale_not_ok():
    # A source served from a too-old park file produced items, so the old code called it OK and a
    # dead local cron stayed invisible for days. It must read STALE (and not as a FAILURE).
    import main

    async def parked_items():
        return [_item("http://a.com")]

    park = ParkedItems(outcome=ParkOutcome.STALE, age_hours=72.0, detail="park file is 72.0h old (>36h)")
    with patch.object(main, "_build_collector_tasks", return_value=([parked_items()], ["youtube"], [_Collector(park)])):
        items, report = await main.run_collectors_with_health(config=None, llm_factory=None)

    assert len(items) == 1
    source = report.sources[0]
    assert source.status == SourceStatus.STALE
    assert source.item_count == 1
    assert "72.0h old" in source.detail
    assert report.has_failures is False
    assert report.stale_sources == ["youtube"]


@pytest.mark.asyncio
async def test_unreadable_park_file_reports_stale_after_live_fallback():
    # An unreadable park file falls through to live collection; the live items are kept, but the
    # source is still flagged STALE so the broken park read surfaces.
    import main

    async def live_items():
        return [_item("http://a.com")]

    park = ParkedItems(outcome=ParkOutcome.ERROR, detail="could not read park file: AccessDenied")
    with patch.object(main, "_build_collector_tasks", return_value=([live_items()], ["rsshub"], [_Collector(park)])):
        items, report = await main.run_collectors_with_health(config=None, llm_factory=None)

    assert len(items) == 1
    assert report.sources[0].status == SourceStatus.STALE
    assert report.has_failures is False


@pytest.mark.asyncio
async def test_fresh_park_file_is_ok():
    import main

    async def parked_items():
        return [_item("http://a.com")]

    park = ParkedItems(outcome=ParkOutcome.FRESH, age_hours=1.0)
    with patch.object(main, "_build_collector_tasks", return_value=([parked_items()], ["youtube"], [_Collector(park)])):
        _items, report = await main.run_collectors_with_health(config=None, llm_factory=None)

    assert report.sources[0].status == SourceStatus.OK
    assert report.stale_sources == []
