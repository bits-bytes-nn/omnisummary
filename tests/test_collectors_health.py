from unittest.mock import patch

import pytest

from shared.constants import SourceType
from shared.models import CollectedItem, SourceStatus


def _item(url: str) -> CollectedItem:
    return CollectedItem(source_type=SourceType.RSS, title="t", url=url)


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
    with patch.object(main, "_build_collector_tasks", return_value=(tasks, labels)):
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

    with patch.object(main, "_build_collector_tasks", return_value=([], [])):
        items, report = await main.run_collectors_with_health(config=None, llm_factory=None)

    assert items == []
    assert report.sources == []
    assert report.has_failures is False
