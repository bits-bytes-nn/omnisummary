import pytest

from collectors.base import gather_collector_results
from shared.constants import SourceType
from shared.models import CollectedItem


def _item(item_id: str) -> CollectedItem:
    return CollectedItem(item_id=item_id, source_type=SourceType.WEB, title="t", url=f"http://e.com/{item_id}")


async def _ok(item_id: str) -> list[CollectedItem]:
    return [_item(item_id)]


async def _fail() -> list[CollectedItem]:
    raise RuntimeError("boom")


class TestGatherCollectorResults:
    @pytest.mark.asyncio
    async def test_partial_failure_passes_through(self):
        items = await gather_collector_results([_ok("a"), _fail()], raise_if_all_failed=True)
        assert {i.item_id for i in items} == {"a"}

    @pytest.mark.asyncio
    async def test_all_failed_raises_when_flagged(self):
        with pytest.raises(RuntimeError, match="All 2 collector tasks failed"):
            await gather_collector_results([_fail(), _fail()], raise_if_all_failed=True)

    @pytest.mark.asyncio
    async def test_all_failed_silent_when_not_flagged(self):
        items = await gather_collector_results([_fail(), _fail()])
        assert items == []

    @pytest.mark.asyncio
    async def test_no_tasks_does_not_raise(self):
        items = await gather_collector_results([], raise_if_all_failed=True)
        assert items == []
