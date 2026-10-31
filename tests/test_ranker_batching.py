from unittest.mock import MagicMock

import pytest

from application.ranker import ContentRanker
from shared.config import PipelineConfig
from shared.constants import SourceType
from shared.models import CollectedItem


def _factory() -> MagicMock:
    factory = MagicMock()
    factory.truncate_to_tokens.side_effect = lambda text, max_tokens: text
    return factory


def _items(count: int, *, text: str = "body") -> list[CollectedItem]:
    return [
        CollectedItem(
            item_id=f"i{n}",
            source_type=SourceType.RSS,
            title=f"t{n}",
            url=f"http://e.com/{n}",
            text=text,
        )
        for n in range(count)
    ]


class TestBatchCaps:
    def test_batches_are_capped_by_item_count(self):
        ranker = ContentRanker(PipelineConfig(ranking_batch_size=3), _factory())
        batches = ranker._make_batches(_items(7))
        assert [len(b) for b in batches] == [3, 3, 1]

    def test_batching_order_is_preserved(self):
        ranker = ContentRanker(PipelineConfig(ranking_batch_size=2), _factory())
        batches = ranker._make_batches(_items(5))
        assert [[i.item_id for i in b] for b in batches] == [["i0", "i1"], ["i2", "i3"], ["i4"]]


@pytest.mark.asyncio
async def test_batches_mix_sources_even_when_ids_cluster_by_source():
    # item_id is the feed GUID, so ids group by source ("https://news.hada.io/…", "https://x.com/…");
    # sorting on the raw id put one source per batch. Each batch must see several sources.
    items = [
        CollectedItem(
            item_id=f"https://{host}/{i:03d}",
            source_type=src,
            title=f"{src.value} {i}",
            url=f"https://{host}/{i:03d}",
            text="body",
        )
        for host, src in (("news.hada.io", SourceType.RSS), ("x.com", SourceType.X))
        for i in range(40)
    ]
    ranker = ContentRanker(PipelineConfig(ranking_batch_size=40), _factory())
    captured: list[list[CollectedItem]] = []

    async def _capture(batch, semaphore, recent_titles):
        captured.append(batch)
        return []

    ranker._rank_batch = _capture  # type: ignore[method-assign]
    await ranker.rank(items, select_count=8)
    assert len(captured) == 2
    assert all({it.source_type for it in batch} == {SourceType.RSS, SourceType.X} for batch in captured)
