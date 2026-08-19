import asyncio
from unittest.mock import MagicMock

import pytest

from pipeline.ranker import ContentRanker
from shared.config import PipelineConfig
from shared.constants import SourceType
from shared.models import CollectedItem


def _factory(*, context_window: int | None = 200_000, on_count=None) -> MagicMock:
    """A factory whose token helpers behave like the real (SYNC, boto3-backed) ones: count_tokens
    ~= chars/4, truncate is a no-op. `on_count` observes each call, e.g. to prove concurrency."""
    factory = MagicMock()

    def _count(text: str) -> int:
        if on_count is not None:
            on_count(text)
        return len(text) // 4

    factory.count_tokens.side_effect = _count
    factory.truncate_to_tokens.side_effect = lambda text, max_tokens: text
    factory.get_model_info.return_value = MagicMock(context_window_size=context_window) if context_window else None
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
    @pytest.mark.asyncio
    async def test_batches_are_capped_by_item_count(self):
        ranker = ContentRanker(PipelineConfig(ranking_batch_size=3), _factory())
        batches = await ranker._make_batches(_items(7))
        assert [len(b) for b in batches] == [3, 3, 1]

    @pytest.mark.asyncio
    async def test_batches_are_capped_by_the_token_budget(self):
        # 400-char texts -> 100 tokens each; a 1000-token window at ratio 0.5 budgets 500 tokens,
        # so five items fill a batch regardless of the (larger) count cap.
        ranker = ContentRanker(
            PipelineConfig(
                ranking_batch_size=40,
                ranking_batch_token_budget_ratio=0.5,
            ),
            _factory(context_window=1000),
        )
        batches = await ranker._make_batches(_items(11, text="x" * 400))
        assert [len(b) for b in batches] == [5, 5, 1]

    @pytest.mark.asyncio
    async def test_the_budget_ratio_and_window_fallback_come_from_config(self):
        # No registry entry for the model: the configured fallback window still bounds the batch.
        ranker = ContentRanker(
            PipelineConfig(
                ranking_batch_size=40, ranking_context_window_fallback=800, ranking_batch_token_budget_ratio=0.5
            ),
            _factory(context_window=None),
        )
        batches = await ranker._make_batches(_items(9, text="x" * 400))
        assert [len(b) for b in batches] == [4, 4, 1]

    @pytest.mark.asyncio
    async def test_a_single_oversized_item_still_gets_its_own_batch(self):
        ranker = ContentRanker(
            PipelineConfig(ranking_batch_size=40, ranking_batch_token_budget_ratio=0.5),
            _factory(context_window=100),
        )
        batches = await ranker._make_batches(_items(3, text="x" * 4000))
        assert [len(b) for b in batches] == [1, 1, 1]

    @pytest.mark.asyncio
    async def test_batching_order_is_preserved(self):
        ranker = ContentRanker(PipelineConfig(ranking_batch_size=2), _factory())
        batches = await ranker._make_batches(_items(5))
        assert [[i.item_id for i in b] for b in batches] == [["i0", "i1"], ["i2", "i3"], ["i4"]]


class TestBatchingDoesNotBlockTheLoop:
    """count_tokens/truncate_to_tokens are sync boto3 CountTokens round-trips. Measuring the day's
    ~90 candidates one at a time from the async caller blocked the event loop for 90+ serialized
    calls before the first Converse request left the process."""

    @pytest.mark.asyncio
    async def test_counts_are_measured_off_the_event_loop(self):
        loop_ticks = 0

        async def _tick() -> None:
            nonlocal loop_ticks
            while True:
                loop_ticks += 1
                await asyncio.sleep(0)

        def _slow_count(_text: str) -> None:
            import time

            time.sleep(0.01)  # stands in for a CountTokens round-trip

        ranker = ContentRanker(PipelineConfig(ranking_batch_size=40), _factory(on_count=_slow_count))
        ticker = asyncio.ensure_future(_tick())
        try:
            batches = await ranker._make_batches(_items(12))
        finally:
            ticker.cancel()

        assert [len(b) for b in batches] == [12]
        # The loop kept running while the counting happened; the old fully-synchronous version
        # never yielded at all, so the ticker could not advance even once.
        assert loop_ticks > 10
