import json
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pipeline.ranker import ContentRanker
from shared.config import PipelineConfig
from shared.constants import SourceType
from shared.models import CollectedItem


def _ranker(raw_output: str, **overrides) -> ContentRanker:
    config = PipelineConfig(**overrides)
    factory = MagicMock()
    # The ranker builds: RankingPrompt.get_prompt() | self.llm | StrOutputParser().
    # A RunnableLambda standing in for the LLM returns an AIMessage that the
    # StrOutputParser unwraps to raw_output, exercising the real rank() path.
    factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=raw_output))
    return ContentRanker(config, factory)


def _items(specs: list[tuple[str, SourceType]]) -> list[CollectedItem]:
    return [
        CollectedItem(item_id=item_id, source_type=src, title=f"t-{item_id}", url=f"http://e.com/{item_id}")
        for item_id, src in specs
    ]


def _rankings(scores: dict[str, float]) -> str:
    return json.dumps({"rankings": [{"item_id": k, "score": v} for k, v in scores.items()]})


class TestRankEndToEnd:
    @pytest.mark.asyncio
    async def test_empty_input_short_circuits(self):
        ranker = _ranker("", top_n=5, min_score=0.6)
        assert await ranker.rank([]) == []

    @pytest.mark.asyncio
    async def test_min_score_filter_applied(self):
        items = _items([("a", SourceType.RSS), ("b", SourceType.RSS), ("c", SourceType.RSS)])
        ranker = _ranker(_rankings({"a": 0.9, "b": 0.55, "c": 0.7}), top_n=5, min_score=0.6, source_slots={})
        result = await ranker.rank(items)
        ids = {r.item.item_id for r in result}
        assert ids == {"a", "c"}  # b (0.55) filtered out

    @pytest.mark.asyncio
    async def test_llm_failure_returns_empty(self):
        items = _items([("a", SourceType.RSS)])
        ranker = _ranker("garbage not json", top_n=5, min_score=0.6)
        assert await ranker.rank(items) == []

    @pytest.mark.asyncio
    async def test_result_never_exceeds_top_n(self):
        items = _items([(f"i{n}", SourceType.RSS) for n in range(8)])
        # source_slots sum (web+x+rss+reddit+youtube = 5) > top_n 3: must still cap at 3.
        ranker = _ranker(
            _rankings({f"i{n}": 0.9 - n * 0.01 for n in range(8)}),
            top_n=3,
            min_score=0.6,
            source_slots={"web": 1, "x": 1, "rss": 1, "reddit": 1, "youtube": 1},
            source_cap_multiplier=5,
        )
        result = await ranker.rank(items)
        assert len(result) == 3

    @pytest.mark.asyncio
    async def test_pinned_item_force_included_below_min_score(self):
        # A pinned item must appear even when its score is below min_score and it would
        # otherwise be filtered out.
        items = _items([("a", SourceType.RSS), ("b", SourceType.RSS)])
        items[1].metadata = {"pinned": True}  # b is pinned
        ranker = _ranker(_rankings({"a": 0.9, "b": 0.2}), top_n=5, min_score=0.6, source_slots={})
        result = await ranker.rank(items)
        ids = {r.item.item_id for r in result}
        assert "b" in ids  # pinned, despite 0.2 < 0.6
        assert "a" in ids

    @pytest.mark.asyncio
    async def test_pinned_item_leads_and_respects_top_n(self):
        # Pinned items lead the result and the total still caps at top_n.
        items = _items([(f"i{n}", SourceType.RSS) for n in range(5)])
        items[4].metadata = {"pinned": True}  # i4 pinned, lowest score
        ranker = _ranker(
            _rankings({"i0": 0.95, "i1": 0.9, "i2": 0.85, "i3": 0.8, "i4": 0.3}),
            top_n=3,
            min_score=0.6,
            source_slots={},
        )
        result = await ranker.rank(items)
        assert len(result) == 3
        assert result[0].item.item_id == "i4"  # pinned leads
        assert "i4" in {r.item.item_id for r in result}

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_desc(self):
        items = _items([("a", SourceType.RSS), ("b", SourceType.REDDIT), ("c", SourceType.WEB)])
        ranker = _ranker(_rankings({"a": 0.7, "b": 0.95, "c": 0.8}), top_n=5, min_score=0.6)
        result = await ranker.rank(items)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_ranking_categories_reach_prompt(self):
        items = _items([("a", SourceType.RSS)])
        captured: dict[str, str] = {}

        def capture(prompt_value):
            captured["text"] = str(prompt_value)
            return AIMessage(content=_rankings({"a": 0.9}))

        config = PipelineConfig(top_n=5, min_score=0.6, ranking_categories=["alpha", "beta", "gamma"])
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(capture)
        ranker = ContentRanker(config, factory)
        await ranker.rank(items)

        assert "alpha, beta, gamma" in captured["text"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("penalty", [0.0, 0.25, 0.5])
    async def test_duplicate_score_penalty_reaches_prompt(self, penalty):
        items = _items([("a", SourceType.RSS)])
        captured: dict[str, str] = {}

        def capture(prompt_value):
            captured["text"] = str(prompt_value)
            return AIMessage(content=_rankings({"a": 0.9}))

        config = PipelineConfig(top_n=5, min_score=0.6, ranking_duplicate_score_penalty=penalty)
        factory = MagicMock()
        factory.get_model.return_value = RunnableLambda(capture)
        ranker = ContentRanker(config, factory)
        await ranker.rank(items)

        assert str(penalty) in captured["text"]

    @pytest.mark.asyncio
    async def test_parallel_batches_merge_all_items(self):
        import re

        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        items = _items([(f"i{n}", SourceType.RSS) for n in range(10)])
        config = PipelineConfig(top_n=20, min_score=0.6, source_slots={}, ranking_batch_size=3)
        factory = MagicMock()

        # Each batch's mock scores exactly the item_ids present in that batch's prompt,
        # so a correct merge yields all 10 (4 batches: 3+3+3+1).
        def score_batch(prompt_value):
            text = str(prompt_value)
            ids = re.findall(r"ID: (i\d+)", text)
            return AIMessage(content=_rankings(dict.fromkeys(ids, 0.8)))

        factory.get_model.return_value = RunnableLambda(score_batch)
        ranker = ContentRanker(config, factory)
        result = await ranker.rank(items)
        assert {r.item.item_id for r in result} == {f"i{n}" for n in range(10)}


class TestGraceIntegration:
    @pytest.mark.asyncio
    async def test_grace_item_reaches_final_selection(self):
        # A source with a guaranteed slot but only a sub-threshold best item (within grace)
        # still lands in the digest via its own slot, end-to-end through rank().
        items = _items([("r1", SourceType.RSS), ("y1", SourceType.YOUTUBE)])
        ranker = _ranker(
            _rankings({"r1": 0.80, "y1": 0.55}),
            top_n=5,
            min_score=0.6,
            source_slot_score_grace=0.1,
            source_slots={"rss": 1, "youtube": 1},
        )
        result = await ranker.rank(items)
        assert {r.item.item_id for r in result} == {"r1", "y1"}

    @pytest.mark.asyncio
    async def test_grace_item_does_not_fill_fallback_slots(self):
        # Quiet day: one strong RSS item + sub-threshold grace items in two slotted sources.
        # Grace items take ONLY their own guaranteed slot — they must NOT be pulled into the
        # relaxed fallback fill to pad the digest toward top_n.
        items = _items([("r1", SourceType.RSS), ("y1", SourceType.YOUTUBE), ("x1", SourceType.X)])
        ranker = _ranker(
            _rankings({"r1": 0.90, "y1": 0.55, "x1": 0.55}),
            top_n=5,
            min_score=0.6,
            source_slot_score_grace=0.1,
            source_slots={"rss": 1, "youtube": 1, "x": 1},
            source_cap_multiplier=3,  # fallback could otherwise pull extra items per source
        )
        result = await ranker.rank(items)
        # Each appears exactly once via its own slot; no duplication / fallback padding.
        ids = sorted(r.item.item_id for r in result)
        assert ids == ["r1", "x1", "y1"]
