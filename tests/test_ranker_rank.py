import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from pipeline.ranker import ContentRanker
from shared.config import PipelineConfig
from shared.constants import SourceType
from shared.models import CollectedItem


def _mock_factory() -> MagicMock:
    """A mock LLM factory whose truncate is a no-op."""
    factory = MagicMock()
    factory.truncate_to_tokens.side_effect = lambda text, max_tokens: text
    return factory


def _ranker(raw_output: str, **overrides) -> ContentRanker:
    config = PipelineConfig(**overrides)
    factory = _mock_factory()
    # The ranker builds: RankingPrompt.get_prompt() | self.llm | StrOutputParser().
    # A RunnableLambda standing in for the LLM returns an AIMessage that the
    # StrOutputParser unwraps to raw_output, exercising the real rank() path.
    factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=raw_output))
    return ContentRanker(config, factory)


def _items(specs: list[tuple[str, SourceType]]) -> list[CollectedItem]:
    # A DISTINCT host per item: resolve_origin_key falls back to the host when a source's own origin
    # metadata is absent, so a shared host would put every fixture item in one origin and let
    # max_per_origin (1 by default) decide these selection tests instead of the behaviour under test.
    return [
        CollectedItem(item_id=item_id, source_type=src, title=f"t-{item_id}", url=f"http://{item_id}.example/x")
        for item_id, src in specs
    ]


def _rankings(scores: dict[str, float]) -> str:
    return json.dumps({"rankings": [{"item_id": k, "score": v} for k, v in scores.items()]})


class TestRankEndToEnd:
    @pytest.mark.asyncio
    async def test_recent_digest_titles_reach_the_ranking_prompt(self):
        # The URL ledger cannot match a story that returns under a new URL; without the titles the
        # Novelty criterion had nothing to judge a rehash against.
        ranker, seen = _ranker_with_outputs([_rankings({"a": 0.9})], source_slots={})
        await ranker.rank(_items([("a", SourceType.RSS)]), recent_titles=["Claude Haiku 5.5 출시"])
        assert "- Claude Haiku 5.5 출시" in seen[0]

    @pytest.mark.asyncio
    async def test_formatting_a_batch_does_not_block_the_event_loop(self):
        # truncate_to_tokens makes blocking boto3 CountTokens calls; formatting on the loop serialized
        # the "parallel" batches (~30s of a 67s ranking phase on 2026-10-12).
        import asyncio
        import time

        ranker, _ = _ranker_with_outputs([_rankings({"a": 0.9})], source_slots={})
        ticks = 0
        advanced_during_truncation: list[bool] = []

        def _slow_truncate(text: str, max_tokens: int) -> str:
            before = ticks
            time.sleep(0.05)
            advanced_during_truncation.append(ticks > before)
            return text

        ranker.llm_factory.truncate_to_tokens.side_effect = _slow_truncate

        async def _tick() -> None:
            nonlocal ticks
            while True:
                ticks += 1
                await asyncio.sleep(0)

        ticker = asyncio.ensure_future(_tick())
        try:
            await ranker.rank(_items([("a", SourceType.RSS)]))
        finally:
            ticker.cancel()
        assert advanced_during_truncation and all(advanced_during_truncation)

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
    async def test_unparseable_llm_output_fails_the_run_instead_of_emptying_it(self):
        # A response that never parses (even after the coverage re-ask) used to return [] with every
        # health counter at zero, so the digest published off a shortened pool and every ranking
        # alert stayed silent. With a single batch that is a total outage, so it raises.
        items = _items([("a", SourceType.RSS)])
        ranker = _ranker("garbage not json", top_n=5, min_score=0.6)
        with pytest.raises(RuntimeError, match="ranking batches failed"):
            await ranker.rank(items)

    @pytest.mark.asyncio
    async def test_result_never_exceeds_top_n(self):
        items = _items([(f"i{n}", SourceType.RSS) for n in range(8)])
        # source_slots sum (web+x+rss+reddit+youtube = 5) > top_n 3: must still cap at 3.
        ranker = _ranker(
            _rankings({f"i{n}": 0.9 - n * 0.01 for n in range(8)}),
            top_n=3,
            min_score=0.6,
            source_slots={"web": 1, "x": 1, "rss": 1, "reddit": 1, "youtube": 1},
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
    async def test_pinned_item_force_included_even_when_ranker_omits_it(self):
        # The ranking LLM dropped the pinned item's id entirely (hallucinated it away, or its
        # batch failed). It never became a RankedItem, but the --pin-url guarantee must still
        # hold: it's synthesized at min_score and force-included.
        items = _items([("a", SourceType.RSS), ("b", SourceType.RSS)])
        items[1].metadata = {"pinned": True}  # b is pinned but the LLM only scores a
        ranker = _ranker(_rankings({"a": 0.9}), top_n=5, min_score=0.6, source_slots={})
        result = await ranker.rank(items)
        ids = {r.item.item_id for r in result}
        assert "b" in ids  # synthesized despite the ranker never scoring it
        b = next(r for r in result if r.item.item_id == "b")
        assert b.score == 0.6  # min_score

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
    async def test_pinned_item_appears_once_and_sub_threshold_items_stay_out(self):
        items = _items([("a", SourceType.RSS), ("y_pin", SourceType.YOUTUBE), ("y_weak", SourceType.YOUTUBE)])
        items[1].metadata = {"pinned": True}
        ranker = _ranker(
            _rankings({"a": 0.9, "y_pin": 0.85, "y_weak": 0.55}),
            top_n=5,
            min_score=0.6,
            source_slots={"web": 1, "x": 1, "rss": 1, "reddit": 1, "youtube": 1},
        )
        result = await ranker.rank(items)
        ids = [r.item.item_id for r in result]
        assert ids.count("y_pin") == 1
        assert "y_weak" not in ids

    @pytest.mark.asyncio
    async def test_pinned_web_item_still_fills_to_top_n_via_relaxed_pass(self):
        # Counting the pin's origin must not starve the digest: when every remaining candidate
        # shares the pin's host, the relaxed final pass (per-origin cap off)
        # still fills to top_n.
        items = _items([("p", SourceType.WEB), ("w1", SourceType.WEB), ("w2", SourceType.WEB)])
        items[0].metadata = {"pinned": True}
        ranker = _ranker(
            _rankings({"p": 0.9, "w1": 0.85, "w2": 0.8}),
            top_n=3,
            min_score=0.6,
            source_slots={"web": 1},
            max_per_origin=1,
        )
        result = await ranker.rank(items)
        assert len(result) == 3
        assert result[0].item.item_id == "p"  # pinned still leads

    @pytest.mark.asyncio
    async def test_pin_consumes_its_own_source_slot(self):
        # rank() reserves room for the pin and seeds the slot counters with it, but the guaranteed-
        # slot pass restarted at 0 — so a pinned web item yielded TWO web stories and pushed the
        # lowest-scoring source out of the core, breaking the diversity guarantee on exactly the
        # days the operator intervened. Distinct hosts keep max_per_origin out of the way.
        pin = CollectedItem(
            item_id="p",
            source_type=SourceType.WEB,
            title="t-p",
            url="https://pinned.example/a",
            metadata={"pinned": True},
        )
        web = CollectedItem(item_id="w1", source_type=SourceType.WEB, title="t-w1", url="https://other.example/a")
        rest = _items(
            [("x1", SourceType.X), ("r1", SourceType.RSS), ("rd1", SourceType.REDDIT), ("y1", SourceType.YOUTUBE)]
        )
        ranker = _ranker(
            _rankings({"p": 0.7, "w1": 0.95, "x1": 0.9, "r1": 0.85, "rd1": 0.8, "y1": 0.75}),
            top_n=5,
            min_score=0.6,
            source_slots={"web": 1, "x": 1, "rss": 1, "reddit": 1, "youtube": 1},
        )
        result = await ranker.rank([pin, web, *rest])
        ids = [r.item.item_id for r in result]
        assert len(result) == 5
        assert ids[0] == "p"
        # Every source keeps its guaranteed slot; the pin's own source does not get a second one.
        assert set(ids) == {"p", "x1", "r1", "rd1", "y1"}

    @pytest.mark.asyncio
    async def test_results_sorted_by_score_desc(self):
        items = _items([("a", SourceType.RSS), ("b", SourceType.REDDIT), ("c", SourceType.WEB)])
        ranker = _ranker(_rankings({"a": 0.7, "b": 0.95, "c": 0.8}), top_n=5, min_score=0.6)
        result = await ranker.rank(items)
        scores = [r.score for r in result]
        assert scores == sorted(scores, reverse=True)

    @pytest.mark.asyncio
    async def test_parallel_batches_merge_all_items(self):
        import re

        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        items = _items([(f"i{n}", SourceType.RSS) for n in range(10)])
        config = PipelineConfig(top_n=20, min_score=0.6, source_slots={}, ranking_batch_size=3)
        factory = _mock_factory()

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

    @pytest.mark.asyncio
    async def test_transient_batch_failure_is_retried_not_dropped(self):
        # A throttle/5xx on the Converse call used to be swallowed into [], silently deleting a
        # whole batch of candidates from the day's pool. It must be retried instead.
        items = _items([("a", SourceType.RSS)])
        attempts = {"n": 0}

        def flaky(_prompt):
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise RuntimeError("ThrottlingException")
            return AIMessage(content=_rankings({"a": 0.9}))

        factory = _mock_factory()
        factory.get_model.return_value = RunnableLambda(flaky)
        config = PipelineConfig(
            top_n=5, min_score=0.6, source_slots={}, ranking_max_retries=3, ranking_retry_backoff_sec=0
        )
        result = await ContentRanker(config, factory).rank(items)
        assert [r.item.item_id for r in result] == ["a"]
        assert attempts["n"] == 3

    @pytest.mark.asyncio
    async def test_all_batches_failing_raises(self):
        # Nothing ranked at all is an outage, not a quiet day: raise so the run reports FAILED
        # instead of publishing an empty digest.
        items = _items([(f"i{n}", SourceType.RSS) for n in range(4)])
        factory = _mock_factory()
        factory.get_model.return_value = RunnableLambda(lambda _: (_ for _ in ()).throw(RuntimeError("bedrock down")))
        config = PipelineConfig(
            top_n=5, min_score=0.6, source_slots={}, ranking_batch_size=2, ranking_retry_backoff_sec=0
        )
        with pytest.raises(RuntimeError, match="All 2 ranking batches failed"):
            await ContentRanker(config, factory).rank(items)

    @pytest.mark.asyncio
    async def test_one_permanently_failed_batch_is_tolerated(self):
        import re

        items = _items([(f"i{n}", SourceType.RSS) for n in range(4)])

        def score_or_die(prompt_value):
            ids = re.findall(r"ID: (i\d+)", str(prompt_value))
            if "i0" in ids:  # the first batch never recovers
                raise RuntimeError("ThrottlingException")
            return AIMessage(content=_rankings(dict.fromkeys(ids, 0.8)))

        factory = _mock_factory()
        factory.get_model.return_value = RunnableLambda(score_or_die)
        config = PipelineConfig(
            top_n=5, min_score=0.6, source_slots={}, ranking_batch_size=2, ranking_retry_backoff_sec=0
        )
        result = await ContentRanker(config, factory).rank(items)
        # The surviving batch still ranks; only the dead batch's two items (i0 and its partner) are missing.
        ids = {r.item.item_id for r in result}
        assert len(ids) == 2 and "i0" not in ids

    @pytest.mark.asyncio
    async def test_bedrock_fan_out_is_bounded_by_config(self):
        import re

        items = _items([(f"i{n}", SourceType.RSS) for n in range(8)])
        state = {"active": 0, "peak": 0}

        async def score(prompt_value):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            ids = re.findall(r"ID: (i\d+)", str(prompt_value))
            return AIMessage(content=_rankings(dict.fromkeys(ids, 0.8)))

        factory = _mock_factory()
        factory.get_model.return_value = RunnableLambda(func=lambda _: None, afunc=score)
        config = PipelineConfig(
            top_n=20, min_score=0.6, source_slots={}, ranking_batch_size=1, ranking_max_concurrency=2
        )
        result = await ContentRanker(config, factory).rank(items)
        assert len(result) == 8  # every batch still scored
        assert state["peak"] <= 2, f"fan-out exceeded ranking_max_concurrency: peak={state['peak']}"

    def test_the_count_cap_is_spread_evenly_instead_of_leaving_a_tiny_tail(self):
        config = PipelineConfig(ranking_batch_size=40)
        items = [
            CollectedItem(item_id=f"i{n:02d}", source_type=SourceType.RSS, title=f"t{n}", url=f"http://e.com/{n}")
            for n in range(41)
        ]
        batches = ContentRanker(config, _mock_factory())._make_batches(items)
        assert sorted(len(b) for b in batches) == [20, 21]


class TestSubThresholdItems:
    @pytest.mark.asyncio
    async def test_a_guaranteed_slot_never_admits_an_item_below_min_score(self):
        # A source with a slot but nothing above the bar goes unrepresented; the slot does not lower
        # the bar for it.
        items = _items([("r1", SourceType.RSS), ("y1", SourceType.YOUTUBE)])
        ranker = _ranker(
            _rankings({"r1": 0.80, "y1": 0.59}),
            top_n=5,
            min_score=0.6,
            source_slots={"rss": 1, "youtube": 1},
        )
        result = await ranker.rank(items)
        assert {r.item.item_id for r in result} == {"r1"}


def _ranker_with_outputs(outputs: list[str], **overrides) -> tuple[ContentRanker, list[str]]:
    """A ranker whose stand-in LLM returns `outputs` one call at a time (the last one repeats),
    plus the list of prompts it saw — so the coverage re-ask can be counted."""
    config = PipelineConfig(**overrides)
    factory = _mock_factory()
    seen: list[str] = []

    def _respond(prompt_value):
        seen.append(str(prompt_value))
        return AIMessage(content=outputs[min(len(seen) - 1, len(outputs) - 1)])

    factory.get_model.return_value = RunnableLambda(_respond)
    return ContentRanker(config, factory), seen


class TestRankingCoverage:
    @pytest.mark.asyncio
    async def test_full_coverage_makes_no_extra_call(self):
        items = _items([("a", SourceType.RSS), ("b", SourceType.RSS)])
        ranker, seen = _ranker_with_outputs([_rankings({"a": 0.9, "b": 0.8})], top_n=5, min_score=0.6, source_slots={})
        result = await ranker.rank(items)
        assert {r.item.item_id for r in result} == {"a", "b"}
        assert len(seen) == 1  # a full-coverage day costs zero extra Bedrock calls

    @pytest.mark.asyncio
    async def test_omitted_items_recovered_by_one_reask(self):
        # The model silently dropped b and c; the re-ask scores b, and that is the ONLY retry —
        # c staying unscored must not trigger a second one.
        items = _items([("a", SourceType.RSS), ("b", SourceType.RSS), ("c", SourceType.RSS)])
        ranker, seen = _ranker_with_outputs(
            [_rankings({"a": 0.9}), _rankings({"b": 0.8})],
            top_n=5,
            min_score=0.6,
            source_slots={},
            ranking_retry_backoff_sec=0,
        )
        result = await ranker.rank(items)
        assert {r.item.item_id for r in result} == {"a", "b"}
        assert len(seen) == 2  # one score pass + exactly one re-ask

    @pytest.mark.asyncio
    async def test_a_single_omission_in_a_large_batch_is_still_re_asked(self):
        # Prod omitted 1 of 40 on five runs; a coverage-ratio gate (0.975 >= 0.9) lost each one.
        ids = [f"i{n:02d}" for n in range(10)]
        items = _items([(i, SourceType.RSS) for i in ids])
        ranker, seen = _ranker_with_outputs(
            [_rankings(dict.fromkeys(ids[:-1], 0.7)), _rankings({ids[-1]: 0.9})],
            top_n=20,
            min_score=0.6,
            source_slots={},
            max_per_origin=20,
        )
        result = await ranker.rank(items)
        assert ids[-1] in {r.item.item_id for r in result}
        assert len(seen) == 2
        assert f"ID: {ids[-1]}" in seen[1] and f"ID: {ids[0]}" not in seen[1]

    @pytest.mark.asyncio
    async def test_failed_reask_keeps_the_partially_scored_batch(self):
        items = _items([("a", SourceType.RSS), ("b", SourceType.RSS)])
        config = PipelineConfig(top_n=5, min_score=0.6, source_slots={}, ranking_retry_backoff_sec=0)
        factory = _mock_factory()
        calls: list[int] = []

        def _respond(prompt_value):
            calls.append(1)
            if len(calls) == 1:
                return AIMessage(content=_rankings({"a": 0.9}))
            raise RuntimeError("ThrottlingException")

        factory.get_model.return_value = RunnableLambda(_respond)
        result = await ContentRanker(config, factory).rank(items)
        # No raise, no lost items: the first pass's outcome survives untouched.
        assert {r.item.item_id for r in result} == {"a"}

    @pytest.mark.asyncio
    async def test_repeated_id_still_counts_as_a_shortfall(self):
        # The model scored a twice and never scored b: 1/2 coverage, so the re-ask must still fire
        # and b must be recovered. With duplicates counted, coverage read 1.0 and b was lost.
        items = _items([("a", SourceType.RSS), ("b", SourceType.RSS)])
        duplicated = json.dumps({"rankings": [{"item_id": "a", "score": 0.9}, {"item_id": "a", "score": 0.7}]})
        ranker, seen = _ranker_with_outputs(
            [duplicated, _rankings({"b": 0.8})],
            top_n=5,
            min_score=0.6,
            source_slots={},
            ranking_retry_backoff_sec=0,
        )
        result = await ranker.rank(items)
        assert sorted(r.item.item_id for r in result) == ["a", "b"]
        assert len(seen) == 2
        assert ranker.health.items_scored == 2


class TestRankingPromptOrigin:
    """What the ranker actually SHOWS the model. The prompt scores "Source Authority", so a web
    item's outlet has to appear — it used to be omitted entirely for web-search results."""

    def test_web_items_carry_an_origin_line(self):
        ranker = _ranker("", source_slots={})
        item = CollectedItem(item_id="w", source_type=SourceType.WEB, title="t", url="https://www.wired.com/story/x")
        assert "Origin: wired.com" in ranker._format_items([item])

    def test_no_origin_line_when_the_url_has_no_host(self):
        ranker = _ranker("", source_slots={})
        item = CollectedItem(item_id="w", source_type=SourceType.WEB, title="t", url="notaurl")
        assert "Origin:" not in ranker._format_items([item])


class TestRankingHealthVerdict:
    """A batch that fails every retry silently deletes ~40 candidates from the day, and the digest
    that follows looks entirely normal — so the verdict has to travel out of rank()."""

    @pytest.mark.asyncio
    async def test_a_permanently_failed_batch_is_recorded_and_logged_as_an_error(self):
        items = _items([("a", SourceType.WEB), ("b", SourceType.WEB), ("c", SourceType.WEB)])
        ranker = _ranker(_rankings({"a": 0.9}), min_score=0.6, ranking_batch_size=1, ranking_max_retries=1)
        original = ranker._score_batch

        async def flaky(batch, semaphore, recent_titles):
            if batch[0].item_id != "a":
                raise RuntimeError("throttled")
            return await original(batch, semaphore, recent_titles)

        with patch.object(ranker, "_score_batch", side_effect=flaky):
            with patch("pipeline.ranker.logger.error") as err:
                await ranker.rank(items)
        assert ranker.health.degraded is True
        assert ranker.health.batches_failed == 2 and ranker.health.batches_total == 3
        assert ranker.health.items_lost == 2
        assert err.called

    @pytest.mark.asyncio
    async def test_a_batch_that_parses_to_nothing_counts_as_lost(self):
        # Both the first pass and the coverage re-ask returned unparseable JSON, so the batch used to
        # return [] without raising: failures stayed empty, items_lost stayed 0 and every ranking
        # alert was silent while a whole batch vanished from the pool.
        items = _items([("a", SourceType.WEB), ("b", SourceType.WEB)])
        config = PipelineConfig(
            top_n=5, min_score=0.6, source_slots={}, ranking_batch_size=1, ranking_retry_backoff_sec=0
        )
        factory = _mock_factory()

        def _respond(prompt_value):
            return AIMessage(content="garbage not json" if "t-b" in str(prompt_value) else _rankings({"a": 0.9}))

        factory.get_model.return_value = RunnableLambda(_respond)
        ranker = ContentRanker(config, factory)
        result = await ranker.rank(items)

        assert {r.item.item_id for r in result} == {"a"}  # the healthy batch still publishes
        assert ranker.health.batches_failed == 1
        assert ranker.health.items_lost == 1
        assert ranker.health.degraded is True

    @pytest.mark.asyncio
    async def test_a_clean_run_is_not_degraded(self):
        items = _items([("a", SourceType.WEB)])
        ranker = _ranker(_rankings({"a": 0.9}), min_score=0.6)
        await ranker.rank(items)
        assert ranker.health.degraded is False
        assert ranker.health.items_lost == 0
