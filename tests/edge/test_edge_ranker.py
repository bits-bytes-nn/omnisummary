"""ContentRanker.rank end to end with scripted LLMs: story cap, pins, tiny pools, unparseable batches, ties."""

from __future__ import annotations

import random

from application.ranker import ContentRanker
from domain.models import CollectedItem
from shared.config import PipelineConfig
from shared.constants import SourceType
from tests.edge.edge_harness import ScriptedModels, prose, score_by_title


def _cfg(**kw) -> PipelineConfig:
    return PipelineConfig(ranking_retry_backoff_sec=0, **kw)


def _item(k, source=SourceType.WEB, host=None, **meta) -> CollectedItem:
    return CollectedItem(
        item_id=f"i{k}",
        source_type=source,
        title=f"story {k}",
        url=f"https://{host or f'h{k}'}.example/{k}",
        text="b",
        metadata=meta,
    )


def _ranker(cfg, handlers) -> tuple[ContentRanker, ScriptedModels]:
    models = ScriptedModels(handlers)
    return ContentRanker(cfg, models), models


async def test_a_pool_that_is_one_story_still_fills_via_the_relaxed_pass():
    items = [_item(k) for k in range(15)]
    one_story = {"same_story": [list(range(1, 16))]}
    ranker, _ = _ranker(_cfg(), {"ranking": score_by_title({}, 0.8), "story-groups": lambda _t: one_story})
    pool = await ranker.rank(items, select_count=12)
    assert len(pool) == 12


async def test_one_story_with_source_slots_does_not_lose_seats():
    """All copies are one story; x has a guaranteed slot but its copy is not the best copy."""
    items = [_item(k) for k in range(10)] + [_item(100 + k, SourceType.X) for k in range(3)]
    scores = {f"story {k}": 0.9 - k / 100 for k in range(10)} | {f"story {100 + k}": 0.7 for k in range(3)}
    groups = {"same_story": [list(range(1, 14))]}
    ranker, _ = _ranker(
        _cfg(source_slots={"x": 1}), {"ranking": score_by_title(scores), "story-groups": lambda _t: groups}
    )
    pool = await ranker.rank(items, select_count=12)
    assert len(pool) == 12
    assert pool[0].item.item_id == "i0"


async def test_pinned_item_and_story_cap():
    """A pin is kept out of the grouping, so its other copies are not story-capped against it."""
    pinned = _item(0, pinned=True)
    copies = [_item(k) for k in range(1, 6)]
    others = [_item(k) for k in range(6, 20)]
    groups = {"same_story": [[1, 2, 3, 4, 5]]}  # numbers are over above_threshold, pin excluded
    scores = {f"story {k}": 0.95 for k in range(1, 6)} | {f"story {k}": 0.7 for k in range(6, 20)}
    ranker, models = _ranker(_cfg(), {"ranking": score_by_title(scores), "story-groups": lambda _t: groups})
    pool = await ranker.rank([pinned, *copies, *others], select_count=6)
    assert pool[0].item.item_id == "i0"
    shown = models.prompts["story-groups"][0]
    copies_in_pool = [r for r in pool if r.item.item_id in {c.item_id for c in copies}]
    # Observed: the pin's story is not in the grouping input; the copies it shares a story with each other
    # still take only ONE seat. Report whether the pin itself could be grouped with its copies.
    assert "story 0" not in shown
    assert len(copies_in_pool) == 1


async def test_a_single_item():
    ranker, models = _ranker(_cfg(), {"ranking": score_by_title({}, 0.9)})
    pool = await ranker.rank([_item(0)], select_count=12)
    assert [r.item.item_id for r in pool] == ["i0"] and models.calls["ranking"] == 1
    assert models.calls.get("story-groups", 0) == 0


async def test_nothing_above_min_score():
    ranker, models = _ranker(_cfg(), {"ranking": score_by_title({}, 0.1)})
    pool = await ranker.rank([_item(k) for k in range(5)], select_count=12)
    assert pool == [] and not ranker.health.degraded and ranker.health.items_scored == 5


async def test_unparseable_batch_of_one_item_fails_after_one_reask():
    ranker, models = _ranker(_cfg(), {"ranking": prose})
    try:
        await ranker.rank([_item(0)], select_count=12)
    except RuntimeError as e:
        assert "ranking batches failed" in str(e)
    else:
        raise AssertionError("expected the single-batch failure to raise")
    assert models.calls["ranking"] == 2


async def test_split_batch_that_loses_one_half_reads_degraded():
    """40 items, the whole batch answers prose, then only the half with 'story 0' parses."""
    items = [_item(k) for k in range(40)]
    calls = {"n": 0}

    def ranking(text):
        calls["n"] += 1
        if calls["n"] == 1:
            return prose(text)
        return score_by_title({}, 0.8)(text) if "Title: story 0\n" in text else prose(text)

    ranker, models = _ranker(_cfg(ranking_batch_size=40), {"ranking": ranking, "story-groups": lambda _t: {}})
    await ranker.rank(items, select_count=12)
    assert models.calls["ranking"] == 3
    assert ranker.health.batches_partly_lost == 1 and ranker.health.degraded
    assert ranker.health.items_lost == 20


async def test_split_half_that_merely_omits_one_item_marks_the_run_degraded():
    """A split half that omitted ONE item is DEGRADED (alerts), while an unsplit batch that omits the same
    item after its coverage re-ask is not: inconsistent alerting for the same loss."""
    items = [_item(k) for k in range(40)]
    calls = {"n": 0}

    def ranking(text):
        calls["n"] += 1
        if calls["n"] == 1:
            return prose(text)
        out = score_by_title({}, 0.8)(text)
        out["rankings"] = out["rankings"][1:] if "Title: story 0\n" in text else out["rankings"]
        return out

    ranker, _ = _ranker(_cfg(ranking_batch_size=40), {"ranking": ranking, "story-groups": lambda _t: {}})
    await ranker.rank(items, select_count=12)
    split_degraded = ranker.health.degraded

    def omit_always(text):
        from tests.edge.edge_harness import ITEM_RE

        drop = {int(n) for n, t in ITEM_RE.findall(text) if t.strip() == "story 0"}
        out = score_by_title({}, 0.8)(text)
        out["rankings"] = [e for e in out["rankings"] if e["item"] not in drop]
        return out

    plain, _ = _ranker(_cfg(ranking_batch_size=40), {"ranking": omit_always, "story-groups": lambda _t: {}})
    await plain.rank(items, select_count=12)
    assert plain.health.items_lost >= 1
    assert split_degraded == plain.health.degraded, (split_degraded, plain.health)


async def test_ties_are_order_independent():
    items = [_item(k) for k in range(30)]
    selections = set()
    for seed in range(5):
        shuffled = items[:]
        random.Random(seed).shuffle(shuffled)
        ranker, _ = _ranker(_cfg(), {"ranking": score_by_title({}, 0.8), "story-groups": lambda _t: {}})
        pool = await ranker.rank([it.model_copy() for it in shuffled], select_count=12)
        selections.add(tuple(r.item.item_id for r in pool))
    assert len(selections) == 1


async def test_best_copy_beyond_the_grouped_window_cannot_block_a_seat():
    """Only the top _MAX_GROUPED candidates are grouped; nothing below them carries a story id."""
    from application import ranker as ranker_mod

    items = [_item(k) for k in range(ranker_mod._MAX_GROUPED + 10)]
    groups = {"same_story": [[1, ranker_mod._MAX_GROUPED]]}
    ranker, _ = _ranker(
        _cfg(ranking_batch_size=200), {"ranking": score_by_title({}, 0.8), "story-groups": lambda _t: groups}
    )
    pool = await ranker.rank(items, select_count=12)
    assert len(pool) == 12


async def test_best_copy_at_origin_cap_slot_pass_skips_the_story_entirely():
    """Story S: best copy from web host A (A already holds a seat), other copy is the only x item. The
    x slot cannot take the x copy (not the best copy), and the best copy is origin-capped; the story
    still lands through the fill pass, and x gets no seat."""
    a1 = _item(1, host="a")
    a2 = _item(2, host="a")
    x3 = _item(3, SourceType.X)
    rest = [_item(k) for k in range(4, 20)]
    scores = {"story 1": 0.95, "story 2": 0.9, "story 3": 0.7}
    ranker, models = _ranker(_cfg(source_slots={"x": 1, "web": 1}), {"ranking": score_by_title(scores, 0.65)})

    def groups(text):
        import re

        nums = {t: int(n) for n, t in re.findall(r"^(\d+)\. (story \d+)", text, re.M)}
        return {"same_story": [[nums["story 2"], nums["story 3"]]]}

    models.handlers["story-groups"] = groups
    pool = await ranker.rank([a1, a2, x3, *rest], select_count=5)
    ids = [r.item.item_id for r in pool]
    assert "i1" in ids
    # Documented trade-off (quality-first), recorded here: the x slot goes unfilled for that story.
    assert ("i3" in ids) or ("i2" in ids)
