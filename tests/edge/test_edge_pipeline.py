"""run_pipeline end to end: real aggregator, ranker, digest generator and trend tracker, scripted LLMs."""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import MagicMock

import pytest

from application import digest_pipeline as runner
from application.digest_generator import DigestContentError
from domain.models import CollectedItem, SourceStatus
from shared.config import Config
from shared.constants import SourceType
from tests.edge.edge_harness import (
    ScriptedModels,
    editor_first_candidates,
    editor_picking,
    no_groups,
    no_violations,
    prose,
    score_by_title,
)
from tests.fakes import MemoryStateStore, nothing_paid

DAY = date(2026, 11, 2)


def _config(**pipeline) -> Config:
    config = Config()
    config.pipeline.enable_threads_post = False
    config.pipeline.ranking_retry_backoff_sec = 0
    config.pipeline.digest_retry_backoff_sec = 0
    for key, value in pipeline.items():
        setattr(config.pipeline, key, value)
    return config


def _items(n: int, source=SourceType.WEB) -> list[CollectedItem]:
    return [
        CollectedItem(
            item_id=f"i{k}", source_type=source, title=f"story {k}", url=f"https://h{k}.example/a", text=f"body {k}"
        )
        for k in range(n)
    ]


def _memory(snapshots=None, *, raises: Exception | None = None):
    memory = MagicMock()
    if raises:
        memory.get_recent_digests.side_effect = raises
    else:
        memory.get_recent_digests.return_value = snapshots or []
    return memory


def _handlers(urls, **over):
    handlers = {
        "ranking": score_by_title({}, 0.8),
        "story-groups": no_groups,
        "digest": editor_picking(urls),
        "grounding": no_violations,
        "trends": lambda _t: {"trends": []},
    }
    handlers.update(over)
    return handlers


async def _run(config, models, items, memory, **kw):
    return await asyncio.wait_for(
        runner.run_pipeline(
            config,
            models,
            items,
            DAY,
            state_store=MemoryStateStore(),
            memory_store=lambda: memory,
            sibling_memory=lambda: [],
            paid_check=nothing_paid,
            dry_run=True,
            **kw,
        ),
        timeout=20,
    )


class _Source:
    def __init__(self, result=None, exc=None):
        self.result, self.exc, self.health_issue = result or [], exc, None

    async def collect(self):
        if self.exc:
            raise self.exc
        return self.result


async def test_every_collector_returns_nothing():
    config = _config()
    items, health = await runner.run_collectors_with_health(config, [("rss", _Source()), ("web_search", _Source())])
    assert items == [] and [s.status for s in health.sources] == [SourceStatus.EMPTY, SourceStatus.EMPTY]
    models = ScriptedModels({})
    assert await _run(config, models, items, _memory()) == (None, None, None)
    assert sum(models.calls.values()) == 0


async def test_one_collector_raises_and_the_rest_still_ship():
    config = _config()
    good = _items(6)
    items, health = await runner.run_collectors_with_health(
        config, [("rss", _Source(exc=RuntimeError("boom"))), ("web_search", _Source(good))]
    )
    assert {s.name: s.status for s in health.sources} == {"rss": SourceStatus.FAILED, "web_search": SourceStatus.OK}
    models = ScriptedModels(_handlers([good[0].url, good[1].url]))
    _, ranked, digest = await _run(config, models, items, _memory())
    assert digest.content.items[0].url == good[0].url and len(digest.content.items) == 2


async def test_every_ranking_batch_answers_prose_fails_fast_and_bounded():
    config = _config(ranking_batch_size=40)
    items = _items(100)
    models = ScriptedModels(_handlers([], ranking=prose))
    with pytest.raises(RuntimeError, match="All 3 ranking batches failed"):
        await _run(config, models, items, _memory())
    # 3 batches; each: one whole-batch call plus one per half, never recursive.
    assert models.calls["ranking"] == 9
    assert models.calls.get("digest", 0) == 0


async def test_every_ranking_call_throttles_fails_bounded():
    config = _config(ranking_batch_size=40)
    models = ScriptedModels(_handlers([], ranking=lambda _t: RuntimeError("ThrottlingException")))
    with pytest.raises(RuntimeError, match="All 3 ranking batches failed"):
        await _run(config, models, _items(100), _memory())
    assert models.calls["ranking"] == 3 * config.pipeline.ranking_max_retries


async def test_editor_headline_matching_no_candidate_fails_the_run_after_bounded_reasks():
    config = _config()
    models = ScriptedModels(_handlers(["https://invented.example/story", "https://h1.example/a"]))
    with pytest.raises(DigestContentError, match="matches no ranked candidate"):
        await _run(config, models, _items(6), _memory())
    assert models.calls["digest"] == config.pipeline.digest_max_retries


@pytest.mark.parametrize(
    "grounding",
    [
        lambda _t: RuntimeError("grounding down"),
        lambda _t: "not json at all",
        lambda _t: {"violations": [{"claim": "x"}], "corrected_digest": "ITEM 7 BODY: 다른 아이템 덮어쓰기"},
        lambda _t: {"violations": [{"claim": "x"}], "corrected_digest": "LEAD: \nITEM 0 BODY:"},
        lambda _t: ["a list"],
    ],
    ids=["raises", "prose", "renumbered", "empty-markers", "list"],
)
async def test_grounding_failure_keeps_the_original_content(grounding):
    config = _config()
    items = _items(6)
    models = ScriptedModels(_handlers([items[0].url, items[1].url], grounding=grounding))
    _, _, digest = await _run(config, models, items, _memory())
    assert digest.content.items[0].body == "본문 문장이다."
    assert digest.content.lead.endswith("오늘의 리드 문장이다. 두 번째 문장이다.")


@pytest.mark.parametrize(
    "groups",
    [
        lambda _t: RuntimeError("down"),
        lambda _t: "garbage",
        lambda _t: {"same_story": "1,2"},
        lambda _t: {"same_story": [[1, 999, -3, "2"], [True, 2]]},
        lambda _t: [[1, 2]],
    ],
    ids=["raises", "prose", "string", "bad-numbers", "list-root"],
)
async def test_story_grouping_failure_still_fills_the_pool(groups):
    config = _config()
    items = _items(20)
    models = ScriptedModels(_handlers([], digest=editor_first_candidates(), **{"story-groups": groups}))
    _, ranked, digest = await _run(config, models, items, _memory())
    assert len(ranked) == config.pipeline.top_n + config.pipeline.digest_candidate_buffer
    assert digest.content.items


async def test_story_grouping_json_true_is_not_read_as_item_1():
    """[True, 2] in a group: isinstance(True, int) is True, so True reads as item 1."""
    from application.ranker import ContentRanker
    from domain.models import RankedItem

    config = _config()
    models = ScriptedModels({"ranking": score_by_title({}), "story-groups": lambda _t: {"same_story": [[True, 2]]}})
    ranker = ContentRanker(config.pipeline, models)
    cands = [RankedItem(item=it, score=0.9 - k / 100) for k, it in enumerate(_items(3))]
    stories = await ranker._story_groups(cands)
    assert stories == {}, f"JSON true was read as item 1: {stories}"


async def test_memory_store_down_for_seed_and_titles():
    config = _config()
    items = _items(6)
    models = ScriptedModels(_handlers([items[0].url]))
    _, ranked, digest = await _run(config, models, items, _memory(raises=RuntimeError("Memory not found")))
    assert digest.content.items[0].url == items[0].url
    assert "(No recent digest on record.)" in models.prompts["ranking"][0]


async def test_memory_factory_itself_raises():
    config = _config()
    items = _items(6)
    models = ScriptedModels(_handlers([items[0].url]))

    def factory():
        raise RuntimeError("no memory named omnisummary")

    result = await asyncio.wait_for(
        runner.run_pipeline(
            config,
            models,
            items,
            DAY,
            state_store=MemoryStateStore(),
            memory_store=factory,
            sibling_memory=lambda: [],
            paid_check=nothing_paid,
            dry_run=True,
        ),
        timeout=20,
    )
    assert result[2].content.items


async def test_non_dry_run_records_the_ledger_and_trends_failure_is_non_fatal():
    config = _config()
    items = _items(6)
    store = MemoryStateStore()
    models = ScriptedModels(_handlers([items[0].url], trends=lambda _t: RuntimeError("trends down")))
    result = await runner.run_pipeline(
        config,
        models,
        items,
        DAY,
        state_store=store,
        memory_store=lambda: _memory(),
        sibling_memory=lambda: [],
        paid_check=nothing_paid,
        dry_run=False,
    )
    assert result[2].content.items
    assert any("published" in k for k in store.blobs), store.blobs.keys()
