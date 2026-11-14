"""The blind judge's label bookkeeping: every recorded A/B verdict depends on it."""

import asyncio
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from shared.config import Config

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "eval_harness.py"


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location("eval_harness", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["eval_harness"] = module
    spec.loader.exec_module(module)
    return module


def _judging(prefer):
    """A judge that scores by `prefer(first, second)` and names its pick by position."""

    def answer(prompt_value):
        text = prompt_value.to_string()
        first, second = (re.search(rf"<output_{n}>\n(.*?)\n</output_{n}>", text, re.S).group(1) for n in (1, 2))
        pick = prefer(first, second)
        return AIMessage(
            content=json.dumps(
                {
                    "scores_1": {"accuracy": 5 if pick == 1 else 1},
                    "scores_2": {"accuracy": 5 if pick == 2 else 1},
                    "winner": pick,
                    "reason": "r",
                }
            )
        )

    return RunnableLambda(answer)


def _cases(harness, monkeypatch, tmp_path, judge, n=8):
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "_edition", lambda: "")
    monkeypatch.setattr(harness.Config, "load", classmethod(lambda cls, *a, **k: Config()))
    models = MagicMock()
    models.get_model.return_value = judge
    monkeypatch.setattr(harness.bootstrap, "language_models", lambda _config: models)
    names = [f"case-{k}.json" for k in range(n)]
    for label, artifact in (("base", "GOOD"), ("arm", "bad")):
        folder = tmp_path / "digest" / label
        folder.mkdir(parents=True)
        for name in names:
            (folder / name).write_text(json.dumps({"case": name, "context": "c", "artifact": f"{artifact} {name}"}))
    return names


def _verdicts(tmp_path):
    return json.loads((tmp_path / "digest" / "judge-base-vs-arm.json").read_text())


def test_a_judge_that_prefers_one_artifact_credits_its_label_in_both_orders(harness, monkeypatch, tmp_path):
    names = _cases(harness, monkeypatch, tmp_path, _judging(lambda a, b: 1 if a.startswith("GOOD") else 2))
    orders = {int(hashlib.sha256(n.encode()).hexdigest(), 16) % 2 == 0 for n in names}
    assert orders == {True, False}, "the fixture must exercise both orders"
    asyncio.run(harness._judge("digest", "base", "arm"))
    rows = _verdicts(tmp_path)
    assert len(rows) == len(names)
    assert all(r["winner"] == "base" for r in rows)
    assert all(r["base"]["accuracy"] == 5 and r["arm"]["accuracy"] == 1 for r in rows)


def test_a_judge_that_always_prefers_the_first_position_splits_by_the_shuffle(harness, monkeypatch, tmp_path):
    names = _cases(harness, monkeypatch, tmp_path, _judging(lambda a, b: 1))
    asyncio.run(harness._judge("digest", "base", "arm"))
    expected = {n: "base" if int(hashlib.sha256(n.encode()).hexdigest(), 16) % 2 == 0 else "arm" for n in names}
    assert {r["case"]: r["winner"] for r in _verdicts(tmp_path)} == expected


def test_a_digest_replay_takes_the_editors_path_and_shows_the_judge_the_editors_cut(harness, monkeypatch, tmp_path):
    from application import digest_pipeline
    from domain.models import CollectedItem, DigestResult, RankedItem
    from shared.constants import SourceType
    from tests.fakes import MemoryStateStore, snapshot

    config = Config()
    config.pipeline.item_text_max_tokens = 3000
    item = CollectedItem(source_type=SourceType.RSS, title="T", url="https://e.com/a", text="x" * 5000)
    store = MemoryStateStore()
    store.write_json(
        f"{digest_pipeline.POOLS_PREFIX}2026-11-10.json", [RankedItem(item=item, score=0.9).model_dump(mode="json")]
    )
    memory = MagicMock()
    memory.get_digest.return_value = snapshot(lead="어제의 리드.", items=[{"title": "Yesterday"}])
    memory.get_recent_digests_dated.return_value = []
    models = MagicMock()
    models.truncate_to_tokens = lambda text, tokens: text[:tokens]
    write = MagicMock(side_effect=lambda *a, **k: asyncio.sleep(0, DigestResult(digest_text="digest")))

    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "_edition", lambda: config.edition)
    monkeypatch.setattr(harness.Config, "load", classmethod(lambda cls, *a, **k: config))
    monkeypatch.setattr(harness, "_other_editions", lambda _config: [])
    monkeypatch.setattr(harness.bootstrap, "language_models", lambda _config: models)
    monkeypatch.setattr(harness.bootstrap, "memory_store", lambda *_a, **_k: memory)
    monkeypatch.setattr(harness.bootstrap, "state_store", lambda _config: store)
    monkeypatch.setattr(harness.bootstrap, "sibling_memory_stores", lambda _config: [])
    monkeypatch.setattr(digest_pipeline, "write_digest", write)

    asyncio.run(harness._digest("arm", ["2026-11-10"]))

    args, kwargs = write.call_args
    assert args[2][0].item.url == "https://e.com/a"
    assert isinstance(args[3], digest_pipeline.EditorContext) and kwargs["today"].isoformat() == "2026-11-10"
    case = json.loads((tmp_path / "digest" / "arm" / "2026-11-10.json").read_text())
    assert "x" * 3000 in case["context"] and "x" * 3001 not in case["context"]
    assert case["artifact"] == "digest"


def test_a_research_run_that_printed_nothing_is_skipped_and_the_rest_are_kept(harness, monkeypatch, tmp_path):
    import subprocess

    runs = iter(
        [
            subprocess.CompletedProcess([], 1, stdout="", stderr="Traceback: boom"),
            subprocess.CompletedProcess(
                [], 0, stdout=json.dumps({"status": "succeeded", "markdown": "# R"}), stderr=""
            ),
        ]
    )
    monkeypatch.setattr(harness, "ROOT", tmp_path)
    monkeypatch.setattr(harness, "_edition", lambda: "")
    monkeypatch.setattr(harness.subprocess, "run", lambda *a, **k: next(runs))
    harness._research("arm", ["crashed topic", "good topic"])
    cases = [json.loads(p.read_text()) for p in (tmp_path / "research" / "arm").glob("*.json")]
    assert [c["case"] for c in cases] == ["good topic"] and cases[0]["status"] == "succeeded"
