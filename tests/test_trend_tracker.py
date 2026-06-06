import json
from datetime import date
from unittest.mock import MagicMock

import pytest

from pipeline.trend_tracker import TRENDS_KEY, TrendTracker
from shared.config import PipelineConfig
from shared.models import Trend, TrendEvidence, TrendMemory, TrendStatus


class _FakeStore:
    def __init__(self, initial: dict[str, str] | None = None) -> None:
        self.data: dict[str, str] = dict(initial or {})

    def read(self, key: str) -> str | None:
        return self.data.get(key)

    def write(self, key: str, content: str) -> None:
        self.data[key] = content

    def exists(self, key: str) -> bool:
        return key in self.data


def _patched_tracker(store, observations, monkeypatch, **overrides) -> TrendTracker:
    config = PipelineConfig(**overrides)
    factory = MagicMock()
    factory.get_model.return_value = MagicMock()
    tracker = TrendTracker(config, factory, store)
    payload = observations if isinstance(observations, str) else json.dumps({"observations": observations})

    class _Chain:
        def __or__(self, _other):
            return self

        async def ainvoke(self, _inputs):
            return payload

    monkeypatch.setattr("pipeline.trend_tracker.TrendClassifyPrompt.get_prompt", lambda: _Chain())
    return tracker


class TestMakeSlug:
    def test_basic_slug(self):
        assert TrendTracker._make_slug("Open Weight Models!", set()) == "open-weight-models"

    def test_dedup_against_existing(self):
        assert TrendTracker._make_slug("Agents", {"agents"}) == "agents-2"
        assert TrendTracker._make_slug("Agents", {"agents", "agents-2"}) == "agents-3"

    def test_empty_title_fallback(self):
        assert TrendTracker._make_slug("!!!", set()) == "trend"


class TestUpdateTrends:
    @pytest.mark.asyncio
    async def test_creates_new_trend_with_code_stamped_date(self, monkeypatch):
        store = _FakeStore()
        obs = [{"trend_id": "", "new_title": "Open Weight Models", "summary": "Meta released a new model."}]
        tracker = _patched_tracker(store, obs, monkeypatch)
        await tracker.update_trends("digest body", "2026-06-05")

        memory = TrendMemory.model_validate_json(store.data[TRENDS_KEY])
        assert len(memory.trends) == 1
        t = memory.trends[0]
        assert t.id == "open-weight-models"
        assert t.first_seen == t.last_seen == "2026-06-05"
        assert t.evidence[0].date == "2026-06-05"
        assert t.evidence[0].summary == "Meta released a new model."

    @pytest.mark.asyncio
    async def test_extends_existing_trend(self, monkeypatch):
        existing = TrendMemory(
            trends=[
                Trend(
                    id="agents",
                    title="Agents",
                    first_seen="2026-06-01",
                    last_seen="2026-06-01",
                    evidence=[TrendEvidence(date="2026-06-01", summary="old")],
                )
            ]
        )
        store = _FakeStore({TRENDS_KEY: existing.model_dump_json()})
        obs = [{"trend_id": "agents", "new_title": "", "summary": "new framework launched"}]
        tracker = _patched_tracker(store, obs, monkeypatch)
        await tracker.update_trends("d", "2026-06-03")

        memory = TrendMemory.model_validate_json(store.data[TRENDS_KEY])
        t = memory.by_id("agents")
        assert t is not None
        assert len(t.evidence) == 2
        assert t.last_seen == "2026-06-03"
        assert t.first_seen == "2026-06-01"

    @pytest.mark.asyncio
    async def test_idempotent_double_run(self, monkeypatch):
        store = _FakeStore()
        obs = [{"trend_id": "", "new_title": "Topic", "summary": "same evidence"}]
        tracker = _patched_tracker(store, obs, monkeypatch)
        await tracker.update_trends("d", "2026-06-05")

        again = [{"trend_id": "topic", "new_title": "", "summary": "same evidence"}]
        tracker2 = _patched_tracker(store, again, monkeypatch)
        await tracker2.update_trends("d", "2026-06-05")

        memory = TrendMemory.model_validate_json(store.data[TRENDS_KEY])
        assert len(memory.trends) == 1
        assert len(memory.trends[0].evidence) == 1

    @pytest.mark.asyncio
    async def test_rerun_same_date_reworded_does_not_double_append(self, monkeypatch):
        # The real failure mode: a re-run of the same date yields a REWORDED summary.
        # Same-date evidence is dropped before re-ingest, so the count stays at 1.
        existing = TrendMemory(
            trends=[
                Trend(
                    id="agents",
                    title="Agents",
                    first_seen="2026-06-05",
                    last_seen="2026-06-05",
                    evidence=[TrendEvidence(date="2026-06-05", summary="first wording")],
                )
            ]
        )
        store = _FakeStore({TRENDS_KEY: existing.model_dump_json()})
        obs = [{"trend_id": "agents", "new_title": "", "summary": "completely different wording"}]
        tracker = _patched_tracker(store, obs, monkeypatch)
        await tracker.update_trends("d", "2026-06-05")

        memory = TrendMemory.model_validate_json(store.data[TRENDS_KEY])
        t = memory.by_id("agents")
        assert t is not None
        same_date = [ev for ev in t.evidence if ev.date == "2026-06-05"]
        assert len(same_date) == 1
        assert same_date[0].summary == "completely different wording"

    @pytest.mark.asyncio
    async def test_malformed_llm_output_no_crash(self, monkeypatch):
        store = _FakeStore()
        tracker = _patched_tracker(store, "not json at all", monkeypatch)
        await tracker.update_trends("d", "2026-06-05")
        memory = TrendMemory.model_validate_json(store.data[TRENDS_KEY])
        assert memory.trends == []


class TestLifecycle:
    def test_status_transitions_by_date(self):
        config = PipelineConfig(trend_cooling_days=7, trend_retention_days=30)
        tracker = TrendTracker(config, MagicMock(), _FakeStore())
        active = Trend(id="a", title="A", first_seen="2026-06-01", last_seen="2026-06-04")
        cooling = Trend(id="c", title="C", first_seen="2026-05-01", last_seen="2026-05-25")
        archived = Trend(id="z", title="Z", first_seen="2026-04-01", last_seen="2026-05-01")
        today = date(2026, 6, 5)
        assert tracker._compute_status(active, today) == TrendStatus.ACTIVE
        assert tracker._compute_status(cooling, today) == TrendStatus.COOLING
        assert tracker._compute_status(archived, today) == TrendStatus.ARCHIVED

    @pytest.mark.asyncio
    async def test_max_evidence_cap_drops_oldest(self, monkeypatch):
        ev = [TrendEvidence(date=f"2026-06-0{i}", summary=f"e{i}") for i in range(1, 6)]
        existing = TrendMemory(
            trends=[Trend(id="t", title="T", first_seen="2026-06-01", last_seen="2026-06-05", evidence=ev)]
        )
        store = _FakeStore({TRENDS_KEY: existing.model_dump_json()})
        obs = [{"trend_id": "t", "new_title": "", "summary": "newest"}]
        tracker = _patched_tracker(store, obs, monkeypatch, trend_max_evidence=3)
        await tracker.update_trends("d", "2026-06-06")

        memory = TrendMemory.model_validate_json(store.data[TRENDS_KEY])
        t = memory.by_id("t")
        assert t is not None
        assert len(t.evidence) == 3
        assert t.evidence[-1].summary == "newest"
        assert t.evidence[0].summary == "e4"

    @pytest.mark.asyncio
    async def test_active_cap_archives_lowest_momentum(self, monkeypatch):
        trends = [
            Trend(
                id=f"t{i}",
                title=f"T{i}",
                first_seen="2026-05-01",
                last_seen=f"2026-06-0{i}",
                evidence=[TrendEvidence(date=f"2026-06-0{i}", summary="e")],
            )
            for i in range(1, 4)
        ]
        store = _FakeStore({TRENDS_KEY: TrendMemory(trends=trends).model_dump_json()})
        tracker = _patched_tracker(store, [], monkeypatch, trend_max_active_trends=2, trend_cooling_days=60)
        await tracker.update_trends("d", "2026-06-05")

        memory = TrendMemory.model_validate_json(store.data[TRENDS_KEY])
        active = {t.id for t in memory.trends if t.status == TrendStatus.ACTIVE}
        assert active == {"t2", "t3"}


class TestGetTrendsContext:
    def test_excludes_archived_and_sorts_by_momentum(self):
        config = PipelineConfig(trend_momentum_half_life_days=7.0)
        trends = [
            Trend(
                id="low",
                title="Low",
                status=TrendStatus.ACTIVE,
                first_seen="2026-05-01",
                last_seen="2026-05-20",
                evidence=[TrendEvidence(date="2026-05-20", summary="old")],
            ),
            Trend(
                id="high",
                title="High",
                status=TrendStatus.ACTIVE,
                first_seen="2026-06-01",
                last_seen=date.today().isoformat(),
                evidence=[TrendEvidence(date=date.today().isoformat(), summary="fresh")],
            ),
            Trend(
                id="gone", title="Gone", status=TrendStatus.ARCHIVED, first_seen="2026-01-01", last_seen="2026-02-01"
            ),
        ]
        store = _FakeStore({TRENDS_KEY: TrendMemory(trends=trends).model_dump_json()})
        tracker = TrendTracker(config, MagicMock(), store)
        ctx = tracker.get_trends_context()
        assert "Gone" not in ctx
        assert ctx.index("High") < ctx.index("Low")

    def test_empty_when_no_visible_trends(self):
        store = _FakeStore({TRENDS_KEY: TrendMemory().model_dump_json()})
        tracker = TrendTracker(PipelineConfig(), MagicMock(), store)
        assert tracker.get_trends_context() == ""
