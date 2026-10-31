"""Orchestration tests for application.digest_pipeline — the single path both the CLI and the digest
Lambda take from collected items to a stored digest. Only the LLM collaborators (ranker, digest
generator, trend tracker) are stubbed; the aggregation, cross-day dedup, ledger, leads log and
state-store wiring under test run for real against a temp-dir state store."""

from __future__ import annotations

import tempfile
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters.state_store import LocalStateStore
from application import digest_pipeline as runner
from application.history import PUBLISHED_URLS_KEY, RECENT_LEADS_KEY
from domain.models import CollectedItem, DigestContent, DigestItem, DigestResult, RankedItem
from shared.config import Config
from shared.constants import SourceType

DIGEST_DATE = date(2026, 6, 3)


def _item(url: str, title: str = "t") -> CollectedItem:
    return CollectedItem(source_type=SourceType.WEB, title=title, url=url, text="body")


def _ranked(items: list[CollectedItem]) -> list[RankedItem]:
    return [RankedItem(item=it, score=0.9) for it in items]


def _digest(items: list[CollectedItem], lead: str = "리드 문장이다.") -> DigestResult:
    content = DigestContent(
        lead=lead,
        items=[DigestItem(title=it.title, url=it.url, body="본문") for it in items],
    )
    return DigestResult(digest_text="digest body", ranked_items=_ranked(items), content=content)


class _Pipeline:
    """Stubbed LLM collaborators + a real temp-dir state store, patched into application.digest_pipeline for one run."""

    def __init__(self, tmpdir: str, *, digest: DigestResult | None = None, ranked: list[RankedItem] | None = None):
        self.store = LocalStateStore(tmpdir)
        self.ranked = ranked
        self.digest = digest
        self.ranker = MagicMock()
        self.generator = MagicMock()
        self.tracker = MagicMock()
        self.tracker.get_trends_context.return_value = "trends"
        self.tracker.update_trends = AsyncMock()
        self.memory = MagicMock()
        self.memory.get_recent_digests.return_value = []

    def __enter__(self):
        ranked = self.ranked
        digest = self.digest

        async def rank(items, select_count=None, recent_titles=()):
            return ranked if ranked is not None else _ranked(items)

        async def generate(ranked_items, **kwargs):
            self.generate_kwargs = kwargs
            return digest if digest is not None else _digest([r.item for r in ranked_items])

        self.ranker.rank = AsyncMock(side_effect=rank)
        self.generator.generate = AsyncMock(side_effect=generate)
        self._patches = [
            patch.object(runner, "ContentRanker", return_value=self.ranker),
            patch.object(runner, "DigestGenerator", return_value=self.generator),
            patch.object(runner, "TrendTracker", return_value=self.tracker),
        ]
        for patcher in self._patches:
            patcher.start()
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def _config(**pipeline_overrides) -> Config:
    config = Config()
    config.pipeline.enable_threads_post = False
    for key, value in pipeline_overrides.items():
        setattr(config.pipeline, key, value)
    return config


async def _run(pipeline: _Pipeline, config: Config, items: list[CollectedItem], **kwargs):
    return await runner.run_pipeline(
        config,
        MagicMock(),
        items,
        DIGEST_DATE,
        state_store=pipeline.store,
        memory_store=lambda: pipeline.memory,
        **kwargs,
    )


class TestRunPipelineShortCircuits:
    @pytest.mark.asyncio
    async def test_no_items_after_aggregation_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                result = await _run(p, _config(), [])
        assert result == (None, None, None)
        p.ranker.rank.assert_not_called()  # never pays for a ranking call on an empty day

    @pytest.mark.asyncio
    async def test_nothing_above_threshold_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir, ranked=[]) as p:
                result = await _run(p, _config(), [_item("https://a.example/1")])
        assert result == (None, None, None)
        p.generator.generate.assert_not_called()  # no digest generated from nothing


class TestRunPipelineHappyPath:
    @pytest.mark.asyncio
    async def test_records_ledger_leads_and_updates_trends(self):
        items = [_item("https://a.example/1", "A"), _item("https://b.example/2", "B")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                collected, ranked, digest = await _run(p, _config(), items)
                ledger = p.store.read_json(PUBLISHED_URLS_KEY)
                leads = p.store.read_json(RECENT_LEADS_KEY)
        assert len(collected) == 2 and len(ranked) == 2 and digest is not None
        # Every published URL is stamped with today's date so tomorrow's run skips the story.
        assert set(ledger) == {"https://a.example/1", "https://b.example/2"}
        assert set(ledger.values()) == {DIGEST_DATE.isoformat()}
        assert [e["date"] for e in leads] == [DIGEST_DATE.isoformat()]
        p.tracker.update_trends.assert_awaited_once()
        assert p.tracker.update_trends.await_args.args[1] == DIGEST_DATE.isoformat()

    @pytest.mark.asyncio
    async def test_a_failed_trend_update_does_not_cost_the_days_digest(self):
        # Trends only feed tomorrow's lead; a throttled Sonnet call there used to raise after the
        # digest was paid for, and nothing was persisted or posted that day.
        items = [_item("https://a.example/1", "A")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                p.tracker.update_trends.side_effect = RuntimeError("throttled")
                _, _, digest = await _run(p, _config(), items)
                ledger = p.store.read_json(PUBLISHED_URLS_KEY)
        assert digest is not None
        assert set(ledger) == {"https://a.example/1"}

    @pytest.mark.asyncio
    async def test_recent_leads_are_fed_back_to_the_generator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                p.store.write_json(RECENT_LEADS_KEY, [{"date": "2026-06-02", "lead": "어제의 리드다."}])
                await _run(p, _config(), [_item("https://a.example/1")])
        assert p.generate_kwargs["recent_leads"] == ["어제의 리드다."]
        assert p.generate_kwargs["trends_context"] == "trends"
        assert p.generate_kwargs["today"] == DIGEST_DATE

    @pytest.mark.asyncio
    async def test_lead_is_stored_without_the_agi_countdown_prefix(self):
        # The countdown is a fixed daily template; storing it would make every lead look similar
        # to the novelty check, defeating the anti-repetition feedback.
        config = _config(agi_countdown_date="2029-01-01", agi_countdown_template="AGI 등장 {days}일 전이다. ")
        intro = runner.agi_countdown_intro(
            config.pipeline.agi_countdown_date,
            config.pipeline.agi_countdown_template,
            DIGEST_DATE,
            config.pipeline.agi_countdown_after,
        )
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir, digest=_digest(items, lead=intro + "본론 리드다.")) as p:
                await _run(p, config, items)
                leads = p.store.read_json(RECENT_LEADS_KEY)
        assert intro  # the countdown is actually configured, so the test isn't vacuous
        assert leads[0]["lead"] == "본론 리드다."

    @pytest.mark.asyncio
    async def test_select_count_over_selects_by_the_candidate_buffer(self):
        config = _config(top_n=5, digest_candidate_buffer=3)
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                await _run(p, config, [_item("https://a.example/1")])
        assert p.ranker.rank.await_args.kwargs == {"select_count": 8, "recent_titles": []}


class TestRunPipelineDedup:
    @pytest.mark.asyncio
    async def test_ledger_urls_are_excluded_from_aggregation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                p.store.write_json(PUBLISHED_URLS_KEY, {"https://old.example/1": "2026-06-01"})
                collected, _ranked, _digest_result = await _run(
                    p, _config(), [_item("https://old.example/1"), _item("https://new.example/2")]
                )
        assert [it.url for it in collected] == ["https://new.example/2"]

    @pytest.mark.asyncio
    async def test_memory_snapshot_urls_are_excluded_and_normalized(self):
        # Cross-day dedup must also self-heal from AgentCore Memory snapshots, matching http/https
        # and trailing-slash variants of a story published earlier in the window.
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                p.memory.get_recent_digests.return_value = [
                    {"digest_result": {"content": {"items": [{"url": "http://old.example/1/"}]}}}
                ]
                collected, _r, _d = await _run(
                    p, _config(), [_item("https://old.example/1"), _item("https://new.example/2")]
                )
        assert [it.url for it in collected] == ["https://new.example/2"]
        # The ledger's window: ages 1..ttl-1, so a story is allowed back on the TTL day by both.
        assert p.memory.get_recent_digests.call_args.args == (DIGEST_DATE, 5)

    @pytest.mark.asyncio
    async def test_memory_failure_is_non_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                p.memory.get_recent_digests.side_effect = RuntimeError("agentcore down")
                collected, ranked, digest = await _run(p, _config(), [_item("https://a.example/1")])
        assert collected and ranked and digest  # the digest still ships


class TestRunPipelineDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_writes_no_state(self):
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                collected, ranked, digest = await _run(p, _config(), items, dry_run=True)
                assert p.store.read_json(PUBLISHED_URLS_KEY) is None
                assert p.store.read_json(RECENT_LEADS_KEY) is None
        assert collected and ranked and digest
        p.tracker.update_trends.assert_not_awaited()


class TestPublishInline:
    """The local run's visual and Threads post. In AWS the digest Lambda fires the visual Lambda
    instead, so this is main.py's path only, and its exit code is the only alert."""

    @staticmethod
    def _maker(outcome=None, error=None):
        maker = MagicMock()
        maker.run = AsyncMock(return_value=True, side_effect=error)
        maker.threads_outcome = outcome
        return maker

    @pytest.mark.asyncio
    async def test_the_visual_runs_for_the_date_and_carries_force_republish(self):
        items = [_item("https://a.example/1")]
        maker = self._maker()
        await runner.publish_inline(maker, _ranked(items), _digest(items), DIGEST_DATE, force_republish=True)
        maker.run.assert_awaited_once()
        assert maker.run.await_args.kwargs["force_republish"] is True
        assert maker.run.await_args.kwargs["today"] == DIGEST_DATE

    @pytest.mark.asyncio
    async def test_a_failed_publish_is_reported(self):
        # Locally there is no alert: a swallowed failure ended main.py with exit 0 and "completed".
        items = [_item("https://a.example/1")]
        with pytest.raises(runner.PublishIncomplete, match="openai down"):
            await runner.publish_inline(
                self._maker(error=RuntimeError("openai down")), _ranked(items), _digest(items), DIGEST_DATE
            )

    @pytest.mark.asyncio
    async def test_an_incomplete_post_is_reported(self):
        from domain.models import ThreadsDelivery

        items = [_item("https://a.example/1")]
        maker = self._maker(ThreadsDelivery(5, 6, root_id="r", detached=("s",)))
        with pytest.raises(runner.PublishIncomplete, match="5/6 posts.*standalone s"):
            await runner.publish_inline(maker, _ranked(items), _digest(items), DIGEST_DATE)


class TestRunPipelineHistory:
    @pytest.mark.asyncio
    async def test_history_write_failure_is_non_fatal(self):
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                with patch.object(p.store, "write_json", side_effect=RuntimeError("disk full")):
                    collected, ranked, digest = await _run(p, _config(), items)
        assert collected and ranked and digest


class TestRecentStoryTitles:
    """The editor is told what recent digests ran, read off the snapshots the cross-day dedup seed
    already fetched — no extra call, and no change to the URL ledger."""

    def test_titles_come_from_every_snapshot_in_the_window_newest_first(self):
        # A weekly roundup re-surfaces a story under a new URL; one day of titles missed three
        # repeats on 2026-10-11.
        newest = {"digest_result": {"content": {"items": [{"title": "어제 1"}, {"title": "어제 2"}]}}}
        older = {"digest_result": {"content": {"items": [{"title": "그제 1"}, {"title": "어제 1"}]}}}
        assert runner._recent_story_titles([newest, older]) == ["어제 1", "어제 2", "그제 1"]

    def test_a_story_less_or_missing_snapshot_degrades_to_nothing(self):
        assert runner._recent_story_titles([]) == []
        assert runner._recent_story_titles([{"digest_result": None}]) == []
        assert runner._recent_story_titles([{"digest_result": {"content": {"items": []}}}]) == []


class TestResolveDigestWindow:
    """The collection cutoff is load-bearing: every collector's lookback_hours counts back from it,
    so an off-by-a-day reference time silently changes what the digest can even see. Extracted as a
    pure helper (from the CLI and digest_handler._run, which each had their own copy)."""

    def test_defaults_to_today_in_the_configured_timezone(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        config = Config()
        config.aws.timezone = "Asia/Seoul"
        digest_date, reference_time = runner.resolve_digest_window(config)
        assert digest_date == datetime.now(ZoneInfo("Asia/Seoul")).date()
        assert reference_time.tzinfo == ZoneInfo("Asia/Seoul")

    def test_explicit_date_gets_that_days_own_window(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        config = Config()
        config.aws.timezone = "Asia/Seoul"
        digest_date, reference_time = runner.resolve_digest_window(config, "2026-06-03")
        assert digest_date == date(2026, 6, 3)
        # Midnight at the END of the digest date, so a 19:00 run still sees the whole day and a
        # re-run of a past day sees that day rather than a window ending now.
        assert reference_time == datetime(2026, 6, 4, tzinfo=ZoneInfo("Asia/Seoul"))

    def test_the_boundary_follows_the_configured_timezone(self):
        from datetime import UTC, datetime
        from zoneinfo import ZoneInfo

        config = Config()
        config.aws.timezone = "America/New_York"
        _, reference_time = runner.resolve_digest_window(config, "2026-06-03")
        assert reference_time == datetime(2026, 6, 4, tzinfo=ZoneInfo("America/New_York"))
        # 04:00 UTC on the 4th, not 00:00 — the same date means a different instant per timezone.
        assert reference_time.astimezone(UTC) == datetime(2026, 6, 4, 4, 0, tzinfo=UTC)


class TestBoundedExecutor:
    @pytest.mark.asyncio
    async def test_one_bounded_pool_is_installed_from_config(self):
        # Each collector's max_concurrency bounds only its own fan-out; every asyncio.to_thread call
        # lands in the loop's DEFAULT executor (6 threads on a 2-vCPU Lambda), so the per-collector
        # bounds did not hold globally.
        import asyncio

        config = Config()
        config.collectors.thread_pool_max_workers = 9
        await runner.run_collectors_with_health(config, [])
        executor = asyncio.get_running_loop()._default_executor
        assert executor is not None
        assert executor._max_workers == 9


class TestAlreadyPublished:
    def test_reads_the_threads_ledger_for_the_date(self):
        from adapters.state_store import LocalStateStore
        from application.history import ThreadsPostLedger

        with tempfile.TemporaryDirectory() as tmp:
            store = LocalStateStore(tmp)
            ThreadsPostLedger(store).mark(date(2026, 10, 26), "run-1")
            config = Config()
            config.pipeline.enable_threads_post = True
            assert runner.already_published(config, date(2026, 10, 26), store) is True
            assert runner.already_published(config, date(2026, 10, 27), store) is False
            config.pipeline.enable_threads_post = False  # nothing can post: a re-run rebuilds the record
            assert runner.already_published(config, date(2026, 10, 26), store) is False
