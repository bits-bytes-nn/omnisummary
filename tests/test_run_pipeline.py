"""Orchestration tests for main.run_pipeline — the single path both the CLI and the digest Lambda
take from collected items to a delivered digest. Only the LLM/network collaborators (ranker, digest
generator, trend tracker, delivery) are stubbed; the aggregation, cross-day dedup, ledger, leads log
and state-store wiring under test run for real against a temp-dir state store."""

from __future__ import annotations

import tempfile
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

import main
from shared.config import Config
from shared.constants import SourceType
from shared.history_store import PUBLISHED_URLS_KEY, RECENT_LEADS_KEY
from shared.models import CollectedItem, DigestContent, DigestItem, DigestResult, RankedItem
from shared.state_store import LocalStateStore

DIGEST_DATE = date(2026, 6, 3)


def _item(url: str, title: str = "t") -> CollectedItem:
    return CollectedItem(source_type=SourceType.WEB, title=title, url=url, text="body")


def _ranked(items: list[CollectedItem]) -> list[RankedItem]:
    return [RankedItem(item=it, score=0.9) for it in items]


def _digest(items: list[CollectedItem], lead: str = "리드 문장이다.") -> DigestResult:
    content = DigestContent(
        lead=lead,
        headline_index=1,
        items=[DigestItem(title=it.title, url=it.url, body="본문") for it in items],
    )
    return DigestResult(digest_text="digest body", ranked_items=_ranked(items), content=content)


class _Pipeline:
    """Stubbed LLM collaborators + a real temp-dir state store, patched into main for one run."""

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
        self.send_slack = AsyncMock(return_value=True)
        self.visual = MagicMock()
        self.visual.run = AsyncMock(return_value=True)

    def __enter__(self):
        ranked = self.ranked
        digest = self.digest

        async def rank(items, select_count=None, core_count=None):
            self.ranked_input = items
            return ranked if ranked is not None else _ranked(items)

        async def generate(ranked_items, items, **kwargs):
            self.generate_kwargs = kwargs
            return digest if digest is not None else _digest([r.item for r in ranked_items])

        self.ranker.rank = AsyncMock(side_effect=rank)
        self.generator.generate = AsyncMock(side_effect=generate)
        self._patches = [
            patch.object(main, "create_state_store", return_value=self.store),
            patch.object(main, "create_memory_store", return_value=self.memory),
            patch.object(main, "ContentRanker", return_value=self.ranker),
            patch.object(main, "DigestGenerator", return_value=self.generator),
            patch.object(main, "TrendTracker", return_value=self.tracker),
            patch.object(main, "send_digest_to_slack", self.send_slack),
            patch.object(main, "persist_digest"),
            patch("pipeline.daily_visual.DailyVisualMaker", return_value=self.visual),
        ]
        started = [p.start() for p in self._patches]
        self.persist_digest = started[6]
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def _config(**pipeline_overrides) -> Config:
    config = Config()
    config.pipeline.enable_slack_post = False
    config.pipeline.enable_threads_post = False
    config.pipeline.enable_daily_visual = False
    for key, value in pipeline_overrides.items():
        setattr(config.pipeline, key, value)
    return config


async def _run(pipeline: _Pipeline, config: Config, items: list[CollectedItem], **kwargs):
    return await main.run_pipeline(config, MagicMock(), items, digest_date=DIGEST_DATE, **kwargs)


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
        intro = main.agi_countdown_intro(
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
        assert p.ranker.rank.await_args.kwargs["select_count"] == 8
        # ...while the source-slot guarantees are enforced on the top_n the reader actually gets.
        assert p.ranker.rank.await_args.kwargs["core_count"] == 5


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
        kwargs = p.memory.get_recent_digests.call_args.kwargs
        assert kwargs["exclude_date"] == DIGEST_DATE.isoformat()  # today's own snapshot is kept
        assert kwargs["after_date"] == "2026-05-28"  # floored at digest_date - ttl (6 days)

    @pytest.mark.asyncio
    async def test_memory_failure_is_non_fatal(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                p.memory.get_recent_digests.side_effect = RuntimeError("agentcore down")
                collected, ranked, digest = await _run(p, _config(), [_item("https://a.example/1")])
        assert collected and ranked and digest  # the digest still ships


class TestRunPipelineDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_writes_no_state_and_delivers_nothing(self):
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                collected, ranked, digest = await _run(
                    p, _config(enable_slack_post=True, enable_daily_visual=True), items, dry_run=True
                )
                assert p.store.read_json(PUBLISHED_URLS_KEY) is None
                assert p.store.read_json(RECENT_LEADS_KEY) is None
        assert collected and ranked and digest
        p.tracker.update_trends.assert_not_awaited()
        p.send_slack.assert_not_awaited()
        p.visual.run.assert_not_awaited()


class TestRunPipelineDelivery:
    @pytest.mark.asyncio
    async def test_slack_post_only_when_enabled(self):
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                await _run(p, _config(enable_slack_post=False), items)
                p.send_slack.assert_not_awaited()
            with _Pipeline(tmpdir) as p:
                await _run(p, _config(enable_slack_post=True), items)
                p.send_slack.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_visual_runs_inline_outside_aws_and_carries_force_republish(self):
        # In AWS the digest Lambda fires a separate visual Lambda; locally it must run inline so
        # `uv run python main.py` still produces the visual (and the Threads fan-out it owns).
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                with patch.object(main, "is_running_in_aws", return_value=False):
                    await _run(p, _config(enable_daily_visual=True), items, force_republish=True)
        p.visual.run.assert_awaited_once()
        assert p.visual.run.await_args.kwargs["force_republish"] is True
        assert p.visual.run.await_args.kwargs["today"] == DIGEST_DATE

    @pytest.mark.asyncio
    async def test_visual_is_not_run_inline_in_aws(self):
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                with patch.object(main, "is_running_in_aws", return_value=True):
                    await _run(p, _config(enable_daily_visual=True), items)
        p.visual.run.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_visual_failure_is_non_fatal(self):
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                p.visual.run.side_effect = RuntimeError("openai down")
                with patch.object(main, "is_running_in_aws", return_value=False):
                    collected, ranked, digest = await _run(p, _config(enable_daily_visual=True), items)
        assert collected and ranked and digest  # the digest itself already shipped

    @pytest.mark.asyncio
    async def test_local_run_persists_the_snapshot(self):
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                with patch.object(main, "is_running_in_aws", return_value=False):
                    await _run(p, _config(), items)
        # persist_digest is stubbed by _Pipeline; assert the local branch called it with the
        # on-disk fallback dir (in AWS the handler passes base_dir=None for AgentCore Memory).
        assert p.persist_digest.called
        assert p.persist_digest.call_args.kwargs["base_dir"] == Path("digest_state")

    @pytest.mark.asyncio
    async def test_history_write_failure_is_non_fatal(self):
        items = [_item("https://a.example/1")]
        with tempfile.TemporaryDirectory() as tmpdir:
            with _Pipeline(tmpdir) as p:
                with patch.object(p.store, "write_json", side_effect=RuntimeError("disk full")):
                    collected, ranked, digest = await _run(p, _config(), items)
        assert collected and ranked and digest


class TestRecentStoryTitles:
    """The editor is told what the LAST digest ran, read off the snapshots the cross-day dedup seed
    already fetched — no extra call, and no change to the URL ledger."""

    def test_titles_come_from_the_newest_snapshot(self):
        newest = {"digest_result": {"content": {"items": [{"title": "어제 1"}, {"title": "어제 2"}]}}}
        older = {"digest_result": {"content": {"items": [{"title": "그제 1"}]}}}
        assert main._recent_story_titles([newest, older]) == ["어제 1", "어제 2"]

    def test_a_story_less_or_missing_snapshot_degrades_to_nothing(self):
        assert main._recent_story_titles([]) == []
        assert main._recent_story_titles([{"digest_result": None}]) == []
        assert main._recent_story_titles([{"digest_result": {"content": {"items": []}}}]) == []


class TestResolveDigestWindow:
    """The collection cutoff is load-bearing: every collector's lookback_hours counts back from it,
    so an off-by-a-day reference time silently changes what the digest can even see. Extracted as a
    pure helper (from main.main and digest_handler._run, which each had their own copy)."""

    def test_defaults_to_today_in_the_configured_timezone(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        config = Config()
        config.aws.timezone = "Asia/Seoul"
        digest_date, reference_time = main._resolve_digest_window(config)
        assert digest_date == datetime.now(ZoneInfo("Asia/Seoul")).date()
        assert reference_time.tzinfo == ZoneInfo("Asia/Seoul")

    def test_explicit_date_gets_that_days_own_window(self):
        from datetime import datetime
        from zoneinfo import ZoneInfo

        config = Config()
        config.aws.timezone = "Asia/Seoul"
        digest_date, reference_time = main._resolve_digest_window(config, "2026-06-03")
        assert digest_date == date(2026, 6, 3)
        # Midnight at the END of the digest date, so a 19:00 run still sees the whole day and a
        # re-run of a past day sees that day rather than a window ending now.
        assert reference_time == datetime(2026, 6, 4, tzinfo=ZoneInfo("Asia/Seoul"))

    def test_the_boundary_follows_the_configured_timezone(self):
        from datetime import UTC, datetime
        from zoneinfo import ZoneInfo

        config = Config()
        config.aws.timezone = "America/New_York"
        _, reference_time = main._resolve_digest_window(config, "2026-06-03")
        assert reference_time == datetime(2026, 6, 4, tzinfo=ZoneInfo("America/New_York"))
        # 04:00 UTC on the 4th, not 00:00 — the same date means a different instant per timezone.
        assert reference_time.astimezone(UTC) == datetime(2026, 6, 4, 4, 0, tzinfo=UTC)


class TestBuildCollectorTasks:
    """Patched out by every other caller, so none of these branches ever executed: the --sources
    override, the unknown-source warning, the disabled-source skip, and llm_factory being passed to
    web_search alone."""

    @staticmethod
    def _config() -> Config:
        config = Config()
        for cfg in (
            config.collectors.rss,
            config.collectors.reddit,
            config.collectors.youtube,
            config.collectors.web_search,
        ):
            cfg.enabled = True
        config.collectors.rsshub.enabled = False
        return config

    @staticmethod
    def _close(tasks) -> None:
        # The coroutines are never awaited here; close them so pytest reports no un-awaited warning.
        for task in tasks:
            task.close()

    def test_enabled_sources_are_collected_when_no_override_is_given(self):
        tasks, labels, collectors = main._build_collector_tasks(self._config(), MagicMock())
        self._close(tasks)
        assert labels == ["reddit", "rss", "web_search", "youtube"]  # rsshub is disabled
        assert len(collectors) == len(labels) == len(tasks)

    def test_sources_override_selects_exactly_what_was_asked_for(self):
        tasks, labels, _ = main._build_collector_tasks(self._config(), MagicMock(), ["rss", "reddit"])
        self._close(tasks)
        assert labels == ["rss", "reddit"]  # the caller's order, not the map's

    def test_an_unknown_source_is_warned_about_and_skipped(self):
        with patch.object(main, "logger") as log:
            tasks, labels, _ = main._build_collector_tasks(self._config(), MagicMock(), ["rss", "nope"])
        self._close(tasks)
        assert labels == ["rss"]
        assert any("Unknown source" in str(c.args) for c in log.warning.call_args_list)

    def test_an_explicitly_requested_but_disabled_source_is_skipped(self):
        # --sources rsshub on a deployment where rsshub is disabled must NOT start it.
        tasks, labels, _ = main._build_collector_tasks(self._config(), MagicMock(), ["rsshub", "rss"])
        self._close(tasks)
        assert labels == ["rss"]

    def test_only_web_search_receives_the_llm_factory(self):
        from collectors.web_search import WebSearchCollector

        factory = MagicMock()
        tasks, labels, collectors = main._build_collector_tasks(self._config(), factory, ["web_search", "rss"])
        self._close(tasks)
        web = collectors[labels.index("web_search")]
        assert isinstance(web, WebSearchCollector)
        # The factory is only ever used to build the query-refine model; without it, web_search
        # silently loses query refinement.
        assert web._llm is factory.get_model.return_value
        factory.get_model.assert_called_once()
        assert factory.get_model.call_args.kwargs["stage"] == "query-refine"


class TestBoundedExecutor:
    @pytest.mark.asyncio
    async def test_one_bounded_pool_is_installed_from_config(self):
        # Each collector's max_concurrency bounds only its own fan-out; every asyncio.to_thread call
        # lands in the loop's DEFAULT executor (6 threads on a 2-vCPU Lambda), so the per-collector
        # bounds did not hold globally.
        import asyncio

        config = Config()
        config.collectors.thread_pool_max_workers = 9
        with patch.object(main, "_build_collector_tasks", return_value=([], [], [])):
            await main.run_collectors_with_health(config=config, llm_factory=None)
        executor = asyncio.get_running_loop()._default_executor
        assert executor is not None
        assert executor._max_workers == 9
