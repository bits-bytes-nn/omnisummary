from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from application.daily_visual import DailyVisualMaker
from domain.models import CollectedItem, RankedItem, ThreadsDelivery, VisualBrief
from shared.config import Config
from shared.constants import SourceType
from tests.fakes import FakeRenderer, FakeThreads, MemoryStateStore, fake_models


def _maker(*, renderer_ready: bool = True) -> DailyVisualMaker:
    # An in-memory state store, so visual tests don't read/write the repo's digest_state dir (which
    # made format/idempotency history order-dependent across runs).
    return DailyVisualMaker(
        Config(), fake_models(), MemoryStateStore(), FakeThreads(), FakeRenderer(ready=renderer_ready)
    )


def _rendering_maker() -> DailyVisualMaker:
    """A maker whose publisher always has somewhere to put the image, for tests about what gets
    RENDERED rather than where it is published."""
    maker = _maker()
    maker.publisher.has_a_destination = MagicMock(return_value=True)
    maker.publisher.publish = AsyncMock(return_value=True)
    return maker


def _items(n: int = 3) -> list[RankedItem]:
    return [
        RankedItem(
            item=CollectedItem(
                item_id=f"i{k}", source_type=SourceType.WEB, title=f"Story {k}", url=f"http://e.com/{k}", text="body"
            ),
            score=0.8,
        )
        for k in range(1, n + 1)
    ]


def _content(lead: str = "리드.", implication: str = "시사점이다.", url: str = "http://e.com/1"):
    from domain.models import DigestContent, DigestItem

    return DigestContent(
        lead=lead,
        items=[DigestItem(title="Story 1", url=url, body="본문이다.", implication=implication)],
    )


class TestTheArtDirectorInstruction:
    """The synopsis plus what the art director must keep."""

    def test_guardrails_are_appended_and_are_config_driven(self):
        # Depicting real people is ALLOWED (normal editorial-cartoon practice); what is barred is
        # standing a company or country up as an ethnically-coded human, as a 2026-08-15 visual did.
        maker = _maker()
        instruction = maker._build_instruction({"instruction": "x"})
        assert "GUARDRAILS:" in instruction
        assert "Recognisable depictions of real people are fine" in instruction
        assert "ethnically-coded human" in instruction
        maker.config.pipeline.visual_guardrails = ""
        assert "GUARDRAILS:" not in maker._build_instruction({"instruction": "x"})


class TestRecentComics:
    """The editor saw no earlier comics, so a run of similar stories got the same comic: 2026-10-13,
    -16 and -17 all locked a model behind glass. The posted comics are logged and shown to it."""

    @staticmethod
    async def _run(with_image: bool) -> DailyVisualMaker:
        # The real publisher: whether the root carried the image is the Threads outcome's verdict.
        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.generator.generate = AsyncMock(
            return_value=(b"PNG", VisualBrief(title="T", prompt="p", orientation="portrait", alt_text="1컷: 빵집."))
        )
        outcome = ThreadsDelivery(2, 2, with_image=with_image)
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value={"instruction": "draw"})):
            maker.threads.post_thread = AsyncMock(return_value=outcome)
            await maker.run(_items(), _content(), today=date(2026, 10, 17))
        return maker

    @pytest.mark.asyncio
    async def test_a_comic_that_reached_the_root_post_is_logged(self):
        maker = await self._run(with_image=True)
        assert maker.comic_log.entries() == [{"date": "2026-10-17", "orientation": "portrait", "comic": "1컷: 빵집."}]

    @pytest.mark.asyncio
    async def test_a_text_only_day_logs_nothing(self):
        maker = await self._run(with_image=False)
        assert maker.comic_log.entries() == []

    @pytest.mark.asyncio
    async def test_an_unreadable_log_costs_only_the_history(self):
        maker = _maker()
        maker.comic_log.entries = MagicMock(side_effect=ConnectionError("endpoint down"))
        seen: list[str] = []
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker.llm = RunnableLambda(
            lambda prompt: seen.append(prompt.to_messages()[-1].content) or AIMessage(content='{"instruction": "x"}')
        )
        _plan, instruction, _source = await maker._compose(_items(), _content(), date(2026, 10, 17))
        assert instruction.startswith("x")
        assert "(none yet)" in seen[0]

    @pytest.mark.asyncio
    async def test_the_editor_sees_the_comics_of_earlier_days_only(self):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _maker()
        for day, comic in (("2026-10-15", "교실 만화"), ("2026-10-16", "유리장 만화"), ("2026-10-17", "오늘 만화")):
            maker.comic_log.append({"date": day, "orientation": "portrait", "comic": comic})
        seen: list[str] = []
        maker.llm = RunnableLambda(lambda prompt: seen.append(str(prompt)) or AIMessage(content='{"skip": true}'))
        await maker._compose(_items(), _content(), date(2026, 10, 17))
        assert "- 2026-10-15 (portrait): 교실 만화" in seen[0]
        assert "유리장 만화" in seen[0]
        assert "오늘 만화" not in seen[0]

    def test_no_state_store_means_no_history(self):
        maker = DailyVisualMaker(Config(), fake_models(), None, FakeThreads(), FakeRenderer())
        assert maker.comic_log is None
        assert maker._recent_comics(date(2026, 10, 17)) == []


class TestThePublishPathStandsOnItsOwn:
    """The publish path used to live inside the class that generates the image, so nothing could
    exercise a delivery without also standing up an image generator (and an OpenAI key). It is now
    `DigestPublisher`, built from a config, a state store and the Threads port."""

    @staticmethod
    def _publisher(**pipeline_overrides):
        from application.digest_publisher import DigestPublisher

        config = Config()
        for key, value in pipeline_overrides.items():
            setattr(config.pipeline, key, value)
        return DigestPublisher(config, MemoryStateStore(), FakeThreads())

    @staticmethod
    def _content():
        from domain.models import DigestContent, DigestItem

        return DigestContent(
            lead="오늘의 리드.",
            items=[DigestItem(title="스토리", url="http://e.com/1", body="본문.")],
        )

    @pytest.mark.asyncio
    async def test_a_digest_publishes_with_no_image_at_all(self):
        publisher = self._publisher(enable_threads_post=True)
        publisher.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
        th = publisher.threads.post_thread
        assert await publisher.publish(self._content(), today=date(2026, 6, 10)) is True
        assert th.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_the_image_and_its_alt_text_go_to_the_post(self):
        publisher = self._publisher(enable_threads_post=True)
        publisher.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
        th = publisher.threads.post_thread
        await publisher.publish(self._content(), image_bytes=b"PNG", image_alt="1컷", today=date(2026, 6, 10))
        assert th.await_args.kwargs["image_bytes"] == b"PNG"
        assert th.await_args.kwargs["image_alt"] == "1컷"

    def test_an_already_published_day_has_no_destination_left(self):
        publisher = self._publisher(enable_threads_post=True)
        publisher.threads_ledger.mark(date(2026, 6, 10))
        assert publisher.has_a_destination(self._content(), date(2026, 6, 10), False) is False
        assert publisher.has_a_destination(self._content(), date(2026, 6, 10), True) is True

    def test_a_story_less_digest_leaves_a_verdict_before_anything_is_rendered(self):
        publisher = self._publisher(enable_threads_post=True)
        assert publisher.has_a_destination(None, date(2026, 6, 10), False) is False
        assert publisher.threads_outcome == ThreadsDelivery(0, 1)


class TestVisualFailureNeverCostsTheDigest:
    """Regression: run() returned early on a missing OpenAI key, a failed editor call and an editor
    skip — all BEFORE the only Threads publish path. A visual-only problem therefore cost the whole
    day's digest, silently. The image is an attachment; the text digest must still go out."""

    @staticmethod
    def _content():
        from domain.models import DigestContent, DigestItem

        return DigestContent(
            lead="오늘의 리드.",
            items=[DigestItem(title="스토리", url="http://e.com/1", body="본문.")],
        )

    @staticmethod
    def _threads_maker() -> DailyVisualMaker:
        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        return maker

    @pytest.mark.asyncio
    async def test_a_renderer_that_cannot_run_still_publishes_the_text_digest(self):
        # No OpenAI key, or one the parameter store would not hand over: the renderer says so (and
        # logs which); the image is an attachment, and this is the only Threads publish path.
        maker = _maker(renderer_ready=False)
        maker.config.pipeline.enable_threads_post = True
        maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
        th = maker.threads.post_thread
        with patch.object(maker, "_write_synopsis", new=AsyncMock()) as write:
            result = await maker.run(_items(), self._content(), today=date(2026, 6, 10))
        write.assert_not_awaited()
        maker.generator.renderer.render.assert_not_called()
        assert result is True
        th.assert_awaited_once()
        assert th.await_args.kwargs["root_text"] == "오늘의 리드."
        assert th.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_editor_failure_still_publishes_the_text_digest(self):
        maker = self._threads_maker()
        with patch.object(maker, "_write_synopsis", new=AsyncMock(side_effect=RuntimeError("bedrock down"))):
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            result = await maker.run(_items(), self._content(), today=date(2026, 6, 10))
        assert result is True
        th.assert_awaited_once()
        assert th.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_editor_skip_still_publishes_the_text_digest(self):
        maker = self._threads_maker()
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value={"skip": True})):
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            result = await maker.run(_items(), self._content(), today=date(2026, 6, 10))
        assert result is True
        th.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_render_is_paid_for_when_the_day_is_already_published(self):
        # The top-of-run short-circuit: nothing left to publish, so don't pay for the editor pass
        # and the gpt-image render.
        maker = self._threads_maker()
        maker.threads_ledger.mark(date(2026, 6, 10))
        maker.generator.generate = AsyncMock()
        with patch.object(maker, "_write_synopsis", new=AsyncMock()) as pick:
            maker.threads.post_thread = AsyncMock()
            th = maker.threads.post_thread
            assert await maker.run(_items(), self._content(), today=date(2026, 6, 10)) is False
        pick.assert_not_awaited()
        maker.generator.generate.assert_not_awaited()
        th.assert_not_awaited()


class TestDailyVisualMaker:
    @pytest.mark.asyncio
    async def test_skips_on_empty_items(self):
        assert await _maker().run([]) is False

    @pytest.mark.asyncio
    async def test_no_image_rendered_when_the_editor_skips(self):
        maker = _rendering_maker()
        maker.generator.generate = AsyncMock()
        plan = {"skip": True, "instruction": "draw"}  # a skip wins over a written synopsis
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            await maker.run(_items())
        maker.generator.generate.assert_not_awaited()
        assert maker.publisher.publish.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_visual_draws_the_headline_not_editor_pick(self):
        # The headline is authoritative: even if the editor returns a different item_number,
        # the visual must depict the digest headline (so image and lead stay in sync).
        from domain.models import DigestContent, DigestItem

        maker = _rendering_maker()
        plan = {"skip": False, "item_number": 3, "research": [], "instruction": "draw"}  # editor drifts
        content = DigestContent(
            lead="lead", items=[DigestItem(title="t", url="http://e.com/2", body="b")]
        )  # headline maps to ranked Story 2 by URL
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            result = await maker.run(_items(), content)
        assert result is True
        # source must be the headline (Story 2), NOT the editor's item_number=3
        args, kwargs = maker.generator.generate.call_args
        assert "Story 2" in args[1]

    @pytest.mark.asyncio
    async def test_threads_enabled_fans_out_with_content(self):
        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(
            lead="오늘의 리드.",
            items=[DigestItem(title="스토리", url="http://e.com/1", body="본문.")],
        )
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            await maker.run(_items(), content)
        th.assert_awaited_once()
        # root is the digest lead as-is (the AGI countdown is prepended upstream at digest
        # generation, so it's already part of content.lead — not added here); replies = per item
        root = th.await_args.kwargs["root_text"]
        assert root == "오늘의 리드."
        assert any("스토리" in r for r in th.await_args.kwargs["replies"])
        assert th.await_args.kwargs["image_bytes"] == b"PNG"

    @pytest.mark.asyncio
    async def test_story_less_content_posts_nothing_and_stays_retryable(self):
        # Regression (2026-08-13, 2026-08-17): when the digest carried no stories the visual
        # published its own title+caption as a lone root with zero replies — a story-less "digest"
        # that also burned the day's ledger slot. Nothing must be posted, and the ledger must stay
        # unmarked so the day can still be retried.
        from datetime import date

        from domain.models import DigestContent

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "instruction": "x"}
        empty = DigestContent(lead="리드만 남았다.", items=[])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            await maker.run(_items(), empty, today=date(2026, 6, 10))
        th.assert_not_awaited()
        assert maker.threads_ledger.already_posted(date(2026, 6, 10)) is False
        # …and it leaves a VERDICT behind: content existed, the channel was on, nothing published.
        # With threads_outcome left at None the caller's alert/metric was a silent no-op.
        assert maker.threads_outcome == ThreadsDelivery(0, 1)
        assert maker.threads_outcome.published is False

    @pytest.mark.asyncio
    async def test_a_channel_skip_leaves_no_verdict(self):
        # The already-posted ledger skip and the channel-disabled skip are NOT failures; they must
        # stay silent so the daily alert/metric only speaks when something actually went wrong.
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.threads_ledger.mark(date(2026, 6, 10))
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        posted = await maker.run(_items(), content, today=date(2026, 6, 10))
        assert posted is False
        assert maker.threads_outcome is None

    @pytest.mark.asyncio
    async def test_post_exception_still_leaves_a_verdict(self):
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "research": [], "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(side_effect=RuntimeError("boom"))
            posted = await maker.run(_items(), content, today=date(2026, 6, 10))
        assert posted is False
        assert maker.threads_outcome == ThreadsDelivery(0, 2)

    @pytest.mark.asyncio
    async def test_caller_deadline_is_forwarded_to_the_publish_path(self):
        # A plain monotonic float is threaded through — never the Lambda context object — and a
        # None deadline (local runs) must reach the publisher unchanged.
        import time
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "research": [], "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        deadline = time.monotonic() + 900
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait")))
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = gen
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            await maker.run(_items(), content, today=date(2026, 6, 10), deadline=deadline)
        from adapters.threads import THREADS_INDEXING_BUDGET_SEC, THREADS_PUBLISH_RESERVE_SEC

        assert th.await_args.kwargs["deadline"] == deadline
        # The image work must finish early enough to leave the Threads publish its full budget.
        assert gen.await_args.kwargs["deadline"] == deadline - THREADS_INDEXING_BUDGET_SEC - THREADS_PUBLISH_RESERVE_SEC

    @pytest.mark.asyncio
    async def test_a_hung_editor_past_the_deadline_still_posts_the_text(self):
        # The editor and brief calls sit on Bedrock (read_timeout 300 x 3); unbounded, a hang there ran
        # the Lambda out of time and lost the text digest with the image.
        import asyncio
        import time
        from datetime import date

        from adapters.threads import THREADS_INDEXING_BUDGET_SEC, THREADS_PUBLISH_RESERVE_SEC
        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])

        async def _hang(*_a, **_k):
            await asyncio.sleep(60)

        deadline = time.monotonic() + THREADS_INDEXING_BUDGET_SEC + THREADS_PUBLISH_RESERVE_SEC + 0.05
        with patch.object(maker, "_write_synopsis", new=_hang):
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            started = time.monotonic()
            assert await maker.run(_items(), content, today=date(2026, 6, 10), deadline=deadline) is True
        assert time.monotonic() - started < 5  # cut at the deadline, not after the hang
        assert th.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_a_run_that_claims_the_date_during_our_render_stops_our_post(self):
        # The render takes minutes. If another invocation posts and marks the date meanwhile, the
        # check right before posting is the only thing that prevents a second thread.
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])

        async def _render_while_another_run_posts(*args, **kwargs):
            maker.threads_ledger.mark(date(2026, 6, 10), "other-run")
            return b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait")

        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value={"skip": False, "instruction": "x"})):
            maker.generator.generate = _render_while_another_run_posts
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            await maker.run(_items(), content, today=date(2026, 6, 10))
        th.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_threads_skipped_when_already_posted_today(self):
        # A same-day re-run (or async Lambda retry) must not re-post the root+replies set.
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.threads_ledger.mark(date(2026, 6, 10))  # today already posted
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            await maker.run(_items(), content, today=date(2026, 6, 10))
        th.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_date_claimed_by_another_run_after_the_first_check_is_not_posted(self):
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.publisher.has_a_destination = MagicMock(return_value=True)
        maker.threads_ledger.already_posted = MagicMock(side_effect=[False, True])
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value={"skip": True})):
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            await maker.run(_items(), content, today=date(2026, 6, 10))
        th.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_threads_force_republish_bypasses_guard(self):
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.threads_ledger.mark(date(2026, 6, 10))
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            await maker.run(_items(), content, today=date(2026, 6, 10), force_republish=True)
        th.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_force_republish_with_threads_disabled_renders_nothing(self):
        # Force re-posts a day; it does not turn on a channel that is off (it used to pay for the
        # editor, the brief and the render, then post nowhere).
        maker = _maker()
        maker.config.pipeline.enable_threads_post = False
        with patch.object(maker, "_write_synopsis", new=AsyncMock()) as write:
            maker.generator.generate = AsyncMock()
            await maker.run(_items(), _content(), today=date(2026, 6, 10), force_republish=True)
        write.assert_not_awaited()
        maker.generator.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_plan_without_a_synopsis_is_a_skip(self):
        maker = _maker()
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value={"skip": False, "instruction": " "})):
            _, instruction, _ = await maker._compose(_items(), _content(), date(2026, 6, 10), recent_comics=[])
        assert instruction == ""

    @pytest.mark.asyncio
    async def test_force_republish_failure_keeps_prior_mark(self):
        # A force-republish that FAILS must not wipe out the date's existing successful-post
        # mark (only a mark WE added in this run is rolled back).
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.threads_ledger.mark(date(2026, 6, 10))  # prior successful post
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(0, 2))
            await maker.run(_items(), content, today=date(2026, 6, 10), force_republish=True)
        assert maker.threads_ledger.already_posted(date(2026, 6, 10))  # prior mark preserved

    @pytest.mark.asyncio
    async def test_threads_marks_ledger_after_successful_post(self):
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            await maker.run(_items(), content, today=date(2026, 6, 10))
        assert maker.threads_ledger.already_posted(date(2026, 6, 10))

    @pytest.mark.asyncio
    async def test_threads_not_marked_when_post_fails(self):
        # A failed Threads post must stay retryable — the optimistic claim is rolled back.
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(0, 2))
            await maker.run(_items(), content, today=date(2026, 6, 10))
        assert not maker.threads_ledger.already_posted(date(2026, 6, 10))

    @pytest.mark.asyncio
    async def test_threads_date_claimed_before_post_runs(self):
        # The concurrency fix: the date must already be marked at the moment post_to_threads
        # is entered, so a racing concurrent invocation sees it taken and skips.
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        claimed_during_post: list[bool] = []

        async def _post(**_kwargs):
            claimed_during_post.append(maker.threads_ledger.already_posted(date(2026, 6, 10)))
            return ThreadsDelivery(2, 2)

        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = _post
            await maker.run(_items(), content, today=date(2026, 6, 10))
        assert claimed_during_post == [True]

    @pytest.mark.asyncio
    async def test_threads_marker_rolled_back_when_post_raises(self):
        # An exception mid-post must roll the claim back so the date stays retryable.
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(side_effect=RuntimeError("boom"))
            await maker.run(_items(), content, today=date(2026, 6, 10))
        assert not maker.threads_ledger.already_posted(date(2026, 6, 10))

    @pytest.mark.asyncio
    async def test_a_live_root_whose_replies_all_failed_keeps_the_date_marked(self):
        # The lone root is on the account: releasing the date let a re-run post a second root.
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value={"instruction": "x"})):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(1, 2))
            assert await maker.run(_items(), content, today=date(2026, 6, 10)) is False
        assert maker.threads_ledger.already_posted(date(2026, 6, 10))

    @pytest.mark.asyncio
    async def test_a_root_that_may_be_live_keeps_the_date_marked(self):
        # A publish step that failed ambiguously may have posted: releasing the date let a DLQ replay
        # post a second root.
        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value={"instruction": "x"})):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(0, 2, uncertain=True))
            await maker.run(_items(), _content(), today=date(2026, 6, 10))
        assert maker.threads_ledger.already_posted(date(2026, 6, 10))

    @pytest.mark.asyncio
    async def test_threads_disabled_by_default(self):
        maker = _maker()
        plan = {"skip": False, "instruction": "x"}
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            await maker.run(_items())
        th.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_render_failure_publishes_text_only(self):
        maker = _rendering_maker()
        plan = {"skip": False, "item_number": 1, "instruction": "draw"}
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(side_effect=RuntimeError("boom"))
            await maker.run(_items())
        maker.generator.generate.assert_awaited_once()
        assert maker.publisher.publish.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_image_failure_still_posts_threads_text(self):
        # Regression: an image-generation failure (e.g. OpenAI billing hard limit) must NOT sink
        # the Threads text digest — the image is an optional attachment. Threads still posts the
        # lead + per-story replies, with image_bytes=None (text-only container).
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(
            lead="오늘의 리드.",
            items=[DigestItem(title="스토리", url="http://e.com/1", body="본문.")],
        )
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(side_effect=RuntimeError("billing hard limit"))
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            th = maker.threads.post_thread
            await maker.run(_items(), content, today=date(2026, 6, 10))
        th.assert_awaited_once()
        assert th.await_args.kwargs["root_text"] == "오늘의 리드."
        assert th.await_args.kwargs["image_bytes"] is None  # text-only, no attachment
        assert maker.threads_ledger.already_posted(date(2026, 6, 10))  # marked as posted

    @pytest.mark.asyncio
    async def test_threads_only_success_is_reported_as_posted(self):
        # The run returned False even though Threads published, so the Lambda logged "skipped" for
        # every successful day.
        from datetime import date

        from domain.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "instruction": "x"}
        content = DigestContent(lead="리드.", items=[DigestItem(title="s", url="http://e.com/1", body="b")])
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            result = await maker.run(_items(), content, today=date(2026, 6, 10))
        assert result is True

    def test_the_headline_source_is_its_ranked_article_matched_by_url(self):
        # The headline is content.items[0]; its ranked article (the full text) is found by URL.
        assert _maker()._headline(_content(url="http://e.com/3"), _items(3)) == "Story 3\n\nbody"

    def test_the_headline_matches_on_normalized_urls(self):
        # Regression: an exact string compare made a trailing slash / http→https / utm param an
        # "unmatched" headline.
        source = _maker()._headline(_content(url="https://www.e.com/2/?utm_source=x"), _items(3))
        assert source.startswith("Story 2")

    def test_a_ranked_article_stored_without_its_text_yields_to_the_curated_prose(self):
        # An oversize snapshot keeps the ranked items without text, and the editor got a bare title.
        items = _items(3)
        for r in items:
            r.item.text = ""
        assert _maker()._headline(_content(url="http://e.com/1"), items) == "Story 1\n\n본문이다."

    def test_no_content_draws_the_top_ranked_story(self):
        assert _maker()._headline(None, _items(3)).startswith("Story 1")

    @pytest.mark.asyncio
    async def test_unmatched_headline_briefs_the_curated_story_not_ranked_one(self):
        # The old `or 1` fallback briefed the top-ranked story, which desyncs image and lead. With
        # no ranked match the CURATED headline's own prose is the source.
        from domain.models import DigestContent, DigestItem

        maker = _rendering_maker()
        maker.config.pipeline.enable_threads_post = False
        content = DigestContent(
            lead="리드.",
            items=[
                DigestItem(
                    title="합쳐진 헤드라인",
                    url="http://merged.example/story",
                    body="합본 본문이다.",
                    implication="시사점이다.",
                )
            ],
        )
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait")))
        plan = {"skip": False, "instruction": "draw"}
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)) as write:
            maker.generator.generate = gen
            await maker.run(_items(), content, today=date(2026, 6, 10))
        source = gen.await_args.args[1]
        assert "합쳐진 헤드라인" in source and "합본 본문이다." in source
        assert "Story 1" not in source  # never the unrelated top-ranked story
        assert "시사점이다." not in source  # nor the implication, which turned panel 4 into a moral
        assert write.await_args.args[0] == source  # the editor wrote the comic about the same story

    @pytest.mark.asyncio
    async def test_the_editor_is_given_the_headline_body_and_not_the_lead(self):
        # The editor writes the comic, so it needs the headline's body, not just its title.
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _maker()
        maker.config.pipeline.visual_editor_source_max_tokens = 7
        seen: list[str] = []
        maker.llm = RunnableLambda(
            lambda prompt: seen.append(prompt.to_messages()[-1].content) or AIMessage(content='{"skip": true}')
        )
        await maker._compose(_items(), _content(lead="리드 문장이다."), date(2026, 6, 10))
        assert "<headline>\nStory 1\n\nbody\n</headline>" in seen[0]
        # Given the lead, the editor drew its argument one sentence per panel (2026-10-18 to 10-21).
        assert "리드 문장이다." not in seen[0]
        assert "Story 2" not in seen[0]  # the rest of the candidate pool is not the editor's business
        assert maker.models.truncate_to_tokens.call_args.args[1] == 7

    @pytest.mark.asyncio
    async def test_write_synopsis_parses_prose_wrapped_json(self):
        # Real path: the editor LLM returns prose-wrapped JSON; _write_synopsis must extract it.
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _maker()
        maker.llm = RunnableLambda(lambda _: AIMessage(content='Here:\n{"skip": false, "item_number": 1}\ndone'))
        plan = await maker._write_synopsis("source", [])
        assert plan == {"skip": False, "item_number": 1}

    @pytest.mark.asyncio
    async def test_write_synopsis_malformed_plan_becomes_skip(self):
        # An unparseable plan must read as a SKIP: returning {} let run() fall through to a
        # generic instruction and pay for a full gpt-image render off a plan nobody could read.
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _maker()
        maker.llm = RunnableLambda(lambda _: AIMessage(content="no json here at all"))
        assert await maker._write_synopsis("source", []) == {"skip": True}

    @pytest.mark.asyncio
    async def test_a_stringly_typed_skip_flag_does_not_kill_the_visual(self):
        # `if plan.get("skip")` read the STRING "false" as True, so an editor that quoted its
        # booleans silently killed the day's visual with no error anywhere.
        from domain.models import DigestContent, DigestItem

        maker = _rendering_maker()
        maker.config.pipeline.enable_threads_post = False
        content = DigestContent(lead="리드.", items=[DigestItem(title="헤드라인", url="http://e.com/1", body="본문.")])
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait")))
        plan = {"skip": "false", "research": [], "instruction": "draw it"}
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = gen
            await maker.run(_items(), content, today=date(2026, 6, 10))
        gen.assert_awaited()

    @pytest.mark.asyncio
    async def test_a_stringly_typed_character_flag_is_not_an_opt_in(self):
        # `bool(plan.get("use_character"))` on "false" injected the mascot sheet against the
        # editor's judgment.
        maker = _maker()
        instruction = maker._build_instruction({"instruction": "draw", "use_character": "false"})
        assert "THE NARRATOR" not in instruction

    @pytest.mark.asyncio
    async def test_unparseable_plan_renders_nothing(self):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _rendering_maker()
        calls = []
        maker.llm = RunnableLambda(lambda _: calls.append(1) or AIMessage(content="not json"))
        maker.generator.generate = AsyncMock()
        await maker.run(_items())
        maker.generator.generate.assert_not_awaited()  # no wasted render
        assert len(calls) == 1  # and no retry LLM call
        assert maker.publisher.publish.await_args.kwargs["image_bytes"] is None


class TestCharacterInjection:
    @pytest.mark.asyncio
    async def test_character_sheet_injected_when_editor_opts_in(self):
        maker = _rendering_maker()
        maker.config.pipeline.visual_character_enabled = True
        plan = {"skip": False, "instruction": "draw it", "use_character": True}
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            await maker.run(_items())
        instruction = maker.generator.generate.call_args[0][0]
        assert "THE NARRATOR" in instruction
        assert "cardigan" in instruction  # the sheet's signature props came through

    @pytest.mark.asyncio
    async def test_character_not_injected_when_globally_disabled(self):
        maker = _rendering_maker()
        maker.config.pipeline.visual_character_enabled = False
        plan = {"skip": False, "instruction": "draw it", "use_character": True}
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = AsyncMock(
                return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait"))
            )
            await maker.run(_items())
        instruction = maker.generator.generate.call_args[0][0]
        assert "THE NARRATOR" not in instruction


class TestComposeIsWhatShips:
    """scripts/sample_visual_brief.py grades the text from _compose, so it must be what production
    renders from."""

    @pytest.mark.asyncio
    async def test_output_is_byte_identical_to_what_make_visual_sends(self):
        plan = {"skip": False, "instruction": "Draw the story.", "use_character": True}
        post_date = date(2026, 8, 15)
        maker = _rendering_maker()
        maker.config.pipeline.enable_threads_post = False
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait")))
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.generator.generate = gen
            await maker.run(_items(), _content(), today=post_date)
            composed = await maker._compose(_items(), _content(), post_date)
        assert composed == (plan, gen.await_args.args[0], gen.await_args.args[1])


class TestStoryLessDayDoesNotPayForARender:
    """Threads deliberately refuses a story-less digest, so there is no destination left for the
    image — an LLM editor pass plus a gpt-image render would be bought and thrown away."""

    @pytest.mark.asyncio
    async def test_no_render_when_there_are_no_stories(self):
        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        gen = AsyncMock()
        with patch.object(maker, "_write_synopsis", new=AsyncMock()) as pick:
            maker.generator.generate = gen
            result = await maker.run(_items(), None, today=date(2026, 8, 18))
        assert result is False
        pick.assert_not_awaited()
        gen.assert_not_awaited()
        # The verdict is still recorded so the caller's delivery alert is not a no-op.
        assert maker.threads_outcome == ThreadsDelivery(0, 1)

    @pytest.mark.asyncio
    async def test_a_digest_with_stories_is_unaffected(self):
        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        content = TestVisualFailureNeverCostsTheDigest._content()
        plan = {"skip": False, "instruction": "Draw."}
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="portrait")))
        with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
            maker.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(2, 2))
            maker.generator.generate = gen
            result = await maker.run(_items(), content, today=date(2026, 8, 18))
        assert result is True
        gen.assert_awaited_once()


class TestCharacterReference:
    @pytest.mark.asyncio
    async def test_the_reference_sheet_rides_along_only_when_the_narrator_appears(self):
        from datetime import date

        maker = _maker()
        maker.character_reference = b"SHEET"
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", prompt="draw", orientation="tall")))
        maker.generator.generate = gen
        for use_character, expected in ((True, [b"SHEET"]), (False, [])):
            plan = {"skip": False, "instruction": "x", "use_character": use_character}
            with patch.object(maker, "_write_synopsis", new=AsyncMock(return_value=plan)):
                await maker._make_visual(_items(), None, date(2026, 6, 10))
            assert gen.await_args.kwargs["references"] == expected

    def test_the_narrator_is_the_default_when_the_plan_does_not_say(self):
        maker = _maker()
        assert maker._features_character({"instruction": "x"}) is True
        assert maker._features_character({"instruction": "x", "use_character": False}) is False

    def test_a_missing_reference_file_degrades_to_none(self, tmp_path):
        maker = _maker()
        maker.config.pipeline.visual_character_reference = str(tmp_path / "absent.png")
        assert maker._load_character_reference(maker.config) is None

    def test_the_shipped_reference_sheet_loads(self):
        maker = _maker()
        assert maker._load_character_reference(maker.config)


class TestALiveDayStaysProtected:
    """2026-10-31 review: the claim before the post read the ledger, a throttled read skipped the write,
    and the day posted 6/6 with no marker: the next trigger for the date posted it again."""

    @staticmethod
    def _publisher():
        from application.digest_publisher import DigestPublisher

        config = Config()
        config.pipeline.enable_threads_post = True
        return DigestPublisher(config, MemoryStateStore(), FakeThreads())

    @pytest.mark.asyncio
    async def test_a_claim_that_did_not_stick_is_written_after_the_post(self):
        publisher = self._publisher()
        real_mark = publisher.threads_ledger.mark
        calls: list[int] = []

        def mark(*args):
            calls.append(1)
            if len(calls) > 1:  # the claim before the post is lost; the one after it lands
                real_mark(*args)

        publisher.threads_ledger.mark = mark
        await publisher.publish(_content(), today=date(2026, 10, 31))
        assert publisher.threads_ledger.already_posted(date(2026, 10, 31))
        assert publisher.unprotected is False

    @pytest.mark.asyncio
    async def test_a_date_that_cannot_be_recorded_is_reported(self):
        publisher = self._publisher()
        publisher.threads_ledger.mark = MagicMock(return_value=None)
        await publisher.publish(_content(), today=date(2026, 10, 31))
        assert publisher.unprotected is True

    @pytest.mark.asyncio
    async def test_a_day_that_posted_nothing_is_not_reported_as_unprotected(self):
        publisher = self._publisher()
        publisher.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(0, 2))
        publisher.threads_ledger.mark = MagicMock(return_value=None)
        await publisher.publish(_content(), today=date(2026, 10, 31))
        assert publisher.unprotected is False

    @pytest.mark.asyncio
    async def test_a_story_less_digest_posts_nothing_even_past_the_destination_check(self):
        # The 2026-08-13/08-17 guard inside publish() itself, not only has_a_destination.
        from domain.models import DigestContent

        publisher = self._publisher()
        assert await publisher.publish(DigestContent(lead="리드.", items=[]), today=date(2026, 10, 31)) is False
        publisher.threads.post_thread.assert_not_awaited()
        assert publisher.threads_outcome == ThreadsDelivery(0, 1)
