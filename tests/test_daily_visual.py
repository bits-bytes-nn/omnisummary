from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from output.threads_handler import ThreadsDelivery
from pipeline.daily_visual import DailyVisualMaker
from shared.config import Config
from shared.constants import SourceType
from shared.models import CollectedItem, RankedItem, VisualBrief
from shared.state_store import StateStore


class _MemoryStore(StateStore):
    """In-memory state store so visual tests don't read/write the repo's digest_state dir
    (which made format/idempotency history order-dependent across runs)."""

    def __init__(self) -> None:
        self._d: dict[str, str] = {}

    def read(self, key: str) -> str | None:
        return self._d.get(key)

    def write(self, key: str, content: str) -> None:
        self._d[key] = content

    def exists(self, key: str) -> bool:
        return key in self._d


def _maker() -> DailyVisualMaker:
    config = Config()
    factory = MagicMock()
    factory.get_model.return_value = MagicMock()
    with patch("pipeline.daily_visual.create_state_store", return_value=_MemoryStore()):
        maker = DailyVisualMaker(config, factory)
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


class TestEditorialTakeReachesTheArtDirector:
    """The brief used to see only the raw article, so the image illustrated surface facts: the
    2026-08-15 visual drew a four-way photo finish ("they all tied") for a story whose point was
    that release cadence explained the gap. The digest's angle must reach the art director — as
    context, NOT as a constraint the image has to argue."""

    @staticmethod
    async def _instruction_for(content) -> str:
        from datetime import date

        maker = _maker()
        maker.config.pipeline.enable_threads_post = False
        maker.config.pipeline.enable_slack_post = False
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "Draw the story."}
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw")))
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = gen
                await maker.run(_items(), content, today=date(2026, 8, 15))
        return str(gen.await_args.args[0])

    @pytest.mark.asyncio
    async def test_lead_and_implication_are_handed_over_without_the_countdown(self):
        from shared.models import DigestContent, DigestItem

        content = DigestContent(
            lead="AGI 등장 870일 전이다. 격차의 원인은 모델 품질이 아니라 출시 주기다.",
            headline_index=1,
            items=[
                DigestItem(
                    title="GLM-5.3",
                    url="http://e.com/1",
                    body="본문.",
                    implication="신중함에 값을 매기는 쪽이 시장에서 손해를 본다.",
                )
            ],
        )
        instruction = await self._instruction_for(content)
        assert "격차의 원인은 모델 품질이 아니라 출시 주기다." in instruction
        assert "신중함에 값을 매기는 쪽이 시장에서 손해를 본다." in instruction
        # The fixed daily countdown template carries no information about the story.
        assert "AGI 등장" not in instruction
        # Handed over as context the art director may ignore — not a matching requirement.
        assert "does NOT have to" in instruction

    @pytest.mark.asyncio
    async def test_no_take_when_there_is_no_structured_content(self):
        instruction = await self._instruction_for(None)
        assert "THE DIGEST'S OWN ANGLE" not in instruction

    @pytest.mark.asyncio
    async def test_guardrails_are_appended_and_are_config_driven(self):
        # Handing the angle over as pure context is not always enough: a 2026-08-18 run turned a
        # lead about circular vendor financing into a triumphal rocket-and-money poster — the
        # opposite register. The guardrails say what the image must not DO — far weaker than the
        # rejected "the image must argue the lead's thesis" rule. Depicting real people is
        # explicitly ALLOWED (normal editorial-cartoon practice); what is barred is standing a
        # company or country up as an ethnically-coded human, as a 2026-08-15 visual did.
        instruction = await self._instruction_for(None)
        assert "GUARDRAILS:" in instruction
        assert "must not read as celebratory" in instruction
        assert "Recognisable depictions of real people are fine" in instruction
        assert "ethnically-coded human" in instruction

    @pytest.mark.asyncio
    async def test_empty_guardrails_config_appends_nothing(self):
        from datetime import date

        maker = _maker()
        maker.config.pipeline.enable_threads_post = False
        maker.config.pipeline.enable_slack_post = False
        maker.config.pipeline.visual_guardrails = ""
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "Draw the story."}
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw")))
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = gen
                await maker.run(_items(), None, today=date(2026, 8, 18))
        assert "GUARDRAILS:" not in str(gen.await_args.args[0])


class TestVisualFailureNeverCostsTheDigest:
    """Regression: run() returned early on a missing OpenAI key, a failed editor call and an editor
    skip — all BEFORE the only Threads publish path. A visual-only problem therefore cost the whole
    day's digest, silently. The image is an attachment; the text digest must still go out."""

    @staticmethod
    def _content():
        from shared.models import DigestContent, DigestItem

        return DigestContent(
            lead="오늘의 리드.",
            headline_index=1,
            items=[DigestItem(title="스토리", url="http://e.com/1", body="본문.")],
        )

    @staticmethod
    def _threads_maker() -> DailyVisualMaker:
        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.config.pipeline.enable_slack_post = False
        return maker

    @pytest.mark.asyncio
    async def test_missing_openai_key_still_publishes_the_text_digest(self):
        maker = self._threads_maker()
        with patch("pipeline.daily_visual.resolve_secret", return_value=""):
            with patch(
                "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
            ) as th:
                result = await maker.run(_items(), self._content(), today=date(2026, 6, 10))
        assert result is True
        th.assert_awaited_once()
        assert th.await_args.kwargs["root_text"] == "오늘의 리드."
        assert th.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_editor_failure_still_publishes_the_text_digest(self):
        maker = self._threads_maker()
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(side_effect=RuntimeError("bedrock down"))):
                with patch(
                    "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                ) as th:
                    result = await maker.run(_items(), self._content(), today=date(2026, 6, 10))
        assert result is True
        th.assert_awaited_once()
        assert th.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_editor_skip_still_publishes_the_text_digest(self):
        maker = self._threads_maker()
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value={"skip": True})):
                with patch(
                    "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                ) as th:
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
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock()) as pick:
                with patch("output.threads_handler.post_to_threads", new=AsyncMock()) as th:
                    assert await maker.run(_items(), self._content(), today=date(2026, 6, 10)) is False
        pick.assert_not_awaited()
        maker.generator.generate.assert_not_awaited()
        th.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_slack_enabled_keeps_running_even_if_threads_is_already_posted(self):
        # The Threads marker says nothing about the Slack image upload, so the short-circuit must
        # not swallow a Slack-only visual if Slack delivery is turned back on.
        maker = self._threads_maker()
        maker.config.pipeline.enable_slack_post = True
        maker.threads_ledger.mark(date(2026, 6, 10))
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(
                maker, "_pick_story", new=AsyncMock(return_value={"skip": False, "research": [], "instruction": "x"})
            ):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)) as up:
                    with patch("output.threads_handler.post_to_threads", new=AsyncMock()) as th:
                        result = await maker.run(_items(), self._content(), today=date(2026, 6, 10))
        assert result is True
        up.assert_awaited_once()
        th.assert_not_awaited()  # Threads itself is still guarded by the ledger


class TestDailyVisualMaker:
    @pytest.mark.asyncio
    async def test_skips_on_empty_items(self):
        assert await _maker().run([]) is False

    @pytest.mark.asyncio
    async def test_no_image_rendered_when_the_editor_skips(self):
        maker = _maker()
        maker.generator.generate = AsyncMock()
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value={"skip": True})):
                # Threads/Slack both off in this default-config maker → nothing published.
                maker.config.pipeline.enable_slack_post = False
                assert await maker.run(_items()) is False
        maker.generator.generate.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_visual_draws_the_headline_not_editor_pick(self):
        # The headline is authoritative: even if the editor returns a different item_number,
        # the visual must depict the digest headline (so image and lead stay in sync).
        from shared.models import DigestContent, DigestItem

        maker = _maker()
        plan = {"skip": False, "item_number": 3, "research": [], "instruction": "draw"}  # editor drifts
        content = DigestContent(
            lead="lead", headline_index=1, items=[DigestItem(title="t", url="http://e.com/2", body="b")]
        )  # headline maps to ranked Story 2 by URL
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    result = await maker.run(_items(), content)
        assert result is True
        # source must be the headline (Story 2), NOT the editor's item_number=3
        args, kwargs = maker.generator.generate.call_args
        assert "Story 2" in args[1]

    @pytest.mark.asyncio
    async def test_slack_disabled_skips_upload(self):
        maker = _maker()
        maker.config.pipeline.enable_slack_post = False
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)) as up:
                    result = await maker.run(_items())
        assert result is False
        up.assert_not_called()

    @pytest.mark.asyncio
    async def test_threads_enabled_fans_out_with_content(self):
        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="오늘의 리드.",
            headline_index=1,
            items=[DigestItem(title="스토리", url="http://e.com/1", body="본문.")],
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch(
                        "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                    ) as th:
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

        from shared.models import DigestContent

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        empty = DigestContent(lead="리드만 남았다.", headline_index=1, items=[])
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch(
                        "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                    ) as th:
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

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.threads_ledger.mark(date(2026, 6, 10))
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        posted = await maker.run(_items(), content, today=date(2026, 6, 10))
        assert posted is False
        assert maker.threads_outcome is None

    @pytest.mark.asyncio
    async def test_post_exception_still_leaves_a_verdict(self):
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.threads_handler.post_to_threads", new=AsyncMock(side_effect=RuntimeError("boom"))):
                    posted = await maker.run(_items(), content, today=date(2026, 6, 10))
        assert posted is False
        assert maker.threads_outcome == ThreadsDelivery(0, 2)

    @pytest.mark.asyncio
    async def test_caller_deadline_is_forwarded_to_the_publish_path(self):
        # A plain monotonic float is threaded through — never the Lambda context object — and a
        # None deadline (local runs) must reach the publisher unchanged.
        import time
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        deadline = time.monotonic() + 300
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw")))
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = gen
                with patch(
                    "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                ) as th:
                    await maker.run(_items(), content, today=date(2026, 6, 10), deadline=deadline)
        assert th.await_args.kwargs["deadline"] == deadline
        assert gen.await_args.kwargs["deadline"] == deadline

    @pytest.mark.asyncio
    async def test_threads_skipped_when_already_posted_today(self):
        # A same-day re-run (or async Lambda retry) must not re-post the root+replies set.
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.threads_ledger.mark(date(2026, 6, 10))  # today already posted
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch(
                        "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                    ) as th:
                        await maker.run(_items(), content, today=date(2026, 6, 10))
        th.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_threads_force_republish_bypasses_guard(self):
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.threads_ledger.mark(date(2026, 6, 10))
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch(
                        "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                    ) as th:
                        await maker.run(_items(), content, today=date(2026, 6, 10), force_republish=True)
        th.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_force_republish_failure_keeps_prior_mark(self):
        # A force-republish that FAILS must not wipe out the date's existing successful-post
        # mark (only a mark WE added in this run is rolled back).
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        maker.threads_ledger.mark(date(2026, 6, 10))  # prior successful post
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch(
                        "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(0, 2))
                    ):
                        await maker.run(_items(), content, today=date(2026, 6, 10), force_republish=True)
        assert maker.threads_ledger.already_posted(date(2026, 6, 10))  # prior mark preserved

    @pytest.mark.asyncio
    async def test_threads_marks_ledger_after_successful_post(self):
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch(
                        "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                    ):
                        await maker.run(_items(), content, today=date(2026, 6, 10))
        assert maker.threads_ledger.already_posted(date(2026, 6, 10))

    @pytest.mark.asyncio
    async def test_threads_not_marked_when_post_fails(self):
        # A failed Threads post must stay retryable — the optimistic claim is rolled back.
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch(
                        "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(0, 2))
                    ):
                        await maker.run(_items(), content, today=date(2026, 6, 10))
        assert not maker.threads_ledger.already_posted(date(2026, 6, 10))

    @pytest.mark.asyncio
    async def test_threads_date_claimed_before_post_runs(self):
        # The concurrency fix: the date must already be marked at the moment post_to_threads
        # is entered, so a racing concurrent invocation sees it taken and skips.
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        claimed_during_post: list[bool] = []

        async def _post(**_kwargs):
            claimed_during_post.append(maker.threads_ledger.already_posted(date(2026, 6, 10)))
            return ThreadsDelivery(2, 2)

        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch("output.threads_handler.post_to_threads", new=_post):
                        await maker.run(_items(), content, today=date(2026, 6, 10))
        assert claimed_during_post == [True]

    @pytest.mark.asyncio
    async def test_threads_marker_rolled_back_when_post_raises(self):
        # An exception mid-post must roll the claim back so the date stays retryable.
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch(
                        "output.threads_handler.post_to_threads",
                        new=AsyncMock(side_effect=RuntimeError("boom")),
                    ):
                        await maker.run(_items(), content, today=date(2026, 6, 10))
        assert not maker.threads_ledger.already_posted(date(2026, 6, 10))

    @pytest.mark.asyncio
    async def test_threads_disabled_by_default(self):
        maker = _maker()
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch(
                        "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                    ) as th:
                        await maker.run(_items())
        th.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generation_failure_returns_false(self):
        maker = _maker()
        plan = {"skip": False, "item_number": 1, "instruction": "draw"}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(side_effect=RuntimeError("boom"))
                assert await maker.run(_items()) is False

    @pytest.mark.asyncio
    async def test_image_failure_still_posts_threads_text(self):
        # Regression: an image-generation failure (e.g. OpenAI billing hard limit) must NOT sink
        # the Threads text digest — the image is an optional attachment. Threads still posts the
        # lead + per-story replies, with image_bytes=None (text-only container).
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="오늘의 리드.",
            headline_index=1,
            items=[DigestItem(title="스토리", url="http://e.com/1", body="본문.")],
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(side_effect=RuntimeError("billing hard limit"))
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)) as up:
                    with patch(
                        "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))
                    ) as th:
                        await maker.run(_items(), content, today=date(2026, 6, 10))
        th.assert_awaited_once()
        assert th.await_args.kwargs["root_text"] == "오늘의 리드."
        assert th.await_args.kwargs["image_bytes"] is None  # text-only, no attachment
        up.assert_not_called()  # Slack image upload skipped (no image)
        assert maker.threads_ledger.already_posted(date(2026, 6, 10))  # marked as posted

    @pytest.mark.asyncio
    async def test_threads_only_success_is_reported_as_posted(self):
        # With Slack delivery off (the shipped config) the run returned False even though Threads
        # published, so the Lambda logged "skipped" for every successful day.
        from datetime import date

        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_slack_post = False
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="s", url="http://e.com/1", body="b")]
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))):
                    result = await maker.run(_items(), content, today=date(2026, 6, 10))
        assert result is True

    @pytest.mark.asyncio
    async def test_format_log_deduped_per_day(self):
        # A same-day re-run must REPLACE its format entry, not push a duplicate that crowds the
        # variation window (same convention as the recent-leads log).
        from datetime import date

        maker = _maker()
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x", "format": "poster"}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    await maker.run(_items(), today=date(2026, 6, 10))
                    await maker.run(_items(), today=date(2026, 6, 10))
        entries = maker.format_log.entries()
        assert len(entries) == 1
        assert entries[0]["date"] == "2026-06-10"

    @pytest.mark.asyncio
    async def test_gather_context_dispatches_by_source(self):
        # The editor agentically picks a source per research step; _gather_context routes
        # each to the matching backend (papers -> Semantic Scholar, community/news -> Tavily).
        maker = _maker()
        research = [
            {"source": "papers", "query": "diffusion scaling"},
            {"source": "community", "query": "reactions"},
            {"source": "news", "query": "launch"},
        ]
        with patch("shared.research._search_papers", new=AsyncMock(return_value="PAPERS")) as papers:
            with patch("shared.research._tavily_search", new=AsyncMock(side_effect=["COMMUNITY", "NEWS"])) as tav:
                context = await maker._gather_context(research)

        assert "PAPERS" in context and "COMMUNITY" in context and "NEWS" in context
        papers.assert_awaited_once_with("diffusion scaling")
        # community step must pass the configured community domains; news step uses topic=news
        community_call, news_call = tav.await_args_list
        assert community_call.kwargs.get("include_domains") == maker.config.agent.community_search_domains
        assert news_call.kwargs.get("topic") == "news"

    @pytest.mark.asyncio
    async def test_gather_context_empty_research_returns_empty(self):
        assert await _maker()._gather_context([]) == ""

    @pytest.mark.asyncio
    async def test_gather_context_clamps_research_steps(self):
        # The prompt asks for 1-3 steps; a chatty plan must not fan out into ten live searches.
        maker = _maker()
        research = [{"source": "news", "query": f"q{i}"} for i in range(10)]
        with patch("shared.research._tavily_search", new=AsyncMock(return_value="NEWS")) as tav:
            await maker._gather_context(research)
        assert tav.await_count == maker.config.pipeline.visual_research_max_steps

    @pytest.mark.asyncio
    async def test_gather_context_skips_failed_step(self):
        # A backend that raises must be skipped, not abort the whole gather.
        maker = _maker()
        research = [{"source": "papers", "query": "q1"}, {"source": "news", "query": "q2"}]
        with patch("shared.research._search_papers", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with patch("shared.research._tavily_search", new=AsyncMock(return_value="NEWS")):
                context = await maker._gather_context(research)
        assert context == "NEWS"

    def test_headline_ranked_index_maps_by_url(self):
        # content.headline_index is into the curated content.items; it must map back to the
        # matching ranked_items position by URL, not be used directly as a ranked index.
        from shared.models import DigestContent, DigestItem

        ranked = _items(3)  # urls http://e.com/1..3
        content = DigestContent(
            lead="l",
            headline_index=1,  # first (and only) curated item ...
            items=[DigestItem(title="t", url="http://e.com/3", body="b")],  # ... which is ranked #3
        )
        assert DailyVisualMaker._headline_ranked_index(content, ranked) == 3

    def test_headline_ranked_index_zero_when_unmatched(self):
        from shared.models import DigestContent, DigestItem

        ranked = _items(2)
        content = DigestContent(
            lead="l", headline_index=1, items=[DigestItem(title="t", url="http://x/none", body="b")]
        )
        assert DailyVisualMaker._headline_ranked_index(content, ranked) == 0
        assert DailyVisualMaker._headline_ranked_index(None, ranked) == 0

    def test_headline_ranked_index_matches_on_normalized_urls(self):
        # Regression: matching was an exact string compare, so a trailing slash / http→https /
        # utm param made the headline "unmatched" and the visual drew ranked #1 — a DIFFERENT
        # story than the lead.
        from shared.models import DigestContent, DigestItem

        ranked = _items(3)  # http://e.com/1..3
        content = DigestContent(
            lead="l",
            headline_index=1,
            items=[DigestItem(title="t", url="https://www.e.com/2/?utm_source=x", body="b")],
        )
        assert DailyVisualMaker._headline_ranked_index(content, ranked) == 2

    @pytest.mark.asyncio
    async def test_unmatched_headline_briefs_the_curated_story_not_ranked_one(self):
        # The old `or 1` fallback briefed the top-ranked story, which desyncs image and lead. With
        # no ranked match the CURATED headline's own prose is the source.
        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_slack_post = False
        maker.config.pipeline.enable_threads_post = False
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[
                DigestItem(
                    title="합쳐진 헤드라인",
                    url="http://merged.example/story",
                    body="합본 본문이다.",
                    implication="시사점이다.",
                )
            ],
        )
        gen = AsyncMock(return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw")))
        plan = {"skip": False, "research": [], "instruction": ""}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)) as pick:
                maker.generator.generate = gen
                await maker.run(_items(), content, today=date(2026, 6, 10))
        source = gen.await_args.args[1]
        assert "합쳐진 헤드라인" in source and "합본 본문이다." in source
        assert "Story 1" not in source  # never the unrelated top-ranked story
        # No headline marker is handed to the editor when nothing matched.
        assert pick.await_args.args[1] == 0

    @pytest.mark.asyncio
    async def test_pick_story_parses_prose_wrapped_json(self):
        # Real path: the editor LLM returns prose-wrapped JSON; _pick_story must extract it.
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _maker()
        maker.llm = RunnableLambda(lambda _: AIMessage(content='Here:\n{"skip": false, "item_number": 1}\ndone'))
        plan = await maker._pick_story(_items())
        assert plan == {"skip": False, "item_number": 1}

    @pytest.mark.asyncio
    async def test_pick_story_malformed_plan_becomes_skip(self):
        # An unparseable plan must read as a SKIP: returning {} let run() fall through to a
        # generic instruction and pay for a full gpt-image render off a plan nobody could read.
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _maker()
        maker.llm = RunnableLambda(lambda _: AIMessage(content="no json here at all"))
        assert await maker._pick_story(_items()) == {"skip": True}

    @pytest.mark.asyncio
    async def test_unparseable_plan_renders_nothing(self):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _maker()
        maker.llm = RunnableLambda(lambda _: AIMessage(content="not json"))
        maker.generator.generate = AsyncMock()
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            assert await maker.run(_items()) is False
        maker.generator.generate.assert_not_awaited()  # no wasted render, and no retry LLM call


class TestFormatRotation:
    def test_least_recent_orientation_picks_unused(self):
        maker = _maker()  # image_sizes default: square, landscape, portrait
        # Recent used landscape + portrait (oldest-first); square is unused → pick it.
        recent = [{"orientation": "landscape"}, {"orientation": "portrait"}]
        assert maker._least_recent_orientation(recent) == "square"

    def test_least_recent_orientation_all_used_picks_oldest(self):
        maker = _maker()
        # All three used; entries are oldest-first, so the first (portrait) is least-recent.
        recent = [{"orientation": "portrait"}, {"orientation": "square"}, {"orientation": "landscape"}]
        assert maker._least_recent_orientation(recent) == "portrait"

    def test_least_recent_orientation_uses_last_use_not_first_appearance(self):
        # Regression: scanning the window and returning its FIRST entry picked the orientation
        # that appeared earliest — which is the MOST recently used one when it recurs later
        # (square here was yesterday's shape). LRU must compare each orientation's LAST use.
        maker = _maker()
        recent = [
            {"orientation": "square"},
            {"orientation": "landscape"},
            {"orientation": "portrait"},
            {"orientation": "square"},
        ]
        assert maker._least_recent_orientation(recent) == "landscape"

    def test_least_recent_orientation_empty_history_nudges_nothing(self):
        # With no history there is nothing to vary FROM, so returning the first configured
        # orientation ('square') was an arbitrary lock on a first run.
        maker = _maker()
        assert maker._least_recent_orientation([]) == ""

    def test_least_recent_orientation_no_orientations_configured(self):
        maker = _maker()
        maker.config.pipeline.image_sizes = {}
        assert maker._least_recent_orientation([{"orientation": "square"}]) == ""

    def test_format_guidance_empty_history(self):
        maker = _maker()
        assert "No recent visuals" in maker._format_guidance([], "")

    def test_format_guidance_lists_recent_and_preferred(self):
        maker = _maker()
        recent = [{"orientation": "landscape", "format": "poster"}]
        out = maker._format_guidance(recent, "square")
        assert "landscape/poster" in out
        assert "square" in out  # preferred orientation surfaced


class TestPanelNudge:
    def test_nudges_toward_multi_when_recent_skews_single(self):
        maker = _maker()
        recent = [{"orientation": "square", "multi_panel": False} for _ in range(5)]
        out = maker._panel_nudge(recent, 0.34)
        assert "MULTI-PANEL" in out

    def test_nudges_toward_single_when_recent_skews_multi(self):
        maker = _maker()
        recent = [{"orientation": "square", "multi_panel": True} for _ in range(5)]
        out = maker._panel_nudge(recent, 0.34)
        assert "single striking frame" in out

    def test_no_nudge_when_disabled(self):
        maker = _maker()
        recent = [{"orientation": "square", "multi_panel": False}]
        assert maker._panel_nudge(recent, 0.0) == ""

    def test_no_nudge_without_panel_history(self):
        # Old entries predate the multi_panel field → no basis to nudge.
        maker = _maker()
        recent = [{"orientation": "square", "format": "poster"}]
        assert maker._panel_nudge(recent, 0.34) == ""

    def test_format_guidance_appends_nudge(self):
        maker = _maker()
        recent = [{"orientation": "landscape", "format": "poster", "multi_panel": False}]
        out = maker._format_guidance(recent, "square", 0.34)
        assert "landscape/poster" in out
        assert "MULTI-PANEL" in out


class TestCharacterNudge:
    def test_nudges_to_bring_character_when_absent(self):
        maker = _maker()
        recent = [{"orientation": "square", "use_character": False} for _ in range(4)]
        out = maker._character_nudge(recent, 0.5)
        assert "recurring character" in out and "bring him in" in out

    def test_nudges_away_when_overused(self):
        maker = _maker()
        recent = [{"orientation": "square", "use_character": True} for _ in range(4)]
        out = maker._character_nudge(recent, 0.5)
        assert "character-free" in out

    def test_no_nudge_when_disabled(self):
        maker = _maker()
        recent = [{"orientation": "square", "use_character": False}]
        assert maker._character_nudge(recent, 0.0) == ""

    def test_no_nudge_without_character_history(self):
        maker = _maker()
        recent = [{"orientation": "square", "format": "poster"}]
        assert maker._character_nudge(recent, 0.5) == ""


class TestCharacterInjection:
    @pytest.mark.asyncio
    async def test_character_sheet_injected_when_editor_opts_in(self):
        maker = _maker()
        maker.config.pipeline.visual_character_enabled = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "draw it", "use_character": True}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    await maker.run(_items())
        instruction = maker.generator.generate.call_args[0][0]
        assert "RECURRING CHARACTER" in instruction
        assert "cardigan" in instruction  # the sheet's signature props came through

    @pytest.mark.asyncio
    async def test_character_not_injected_when_globally_disabled(self):
        maker = _maker()
        maker.config.pipeline.visual_character_enabled = False
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "draw it", "use_character": True}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    await maker.run(_items())
        instruction = maker.generator.generate.call_args[0][0]
        assert "RECURRING CHARACTER" not in instruction
