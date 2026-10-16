from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from langchain_core.output_parsers import StrOutputParser

from agent.visuals import VisualGenerator
from output.digest_delivery import DigestPublisher
from pipeline.aggregator import normalize_url
from shared import (
    BedrockLanguageModelFactory,
    DigestContent,
    DigestItem,
    RankedItem,
    SecretUnavailableError,
    VisualBrief,
    VisualEditorPrompt,
    agi_countdown_intro,
    coerce_bool,
    create_state_store,
    editorial_lead,
    logger,
    parse_json_from_llm_output,
    resolve_secret,
)
from shared.config import Config
from shared.history_store import ThreadsPostLedger
from shared.state_store import StateStore

if TYPE_CHECKING:
    from output.threads_handler import ThreadsDelivery


class DailyVisualMaker:
    """Renders the day's visual for the digest headline (normally a four-panel comic), then
    hands it to the digest publisher as an attachment.

    Best-effort in BOTH directions: a visual failure (no OpenAI key, no fit, search/render error)
    is logged and the digest still publishes TEXT-ONLY, and a delivery failure never escapes into
    the pipeline. Publication itself belongs to `output.digest_delivery.DigestPublisher`, which takes
    the image as an OPTIONAL argument — so no failure on this side can cost the day's digest."""

    def __init__(self, config: Config, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm_factory = llm_factory
        self.llm = llm_factory.get_model(config.pipeline.digest_model, stage="visual-editor")
        # The publisher's Threads ledger needs the state store; a failed init degrades to publishing
        # without the ledger rather than crashing the visual.
        store: StateStore | None
        try:
            store = create_state_store(config)
        except Exception:
            logger.error("State store unavailable (init failed); publishing without the Threads ledger", exc_info=True)
            store = None
        self.publisher = DigestPublisher(config, store)
        self.generator = VisualGenerator(
            llm_factory,
            config.pipeline.digest_model,
            image_model=config.pipeline.image_model,
            image_sizes=config.pipeline.image_sizes,
            source_max_tokens=config.pipeline.visual_synopsis_source_max_tokens,
            on_image_language=config.pipeline.visual_on_image_language,
            moderation_softening_instruction=config.pipeline.visual_moderation_softening_instruction,
            style_guidance=config.pipeline.visual_synopsis_style_guidance,
            style_aesthetic=config.pipeline.visual_synopsis_style_aesthetic,
            image_timeout_sec=config.pipeline.visual_image_timeout_sec,
            image_max_retries=config.pipeline.visual_image_max_retries,
            image_quality=config.pipeline.visual_image_quality,
        )
        self.character_reference = self._load_character_reference(config)

    async def run(
        self,
        ranked_items: list[RankedItem],
        content: DigestContent | None = None,
        *,
        today: date | None = None,
        force_republish: bool = False,
        deadline: float | None = None,
    ) -> bool:
        """Make the day's visual and publish the digest. `deadline` is an OPTIONAL monotonic
        timestamp (time.monotonic() + seconds left) that bounds the render + publish path when the
        caller runs under a hard timeout — the visual Lambda passes its remaining time. None (local
        runs, main.py) means unbounded, exactly as before."""
        if not ranked_items:
            return False

        post_date = today or datetime.now(ZoneInfo(self.config.aws.timezone)).date()
        # Asked BEFORE the render: a day with nothing left to publish must not pay for an LLM editor
        # pass plus a gpt-image render it can never deliver.
        if not self.publisher.has_a_destination(content, post_date, force_republish):
            return False

        # The whole visual stage, not just the render, is bounded: an editor or brief call hanging on
        # Bedrock (read_timeout 300 x 3 attempts) could otherwise run the Lambda out of time and lose
        # the text digest too. Past the deadline the day posts text-only.
        image_deadline = self._image_deadline(deadline)
        try:
            image_bytes, brief = await asyncio.wait_for(
                self._make_visual(ranked_items, content, post_date, deadline=deadline),
                timeout=None if image_deadline is None else max(0.0, image_deadline - time.monotonic()),
            )
        except TimeoutError:
            logger.error("Daily visual ran past its deadline; publishing the digest text-only")
            image_bytes, brief = None, None

        return await self.publisher.publish(
            content,
            image_bytes=image_bytes,
            image_alt=brief.alt_text if image_bytes and brief else "",
            today=post_date,
            force_republish=force_republish,
            deadline=deadline,
        )

    @property
    def threads_outcome(self) -> ThreadsDelivery | None:
        """The publisher's last Threads verdict, for the caller's metrics/alerts."""
        return self.publisher.threads_outcome

    @property
    def threads_ledger(self) -> ThreadsPostLedger | None:
        return self.publisher.threads_ledger

    async def _make_visual(
        self,
        ranked_items: list[RankedItem],
        content: DigestContent | None,
        post_date: date,
        *,
        deadline: float | None = None,
    ) -> tuple[bytes | None, VisualBrief | None]:
        """Render the day's image, or (None, None). EVERY failure is contained here — a missing
        OpenAI key, an unusable editor plan, a render error — because the image is an ATTACHMENT to
        the digest, and this is the only Threads publish path. run() used to return early on all
        three, so a visual-only failure silently cost the whole day's digest."""
        # strict=True so an SSM read FAILURE (throttled, denied, wrong region) is distinguishable
        # from "no key configured": the lenient read returned "" for both, so a broken parameter
        # store silently produced text-only digests that looked like a deliberate config. The
        # exception is caught right here — a strict read must never cost the text digest.
        try:
            api_key = resolve_secret("OPENAI_API_KEY", "openai-api-key", strict=True)
        except SecretUnavailableError as e:
            logger.error("Could not read OPENAI_API_KEY (%s) — publishing the digest text-only", e)
            return None, None
        if not api_key:
            logger.info("OPENAI_API_KEY not set — publishing the digest text-only, without a visual")
            return None, None

        # The visual MUST depict the digest's headline so the image and the lead stay in sync.
        # content.headline_index is into the curated content.items (may be merged/reordered), so
        # map it back to a ranked_items position by normalized URL.
        headline_index = self._headline_ranked_index(content, ranked_items)
        marker_index, headline_title, source = self._headline_brief(content, ranked_items, headline_index)
        take = self._editorial_take(content, post_date)
        try:
            plan = await self._pick_story(ranked_items, marker_index, headline_source=source, editorial_take=take)
        except Exception:
            # Best-effort: a visual failure must never block the digest, so catch broadly here.
            logger.warning("Daily visual editor failed; publishing the digest text-only", exc_info=True)
            return None, None

        # coerce_bool, not truthiness: an editor that writes the STRING "false" here silently
        # killed the day's visual, with no error anywhere.
        if coerce_bool(plan.get("skip")):
            logger.info("Daily visual: editor could not illustrate the headline; publishing the digest text-only")
            return None, None

        instruction = self._build_instruction(plan, content, post_date, headline_title)
        references = [self.character_reference] if self._features_character(plan) and self.character_reference else []

        image_bytes: bytes | None = None
        brief: VisualBrief | None = None
        try:
            image_bytes, brief = await self.generator.generate(
                instruction, source, references=references, deadline=self._image_deadline(deadline)
            )
        except Exception:
            # The image is an optional attachment; its failure must NOT sink the Threads text
            # digest, which stands on its own. Fall through with no image so _post_threads still
            # posts the lead + per-story replies (text-only).
            logger.warning("Daily visual generation failed; posting Threads text-only", exc_info=True)

        return image_bytes, brief

    def _build_instruction(
        self, plan: dict, content: DigestContent | None, post_date: date, headline_title: str
    ) -> str:
        """The art-director instruction production actually sends.

        A PURE extraction from _make_visual — no I/O, no state — so scripts/sample_visual_brief.py
        can grade the very string production sends, not the bare `plan["instruction"]` without the
        editorial angle, the guardrails or the character sheet."""
        instruction = plan.get("instruction", "") or f"A 2x2 four-panel comic about: {headline_title}"

        # The art director only ever saw the raw article, so it illustrates surface facts: the
        # 2026-08-15 visual drew a four-way photo finish ("they all tied") for a story whose point
        # was that RELEASE CADENCE, not model quality, explained the gap. Hand over the digest's own
        # angle as INFORMATION, not a constraint — the image needn't match the lead's thesis, but it
        # shouldn't be blind to it.
        take = self._editorial_take(content, post_date)
        if take:
            instruction += (
                "\n\nTHE DIGEST'S OWN ANGLE on this story (context you may use or ignore — "
                f"the image does NOT have to argue this point):\n{take}"
            )
        if self.config.pipeline.visual_guardrails:
            instruction += f"\n\nGUARDRAILS: {self.config.pipeline.visual_guardrails}"
        # The narrator is the strip's viewpoint character unless the editor judged he'd get in the way.
        # The sheet is written into the brief; his exact look rides along as the reference image.
        if self._features_character(plan):
            instruction += (
                "\n\nTHE NARRATOR is the viewpoint character, inside the scenes (not in a void): "
                f"{self.config.pipeline.visual_character_sheet}"
            )
        return instruction

    @staticmethod
    def _image_deadline(deadline: float | None) -> float | None:
        """When the image work must be done: the caller's deadline minus what the Threads publish
        still needs afterwards, so a slow correction can never eat the day's post."""
        from output.threads_handler import THREADS_INDEXING_BUDGET_SEC, THREADS_PUBLISH_RESERVE_SEC

        return None if deadline is None else deadline - THREADS_INDEXING_BUDGET_SEC - THREADS_PUBLISH_RESERVE_SEC

    def _features_character(self, plan: dict) -> bool:
        return self.config.pipeline.visual_character_enabled and coerce_bool(plan.get("use_character"), default=True)

    @staticmethod
    def _load_character_reference(config: Config) -> bytes | None:
        """The narrator's reference sheet, or None: a missing file costs consistency, never the visual."""
        path = Path(config.pipeline.visual_character_reference)
        if not path.is_absolute():
            path = Path(__file__).resolve().parents[1] / path
        try:
            return path.read_bytes()
        except OSError:
            logger.warning("Character reference '%s' unreadable; rendering without it", path)
            return None

    @staticmethod
    def _curated_headline(content: DigestContent | None) -> DigestItem | None:
        """The curated story the lead is about (content.items[headline_index]), or None."""
        if not content or not content.items:
            return None
        idx = content.headline_index if 1 <= content.headline_index <= len(content.items) else 1
        return content.items[idx - 1]

    def _headline_brief(
        self, content: DigestContent | None, ranked_items: list[RankedItem], headline_index: int
    ) -> tuple[int, str, str]:
        """(marker_index, title, source_text) for the story the visual must depict.

        marker_index is the ranked position to flag as TODAY'S HEADLINE for the editor (0 = none).
        When the curated headline has no matching ranked source, the CURATED story's own prose is
        the source: the old `or 1` fallback briefed ranked #1 instead, i.e. a different story than
        the lead — exactly the image/text desync this whole path exists to prevent."""
        if headline_index:
            item = ranked_items[headline_index - 1].item
            return headline_index, item.title, f"{item.title}\n\n{item.text}"
        curated = self._curated_headline(content)
        if curated is not None:
            body = "\n\n".join(p for p in (curated.body.strip(), curated.implication.strip()) if p)
            return 0, curated.title, f"{curated.title}\n\n{body}"
        # No structured content at all (visual-only run): the top-ranked story is the headline.
        item = ranked_items[0].item
        return 1, item.title, f"{item.title}\n\n{item.text}"

    def _countdown_intro(self, post_date: date) -> str:
        """The AGI-countdown gag for this date, exactly as the digest generator computed it, so the
        editorial take can be read without it."""
        return agi_countdown_intro(
            self.config.pipeline.agi_countdown_date,
            self.config.pipeline.agi_countdown_template,
            post_date,
            self.config.pipeline.agi_countdown_after,
        )

    def _editorial_take(self, content: DigestContent | None, post_date: date) -> str:
        """The digest's angle on the headline story: its lead plus the headline item's closing
        implication. The AGI-countdown prefix is dropped — it's the same fixed template every day
        and carries no information about today's story. Empty when there is no structured content
        (a visual can still be made; it just won't know the take)."""
        if not content or not content.items:
            return ""
        intro = self._countdown_intro(post_date)
        idx = content.headline_index if 1 <= content.headline_index <= len(content.items) else 1
        headline = content.items[idx - 1]
        parts = [editorial_lead(content.lead, intro).strip(), headline.implication.strip()]
        return "\n\n".join(p for p in parts if p)

    @staticmethod
    def _headline_ranked_index(content: DigestContent | None, ranked_items: list[RankedItem]) -> int:
        """Ranked position (1-based) of the curated headline, or 0 when it matches none.

        Matched on the aggregator's normalize_url, not an exact string: the editor echoes the URL
        back and a trailing slash / http→https / utm-param difference is enough to miss a match
        that is obviously the same article. Already-identical URLs are unaffected."""
        if not content or not content.items:
            return 0
        idx = content.headline_index
        if not (1 <= idx <= len(content.items)):
            return 0
        url = normalize_url(content.items[idx - 1].url)
        for i, r in enumerate(ranked_items, start=1):
            if normalize_url(r.item.url) == url:
                return i
        return 0

    def _editor_items_text(
        self,
        ranked_items: list[RankedItem],
        headline_index: int,
        headline_source: str,
        editorial_take: str,
    ) -> str:
        """What the visual editor is shown: the day's stories as title rows, plus the HEADLINE story
        in full and the digest's own angle on it.

        The editor writes the comic, so it needs the headline's body, not just its title row. Handing
        the headline over explicitly also closes the no-ranked-match case: with headline_index 0 NO row carried the marker the
        prompt promises, so the editor's brief and the art director's source material described
        different articles."""
        rows = [
            f"{i}. [{r.item.source_type.value}] {r.item.title}"
            + (" ← TODAY'S HEADLINE — illustrate this one" if i == headline_index else "")
            for i, r in enumerate(ranked_items, start=1)
        ]
        blocks = ["\n".join(rows)]
        if headline_source.strip():
            headline = self.llm_factory.truncate_to_tokens(
                headline_source.strip(), self.config.pipeline.visual_editor_source_max_tokens
            )
            blocks.append(f"TODAY'S HEADLINE STORY IN FULL — this is the one to illustrate:\n{headline}")
        if editorial_take:
            blocks.append(f"THE DIGEST'S OWN ANGLE on it:\n{editorial_take}")
        return "\n\n".join(blocks)

    async def _pick_story(
        self,
        ranked_items: list[RankedItem],
        headline_index: int = 0,
        *,
        headline_source: str = "",
        editorial_take: str = "",
    ) -> dict:
        # The editor briefs the HEADLINE handed to it (it doesn't choose the story); the visual must
        # match the lead, which is about this same headline.
        items_text = self._editor_items_text(ranked_items, headline_index, headline_source, editorial_take)
        chain = VisualEditorPrompt.get_prompt() | self.llm | StrOutputParser()
        raw = await chain.ainvoke(
            {
                "items_text": items_text,
                "audience": self.config.pipeline.visual_audience_description,
                "on_image_language": self.config.pipeline.visual_on_image_language,
            }
        )
        try:
            return parse_json_from_llm_output(raw)
        except json.JSONDecodeError:
            # Treat an unparseable plan as a skip. Returning {} let run() fall through with a
            # generic fallback instruction and pay for a full gpt-image render off a plan nobody
            # could read; skipping costs nothing extra (no second LLM call either).
            logger.warning("Daily visual editor returned unparseable JSON; skipping", exc_info=True)
            return {"skip": True}
