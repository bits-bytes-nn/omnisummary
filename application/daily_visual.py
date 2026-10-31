from __future__ import annotations

import asyncio
import json
import time
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from langchain_core.output_parsers import StrOutputParser

from application.digest_publisher import DigestPublisher
from application.history import RECENT_COMICS_KEY, RollingLog, ThreadsPostLedger
from application.ports import ImageRenderer, LanguageModels, StateStore, ThreadsPublisher
from application.prompts import VisualEditorPrompt
from application.visual_generator import VisualGenerator
from domain.aggregator import normalize_url
from domain.models import DigestContent, DigestItem, RankedItem, ThreadsDelivery, VisualBrief
from shared import coerce_bool, logger, parse_json_from_llm_output
from shared.config import Config


class DailyVisualMaker:
    """Renders the day's visual for the digest headline (normally a four-panel comic), then
    hands it to the digest publisher as an attachment.

    Best-effort in BOTH directions: a visual failure (no OpenAI key, an editor skip, a render error)
    is logged and the digest still publishes TEXT-ONLY, and a delivery failure never escapes into
    the pipeline. Publication itself belongs to `DigestPublisher`, which takes the image as an
    OPTIONAL argument — so no failure on this side can cost the day's digest.

    `store` is None when the state store could not be built: the day then publishes without the
    Threads ledger and the comic history rather than not at all."""

    def __init__(
        self,
        config: Config,
        models: LanguageModels,
        store: StateStore | None,
        threads: ThreadsPublisher,
        renderer: ImageRenderer,
    ) -> None:
        self.config = config
        self.models = models
        self.threads = threads
        self.llm = models.get_model(config.pipeline.digest_model, stage="visual-editor")
        self.publisher = DigestPublisher(config, store, threads)
        # What the last few comics drew, shown to the editor so today's does not repeat their sets,
        # props and gags (2026-10-13, -16 and -17 all locked a model in a glass case).
        self.comic_log = RollingLog(store, RECENT_COMICS_KEY, config.pipeline.recent_leads_window) if store else None
        self.generator = VisualGenerator(
            models,
            config.pipeline.digest_model,
            renderer,
            image_sizes=config.pipeline.image_sizes,
            source_max_tokens=config.pipeline.visual_synopsis_source_max_tokens,
            on_image_language=config.pipeline.visual_on_image_language,
            moderation_softening_instruction=config.pipeline.visual_moderation_softening_instruction,
            style_guidance=config.pipeline.visual_synopsis_style_guidance,
            style_aesthetic=config.pipeline.visual_synopsis_style_aesthetic,
            image_timeout_sec=config.pipeline.visual_image_timeout_sec,
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

        published = await self.publisher.publish(
            content,
            image_bytes=image_bytes,
            image_alt=brief.alt_text if image_bytes and brief else "",
            today=post_date,
            force_republish=force_republish,
            deadline=deadline,
        )
        outcome = self.publisher.threads_outcome
        if brief and outcome is not None and outcome.with_image:
            self._record_comic(post_date, brief)
        return published

    def _record_comic(self, post_date: date, brief: VisualBrief) -> None:
        if self.comic_log is None:
            return
        try:
            self.comic_log.append(
                {
                    "date": post_date.isoformat(),
                    "orientation": brief.orientation,
                    "comic": brief.alt_text or brief.title,
                },
                dedup_key="date",
            )
        except Exception:
            logger.warning("Failed to record the day's comic in the recent-comics log (non-fatal)", exc_info=True)

    def _recent_comics(self, post_date: date) -> list[dict]:
        """The logged comics of the days before `post_date`, oldest first. Best-effort: the history
        only steers the editor, so a store that cannot be read must not cost the day's comic."""
        try:
            entries = self.comic_log.entries() if self.comic_log else []
        except Exception:
            logger.warning("Recent-comics log unreadable; writing the comic without it", exc_info=True)
            return []
        return [e for e in entries if isinstance(e, dict) and str(e.get("date", "")) < post_date.isoformat()]

    @property
    def threads_outcome(self) -> ThreadsDelivery | None:
        """The publisher's last Threads verdict, for the caller's metrics/alerts."""
        return self.publisher.threads_outcome

    @property
    def threads_ledger(self) -> ThreadsPostLedger | None:
        return self.publisher.threads_ledger

    @property
    def threads_unprotected(self) -> bool:
        """Posts are live but the date is not recorded as posted, so a replay would post again."""
        return self.publisher.unprotected

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
        if not self.generator.renderer.ready():
            return None, None

        try:
            plan, instruction, source = await self._compose(ranked_items, content, post_date)
        except Exception:
            # Best-effort: a visual failure must never block the digest, so catch broadly here.
            logger.warning("Daily visual editor failed; publishing the digest text-only", exc_info=True)
            return None, None
        if not instruction:
            logger.info("Daily visual: editor could not illustrate the headline; publishing the digest text-only")
            return None, None

        image_bytes: bytes | None = None
        brief: VisualBrief | None = None
        try:
            image_bytes, brief = await self.generator.generate(
                instruction, source, references=self._references(plan), deadline=self._image_deadline(deadline)
            )
        except Exception:
            # The image is an optional attachment; its failure must NOT sink the Threads text
            # digest, which stands on its own. Fall through with no image so the publisher still
            # posts the lead + per-story replies (text-only).
            logger.warning("Daily visual generation failed; posting Threads text-only", exc_info=True)

        return image_bytes, brief

    async def _compose(
        self,
        ranked_items: list[RankedItem],
        content: DigestContent | None,
        post_date: date,
        recent_comics: list[dict] | None = None,
    ) -> tuple[dict, str, str]:
        """(editor plan, art-director instruction, source material): everything the render is made
        from, shared with scripts/sample_visual_brief.py so the sampler grades what production sends.
        `recent_comics` defaults to the log's; the sampler passes its own to replay a past day. The
        instruction is "" when the editor skips."""
        source = self._headline(content, ranked_items)
        if recent_comics is None:
            recent_comics = self._recent_comics(post_date)
        plan = await self._write_synopsis(source, recent_comics)
        # coerce_bool, not truthiness: an editor that writes the STRING "false" here silently
        # killed the day's visual, with no error anywhere.
        if coerce_bool(plan.get("skip")) or not str(plan.get("instruction") or "").strip():
            # A plan with no synopsis is skipped like an unparseable one: rendering a generic
            # "comic about <title>" paid for an image nobody wrote.
            return plan, "", source
        return plan, self._build_instruction(plan), source

    def _build_instruction(self, plan: dict) -> str:
        """The editor's synopsis plus what the art director must keep: the guardrails and, when the
        narrator appears, his sheet."""
        instruction = plan["instruction"]
        if self.config.pipeline.visual_guardrails:
            instruction += f"\n\nGUARDRAILS: {self.config.pipeline.visual_guardrails}"
        # The sheet is written into the brief; his exact look rides along as the reference image.
        if self._features_character(plan):
            instruction += (
                "\n\nTHE NARRATOR, in the panels the synopsis puts him in: "
                f"{self.config.pipeline.visual_character_sheet}"
            )
        return instruction

    def _image_deadline(self, deadline: float | None) -> float | None:
        """When the image work must be done: the caller's deadline minus what the Threads publish
        still needs afterwards, so a slow render can never eat the day's post."""
        return None if deadline is None else deadline - self.threads.publish_budget_sec

    def _features_character(self, plan: dict) -> bool:
        return self.config.pipeline.visual_character_enabled and coerce_bool(plan.get("use_character"), default=False)

    def _references(self, plan: dict) -> list[bytes]:
        """The input images the render carries: the narrator's reference sheet when he appears."""
        return [self.character_reference] if self._features_character(plan) and self.character_reference else []

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
        """The curated story the lead is about (content.items[0]), or None."""
        return content.items[0] if content and content.items else None

    def _headline(self, content: DigestContent | None, ranked_items: list[RankedItem]) -> str:
        """The source text (title, then body) of the story the visual must depict: the curated headline, read from
        its ranked item (the full article) when one matches.

        Matched on the aggregator's normalize_url: the editor echoes the URL back, and a trailing
        slash or a utm parameter is enough to miss an exact match. With no match the CURATED story's
        own prose is the source; falling back to ranked #1 drew a different story than the lead."""
        curated = self._curated_headline(content)
        if curated is None:
            # No structured content at all (visual-only run): the top-ranked story is the headline.
            item = ranked_items[0].item
            return f"{item.title}\n\n{item.text}"
        url = normalize_url(curated.url)
        for r in ranked_items:
            # A snapshot over the store's size limit keeps ranked items without their text; the
            # curated prose then beats a bare title.
            if normalize_url(r.item.url) == url and (r.item.text or "").strip():
                return f"{r.item.title}\n\n{r.item.text}"
        # Not the implication: it is a recommendation, and given it the editor's panel 4 turned into a moral.
        return f"{curated.title}\n\n{curated.body.strip()}"

    async def _write_synopsis(self, headline_source: str, recent_comics: list[dict]) -> dict:
        """The editor's plan for the comic about the headline (it does not choose the story). It does not
        see the digest's lead: given the lead, it drew the lead's argument one sentence per panel (four
        comics, 2026-10-18 to 10-21), each panel adding a fact, a company and a new set."""
        chain = VisualEditorPrompt.get_prompt() | self.llm | StrOutputParser()
        raw = await chain.ainvoke(
            {
                "headline": self.models.truncate_to_tokens(
                    headline_source.strip(), self.config.pipeline.visual_editor_source_max_tokens
                ),
                "recent_comics": _format_recent_comics(recent_comics),
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


def _format_recent_comics(entries: list[dict]) -> str:
    lines = [
        f"- {e.get('date', '')} ({e.get('orientation', '')}): {e.get('comic', '')}" for e in entries if e.get("comic")
    ]
    return "\n".join(lines) or "(none yet)"
