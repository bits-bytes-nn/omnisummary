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
    coerce_bool,
    create_state_store,
    logger,
    parse_json_from_llm_output,
    resolve_secret,
)
from shared.config import Config
from shared.history_store import RECENT_COMICS_KEY, RollingLog, ThreadsPostLedger
from shared.state_store import StateStore

if TYPE_CHECKING:
    from output.threads_handler import ThreadsDelivery


class DailyVisualMaker:
    """Renders the day's visual for the digest headline (normally a four-panel comic), then
    hands it to the digest publisher as an attachment.

    Best-effort in BOTH directions: a visual failure (no OpenAI key, an editor skip, a render error)
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
        # What the last few comics drew, shown to the editor so today's does not repeat their sets,
        # props and gags (2026-10-13, -16 and -17 all locked a model in a glass case).
        self.comic_log = RollingLog(store, RECENT_COMICS_KEY, config.pipeline.recent_leads_window) if store else None
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
        headline_title, source = self._headline(content, ranked_items)
        if recent_comics is None:
            recent_comics = self._recent_comics(post_date)
        plan = await self._write_synopsis(source, recent_comics)
        # coerce_bool, not truthiness: an editor that writes the STRING "false" here silently
        # killed the day's visual, with no error anywhere.
        if coerce_bool(plan.get("skip")):
            return plan, "", source
        return plan, self._build_instruction(plan, headline_title), source

    def _build_instruction(self, plan: dict, headline_title: str) -> str:
        """The editor's synopsis plus what the art director must keep: the guardrails and, when the
        narrator appears, his sheet."""
        instruction = plan.get("instruction", "") or f"A 2x2 four-panel comic about: {headline_title}"
        if self.config.pipeline.visual_guardrails:
            instruction += f"\n\nGUARDRAILS: {self.config.pipeline.visual_guardrails}"
        # The sheet is written into the brief; his exact look rides along as the reference image.
        if self._features_character(plan):
            instruction += (
                "\n\nTHE NARRATOR, in the panels the synopsis puts him in: "
                f"{self.config.pipeline.visual_character_sheet}"
            )
        return instruction

    @staticmethod
    def _image_deadline(deadline: float | None) -> float | None:
        """When the image work must be done: the caller's deadline minus what the Threads publish
        still needs afterwards, so a slow render can never eat the day's post."""
        from output.threads_handler import THREADS_INDEXING_BUDGET_SEC, THREADS_PUBLISH_RESERVE_SEC

        return None if deadline is None else deadline - THREADS_INDEXING_BUDGET_SEC - THREADS_PUBLISH_RESERVE_SEC

    def _features_character(self, plan: dict) -> bool:
        return self.config.pipeline.visual_character_enabled and coerce_bool(plan.get("use_character"), default=True)

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
        """The curated story the lead is about (content.items[headline_index]), or None."""
        if not content or not content.items:
            return None
        idx = content.headline_index if 1 <= content.headline_index <= len(content.items) else 1
        return content.items[idx - 1]

    def _headline(self, content: DigestContent | None, ranked_items: list[RankedItem]) -> tuple[str, str]:
        """(title, source text) of the story the visual must depict: the curated headline, read from
        its ranked item (the full article) when one matches.

        Matched on the aggregator's normalize_url: the editor echoes the URL back, and a trailing
        slash or a utm parameter is enough to miss an exact match. With no match the CURATED story's
        own prose is the source; falling back to ranked #1 drew a different story than the lead."""
        curated = self._curated_headline(content)
        if curated is None:
            # No structured content at all (visual-only run): the top-ranked story is the headline.
            item = ranked_items[0].item
            return item.title, f"{item.title}\n\n{item.text}"
        url = normalize_url(curated.url)
        for r in ranked_items:
            if normalize_url(r.item.url) == url:
                return r.item.title, f"{r.item.title}\n\n{r.item.text}"
        body = "\n\n".join(p for p in (curated.body.strip(), curated.implication.strip()) if p)
        return curated.title, f"{curated.title}\n\n{body}"

    async def _write_synopsis(self, headline_source: str, recent_comics: list[dict]) -> dict:
        """The editor's plan for the comic about the headline (it does not choose the story). It does not
        see the digest's lead: given the lead, it drew the lead's argument one sentence per panel (four
        comics, 2026-10-18 to 10-21), each panel adding a fact, a company and a new set."""
        chain = VisualEditorPrompt.get_prompt() | self.llm | StrOutputParser()
        raw = await chain.ainvoke(
            {
                "headline": self.llm_factory.truncate_to_tokens(
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
