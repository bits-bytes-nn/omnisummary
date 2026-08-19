from __future__ import annotations

import asyncio
import json
from datetime import date, datetime
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
    RollingLog,
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
from shared.history_store import VISUAL_FORMATS_KEY, ThreadsPostLedger
from shared.state_store import StateStore

if TYPE_CHECKING:
    from output.threads_handler import ThreadsDelivery


class DailyVisualMaker:
    """Picks one digest story and renders a fun daily visual (meme / parody /
    illustration / N-panel cartoon), then hands it to the digest publisher as an attachment.

    Best-effort in BOTH directions: a visual failure (no OpenAI key, no fit, search/render error)
    is logged and the digest still publishes TEXT-ONLY, and a delivery failure never escapes into
    the pipeline. Publication itself belongs to `output.digest_delivery.DigestPublisher`, which takes
    the image as an OPTIONAL argument — so no failure on this side can cost the day's digest."""

    def __init__(self, config: Config, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm_factory = llm_factory
        self.llm = llm_factory.get_model(config.pipeline.digest_model, stage="visual-editor")
        # Format-variation history is best-effort: if the state store can't be created
        # (misconfigured bucket/profile), degrade to no history rather than crash the visual.
        store: StateStore | None
        try:
            store = create_state_store(config)
            self.format_log: RollingLog | None = RollingLog(
                store, VISUAL_FORMATS_KEY, config.pipeline.visual_format_window
            )
        except Exception:
            logger.error("Visual format history is unavailable (state store init failed)", exc_info=True)
            store = None
            self.format_log = None
        # The publisher shares this run's store, so one init failure doesn't produce two of them.
        self.publisher = DigestPublisher(config, store)
        self.generator = VisualGenerator(
            llm_factory,
            config.pipeline.digest_model,
            image_model=config.pipeline.image_model,
            image_sizes=config.pipeline.image_sizes,
            source_max_tokens=config.pipeline.visual_synopsis_source_max_tokens,
            context_max_tokens=config.pipeline.visual_synopsis_context_max_tokens,
            caption_language=config.pipeline.visual_caption_language,
            on_image_language=config.pipeline.visual_on_image_language,
            moderation_softening_instruction=config.pipeline.visual_moderation_softening_instruction,
            style_guidance=config.pipeline.visual_synopsis_style_guidance,
            humor_guidance=config.pipeline.visual_synopsis_humor_guidance,
            style_aesthetic=config.pipeline.visual_synopsis_style_aesthetic,
            image_timeout_sec=config.pipeline.visual_image_timeout_sec,
            image_max_retries=config.pipeline.visual_image_max_retries,
            image_quality=config.pipeline.visual_image_quality,
        )

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

        image_bytes, brief = await self._make_visual(ranked_items, content, post_date, deadline=deadline)

        return await self.publisher.publish(
            content,
            image_bytes=image_bytes,
            brief=brief,
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
        recent_formats = self.format_log.entries() if self.format_log else []
        preferred_orientation = self._least_recent_orientation(recent_formats)
        take = self._editorial_take(content, post_date)
        try:
            plan = await self._pick_story(
                ranked_items,
                marker_index,
                recent_formats,
                preferred_orientation,
                headline_source=source,
                editorial_take=take,
            )
        except Exception:
            # Best-effort: a visual failure must never block the digest, so catch broadly here.
            logger.warning("Daily visual editor failed; publishing the digest text-only", exc_info=True)
            return None, None

        # coerce_bool, not truthiness: an editor that writes the STRING "false" here silently
        # killed the day's visual, with no error anywhere.
        if coerce_bool(plan.get("skip")):
            logger.info("Daily visual: editor could not illustrate the headline; publishing the digest text-only")
            return None, None

        try:
            context = await self._gather_context(plan.get("research", []))
        except Exception:
            # Extra context is a nice-to-have; a research backend outage must not reach the digest.
            logger.warning("Daily visual context gathering failed; continuing without it", exc_info=True)
            context = ""
        instruction, use_character = self._build_instruction(
            plan, content, post_date, headline_title, recent_formats, preferred_orientation
        )

        image_bytes: bytes | None = None
        brief: VisualBrief | None = None
        try:
            image_bytes, brief = await self.generator.generate(instruction, source, context, deadline=deadline)
        except Exception:
            # The image is an optional attachment; its failure must NOT sink the Threads text
            # digest, which stands on its own. Fall through with no image so _post_threads still
            # posts the lead + per-story replies (text-only). Slack image upload is skipped below.
            logger.warning("Daily visual generation failed; posting Threads text-only", exc_info=True)

        # Record the chosen format so tomorrow can deliberately differ. Best-effort. Only when a
        # brief was actually rendered — a text-only fallback has no format to record. Deduped by
        # date so a same-day re-run replaces its entry instead of pushing a duplicate that crowds
        # the variation window (same convention as the recent-leads log).
        if brief and self.format_log:
            try:
                self.format_log.append(
                    {
                        "date": post_date.isoformat(),
                        "orientation": brief.orientation,
                        "format": plan.get("format", ""),
                        "multi_panel": coerce_bool(plan.get("multi_panel")),
                        "use_character": use_character,
                    },
                    dedup_key="date",
                )
            except Exception:
                logger.warning("Failed to record visual format history (non-fatal)", exc_info=True)
        return image_bytes, brief

    def _build_instruction(
        self,
        plan: dict,
        content: DigestContent | None,
        post_date: date,
        headline_title: str,
        recent_formats: list[dict],
        preferred_orientation: str,
    ) -> tuple[str, bool]:
        """The art-director instruction production actually sends, plus whether the recurring
        character was requested.

        A PURE extraction from _make_visual — no I/O, no state — so scripts/sample_visual_brief.py
        can grade the very string production sends. The sampler used to brief the bare
        `plan["instruction"]`, i.e. without the editorial angle, the guardrails, the format nudge or
        the character sheet: it was scoring a prompt that never ships."""
        instruction = plan.get("instruction", "") or f"A fun visual about: {headline_title}"

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
        # The art-director picks orientation, but it anchors to the same one for days. Steer it to
        # the least-recently-used aspect ratio so consecutive visuals actually vary in shape.
        if preferred_orientation:
            instruction += (
                f"\n\nVARY THE FORMAT: recent daily visuals used {self._recent_orientations(recent_formats)}. "
                f"Make TODAY visually different — use a '{preferred_orientation}' orientation and a different "
                "composition (panel count / genre) than those."
            )

        # The recurring mascot appears only when the editor judged it fits this story (use_character).
        # Inject the character sheet so the image model draws the SAME person; identity rides on the
        # signature props, so it survives the daily-varying art style.
        use_character = coerce_bool(plan.get("use_character")) and self.config.pipeline.visual_character_enabled
        if use_character:
            instruction += (
                "\n\nFEATURE THE RECURRING CHARACTER as a witness reacting inside this scene "
                f"(do not just draw him in a void): {self.config.pipeline.visual_character_sheet}"
            )
        return instruction, use_character

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
        """The AGI-countdown gag for this date, exactly as the digest generator computed it — the one
        part of the lead CODE owns, so the renderer can identify (and drop) it if the lead overflows."""
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

    def _least_recent_orientation(self, recent_formats: list[dict]) -> str:
        """Return an orientation not used in the recent window (least-recently-used), so the
        next visual differs in shape. Empty when no orientations are configured, or when there is
        NO history at all: with nothing to vary FROM, nudging toward whichever orientation happens
        to be listed first is an arbitrary lock (and contradicts the 'pick whatever fits this
        story' guidance the format block emits on a first run)."""
        all_orientations = list(self.config.pipeline.image_sizes)
        if not all_orientations or not recent_formats:
            return ""
        used = [f.get("orientation", "") for f in recent_formats]
        unused = [o for o in all_orientations if o not in used]
        if unused:
            return unused[0]
        # All used recently → pick the one whose LAST use is oldest. Scanning the window in order
        # and returning its first entry picked the orientation that appeared earliest, which is the
        # MOST recently used one whenever it recurs later in the window (e.g. square, landscape,
        # portrait, square → it answered 'square', yesterday's shape).
        last_use = {o: i for i, o in enumerate(used) if o in all_orientations}
        return min(last_use, key=lambda o: last_use[o]) if last_use else all_orientations[0]

    @staticmethod
    def _recent_orientations(recent_formats: list[dict]) -> str:
        seen = [f.get("orientation", "") for f in recent_formats if f.get("orientation")]
        return ", ".join(seen) if seen else "none recorded"

    @staticmethod
    def _ratio_share(recent_formats: list[dict], key: str) -> float | None:
        """Share of recent entries (that carry `key`) where it's truthy. None when no entry
        records the key yet — so a nudge has no basis and is skipped."""
        flagged = [f for f in recent_formats if key in f]
        if not flagged:
            return None
        return sum(1 for f in flagged if f.get(key)) / len(flagged)

    @classmethod
    def _panel_nudge(cls, recent_formats: list[dict], target_ratio: float) -> str:
        """Soft-steer the single-vs-multi-panel mix. The editor leans single-frame on its own,
        so when the recent multi-panel share is below target, nudge toward a multi-panel
        sequence; when above, toward a single frame. Empty when no history / disabled (target 0)."""
        if target_ratio <= 0:
            return ""
        share = cls._ratio_share(recent_formats, "multi_panel")
        if share is None:
            return ""
        if share < target_ratio:
            return (
                " Recent visuals have skewed to single-frame compositions; if this story has any "
                "sequence, reversal, or setup-and-payoff, lean toward a MULTI-PANEL comic today."
            )
        return " Recent visuals have leaned multi-panel; prefer a single striking frame today unless the story truly needs a sequence."

    @classmethod
    def _character_nudge(cls, recent_formats: list[dict], target_ratio: float) -> str:
        """Soft-steer how often the recurring character appears, toward target_ratio. Empty when
        no history / disabled (target 0) — and the editor still skips him when the story doesn't fit."""
        if target_ratio <= 0:
            return ""
        share = cls._ratio_share(recent_formats, "use_character")
        if share is None:
            return ""
        if share < target_ratio:
            return " The recurring character hasn't appeared lately; if he'd fit this story as a reacting witness, bring him in today."
        return " The recurring character has appeared often lately; lean toward a character-free concept visual today unless he genuinely fits."

    @classmethod
    def _format_guidance(
        cls,
        recent_formats: list[dict],
        preferred_orientation: str,
        panel_target_ratio: float = 0.0,
        character_target_ratio: float = 0.0,
    ) -> str:
        if not recent_formats:
            return "No recent visuals on record — pick whatever format fits this story best."
        recent = "; ".join(
            f"{f.get('orientation', '?')}/{f.get('format', '?')}" for f in recent_formats if f.get("orientation")
        )
        line = f"Recent daily visuals (most recent last): {recent}. Deliberately differ TODAY — "
        if preferred_orientation:
            line += f"prefer a '{preferred_orientation}' orientation and "
        line += "choose a composition/genre you have NOT used recently so consecutive visuals don't look alike."
        return (
            line
            + cls._panel_nudge(recent_formats, panel_target_ratio)
            + cls._character_nudge(recent_formats, character_target_ratio)
        )

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

        The editor writes the joke, the format and the 1-3 research queries, and it used to decide all
        of that from `N. [source] title` rows — the headline's body, the lead and the implication only
        reached the art director later, in _build_instruction. Handing the headline over explicitly
        also closes the no-ranked-match case: with headline_index 0 NO row carried the marker the
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
        recent_formats: list[dict] | None = None,
        preferred_orientation: str = "",
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
                "format_guidance": self._format_guidance(
                    recent_formats or [],
                    preferred_orientation,
                    self.config.pipeline.visual_multi_panel_target_ratio,
                    (
                        self.config.pipeline.visual_character_target_ratio
                        if self.config.pipeline.visual_character_enabled
                        else 0.0
                    ),
                ),
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

    async def _gather_context(self, research: list[dict]) -> str:
        """Run the editor's chosen research steps and concatenate the findings. Each step
        names a source the LLM picked for THIS story — papers (Semantic Scholar), community
        (Reddit/X/HN/Substack), or news — dispatched to the same backends the agent uses.
        Best-effort: a failed or unknown step is skipped, never blocking the visual."""
        steps = [s for s in (research or []) if isinstance(s, dict) and s.get("query")]
        if not steps:
            return ""
        # The prompt asks for 1-3 steps, but nothing stopped a chatty editor from returning ten —
        # each one is a live Tavily/Semantic Scholar call. Clamp to the configured budget.
        max_steps = self.config.pipeline.visual_research_max_steps
        if len(steps) > max_steps:
            logger.info("Visual editor returned %d research steps; clamping to %d", len(steps), max_steps)
            steps = steps[:max_steps]
        results = await asyncio.gather(*(self._run_research_step(s) for s in steps), return_exceptions=True)
        blocks = [r for r in results if isinstance(r, str) and r]
        return "\n\n".join(blocks)

    async def _run_research_step(self, step: dict) -> str:
        from shared.research import semantic_scholar_search, tavily_search

        query = step["query"]
        source = str(step.get("source", "news")).lower()
        if source == "papers":
            return await semantic_scholar_search(query)
        if source == "community":
            return await tavily_search(query, include_domains=self.config.agent.community_search_domains)
        return await tavily_search(query, topic="news")
