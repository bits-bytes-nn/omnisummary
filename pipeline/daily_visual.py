from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from langchain_core.output_parsers import StrOutputParser

from agent.visuals import VisualGenerator
from pipeline.aggregator import normalize_url
from shared import (
    BedrockLanguageModelFactory,
    DigestContent,
    DigestItem,
    RankedItem,
    RollingLog,
    VisualBrief,
    VisualEditorPrompt,
    agi_countdown_intro,
    create_state_store,
    editorial_lead,
    get_correlation_id,
    logger,
    parse_json_from_llm_output,
    resolve_secret,
)
from shared.config import Config
from shared.history_store import VISUAL_FORMATS_KEY, ThreadsPostLedger

if TYPE_CHECKING:
    from output.threads_handler import ThreadsDelivery


class DailyVisualMaker:
    """Picks one digest story and renders a fun daily visual (meme / parody /
    illustration / N-panel cartoon), then posts the digest (image + text) to Slack/Threads.

    Best-effort in BOTH directions: a visual failure (no OpenAI key, no fit, search/render error)
    is logged and the digest still publishes TEXT-ONLY, and a delivery failure never escapes into
    the pipeline. This function is the only Threads publish path, so an early return here means
    the day's digest is never delivered at all."""

    def __init__(self, config: Config, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        # Last Threads publish outcome (posted/expected posts), for the caller's metrics/alerts.
        self.threads_outcome: ThreadsDelivery | None = None
        self.llm = llm_factory.get_model(config.pipeline.digest_model)
        # Format-variation history is best-effort: if the state store can't be created
        # (misconfigured bucket/profile), degrade to no history rather than crash the visual.
        try:
            store = create_state_store(config)
            self.format_log: RollingLog | None = RollingLog(
                store, VISUAL_FORMATS_KEY, config.pipeline.visual_format_window
            )
            self.threads_ledger: ThreadsPostLedger | None = ThreadsPostLedger(store)
        except Exception:
            logger.warning("Visual format history unavailable (state store init failed)", exc_info=True)
            self.format_log = None
            self.threads_ledger = None
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
        )

    async def run(
        self,
        ranked_items: list[RankedItem],
        content: DigestContent | None = None,
        *,
        today: date | None = None,
        force_republish: bool = False,
    ) -> bool:
        if not ranked_items:
            return False

        post_date = today or datetime.now(ZoneInfo(self.config.aws.timezone)).date()
        if self._nothing_left_to_publish(post_date, force_republish):
            logger.info(
                "Threads digest for %s already posted and no other channel is enabled, skipping "
                "(use force to re-publish)",
                post_date,
            )
            return False

        image_bytes, brief = await self._make_visual(ranked_items, content, post_date)

        slack_ok = await self._post(image_bytes, brief)
        threads_ok = await self._post_threads(image_bytes, content, today=post_date, force_republish=force_republish)
        # Success = at least one enabled channel published. Returning only slack_ok reported
        # "skipped" for every Threads-only run (the current config), hiding real outcomes.
        return slack_ok or threads_ok

    def _nothing_left_to_publish(self, post_date: date, force_republish: bool) -> bool:
        """True when the day is provably done: Threads already carries this date's digest AND no
        other channel could take the visual. Checked at the TOP of run() so an already-posted day
        doesn't pay for an LLM editor pass + a gpt-image render it can never publish.

        Deliberately narrow — with enable_slack_post on, the Slack image upload is a separate
        delivery the Threads marker says nothing about, so the run must proceed."""
        if force_republish or self.config.pipeline.enable_slack_post:
            return False
        if not self.config.pipeline.enable_threads_post:
            return False
        return bool(self.threads_ledger and self.threads_ledger.already_posted(post_date))

    async def _make_visual(
        self, ranked_items: list[RankedItem], content: DigestContent | None, post_date: date
    ) -> tuple[bytes | None, VisualBrief | None]:
        """Render the day's image, or (None, None). EVERY failure is contained here — a missing
        OpenAI key, an unusable editor plan, a render error — because the image is an ATTACHMENT to
        the digest, and this is the only Threads publish path. run() used to return early on all
        three, so a visual-only failure silently cost the whole day's digest."""
        if not resolve_secret("OPENAI_API_KEY", "openai-api-key"):
            logger.info("OPENAI_API_KEY not set — publishing the digest text-only, without a visual")
            return None, None

        # The visual MUST depict the digest's headline so the image and the lead stay in sync.
        # content.headline_index is into the curated content.items (may be merged/reordered), so
        # map it back to a ranked_items position by normalized URL.
        headline_index = self._headline_ranked_index(content, ranked_items)
        marker_index, headline_title, source = self._headline_brief(content, ranked_items, headline_index)
        recent_formats = self.format_log.entries() if self.format_log else []
        preferred_orientation = self._least_recent_orientation(recent_formats)
        try:
            plan = await self._pick_story(ranked_items, marker_index, recent_formats, preferred_orientation)
        except Exception:
            # Best-effort: a visual failure must never block the digest, so catch broadly here.
            logger.warning("Daily visual editor failed; publishing the digest text-only", exc_info=True)
            return None, None

        if plan.get("skip"):
            logger.info("Daily visual: editor could not illustrate the headline; publishing the digest text-only")
            return None, None

        try:
            context = await self._gather_context(plan.get("research", []))
        except Exception:
            # Extra context is a nice-to-have; a research backend outage must not reach the digest.
            logger.warning("Daily visual context gathering failed; continuing without it", exc_info=True)
            context = ""
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
        use_character = bool(plan.get("use_character", False)) and self.config.pipeline.visual_character_enabled
        if use_character:
            instruction += (
                "\n\nFEATURE THE RECURRING CHARACTER as a witness reacting inside this scene "
                f"(do not just draw him in a void): {self.config.pipeline.visual_character_sheet}"
            )

        image_bytes: bytes | None = None
        brief: VisualBrief | None = None
        try:
            image_bytes, brief = await self.generator.generate(instruction, source, context)
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
                        "multi_panel": bool(plan.get("multi_panel", False)),
                        "use_character": use_character,
                    },
                    dedup_key="date",
                )
            except Exception:
                logger.warning("Failed to record visual format history (non-fatal)", exc_info=True)
        return image_bytes, brief

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

    def _editorial_take(self, content: DigestContent | None, post_date: date) -> str:
        """The digest's angle on the headline story: its lead plus the headline item's closing
        implication. The AGI-countdown prefix is dropped — it's the same fixed template every day
        and carries no information about today's story. Empty when there is no structured content
        (a visual can still be made; it just won't know the take)."""
        if not content or not content.items:
            return ""
        intro = agi_countdown_intro(
            self.config.pipeline.agi_countdown_date,
            self.config.pipeline.agi_countdown_template,
            post_date,
            self.config.pipeline.agi_countdown_after,
        )
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

    async def _pick_story(
        self,
        ranked_items: list[RankedItem],
        headline_index: int = 0,
        recent_formats: list[dict] | None = None,
        preferred_orientation: str = "",
    ) -> dict:
        # The editor briefs the marked HEADLINE (it doesn't choose the story); the visual must
        # match the lead, which is about this same headline.
        items_text = "\n".join(
            f"{i}. [{r.item.source_type.value}] {r.item.title}"
            + (" ← TODAY'S HEADLINE — illustrate this one" if i == headline_index else "")
            for i, r in enumerate(ranked_items, start=1)
        )
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
        from shared.research import _search_papers, _tavily_search

        query = step["query"]
        source = str(step.get("source", "news")).lower()
        if source == "papers":
            return await _search_papers(query)
        if source == "community":
            return await _tavily_search(query, include_domains=self.config.agent.community_search_domains)
        return await _tavily_search(query, topic="news")

    async def _post(self, image_bytes: bytes | None, brief: VisualBrief | None) -> bool:
        if not self.config.pipeline.enable_slack_post:
            return False
        if not image_bytes or not brief:
            return False
        from output.slack_handler import send_image_to_slack

        title = brief.title
        caption = brief.caption
        emoji = self.config.pipeline.visual_caption_emoji
        bot_token = self.config.slack.bot_token
        channel_id = self.config.slack.channel_id
        return await send_image_to_slack(
            image_bytes,
            channel_id=channel_id,
            title=title,
            comment=f"{emoji} *{title}*\n{caption}",
            bot_token=bot_token,
        )

    async def _post_threads(
        self,
        image_bytes: bytes | None,
        content: DigestContent | None,
        *,
        today: date | None = None,
        force_republish: bool = False,
    ) -> bool:
        if not self.config.pipeline.enable_threads_post:
            return False
        from output.renderers import render_threads_posts
        from output.threads_handler import post_to_threads

        # Idempotency: a same-day re-run (manual `main.py`) or an automatic async retry of the
        # visual Lambda after a timeout would otherwise post the whole root+replies set again.
        # Skip if today's digest already went to Threads, unless explicitly forced.
        post_date = today or datetime.now(ZoneInfo(self.config.aws.timezone)).date()
        if self.threads_ledger and not force_republish and self.threads_ledger.already_posted(post_date):
            logger.info("Threads digest for %s already posted, skipping (use force to re-publish)", post_date)
            return False

        # Root = visual image + the digest lead (which already carries the AGI-countdown intro,
        # prepended at digest generation); replies = one per story.
        #
        # A digest with no stories is NOT posted. There used to be a fallback that published the
        # visual's own title/caption as a lone root with no replies: on 2026-08-13 and 2026-08-17 a
        # digest whose content failed to parse took that branch and published a story-less post
        # (one of them carrying leaked `</caption>` markup), consuming the day's ledger slot and
        # logging success. Skipping instead keeps the day retryable and never ships a broken digest.
        if not (content and content.items):
            logger.warning("No digest stories to post to Threads for %s; skipping (day stays retryable)", post_date)
            return False
        root_text, replies = render_threads_posts(content)

        bucket = self.config.aws.state_bucket_name or os.environ.get("STATE_BUCKET", "")
        prefix = self.config.aws.s3_prefix.rstrip("/") + "/" if self.config.aws.s3_prefix else ""
        image_key = f"{prefix}threads/{hashlib.sha256(image_bytes).hexdigest()[:16]}.png" if image_bytes else ""

        # Claim the date BEFORE the multi-minute post so concurrent invocations (e.g. a client
        # that retried a timed-out invoke) see it already taken and skip, instead of all passing
        # the already_posted() check above and each posting. Roll back if the post fails so a
        # genuine failure stays retryable — but only if WE added the mark, so a force-republish
        # failure doesn't wipe out a prior day's successful-post record.
        # run_id scopes marker ownership: a rollback only releases the marker THIS run wrote, so a
        # concurrent invocation's failure can't erase the marker of one that succeeded.
        run_id = get_correlation_id() or ""
        was_marked = bool(self.threads_ledger and self.threads_ledger.already_posted(post_date))
        if self.threads_ledger and not was_marked:
            try:
                self.threads_ledger.mark(post_date, run_id)
            except Exception:
                logger.warning("Failed to record Threads post marker (non-fatal)", exc_info=True)

        try:
            outcome = await post_to_threads(
                root_text=root_text,
                replies=replies,
                image_bytes=image_bytes,
                image_bucket=bucket,
                image_key=image_key,
            )
        except Exception:
            # Best-effort like the rest of the visual path: roll the claim back so the post
            # stays retryable, log, and don't let a Threads failure escape into run().
            logger.warning("Threads post failed", exc_info=True)
            if not was_marked:
                self._release_threads_marker(post_date, run_id)
            return False
        # Expose the (posted, expected) counts so the caller can report/alert on a partial chain.
        # Branch on .published explicitly — the outcome tuple itself is ALWAYS truthy, so `if
        # outcome:` would treat a 0-of-6 post as a success and skip the ledger rollback.
        self.threads_outcome = outcome
        if not outcome.published and not was_marked:
            self._release_threads_marker(post_date, run_id)
        return outcome.published

    def _release_threads_marker(self, post_date: date, run_id: str = "") -> None:
        if not self.threads_ledger:
            return
        try:
            self.threads_ledger.unmark(post_date, run_id)
        except Exception:
            logger.warning("Failed to roll back Threads post marker (non-fatal)", exc_info=True)
