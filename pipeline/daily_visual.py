from __future__ import annotations

import asyncio
import hashlib
import json
import os
from datetime import date, datetime
from zoneinfo import ZoneInfo

from langchain_core.output_parsers import StrOutputParser

from agent.visuals import VisualGenerator
from shared import (
    BedrockLanguageModelFactory,
    DigestContent,
    RankedItem,
    RollingLog,
    VisualBrief,
    VisualEditorPrompt,
    create_state_store,
    get_correlation_id,
    logger,
    parse_json_from_llm_output,
    resolve_secret,
)
from shared.config import Config
from shared.history_store import VISUAL_FORMATS_KEY, ThreadsPostLedger


class DailyVisualMaker:
    """Picks one digest story and renders a fun daily visual (meme / parody /
    illustration / N-panel cartoon), then posts it to Slack. Best-effort: any failure
    (no OpenAI key, no fit, search/render error) is logged and skipped — it must never
    break the digest pipeline."""

    def __init__(self, config: Config, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
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
        if not resolve_secret("OPENAI_API_KEY", "openai-api-key"):
            logger.info("OPENAI_API_KEY not set, skipping daily visual")
            return False

        # The visual MUST depict the digest's headline so the image and the lead stay in sync.
        # content.headline_index is into the curated content.items (may be merged/reordered), so
        # map it back to a ranked_items position by URL; fall back to the top-ranked item.
        headline_index = self._headline_ranked_index(content, ranked_items) or 1
        recent_formats = self.format_log.entries() if self.format_log else []
        preferred_orientation = self._least_recent_orientation(recent_formats)
        try:
            plan = await self._pick_story(ranked_items, headline_index, recent_formats, preferred_orientation)
        except Exception:
            # Best-effort: a visual failure must never block the digest, so catch broadly here.
            logger.warning("Daily visual editor failed", exc_info=True)
            return False

        if plan.get("skip"):
            logger.info("Daily visual: editor could not illustrate the headline, skipping")
            return False

        # The headline is authoritative; the editor only briefs HOW to draw it. Ignore any
        # off-headline item_number the editor might return.
        ranked = ranked_items[headline_index - 1]
        source = f"{ranked.item.title}\n\n{ranked.item.text}"
        context = await self._gather_context(plan.get("research", []))
        instruction = plan.get("instruction", "") or f"A fun visual about: {ranked.item.title}"
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

        slack_ok = await self._post(image_bytes, brief)
        await self._post_threads(image_bytes, brief, content, today=today, force_republish=force_republish)
        # Record the chosen format so tomorrow can deliberately differ. Best-effort. Only when a
        # brief was actually rendered — a text-only fallback has no format to record.
        if brief and self.format_log:
            try:
                self.format_log.append(
                    {
                        "orientation": brief.orientation,
                        "format": plan.get("format", ""),
                        "multi_panel": bool(plan.get("multi_panel", False)),
                        "use_character": use_character,
                    }
                )
            except Exception:
                logger.warning("Failed to record visual format history (non-fatal)", exc_info=True)
        return slack_ok

    def _least_recent_orientation(self, recent_formats: list[dict]) -> str:
        """Return an orientation not used in the recent window (least-recently-used), so the
        next visual differs in shape. Empty when no orientations are configured."""
        all_orientations = list(self.config.pipeline.image_sizes)
        if not all_orientations:
            return ""
        used = [f.get("orientation", "") for f in recent_formats]
        unused = [o for o in all_orientations if o not in used]
        if unused:
            return unused[0]
        # All used recently → pick the one used longest ago (earliest in the window).
        for entry in used:
            if entry in all_orientations:
                return entry
        return all_orientations[0]

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
        if not content or not content.items:
            return 0
        idx = content.headline_index
        if not (1 <= idx <= len(content.items)):
            return 0
        url = content.items[idx - 1].url
        for i, r in enumerate(ranked_items, start=1):
            if r.item.url == url:
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
            logger.warning("Daily visual editor returned unparseable JSON", exc_info=True)
            return {}

    async def _gather_context(self, research: list[dict]) -> str:
        """Run the editor's chosen research steps and concatenate the findings. Each step
        names a source the LLM picked for THIS story — papers (Semantic Scholar), community
        (Reddit/X/HN/Substack), or news — dispatched to the same backends the agent uses.
        Best-effort: a failed or unknown step is skipped, never blocking the visual."""
        steps = [s for s in (research or []) if isinstance(s, dict) and s.get("query")]
        if not steps:
            return ""
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
        brief: VisualBrief | None,
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
        # prepended at digest generation); replies = one per story. When no structured content is
        # available, fall back to the visual's own title/caption as the root.
        if content and content.items:
            root_text, replies = render_threads_posts(content)
        elif brief:
            root_text, replies = f"{brief.title}\n\n{brief.caption}", []
        else:
            # No structured digest AND no visual brief (image generation failed) → nothing to post.
            logger.info("No digest content or visual brief for Threads; skipping")
            return False

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
            posted = await post_to_threads(
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
        if not posted and not was_marked:
            self._release_threads_marker(post_date, run_id)
        return posted

    def _release_threads_marker(self, post_date: date, run_id: str = "") -> None:
        if not self.threads_ledger:
            return
        try:
            self.threads_ledger.unmark(post_date, run_id)
        except Exception:
            logger.warning("Failed to roll back Threads post marker (non-fatal)", exc_info=True)
