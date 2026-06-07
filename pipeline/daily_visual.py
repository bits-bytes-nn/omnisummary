from __future__ import annotations

import asyncio
import hashlib
import json
import os

from langchain_core.output_parsers import StrOutputParser

from agent.visuals import VisualGenerator
from shared import (
    BedrockLanguageModelFactory,
    DigestContent,
    RankedItem,
    VisualBrief,
    VisualEditorPrompt,
    extract_json_from_llm_output,
    logger,
    resolve_secret,
)
from shared.config import Config


class DailyVisualMaker:
    """Picks one digest story and renders a fun daily visual (meme / parody /
    illustration / N-panel cartoon), then posts it to Slack. Best-effort: any failure
    (no OpenAI key, no fit, search/render error) is logged and skipped — it must never
    break the digest pipeline."""

    def __init__(self, config: Config, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm = llm_factory.get_model(config.pipeline.digest_model)
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

    async def run(self, ranked_items: list[RankedItem], content: DigestContent | None = None) -> bool:
        if not ranked_items:
            return False
        if not resolve_secret("OPENAI_API_KEY", "openai-api-key"):
            logger.info("OPENAI_API_KEY not set, skipping daily visual")
            return False

        # The visual MUST depict the digest's headline so the image and the lead stay in sync.
        # content.headline_index is into the curated content.items (may be merged/reordered), so
        # map it back to a ranked_items position by URL; fall back to the top-ranked item.
        headline_index = self._headline_ranked_index(content, ranked_items) or 1
        try:
            plan = await self._pick_story(ranked_items, headline_index)
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

        try:
            image_bytes, brief = await self.generator.generate(instruction, source, context)
        except Exception:
            # Best-effort: a visual failure must never block the digest, so catch broadly here.
            logger.warning("Daily visual generation failed", exc_info=True)
            return False

        slack_ok = await self._post(image_bytes, brief)
        await self._post_threads(image_bytes, brief, content)
        return slack_ok

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

    async def _pick_story(self, ranked_items: list[RankedItem], headline_index: int = 0) -> dict:
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
            }
        )
        try:
            return json.loads(extract_json_from_llm_output(raw))
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
        from agent.agent_tools import _search_papers, _tavily_search

        query = step["query"]
        source = str(step.get("source", "news")).lower()
        if source == "papers":
            return await _search_papers(query)
        if source == "community":
            return await _tavily_search(query, include_domains=self.config.agent.community_search_domains)
        return await _tavily_search(query, topic="news")

    async def _post(self, image_bytes: bytes, brief: VisualBrief) -> bool:
        if not self.config.pipeline.enable_slack_post:
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

    async def _post_threads(self, image_bytes: bytes, brief: VisualBrief, content: DigestContent | None) -> bool:
        if not self.config.pipeline.enable_threads_post:
            return False
        from output.renderers import render_threads_posts
        from output.threads_handler import post_to_threads

        # Root = visual image + the digest lead (which already carries the AGI-countdown intro,
        # prepended at digest generation); replies = one per story. When no structured content is
        # available, fall back to the visual's own title/caption as the root.
        if content and content.items:
            root_text, replies = render_threads_posts(content)
        else:
            root_text, replies = f"{brief.title}\n\n{brief.caption}", []

        bucket = self.config.aws.state_bucket_name or os.environ.get("STATE_BUCKET", "")
        prefix = self.config.aws.s3_prefix.rstrip("/") + "/" if self.config.aws.s3_prefix else ""
        image_key = f"{prefix}threads/{hashlib.sha256(image_bytes).hexdigest()[:16]}.png"
        return await post_to_threads(
            root_text=root_text,
            replies=replies,
            image_bytes=image_bytes,
            image_bucket=bucket,
            image_key=image_key,
        )
