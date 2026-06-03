from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser

from agent.visuals import VisualGenerator
from shared import (
    BedrockLanguageModelFactory,
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
            image_size=config.pipeline.image_size,
            source_max_tokens=config.pipeline.visual_synopsis_source_max_tokens,
            context_max_tokens=config.pipeline.visual_synopsis_context_max_tokens,
            caption_language=config.pipeline.visual_caption_language,
            on_image_language=config.pipeline.visual_on_image_language,
        )

    async def run(self, ranked_items: list[RankedItem]) -> bool:
        if not ranked_items:
            return False
        if not resolve_secret("OPENAI_API_KEY", "openai-api-key"):
            logger.info("OPENAI_API_KEY not set, skipping daily visual")
            return False

        try:
            plan = await self._pick_story(ranked_items)
        except Exception:
            logger.warning("Daily visual editor failed", exc_info=True)
            return False

        if not plan or plan.get("skip"):
            logger.info("Daily visual: no suitable story today, skipping")
            return False

        item_number = plan.get("item_number", 0)
        if not (1 <= item_number <= len(ranked_items)):
            logger.info("Daily visual: editor returned invalid item_number %s, skipping", item_number)
            return False

        ranked = ranked_items[item_number - 1]
        source = f"{ranked.item.title}\n\n{ranked.item.text}"
        context = await self._gather_context(plan.get("search_query", ""))
        instruction = plan.get("instruction", "") or f"A fun visual about: {ranked.item.title}"

        try:
            image_bytes, brief = await self.generator.generate(instruction, source, context)
        except Exception:
            logger.warning("Daily visual generation failed", exc_info=True)
            return False

        return await self._post(image_bytes, brief)

    async def _pick_story(self, ranked_items: list[RankedItem]) -> dict:
        items_text = "\n".join(
            f"{i}. [{r.item.source_type.value}] {r.item.title}" for i, r in enumerate(ranked_items, start=1)
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

    async def _gather_context(self, query: str) -> str:
        if not query:
            return ""
        try:
            from tavily import AsyncTavilyClient

            api_key = resolve_secret("TAVILY_API_KEY", "tavily-api-key")
            if not api_key:
                return ""
            client = AsyncTavilyClient(api_key=api_key)
            max_results = self.config.pipeline.visual_context_max_results
            preview_chars = self.config.pipeline.visual_context_preview_chars
            response = await client.search(query=query, max_results=max_results, topic="news")
            results = response.get("results", [])
            return "\n\n".join(f"- {r.get('title', '')}: {r.get('content', '')[:preview_chars]}" for r in results)
        except Exception:
            logger.warning("Daily visual context search failed", exc_info=True)
            return ""

    async def _post(self, image_bytes: bytes, brief: VisualBrief) -> bool:
        from output.slack_handler import send_image_to_slack

        title = brief.title
        caption = brief.caption
        bot_token = self.config.slack.bot_token
        channel_id = self.config.slack.channel_id
        return await send_image_to_slack(
            image_bytes,
            channel_id=channel_id,
            title=title,
            comment=f"*{title}*\n{caption}",
            bot_token=bot_token,
        )
