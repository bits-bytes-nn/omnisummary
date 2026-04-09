from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urlparse

from langchain_core.output_parsers import StrOutputParser

from shared import (
    BedrockLanguageModelFactory,
    CollectedItem,
    DigestPrompt,
    DigestResult,
    RankedItem,
    SourceType,
    logger,
    sanitize_slack_mrkdwn,
    truncate_text_by_tokens,
)
from shared.config import PipelineConfig


class DigestGenerator:

    def __init__(self, config: PipelineConfig, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm = llm_factory.get_model(config.digest_model)

    async def generate(
        self,
        ranked_items: list[RankedItem],
        all_items: list[CollectedItem],
        trends_context: str = "",
    ) -> DigestResult:
        if not ranked_items:
            logger.warning("No ranked items to generate digest from")
            return DigestResult(
                digest_text="No notable content collected today.",
                ranked_items=[],
                generated_at=datetime.now(UTC),
                total_collected=len(all_items),
                total_ranked=0,
            )

        logger.info(
            "Generating digest from %d ranked items (model: %s)",
            len(ranked_items),
            self.config.digest_model.value,
        )

        items_text = self._format_ranked_items(ranked_items)
        chain = DigestPrompt.get_prompt() | self.llm | StrOutputParser()
        digest_text = await chain.ainvoke(
            {
                "items_text": items_text,
                "trends_context": trends_context or "(No trend data available yet.)",
            }
        )
        digest_text = sanitize_slack_mrkdwn(digest_text)

        logger.info("Digest generated successfully (%d characters)", len(digest_text))

        return DigestResult(
            digest_text=digest_text,
            ranked_items=ranked_items,
            generated_at=datetime.now(UTC),
            total_collected=len(all_items),
            total_ranked=len(ranked_items),
        )

    def _format_ranked_items(self, ranked_items: list[RankedItem]) -> str:
        parts: list[str] = []
        for i, ranked in enumerate(ranked_items):
            item = ranked.item
            snippet = truncate_text_by_tokens(item.text, self.config.item_text_max_tokens)
            source_detail = self._format_source_detail(item)
            parts.append(
                f"=== Item {i + 1} ===\n"
                f"Score: {ranked.score:.2f}\n"
                f"Categories: {', '.join(ranked.categories)}\n"
                f"Reasoning: {ranked.reasoning}\n"
                f"Title: {item.title}\n"
                f"URL: {item.url}\n"
                f"Source: {item.source_type.value}\n"
                f"Source Detail: {source_detail}\n"
                f"Author: {item.author or 'Unknown'}\n"
                f"Text:\n{snippet}\n"
            )
        return "\n".join(parts)

    @staticmethod
    def _format_source_detail(item: CollectedItem) -> str:
        meta = item.metadata
        tag = ""
        metrics: list[str] = []

        if item.source_type == SourceType.REDDIT:
            sub = meta.get("subreddit", "")
            tag = f"`r/{sub}`" if sub else "`Reddit`"
            if meta.get("score"):
                metrics.append(f":thumbsup: {meta['score']}")
            if meta.get("num_comments"):
                metrics.append(f":speech_balloon: {meta['num_comments']}")
        elif item.source_type == SourceType.YOUTUBE:
            tag = "`YouTube`"
            if meta.get("view_count"):
                metrics.append(f":arrow_forward: {meta['view_count']:,}")
        elif item.source_type == SourceType.X:
            tag = f"`@{item.author}`" if item.author else "`X`"
        elif item.source_type == SourceType.RSS:
            feed_title = meta.get("feed_title", "")
            name = feed_title.split(" - ")[0].split(" — ")[0].strip() if feed_title else ""
            if not name:
                feed_url = meta.get("feed_url", "")
                name = urlparse(feed_url).netloc.removeprefix("www.").removeprefix("feeds.") if feed_url else "RSS"
            tag = f"`{name}`"
        elif item.source_type == SourceType.WEB:
            domain = urlparse(item.url).netloc.removeprefix("www.")
            tag = f"`{domain}`" if domain else "`Web`"

        parts = [tag] + metrics

        return " · ".join(parts) if parts else item.source_type.value
