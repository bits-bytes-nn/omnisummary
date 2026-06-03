from __future__ import annotations

from datetime import UTC, datetime
from urllib.parse import urlparse

from langchain_core.output_parsers import StrOutputParser

from shared import (
    YOUTUBE_VIEWS_EMOJI,
    BedrockLanguageModelFactory,
    CollectedItem,
    DigestPrompt,
    DigestResult,
    RankedItem,
    SourceType,
    clean_rss_feed_name,
    format_collected_item,
    logger,
    sanitize_slack_mrkdwn,
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
                "language_rules": self.config.digest_language_rules,
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
            fields = [
                ("Score", f"{ranked.score:.2f}"),
                ("Categories", ", ".join(ranked.categories)),
                ("Reasoning", ranked.reasoning),
                ("Title", item.title),
                ("URL", item.url),
                ("Source", item.source_type.value),
                ("Source Detail", self._format_source_detail(item)),
                ("Author", item.author or "Unknown"),
            ]
            parts.append(
                format_collected_item(item, index=i + 1, max_tokens=self.config.item_text_max_tokens, fields=fields)
            )
        return "\n".join(parts)

    @staticmethod
    def _format_source_detail(item: CollectedItem) -> str:
        meta = item.metadata
        tag = ""
        metrics: list[str] = []

        if item.source_type == SourceType.REDDIT:
            # Reddit is collected via the public .rss feed, which carries no
            # score/num_comments — only the subreddit tag is available.
            sub = meta.get("subreddit", "")
            tag = f"`r/{sub}`" if sub else "`Reddit`"
        elif item.source_type == SourceType.YOUTUBE:
            tag = "`YouTube`"
            if meta.get("view_count"):
                metrics.append(f"{YOUTUBE_VIEWS_EMOJI} {meta['view_count']:,}")
        elif item.source_type == SourceType.X:
            tag = f"`@{item.author}`" if item.author else "`X`"
        elif item.source_type == SourceType.RSS:
            name = clean_rss_feed_name(meta.get("feed_title", ""), meta.get("feed_url", "")) or "RSS"
            tag = f"`{name}`"
        elif item.source_type == SourceType.WEB:
            domain = urlparse(item.url).netloc.removeprefix("www.")
            tag = f"`{domain}`" if domain else "`Web`"

        parts = [tag] + metrics

        return " · ".join(parts) if parts else item.source_type.value
