from __future__ import annotations

import json
from datetime import UTC, datetime
from urllib.parse import urlparse

from langchain_core.output_parsers import StrOutputParser

from shared import (
    YOUTUBE_VIEWS_EMOJI,
    BedrockLanguageModelFactory,
    CollectedItem,
    DigestPrompt,
    DigestResult,
    GroundingCheckPrompt,
    RankedItem,
    SourceType,
    clean_rss_feed_name,
    extract_json_from_llm_output,
    format_collected_item,
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
                "language_rules": self.config.digest_language_rules,
                "audience": self.config.digest_audience_description,
            }
        )
        digest_text = sanitize_slack_mrkdwn(digest_text)

        if self.config.enable_grounding_check:
            digest_text = await self._verify_grounding(digest_text, ranked_items)

        logger.info("Digest generated successfully (%d characters)", len(digest_text))

        return DigestResult(
            digest_text=digest_text,
            ranked_items=ranked_items,
            generated_at=datetime.now(UTC),
            total_collected=len(all_items),
            total_ranked=len(ranked_items),
        )

    async def _verify_grounding(self, digest_text: str, ranked_items: list[RankedItem]) -> str:
        """Check the digest's specific claims against the source items and surgically revise
        unsupported ones. Best-effort: any failure keeps the original digest."""
        try:
            sources = "\n\n".join(
                f"[{i + 1}] {r.item.title}\n{truncate_text_by_tokens(r.item.text, self.config.item_text_max_tokens)}"
                for i, r in enumerate(ranked_items)
            )
            chain = GroundingCheckPrompt.get_prompt() | self.llm | StrOutputParser()
            raw = await chain.ainvoke({"digest_text": digest_text, "sources": sources})
            data = json.loads(extract_json_from_llm_output(raw))
            violations = data.get("violations", [])
            corrected = data.get("corrected_digest", "")
            if not violations or not corrected:
                logger.info("Grounding check: no unsupported claims found")
                return digest_text
            for v in violations:
                logger.info("Grounding check revised claim: %s (%s)", v.get("claim", "")[:80], v.get("issue", "")[:80])
            logger.info("Grounding check revised %d unsupported claim(s)", len(violations))
            return sanitize_slack_mrkdwn(corrected)
        except Exception:
            logger.warning("Grounding check failed; keeping original digest", exc_info=True)
            return digest_text

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
