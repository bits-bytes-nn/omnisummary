from __future__ import annotations

import json

from langchain_core.output_parsers import StrOutputParser

from shared import (
    BedrockLanguageModelFactory,
    CollectedItem,
    RankedItem,
    RankingPrompt,
    SourceType,
    logger,
    truncate_text_by_tokens,
)
from shared.config import PipelineConfig


class ContentRanker:

    def __init__(self, config: PipelineConfig, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm = llm_factory.get_model(config.ranking_model)

    async def rank(self, items: list[CollectedItem]) -> list[RankedItem]:
        if not items:
            logger.warning("No items to rank")
            return []

        logger.info("Ranking %d items with model '%s'", len(items), self.config.ranking_model.value)

        items_text = self._format_items(items)
        chain = RankingPrompt.get_prompt() | self.llm | StrOutputParser()
        raw_output = await chain.ainvoke({"items_text": items_text})

        ranked_items = self._parse_rankings(raw_output, items)
        self._apply_origin_weights(ranked_items)

        above_threshold = [r for r in ranked_items if r.score >= self.config.min_score]
        above_threshold.sort(key=lambda r: (-r.score, r.item.item_id))

        source_scores: dict[str, list[float]] = {}
        for r in ranked_items:
            src = r.item.source_type.value
            source_scores.setdefault(src, []).append(r.score)
        for src, scores in sorted(source_scores.items()):
            above = [s for s in scores if s >= self.config.min_score]
            logger.info(
                "Source '%s': %d items, %d above %.2f, top scores: %s",
                src,
                len(scores),
                len(above),
                self.config.min_score,
                [f"{s:.2f}" for s in sorted(scores, reverse=True)[:5]],
            )

        selected = self._apply_source_slots(above_threshold)

        logger.info(
            "Ranked %d items → %d above min_score %.2f → %d selected (with source slots)",
            len(items),
            len(above_threshold),
            self.config.min_score,
            len(selected),
        )
        for r in selected:
            logger.info(
                "  Selected: [%s] %.2f - '%s'",
                r.item.source_type.value,
                r.score,
                r.item.title[:70],
            )
        return selected

    def _apply_origin_weights(self, ranked_items: list[RankedItem]) -> None:
        weights = self.config.origin_weights
        default_weight = self.config.origin_weight_default
        if not weights and default_weight == 1.0:
            return
        # A weight is a small ADDITIVE tie-breaker, not a multiplier. The LLM prompt
        # already judges Source Authority; multiplying its calibrated score by the
        # weight would double-count authority and distort the scale non-linearly
        # (and inflate mid-range scores most). nudge = (weight-1.0) * factor, clamped.
        nudge_factor = self.config.origin_weight_nudge
        for ranked in ranked_items:
            origin_key = self._resolve_origin_key(ranked.item)
            if not origin_key:
                continue
            weight = weights.get(origin_key, default_weight)
            if weight != 1.0:
                original = ranked.score
                ranked.score = max(0.0, min(1.0, ranked.score + (weight - 1.0) * nudge_factor))
                logger.debug(
                    "Applied origin nudge (w=%.2f) to '%s' (origin='%s'): %.2f → %.2f",
                    weight,
                    ranked.item.title[:50],
                    origin_key,
                    original,
                    ranked.score,
                )

    @staticmethod
    def _resolve_origin_key(item: CollectedItem) -> str | None:
        meta = item.metadata
        if item.source_type == SourceType.YOUTUBE:
            return meta.get("channel_url")
        if item.source_type == SourceType.REDDIT:
            return meta.get("subreddit")
        if item.source_type == SourceType.RSS:
            return meta.get("feed_url")
        if item.source_type == SourceType.X:
            return item.author
        return None

    def _apply_source_slots(self, above_threshold: list[RankedItem]) -> list[RankedItem]:
        source_slots = self.config.source_slots
        if not source_slots:
            return above_threshold[: self.config.top_n]

        selected: list[RankedItem] = []
        selected_ids: set[str] = set()
        source_counts: dict[str, int] = {}
        origin_counts: dict[str, int] = {}
        max_per_origin = self.config.max_per_origin

        def _origin_at_cap(item: RankedItem) -> bool:
            origin_key = self._resolve_origin_key(item.item)
            if not origin_key:
                return False
            return origin_counts.get(origin_key, 0) >= max_per_origin

        def _record(item: RankedItem, source_key: str) -> None:
            selected.append(item)
            selected_ids.add(item.item.item_id)
            source_counts[source_key] = source_counts.get(source_key, 0) + 1
            origin_key = self._resolve_origin_key(item.item)
            if origin_key:
                origin_counts[origin_key] = origin_counts.get(origin_key, 0) + 1

        for source_key, slot_count in source_slots.items():
            taken = 0
            for item in above_threshold:
                if taken >= slot_count or len(selected) >= self.config.top_n:
                    break
                if item.item.source_type.value != source_key or item.item.item_id in selected_ids:
                    continue
                if _origin_at_cap(item):
                    continue
                _record(item, source_key)
                taken += 1

        if len(selected) < self.config.top_n:
            for item in above_threshold:
                if item.item.item_id in selected_ids:
                    continue
                src = item.item.source_type.value
                cap = source_slots.get(src, 1) * self.config.source_cap_multiplier
                if source_counts.get(src, 0) >= cap or _origin_at_cap(item):
                    continue
                _record(item, src)
                if len(selected) >= self.config.top_n:
                    break

        # Final fallback: if diversity caps left the digest below top_n while valid
        # candidates remain, relax the per-origin cap (keep the source cap) so a quiet
        # day with few distinct origins still fills the digest.
        if len(selected) < self.config.top_n:
            for item in above_threshold:
                if item.item.item_id in selected_ids:
                    continue
                src = item.item.source_type.value
                cap = source_slots.get(src, 1) * self.config.source_cap_multiplier
                if source_counts.get(src, 0) >= cap:
                    continue
                _record(item, src)
                if len(selected) >= self.config.top_n:
                    break

        selected.sort(key=lambda r: (-r.score, r.item.item_id))
        return selected[: self.config.top_n]

    def _format_items(self, items: list[CollectedItem]) -> str:
        parts: list[str] = []
        for i, item in enumerate(items):
            snippet = truncate_text_by_tokens(item.text, self.config.item_text_max_tokens)
            engagement = self._format_engagement(item)
            origin = self._format_origin(item)
            entry = (
                f"=== Item {i + 1} ===\n"
                f"ID: {item.item_id}\n"
                f"Title: {item.title}\n"
                f"Source: {item.source_type.value}\n"
                f"Author: {item.author or 'Unknown'}\n"
            )
            if origin:
                entry += f"Origin: {origin}\n"
            if engagement:
                entry += f"Engagement: {engagement}\n"
            entry += f"Text:\n{snippet}\n"
            parts.append(entry)
        return "\n".join(parts)

    @staticmethod
    def _format_origin(item: CollectedItem) -> str:
        meta = item.metadata
        if item.source_type == SourceType.REDDIT:
            return f"r/{meta.get('subreddit', '')}" if meta.get("subreddit") else ""
        if item.source_type == SourceType.YOUTUBE:
            return meta.get("channel_url", "")
        if item.source_type == SourceType.X:
            return f"@{item.author}" if item.author else ""
        if item.source_type == SourceType.RSS:
            return meta.get("feed_title", "") or meta.get("feed_url", "")
        return ""

    @staticmethod
    def _format_engagement(item: CollectedItem) -> str:
        meta = item.metadata
        if item.source_type == SourceType.YOUTUBE and meta.get("view_count"):
            return f"{meta['view_count']:,} views"
        return ""

    def _parse_rankings(self, raw_output: str, items: list[CollectedItem]) -> list[RankedItem]:
        items_by_id = {item.item_id: item for item in items}

        try:
            json_str = raw_output.strip()
            if "```" in json_str:
                json_str = json_str.split("```")[-2] if json_str.count("```") >= 2 else json_str
                json_str = json_str.removeprefix("json").strip()
            start = json_str.find("{")
            end = json_str.rfind("}") + 1
            if start != -1 and end > start:
                json_str = json_str[start:end]

            data = json.loads(json_str)
            rankings = data.get("rankings", [])
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse ranking LLM output: '%s'", exc)
            logger.debug("Raw LLM output:\n%s", raw_output[:500])
            return []

        ranked_items: list[RankedItem] = []
        for entry in rankings:
            try:
                item_id = str(entry["item_id"])
                if item_id not in items_by_id:
                    logger.warning("Unknown item_id in ranking response: '%s'", item_id)
                    continue
                ranked_items.append(
                    RankedItem(
                        item=items_by_id[item_id],
                        score=float(entry["score"]),
                        reasoning=entry.get("reasoning", ""),
                        categories=entry.get("categories", []),
                    )
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed ranking entry: %s (%s)", entry, exc)

        return ranked_items
