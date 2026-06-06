from __future__ import annotations

import asyncio
import json
from collections import defaultdict

from langchain_core.output_parsers import StrOutputParser

from shared import (
    LOGGING_TRUNCATION_CHARS,
    BedrockLanguageModelFactory,
    CollectedItem,
    RankedItem,
    RankingPrompt,
    SourceType,
    extract_json_from_llm_output,
    format_collected_item,
    format_origin_label,
    logger,
    normalize_title,
    resolve_origin_key,
)
from shared.config import PipelineConfig

DEFAULT_SOURCE_SLOT = 1


class ContentRanker:

    def __init__(self, config: PipelineConfig, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm = llm_factory.get_model(config.ranking_model)

    async def rank(self, items: list[CollectedItem]) -> list[RankedItem]:
        if not items:
            logger.warning("No items to rank")
            return []

        logger.info("Ranking %d items with model '%s'", len(items), self.config.ranking_model.value)

        # Scoring is absolute (the prompt calibrates each item to fixed 0-1 criteria), so
        # large inputs are split into batches scored CONCURRENTLY and merged — a single
        # call over 100+ items dominated the Lambda runtime. Results are independent.
        # Sort by normalized title first so near-duplicate stories co-locate in the same
        # batch, where the prompt's same-topic clustering/dedup can still see both.
        batch_size = self.config.ranking_batch_size
        ordered = sorted(items, key=lambda it: (normalize_title(it.title), it.item_id))
        batches = [ordered[i : i + batch_size] for i in range(0, len(ordered), batch_size)]
        if len(batches) > 1:
            logger.info("Ranking in %d parallel batches of up to %d", len(batches), batch_size)
        results = await asyncio.gather(*(self._rank_batch(b) for b in batches))

        ranked_items: list[RankedItem] = [r for batch in results for r in batch]
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
                r.item.title[: LOGGING_TRUNCATION_CHARS["title"]],
            )
        return selected

    async def _rank_batch(self, items: list[CollectedItem]) -> list[RankedItem]:
        items_text = self._format_items(items)
        chain = RankingPrompt.get_prompt() | self.llm | StrOutputParser()
        try:
            raw_output = await chain.ainvoke(
                {
                    "items_text": items_text,
                    "engagement_guidance": self._engagement_guidance(),
                    "ranking_categories": ", ".join(self.config.ranking_categories),
                    "duplicate_score_penalty": self.config.ranking_duplicate_score_penalty,
                    "scoring_rubric": self.config.ranking_scoring_rubric,
                    "audience": self.config.ranking_audience_description,
                }
            )
        except Exception:
            logger.warning("Ranking batch of %d items failed", len(items), exc_info=True)
            return []
        return self._parse_rankings(raw_output, items)

    def _engagement_guidance(self) -> str:
        tiers = sorted(self.config.engagement_tiers)
        parts = [f"{views:,}+ views → +{bonus}" for views, bonus in tiers]
        return "Items with view counts: " + ", ".join(parts) + "."

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
            origin_key = resolve_origin_key(ranked.item)
            if not origin_key:
                continue
            weight = weights.get(origin_key, default_weight)
            if weight != 1.0:
                original = ranked.score
                ranked.score = max(0.0, min(1.0, ranked.score + (weight - 1.0) * nudge_factor))
                logger.debug(
                    "Applied origin nudge (w=%.2f) to '%s' (origin='%s'): %.2f → %.2f",
                    weight,
                    ranked.item.title[: LOGGING_TRUNCATION_CHARS["title_short"]],
                    origin_key,
                    original,
                    ranked.score,
                )

    def _apply_source_slots(self, above_threshold: list[RankedItem]) -> list[RankedItem]:
        source_slots = self.config.source_slots
        if not source_slots:
            return above_threshold[: self.config.top_n]

        selected: list[RankedItem] = []
        selected_ids: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)
        origin_counts: dict[str, int] = defaultdict(int)
        max_per_origin = self.config.max_per_origin

        def origin_at_cap(item: RankedItem) -> bool:
            origin_key = resolve_origin_key(item.item)
            if not origin_key:
                return False
            return origin_counts[origin_key] >= max_per_origin

        def record(item: RankedItem, source_key: str) -> None:
            selected.append(item)
            selected_ids.add(item.item.item_id)
            source_counts[source_key] += 1
            origin_key = resolve_origin_key(item.item)
            if origin_key:
                origin_counts[origin_key] += 1

        for source_key, slot_count in source_slots.items():
            taken = 0
            for item in above_threshold:
                if taken >= slot_count or len(selected) >= self.config.top_n:
                    break
                if item.item.source_type.value != source_key or item.item.item_id in selected_ids:
                    continue
                if origin_at_cap(item):
                    continue
                record(item, source_key)
                taken += 1

        if len(selected) < self.config.top_n:
            for item in above_threshold:
                if item.item.item_id in selected_ids:
                    continue
                src = item.item.source_type.value
                cap = source_slots.get(src, DEFAULT_SOURCE_SLOT) * self.config.source_cap_multiplier
                if source_counts[src] >= cap or origin_at_cap(item):
                    continue
                record(item, src)
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
                cap = source_slots.get(src, DEFAULT_SOURCE_SLOT) * self.config.source_cap_multiplier
                if source_counts[src] >= cap:
                    continue
                record(item, src)
                if len(selected) >= self.config.top_n:
                    break

        selected.sort(key=lambda r: (-r.score, r.item.item_id))
        return selected[: self.config.top_n]

    def _format_items(self, items: list[CollectedItem]) -> str:
        parts: list[str] = []
        for i, item in enumerate(items):
            engagement = self._format_engagement(item)
            origin = format_origin_label(item)
            fields = [
                ("ID", item.item_id),
                ("Title", item.title),
                ("Source", item.source_type.value),
                ("Author", item.author or "Unknown"),
            ]
            if origin:
                fields.append(("Origin", origin))
            if engagement:
                fields.append(("Engagement", engagement))
            parts.append(
                format_collected_item(item, index=i + 1, max_tokens=self.config.item_text_max_tokens, fields=fields)
            )
        return "\n".join(parts)

    @staticmethod
    def _format_engagement(item: CollectedItem) -> str:
        meta = item.metadata
        if item.source_type == SourceType.YOUTUBE and meta.get("view_count"):
            return f"{meta['view_count']:,} views"
        return ""

    def _parse_rankings(self, raw_output: str, items: list[CollectedItem]) -> list[RankedItem]:
        items_by_id = {item.item_id: item for item in items}

        try:
            data = json.loads(extract_json_from_llm_output(raw_output))
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
