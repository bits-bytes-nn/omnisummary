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
    RankingHealth,
    RankingPrompt,
    SourceType,
    format_collected_item,
    format_origin_label,
    logger,
    normalize_title,
    parse_json_from_llm_output,
    resolve_origin_key,
    retry_async,
)
from shared.config import PipelineConfig

DEFAULT_SOURCE_SLOT = 1


class ContentRanker:

    def __init__(self, config: PipelineConfig, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm_factory = llm_factory
        self.llm = llm_factory.get_model(config.ranking_model)
        # How complete the last rank() was, for the caller to report/alert on. A digest built from a
        # pool that lost a whole batch of candidates must not read as a clean success.
        self.health = RankingHealth()

    def _truncate(self, text: str, max_tokens: int) -> str:
        return self.llm_factory.truncate_to_tokens(text, max_tokens)

    async def rank(
        self, items: list[CollectedItem], select_count: int | None = None, core_count: int | None = None
    ) -> list[RankedItem]:
        """Select the day's candidates.

        `select_count` is how many candidates the digest editor is handed (top_n + buffer);
        `core_count` is how many stories the READER gets (top_n). The source-slot guarantees are
        enforced on the CORE, not on the whole candidate list — with slots applied to top_n+buffer,
        every source's guaranteed slot could be satisfied by an item the editor then never used, so
        the digest itself carried no such guarantee. The buffer is still handed over in full (marked
        `backfill`) so merging same-event items can still be topped up to top_n."""
        if not items:
            logger.warning("No items to rank")
            return []
        limit = select_count or self.config.top_n
        core_limit = min(core_count or limit, limit)

        logger.info("Ranking %d items with model '%s'", len(items), self.config.ranking_model.value)

        # Scoring is absolute (the prompt calibrates each item to fixed 0-1 criteria), so
        # large inputs are split into batches scored CONCURRENTLY and merged — a single
        # call over 100+ items dominated the Lambda runtime. Results are independent.
        # Sort by normalized title first so near-duplicate stories co-locate in the same
        # batch, where the prompt's same-topic clustering/dedup can still see both.
        ordered = sorted(items, key=lambda it: (normalize_title(it.title), it.item_id))
        batches = self._make_batches(ordered)
        if len(batches) > 1:
            logger.info(
                "Ranking in %d parallel batches (<=%d items each)", len(batches), self.config.ranking_batch_size
            )
        # Bound the Bedrock fan-out: created HERE (on the running loop, never at import/__init__).
        semaphore = asyncio.Semaphore(self.config.ranking_max_concurrency)
        results = await asyncio.gather(*(self._rank_batch(b, semaphore) for b in batches), return_exceptions=True)

        # A batch that failed every retry is a permanent failure. One among many is tolerated (the
        # rest of the day's candidates still rank), but if EVERY batch failed there is nothing to
        # rank — raise so the run reports FAILED instead of silently publishing an empty digest.
        # Mirrors gather_collector_results(raise_if_all_failed=True).
        failures = [r for r in results if isinstance(r, BaseException)]
        lost = sum(len(b) for b, r in zip(batches, results, strict=True) if isinstance(r, BaseException))
        if failures:
            # ERROR, not warning: those candidates are GONE from the day's pool, and the digest that
            # follows looks perfectly normal — so this line is the only trace that the pool was
            # short. self.health carries the same verdict to the caller for alerting.
            logger.error(
                "%d of %d ranking batches failed permanently; %d candidate(s) never reached the digest: %s",
                len(failures),
                len(batches),
                lost,
                failures[0],
            )
        if failures and len(failures) == len(results):
            raise RuntimeError(f"All {len(failures)} ranking batches failed: {failures[0]}")

        ranked_items: list[RankedItem] = [r for batch in results if not isinstance(batch, BaseException) for r in batch]
        self.health = RankingHealth(
            batches_total=len(batches),
            batches_failed=len(failures),
            items_total=len(items),
            items_scored=len(ranked_items),
            items_lost=lost,
        )
        self._apply_origin_weights(ranked_items)

        # Pinned items (user-specified via --pin-url) are guaranteed a slot regardless of score
        # or diversity caps — they're kept aside and prepended after slotting fills the rest.
        pinned = [r for r in ranked_items if r.item.metadata.get("pinned")]
        # Reconcile against the pinned INPUTS: if the ranking LLM dropped a pinned item (omitted
        # its id, hallucinated it away, or its whole batch threw and returned []), it never became
        # a RankedItem and the user's explicit pin would be silently lost. Synthesize a RankedItem
        # at min_score for any such pin so the force-inclusion guarantee actually holds.
        scored_ids = {r.item.item_id for r in ranked_items}
        missing_pins = [it for it in items if it.metadata.get("pinned") and it.item_id not in scored_ids]
        if missing_pins:
            logger.warning(
                "Ranking did not score %d pinned item(s); force-including at min_score: %s",
                len(missing_pins),
                [it.url for it in missing_pins],
            )
            pinned.extend(
                RankedItem(item=it, score=self.config.min_score, reasoning="Pinned (not scored by ranker)")
                for it in missing_pins
            )
        pinned.sort(key=lambda r: (-r.score, r.item.item_id))
        pinned_ids = {r.item.item_id for r in pinned}

        above_threshold = [
            r for r in ranked_items if r.score >= self.config.min_score and r.item.item_id not in pinned_ids
        ]
        grace = self._grace_candidates(ranked_items, above_threshold, pinned)
        above_threshold.extend(grace)
        above_threshold.sort(key=lambda r: (-r.score, r.item.item_id))
        # Grace items are below min_score; they may ONLY fill their own source's guaranteed slot,
        # never the relaxed fallback fill (which would pad a quiet day with several weak items).
        grace_ids = {r.item.item_id for r in grace}

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

        # Reserve slots for the pinned items so the source-slotting fills only the remainder,
        # then prepend the pinned items so they always lead and never get crowded out.
        remaining = max(0, core_limit - len(pinned))
        filled = self._apply_source_slots(above_threshold, remaining, grace_ids, pinned)
        core = pinned + filled
        extras = self._backfill_candidates(above_threshold, core, grace_ids, limit - len(core))
        selected = core + extras

        if pinned:
            logger.info("Force-included %d pinned item(s): %s", len(pinned), [r.item.url for r in pinned])
        logger.info(
            "Ranked %d items → %d above min_score %.2f → %d core (with source slots) + %d backfill",
            len(items),
            len(above_threshold),
            self.config.min_score,
            len(core),
            len(extras),
        )
        for r in selected:
            logger.info(
                "  Selected%s: [%s] %.2f - '%s'",
                " (backfill)" if r.backfill else "",
                r.item.source_type.value,
                r.score,
                r.item.title[: LOGGING_TRUNCATION_CHARS["title"]],
            )
        return selected

    @staticmethod
    def _backfill_candidates(
        above_threshold: list[RankedItem], core: list[RankedItem], grace_ids: set[str], room: int
    ) -> list[RankedItem]:
        """The extra candidates handed to the editor beyond the core, in score order, flagged as
        backfill. They exist so a merge of two same-event items can still be topped up to top_n
        distinct stories; the diversity guarantees belong to the core, so these deliberately ignore
        the slot/origin caps. Grace items (below min_score) are never backfill — they may only earn
        their own source's guaranteed slot."""
        if room <= 0:
            return []
        chosen_ids = {r.item.item_id for r in core}
        extras: list[RankedItem] = []
        for item in above_threshold:
            if len(extras) >= room:
                break
            if item.item.item_id in chosen_ids or item.item.item_id in grace_ids:
                continue
            item.backfill = True
            extras.append(item)
        return extras

    async def _rank_batch(self, items: list[CollectedItem], semaphore: asyncio.Semaphore) -> list[RankedItem]:
        """Score one batch and reconcile its COVERAGE: an LLM that quietly omits item ids returns a
        perfectly valid response, so those candidates used to vanish from the day's pool without a
        trace. The shortfall is always logged, and when coverage falls below
        ranking_min_coverage_ratio the omitted items get ONE extra re-ask (never more). A failed or
        still-short re-ask leaves the original outcome untouched — it can never fail the batch."""
        ranked = await self._score_batch(items, semaphore)
        scored_ids = {r.item.item_id for r in ranked}
        missing = [it for it in items if it.item_id not in scored_ids]
        if not missing:
            return ranked

        logger.warning(
            "Ranking batch scored %d/%d items; %d omitted by the model: %s",
            len(ranked),
            len(items),
            len(missing),
            [it.item_id for it in missing],
        )
        coverage = len(ranked) / len(items)
        if coverage >= self.config.ranking_min_coverage_ratio:
            return ranked

        logger.info(
            "Re-asking the ranker once for %d omitted item(s) (coverage %.2f < %.2f)",
            len(missing),
            coverage,
            self.config.ranking_min_coverage_ratio,
        )
        try:
            recovered = await self._score_batch(missing, semaphore)
        except Exception:
            # The re-ask is a best-effort top-up: its failure must not turn a partially-scored
            # batch into a failed one (rank() would then count it toward the all-batches-failed
            # outage check), so keep exactly what the first pass produced.
            logger.warning("Ranking coverage re-ask failed; keeping the partially scored batch", exc_info=True)
            return ranked

        extra = [r for r in recovered if r.item.item_id not in scored_ids]
        logger.info("Coverage re-ask recovered %d of %d omitted item(s)", len(extra), len(missing))
        return ranked + extra

    async def _score_batch(self, items: list[CollectedItem], semaphore: asyncio.Semaphore) -> list[RankedItem]:
        """Score one batch, retrying the Converse call before giving up. The failure used to be
        swallowed into [] — a single throttle or transient 5xx silently deleted a whole batch of
        candidates from the day's pool. Now it retries, and a permanent failure PROPAGATES so
        rank() can decide (one bad batch is tolerated; all of them is an outage)."""
        items_text = self._format_items(items)
        chain = RankingPrompt.get_prompt() | self.llm | StrOutputParser()

        async def _invoke() -> str:
            return await chain.ainvoke(
                {
                    "items_text": items_text,
                    "engagement_guidance": self._engagement_guidance(),
                    "ranking_categories": ", ".join(self.config.ranking_categories),
                    "duplicate_score_penalty": self.config.ranking_duplicate_score_penalty,
                    "scoring_rubric": self.config.ranking_scoring_rubric,
                    "audience": self.config.ranking_audience_description,
                }
            )

        async with semaphore:
            raw_output = await retry_async(
                _invoke,
                max_retries=self.config.ranking_max_retries,
                backoff_sec=self.config.ranking_retry_backoff_sec,
                description=f"Ranking batch of {len(items)} items",
            )
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

    def _grace_candidates(
        self, ranked_items: list[RankedItem], above_threshold: list[RankedItem], pinned: list[RankedItem]
    ) -> list[RankedItem]:
        """Per-source safety net: for each source that has a guaranteed slot but landed NOTHING
        above min_score, admit its single best item if it's within source_slot_score_grace of the
        threshold. The absolute-scoring prompt systematically under-rates conversational sources
        (video/podcast transcripts vs tight articles); this keeps a strong-but-0.55 item eligible
        without lowering the global bar for everyone. Generalizes to any under-scored source.

        Pinned items are already guaranteed a slot, so a source they cover is NOT empty — treat it
        as covered and exclude pinned ids from the candidate pool. Otherwise a pinned item that is
        its source's only above-threshold entry (it's stripped from above_threshold) makes the
        source look shut out, and grace would re-admit the pin itself (double emission) or pad the
        source with a below-threshold filler it should not get."""
        grace = self.config.source_slot_score_grace
        if not grace or not self.config.source_slots:
            return []
        floor = self.config.min_score - grace
        pinned_ids = {r.item.item_id for r in pinned}
        have_above = {r.item.source_type.value for r in above_threshold}
        have_above |= {r.item.source_type.value for r in pinned}
        extra: list[RankedItem] = []
        for src, slot in self.config.source_slots.items():
            if slot < 1 or src in have_above:
                continue
            candidates = [
                r
                for r in ranked_items
                if r.item.source_type.value == src and floor <= r.score and r.item.item_id not in pinned_ids
            ]
            if candidates:
                best = max(candidates, key=lambda r: r.score)
                extra.append(best)
                logger.info(
                    "Source '%s' had nothing above %.2f; admitting best item at %.2f (grace floor %.2f)",
                    src,
                    self.config.min_score,
                    best.score,
                    floor,
                )
        return extra

    def _apply_source_slots(
        self,
        above_threshold: list[RankedItem],
        limit: int,
        grace_ids: set[str] | None = None,
        pinned: list[RankedItem] | None = None,
    ) -> list[RankedItem]:
        grace_ids = grace_ids or set()
        source_slots = self.config.source_slots
        if not source_slots:
            return above_threshold[:limit]

        selected: list[RankedItem] = []
        selected_ids: set[str] = set()
        source_counts: dict[str, int] = defaultdict(int)
        origin_counts: dict[str, int] = defaultdict(int)
        max_per_origin = self.config.max_per_origin

        # Pinned items are prepended to the result by rank() and never enter this fill, so their
        # source/origin used to go uncounted — a pin plus a slotted item from the SAME origin both
        # landed, breaking max_per_origin exactly where the user forced an item in. Seed the
        # counters with them (without selecting them); the relaxed final pass still fills to the
        # limit when the caps would otherwise leave the digest short.
        for item in pinned or []:
            source_counts[item.item.source_type.value] += 1
            pinned_origin = resolve_origin_key(item.item)
            if pinned_origin:
                origin_counts[pinned_origin] += 1

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

        for source_key, slot_count in self._slot_order(above_threshold, source_slots):
            taken = 0
            for item in above_threshold:
                if taken >= slot_count or len(selected) >= limit:
                    break
                if item.item.source_type.value != source_key or item.item.item_id in selected_ids:
                    continue
                if origin_at_cap(item):
                    continue
                record(item, source_key)
                taken += 1

        def fill(*, respect_origin: bool, respect_source: bool) -> int:
            """Walk the remaining candidates in score order and take what the given caps allow.
            One loop for every fill pass — the three passes differ ONLY in which caps they honour.
            Grace items (below min_score) are never filler: they may earn their own source's
            guaranteed slot and nothing more. Returns how many items this pass added."""
            added = 0
            for item in above_threshold:
                if len(selected) >= limit:
                    break
                if item.item.item_id in selected_ids or item.item.item_id in grace_ids:
                    continue
                src = item.item.source_type.value
                cap = source_slots.get(src, DEFAULT_SOURCE_SLOT) * self.config.source_cap_multiplier
                if respect_source and source_counts[src] >= cap:
                    continue
                if respect_origin and origin_at_cap(item):
                    continue
                record(item, src)
                added += 1
            return added

        if len(selected) < limit:
            fill(respect_origin=True, respect_source=True)

        # If diversity caps left the digest below the limit while valid candidates remain, relax
        # the per-origin cap (keep the source cap) so a quiet day with few distinct origins still
        # fills the digest.
        if len(selected) < limit:
            fill(respect_origin=False, respect_source=True)

        # Last resort: relax the SOURCE cap too. A collector outage (rsshub/reddit empty) leaves
        # every remaining candidate on one source, so the source cap alone could hold the digest
        # short — fewer stories to read, for a diversity that has no candidates to spend on.
        # Ordered last and entered only while short, so it can never change a day that filled
        # normally; candidates that still satisfy max_per_origin go first.
        if len(selected) < limit:
            relaxed = fill(respect_origin=True, respect_source=False)
            relaxed += fill(respect_origin=False, respect_source=False)
            if relaxed:
                logger.info(
                    "Source caps relaxed to fill the digest: %d extra item(s) taken (%d/%d selected) — "
                    "likely a partial collector outage",
                    relaxed,
                    len(selected),
                    limit,
                )

        selected.sort(key=lambda r: (-r.score, r.item.item_id))
        return selected[:limit]

    @staticmethod
    def _slot_order(above_threshold: list[RankedItem], source_slots: dict[str, int]) -> list[tuple[str, int]]:
        """Order the guaranteed-slot pass by each source's BEST candidate score (descending), with
        the source key as a deterministic tie-break — not by config key order, which is arbitrary.

        It only changes anything when the limit cannot cover every slot (a short digest via
        --select-count, or slots reserved by pinned items): the last-listed sources then went home
        empty regardless of how strong their candidates were, purely because of where they sat in
        the YAML. With limit >= sum(source_slots) — the live config — every source still fills its
        own slots and the selection is identical."""
        best: dict[str, float] = {}
        for r in above_threshold:
            src = r.item.source_type.value
            if src in source_slots:
                best[src] = max(best.get(src, 0.0), r.score)
        return sorted(source_slots.items(), key=lambda kv: (-best.get(kv[0], 0.0), kv[0]))

    def _make_batches(self, ordered: list[CollectedItem]) -> list[list[CollectedItem]]:
        """Split ranking input into batches capped by BOTH item count (ranking_batch_size) and a
        cumulative input-token budget. A fixed count alone can blow the model's context window:
        ranking_batch_size(40) × item_text_max_tokens(10k) = 400k > Opus 200k, and a batch that
        overflows fails the Converse call → _rank_batch drops the WHOLE batch silently. Bound the
        batch by ~70% of the context window (leaving room for the system prompt + JSON output).
        Each item's truncated-text count is cached (truncate runs anyway), so this adds no API cost."""
        model_info = self.llm_factory.get_model_info(self.config.ranking_model)
        window = model_info.context_window_size if model_info else 200_000
        token_budget = int(window * 0.7)
        count_cap = self.config.ranking_batch_size
        batches: list[list[CollectedItem]] = []
        current: list[CollectedItem] = []
        current_tokens = 0
        for item in ordered:
            truncated = self._truncate(item.text or "", self.config.item_text_max_tokens)
            item_tokens = self.llm_factory.count_tokens(truncated) if truncated else 0
            # Start a new batch when adding this item would exceed either cap (but never emit an
            # empty batch — a single item over budget still goes in its own batch).
            if current and (len(current) >= count_cap or current_tokens + item_tokens > token_budget):
                batches.append(current)
                current, current_tokens = [], 0
            current.append(item)
            current_tokens += item_tokens
        if current:
            batches.append(current)
        return batches

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
                format_collected_item(
                    item,
                    index=i + 1,
                    max_tokens=self.config.item_text_max_tokens,
                    fields=fields,
                    truncate=self._truncate,
                )
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
            data = parse_json_from_llm_output(raw_output)
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
