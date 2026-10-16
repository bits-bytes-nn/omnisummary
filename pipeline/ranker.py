from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from langchain_core.output_parsers import StrOutputParser

from shared import (
    BedrockLanguageModelFactory,
    CollectedItem,
    RankedItem,
    RankingHealth,
    RankingPrompt,
    SourceType,
    format_collected_item,
    format_origin_label,
    logger,
    parse_json_from_llm_output,
    resolve_origin_key,
    retry_async,
)
from shared.config import PipelineConfig

from .aggregator import normalize_url
from .digest_generator import format_recent_titles


class ContentRanker:

    def __init__(self, config: PipelineConfig, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm_factory = llm_factory
        self.llm = llm_factory.get_model(config.ranking_model, stage="ranking")
        # How complete the last rank() was, for the caller to report/alert on. A digest built from a
        # pool that lost a whole batch of candidates must not read as a clean success.
        self.health = RankingHealth()

    def _truncate(self, text: str, max_tokens: int) -> str:
        return self.llm_factory.truncate_to_tokens(text, max_tokens)

    async def rank(
        self, items: list[CollectedItem], select_count: int | None = None, recent_titles: Sequence[str] = ()
    ) -> list[RankedItem]:
        """Select the candidate pool the digest editor picks from (`select_count`, top_n + buffer).
        `recent_titles` are the stories recent digests ran, so Novelty can be judged against them.

        The source slots and max_per_origin shape the POOL, so every source's best item is
        considered; the editor then picks the strongest top_n distinct stories from it. Quality wins
        over a guaranteed seat: enforcing the slots on what ships put a 0.62 link-only post ahead of a
        0.70 report, and the editor overrode the guarantee on 14 of 30 runs anyway."""
        if not items:
            logger.warning("No items to rank")
            return []
        limit = select_count or self.config.top_n

        logger.info("Ranking %d items with model '%s'", len(items), self.config.ranking_model.value)

        # Scoring is absolute (the prompt calibrates each item to fixed 0-1 criteria), so
        # large inputs are split into batches scored CONCURRENTLY and merged — a single
        # call over 100+ items dominated the Lambda runtime. Results are independent.
        # Ordered by a HASH of item_id so batch membership is deterministic and every batch mixes
        # sources and scripts. item_id itself is the feed GUID (an article URL for RSS/X, a post or
        # video id elsewhere), and sorting on it put 33 GeekNews items in one batch and all of X in
        # another.
        ordered = sorted(items, key=lambda it: hashlib.sha256(it.item_id.encode()).hexdigest())
        batches = self._make_batches(ordered)
        if len(batches) > 1:
            logger.info(
                "Ranking in %d parallel batches (<=%d items each)", len(batches), self.config.ranking_batch_size
            )
        # Bound the Bedrock fan-out: created HERE (on the running loop, never at import/__init__).
        semaphore = asyncio.Semaphore(self.config.ranking_max_concurrency)
        recent = format_recent_titles(list(recent_titles))
        results = await asyncio.gather(
            *(self._rank_batch(b, semaphore, recent) for b in batches), return_exceptions=True
        )

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

        # Reserve slots for the pinned items so the source-slotting fills only the remainder,
        # then prepend the pinned items so they always lead and never get crowded out.
        remaining = max(0, limit - len(pinned))
        selected = pinned + self._apply_source_slots(above_threshold, remaining, pinned)

        if pinned:
            logger.info("Force-included %d pinned item(s): %s", len(pinned), [r.item.url for r in pinned])
        logger.info(
            "Ranked %d items → %d above min_score %.2f → %d candidates (with source slots)",
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

    async def _rank_batch(
        self, items: list[CollectedItem], semaphore: asyncio.Semaphore, recent_titles: str
    ) -> list[RankedItem]:
        """Score one batch and reconcile its COVERAGE: an LLM that quietly omits item ids returns a
        perfectly valid response, so those candidates used to vanish from the day's pool without a
        trace. Any omitted items get ONE extra re-ask covering exactly them (never more). A failed or
        still-short re-ask leaves the original outcome untouched — it can never fail the batch."""
        ranked = await self._score_batch(items, semaphore, recent_titles)
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
        logger.info("Re-asking the ranker once for %d omitted item(s)", len(missing))
        recovered: list[RankedItem] = []
        try:
            recovered = await self._score_batch(missing, semaphore, recent_titles)
        except Exception:
            # The re-ask is a best-effort top-up: its failure must not turn a partially-scored
            # batch into a failed one (rank() would then count it toward the all-batches-failed
            # outage check), so keep exactly what the first pass produced.
            logger.warning("Ranking coverage re-ask failed; keeping the partially scored batch", exc_info=True)

        extra = [r for r in recovered if r.item.item_id not in scored_ids]
        logger.info("Coverage re-ask recovered %d of %d omitted item(s)", len(extra), len(missing))
        scored = ranked + extra
        if not scored:
            # NOTHING parsed, twice over (a malformed JSON response both times): returning [] left
            # `failures` empty and items_lost at 0, so RankingHealth read clean and every ranking
            # alert stayed silent while a whole batch of candidates vanished from the day's pool.
            # Raising makes rank() count the batch as failed — one bad batch is still tolerated.
            raise RuntimeError(f"Ranking batch of {len(items)} items produced no parsed rankings")
        return scored

    async def _score_batch(
        self, items: list[CollectedItem], semaphore: asyncio.Semaphore, recent_titles: str
    ) -> list[RankedItem]:
        """Score one batch, retrying the Converse call before giving up. The failure used to be
        swallowed into [] — a single throttle or transient 5xx silently deleted a whole batch of
        candidates from the day's pool. Now it retries, and a permanent failure PROPAGATES so
        rank() can decide (one bad batch is tolerated; all of them is an outage)."""
        # Off the event loop: truncation makes blocking CountTokens calls, which serialized the batches.
        items_text = await asyncio.to_thread(self._format_items, items)
        chain = RankingPrompt.get_prompt() | self.llm | StrOutputParser()

        async def _invoke() -> str:
            return await chain.ainvoke(
                {
                    "items_text": items_text,
                    "scoring_rubric": self.config.ranking_scoring_rubric,
                    "audience": self.config.ranking_audience_description,
                    "recent_titles": recent_titles,
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

    def _apply_source_slots(
        self,
        above_threshold: list[RankedItem],
        limit: int,
        pinned: list[RankedItem] | None = None,
    ) -> list[RankedItem]:
        source_slots = self.config.source_slots
        # NOTE: an empty source_slots skips only the GUARANTEED-SLOT pass below (the loop simply has
        # nothing to iterate) — the origin-cap fill passes still run. Returning above_threshold[:limit]
        # here instead dropped max_per_origin along with the slots, so one feed could take every
        # story: the exact failure the origin cap exists to prevent.
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

        for source_key, slot_count in source_slots.items():
            # Start from what the pins already consumed of THIS source's slots. Starting at 0
            # ignored the seed above, so a pinned `web` item plus web's guaranteed slot produced two
            # web stories and pushed the lowest-scoring source out of the core entirely — breaking
            # the diversity guarantee exactly on the days the operator intervened.
            taken = source_counts[source_key]
            for item in above_threshold:
                if taken >= slot_count or len(selected) >= limit:
                    break
                if item.item.source_type.value != source_key or item.item.item_id in selected_ids:
                    continue
                if origin_at_cap(item):
                    continue
                record(item, source_key)
                taken += 1

        def fill(*, respect_origin: bool) -> None:
            """Walk the remaining candidates in score order and take what the origin cap allows."""
            for item in above_threshold:
                if len(selected) >= limit:
                    break
                if item.item.item_id in selected_ids or (respect_origin and origin_at_cap(item)):
                    continue
                record(item, item.item.source_type.value)

        # The rest of the pool goes by score; max_per_origin keeps one publisher from taking it, and
        # is relaxed only when a quiet day has too few distinct origins to fill the pool.
        fill(respect_origin=True)
        fill(respect_origin=False)

        selected.sort(key=lambda r: (-r.score, r.item.item_id))
        return selected[:limit]

    def _make_batches(self, ordered: list[CollectedItem]) -> list[list[CollectedItem]]:
        """Split ranking input into batches of at most ranking_batch_size, spread EVENLY
        (ceil(n / ceil(n / size))): filling each batch to the cap left 1-item tail batches, scored with
        nothing beside them to calibrate against. A batch cannot overflow the window: ranking_batch_size
        x item_text_max_tokens is far below it."""
        if not ordered:
            return []
        size = math.ceil(len(ordered) / math.ceil(len(ordered) / self.config.ranking_batch_size))
        return [ordered[i : i + size] for i in range(0, len(ordered), size)]

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

    @staticmethod
    def _resolve_item_id(raw_id: str, items: list[CollectedItem]) -> tuple[str, str]:
        """(item_id, how) for an id the model answered with, or ("", "") when nothing in the batch
        matches it.

        The `=== Item N ===` header the prompt builder emits invites the model to answer with the
        DISPLAY ORDINAL, and production logs show it doing exactly that ('30', '27', '18', '9', '5'),
        alongside a truncated 14-character id and full URLs. All of those were discarded with a
        warning, so those candidates left the day's pool silently.

        Exact match first, so a genuine id can never be reinterpreted as an ordinal. A prefix is
        accepted only when it is UNAMBIGUOUS across the batch."""
        by_id = {item.item_id: item for item in items}
        if raw_id in by_id:
            return raw_id, "exact"
        if raw_id.isdigit():
            ordinal = int(raw_id)
            if 1 <= ordinal <= len(items):
                return items[ordinal - 1].item_id, "ordinal"
        wanted_url = normalize_url(raw_id)
        by_url = {normalize_url(item.url): item.item_id for item in items if item.url}
        if wanted_url in by_url:
            return by_url[wanted_url], "url"
        prefixed = [item.item_id for item in items if raw_id and item.item_id.startswith(raw_id)]
        if len(prefixed) == 1:
            return prefixed[0], "prefix"
        return "", ""

    def _parse_rankings(self, raw_output: str, items: list[CollectedItem]) -> list[RankedItem]:
        items_by_id = {item.item_id: item for item in items}

        try:
            data = parse_json_from_llm_output(raw_output)
            rankings = data.get("rankings", []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse ranking LLM output: '%s'", exc)
            logger.debug("Raw LLM output:\n%s", raw_output[:500])
            return []

        ranked_items: list[RankedItem] = []
        # One RankedItem per INPUT id, at most. A model that repeats an id (and, in practice, omits
        # another to compensate) otherwise made len(ranked) == len(items): coverage read 1.0, the
        # re-ask never fired, items_scored overstated the pool, and the editor was handed the same
        # story twice. First entry wins — deterministic, and it is the one the model committed to.
        seen_ids: set[str] = set()
        duplicate_ids: list[str] = []
        recovered: Counter[str] = Counter()
        for entry in rankings:
            try:
                raw_id = str(entry["item_id"])
                item_id, how = self._resolve_item_id(raw_id, items)
                if not item_id:
                    logger.warning("Unknown item_id in ranking response: '%s'", raw_id)
                    continue
                if how != "exact":
                    recovered[how] += 1
                if item_id in seen_ids:
                    duplicate_ids.append(item_id)
                    continue
                ranked_items.append(
                    RankedItem(
                        item=items_by_id[item_id],
                        score=float(entry["score"]),
                        reasoning=entry.get("reasoning", ""),
                    )
                )
                seen_ids.add(item_id)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed ranking entry: %s (%s)", entry, exc)

        if duplicate_ids:
            logger.warning(
                "Ranking response repeated %d item_id(s); kept the first entry for each: %s",
                len(duplicate_ids),
                duplicate_ids,
            )
        if recovered:
            # WHICH path recovered them: a persistent 'ordinal' count says the model is answering with
            # the display number, which is a prompt-shape finding, not a transient slip.
            logger.info(
                "Ranking response used non-exact item ids; resolved against the batch by %s",
                ", ".join(f"{how}={count}" for how, count in sorted(recovered.items())),
            )
        return ranked_items
