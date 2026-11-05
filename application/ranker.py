from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence

from langchain_core.output_parsers import StrOutputParser

from application.digest_generator import format_recent_titles
from application.ports import LanguageModels, PaidPageCheck
from application.prompts import RankingPrompt, StoryGroupPrompt
from domain.formatting import format_collected_item, format_origin_label, resolve_origin_key
from domain.models import CollectedItem, RankedItem, RankingHealth, StoryGroups
from shared import SourceType, logger, parse_json_from_llm_output, retry_async
from shared.config import PipelineConfig

# The grouping call sees the top of the above-threshold list in score order; a candidate below it could
# not have taken a pool seat ahead of its higher-scored copy anyway.
_MAX_GROUPED = 120
# Feed posts checked for a paid page: the top of the above-threshold list, a few pools deep, so a paid
# post is out before it takes its feed's or its story's seat.
_PAID_CHECK_POOLS = 3


def _tiebreak(item_id: str) -> str:
    """Equal scores ordered by a hash of the id: by the raw id, Reddit's ids sorted ahead of every URL-
    and hash-keyed source, so Reddit won every tie at the pool's cutoff."""
    return hashlib.sha256(item_id.encode()).hexdigest()


class ContentRanker:

    def __init__(self, config: PipelineConfig, models: LanguageModels, paid_check: PaidPageCheck | None = None) -> None:
        self.config = config
        self.models = models
        self.paid_check = paid_check
        self.llm = models.get_model(config.ranking_model, stage="ranking")
        # How complete the last rank() was, for the caller to report/alert on. A digest built from a
        # pool that lost a whole batch of candidates must not read as a clean success.
        self.health = RankingHealth()
        # Every item the last rank() scored, kept so a run can store them: the pool alone cannot replay a
        # change to the slots, the caps or the threshold.
        self.scored: list[RankedItem] = []
        self._split_batches_lost = 0

    def _truncate(self, text: str, max_tokens: int) -> str:
        return self.models.truncate_to_tokens(text, max_tokens)

    def _ranking_truncate(self, text: str, max_tokens: int) -> str:
        """The ranker scores on the opening of each item. With ranking_text_max_chars set that cut is by
        characters, with no CountTokens calls: measuring 300 of 700 weekday investing items cost about
        1,600 calls and held the first ranking response back 3.5 minutes inside a 15-minute Lambda."""
        chars = self.config.ranking_text_max_chars
        if chars is None:
            return self._truncate(text, max_tokens)
        if len(text) <= chars:
            return text
        space = text.rfind(" ", 0, chars)
        return text[: space if space > chars // 2 else chars].rstrip()

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
        self._split_batches_lost = 0

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
        self.scored = ranked_items
        self.health = RankingHealth(
            batches_total=len(batches),
            batches_failed=len(failures),
            batches_partly_lost=self._split_batches_lost,
            items_total=len(items),
            items_scored=len(ranked_items),
            # Every candidate left unscored, a failed batch's and a split batch's bad half alike.
            items_lost=max(lost, len(items) - len(ranked_items)),
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
        pinned.sort(key=lambda r: (-r.score, _tiebreak(r.item.item_id)))
        pinned_ids = {r.item.item_id for r in pinned}

        above_threshold = [
            r for r in ranked_items if r.score >= self.config.min_score and r.item.item_id not in pinned_ids
        ]
        above_threshold.sort(key=lambda r: (-r.score, _tiebreak(r.item.item_id)))
        if self.paid_check is not None:
            # A newsletter feed carries a paid post's free preview, so the post scores like any other and
            # then links readers to a paywall (SemiAnalysis on 11-04, Interconnects on 11-05). Checked here,
            # before the pool is built, so the post's feed and story seats go to a free copy; and only
            # near the top, since checking every collected feed page puts hundreds of requests on a few
            # hosts. Web results are checked at collection.
            head = above_threshold[: _PAID_CHECK_POOLS * limit]
            paid = await self.paid_check([r.item.url for r in head if r.item.source_type == SourceType.RSS])
            above_threshold = [r for r in above_threshold if r.item.url not in paid]

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
        stories = await self._story_groups(above_threshold) if len(above_threshold) > remaining else {}
        selected = pinned + self._apply_source_slots(above_threshold, remaining, pinned, stories)

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

    async def _story_groups(self, candidates: list[RankedItem]) -> dict[str, int]:
        """item_id -> story number for the candidates that report the same story as another one.

        URL and title dedup miss the same story from several outlets, so one story took several of the
        12 pool seats: 27 of 216 AI seats over 18 runs, and Korea's October exports took 6 seats from 6
        hosts on 2026-11-02 while 52 items cleared the bar. Asked once over every candidate together, not
        per ranking batch, whose boundaries made a same-event penalty inconsistent. Best-effort: on any
        failure the pool is filled as before."""
        shown = candidates[:_MAX_GROUPED]
        items_text = "\n".join(
            f"{n}. {r.item.title.strip()[:160]} ({format_origin_label(r.item)})" for n, r in enumerate(shown, start=1)
        )
        try:
            # json_schema is Bedrock's constrained output: free-form JSON came back malformed on two of
            # three weekday investing runs (trailing notes, a missing comma) and the pool lost the cap.
            llm = self.models.get_model(self.config.digest_model, stage="story-groups").with_structured_output(
                StoryGroups, method="json_schema"
            )
            verdict = await (StoryGroupPrompt.get_prompt() | llm).ainvoke({"items_text": items_text})
            groups = StoryGroups.model_validate(verdict).same_story if verdict is not None else []
        except Exception as e:
            logger.warning("Same-story grouping failed (%s); filling the pool without it", e)
            return {}
        stories: dict[str, int] = {}
        for story, group in enumerate(groups if isinstance(groups, list) else []):
            numbers = (
                {n for n in group if type(n) is int and 1 <= n <= len(shown)} if isinstance(group, list) else set()
            )
            if len(numbers) > 1:
                for n in numbers:
                    stories.setdefault(shown[n - 1].item.item_id, story)
        if stories:
            logger.info("Same-story groups: %d candidate(s) in %d stories", len(stories), len(set(stories.values())))
        return stories

    async def _rank_batch(
        self, items: list[CollectedItem], semaphore: asyncio.Semaphore, recent_titles: str
    ) -> list[RankedItem]:
        """Score one batch and reconcile its COVERAGE: an LLM that quietly omits items returns a
        perfectly valid response, so those candidates used to vanish from the day's pool without a
        trace. Any omitted items get ONE extra re-ask covering exactly them (never more). A failed or
        still-short re-ask leaves the original outcome untouched — it can never fail the batch."""
        ranked = await self._score_batch(items, semaphore, recent_titles)
        if not ranked and len(items) > 1:
            # Nothing parsed (2 of 5 AI runs lost a whole batch this way, both attempts empty or prose).
            # Re-asking the identical batch repeats whatever in it caused that, so score each half once
            # instead: a bad item then costs its half, not 40 neighbours. Once, not recursively: a
            # failure that hits every batch (a model answering in prose) would otherwise cost 3n calls
            # and outrun the Lambda instead of failing fast.
            half = len(items) // 2
            logger.warning("Ranking batch of %d items parsed to nothing; scoring its halves separately", len(items))
            outcomes = await asyncio.gather(
                *(self._score_batch(part, semaphore, recent_titles) for part in (items[:half], items[half:])),
                return_exceptions=True,
            )
            scored = [r for outcome in outcomes if not isinstance(outcome, BaseException) for r in outcome]
            if not scored:
                raise RuntimeError(f"Ranking batch of {len(items)} items produced no parsed rankings")
            if any(isinstance(o, BaseException) or not o for o in outcomes):
                self._split_batches_lost += 1
            return scored
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

        # The re-ask only scored the missing items, so nothing it returns is already in `ranked`.
        logger.info("Coverage re-ask recovered %d of %d omitted item(s)", len(recovered), len(missing))
        scored = ranked + recovered
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
                    "criteria": self.config.ranking_criteria,
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
        stories: dict[str, int] | None = None,
    ) -> list[RankedItem]:
        source_slots = self.config.source_slots
        stories = stories or {}
        stories_taken: set[int] = set()
        # A story competes with its best-scored copy: in the guaranteed-slot pass an X post at 0.65 from
        # an earlier source would otherwise take the seat its 0.85 article copy needed.
        best_copy: dict[int, str] = {}
        for candidate in above_threshold:
            story = stories.get(candidate.item.item_id)
            if story is not None:
                best_copy.setdefault(story, candidate.item.item_id)
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

        def story_taken(item: RankedItem) -> bool:
            story = stories.get(item.item.item_id)
            return story is not None and story in stories_taken

        def record(item: RankedItem, source_key: str) -> None:
            selected.append(item)
            selected_ids.add(item.item.item_id)
            if item.item.item_id in stories:
                stories_taken.add(stories[item.item.item_id])
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
                story = stories.get(item.item.item_id)
                if (
                    origin_at_cap(item)
                    or story_taken(item)
                    or (story is not None and best_copy[story] != item.item.item_id)
                ):
                    continue
                record(item, source_key)
                taken += 1

        def fill(*, respect_origin: bool, respect_story: bool = True, until: int = limit) -> None:
            """Walk the remaining candidates in score order and take what the caps allow, up to `until`."""
            for item in above_threshold:
                if len(selected) >= until:
                    break
                if item.item.item_id in selected_ids or (respect_origin and origin_at_cap(item)):
                    continue
                if respect_story and story_taken(item):
                    continue
                record(item, item.item.source_type.value)

        # The rest of the pool goes by score; max_per_origin keeps one publisher from taking it, and one
        # story keeps one seat. On a quiet day the caps are relaxed only far enough for the editor to have
        # top_n stories, never to fill the buffer: filling it let the highest-volume origins (r/LocalLLaMA,
        # one X curator, GeekNews) take 27 extra seats on 7 of 15 AI days, and 4 of 11-01's 5 stories came
        # from two origins. The origin cap is relaxed first, the story cap last.
        enough = min(limit, self.config.top_n)
        fill(respect_origin=True)
        fill(respect_origin=False, until=enough)
        fill(respect_origin=False, respect_story=False, until=enough)

        selected.sort(key=lambda r: (-r.score, _tiebreak(r.item.item_id)))
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
                    truncate=self._ranking_truncate,
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
        """Each entry names its item by the `=== Item N ===` number the batch was shown with."""
        try:
            data = parse_json_from_llm_output(raw_output)
            rankings = data.get("rankings", []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, KeyError) as exc:
            logger.error("Failed to parse ranking LLM output: '%s'", exc)
            # WARNING, not DEBUG: an unparseable batch is the one case whose cause has to be read later.
            logger.warning("Unparseable ranking output (%d chars): %r", len(raw_output), raw_output[:300])
            return []

        ranked_items: list[RankedItem] = []
        # One RankedItem per INPUT item, at most. A model that repeats an item (and, in practice, omits
        # another to compensate) otherwise made len(ranked) == len(items): coverage read 1.0, the
        # re-ask never fired, items_scored overstated the pool, and the editor was handed the same
        # story twice. First entry wins — deterministic, and it is the one the model committed to.
        seen: set[int] = set()
        duplicates: list[int] = []
        for entry in rankings:
            try:
                number = entry["item"]
                if isinstance(number, str) and number.isdigit():
                    number = int(number)
                elif isinstance(number, float) and number.is_integer():
                    number = int(number)
                if number == 0 and type(number) is int:
                    # Zero-based: every other number points at the item before the one meant. Drop the
                    # whole answer so the coverage re-ask scores the batch again.
                    logger.warning("Ranking response numbered items from 0; discarding the response")
                    return []
                # A bool or a fraction is not an item number (int(True) and int(2.7) would map silently).
                if type(number) is not int or not 1 <= number <= len(items):
                    logger.warning("Skipping ranking entry with item number %r", entry["item"])
                    continue
                if number in seen:
                    duplicates.append(number)
                    continue
                ranked_items.append(
                    RankedItem(
                        item=items[number - 1],
                        score=float(entry["score"]),
                        reasoning=entry.get("reasoning", ""),
                    )
                )
                seen.add(number)
            except (KeyError, ValueError, TypeError) as exc:
                logger.warning("Skipping malformed ranking entry: %s (%s)", entry, exc)

        if duplicates:
            logger.warning(
                "Ranking response repeated %d item(s); kept the first entry for each: %s", len(duplicates), duplicates
            )
        return ranked_items
