from __future__ import annotations

from collections import Counter
from datetime import UTC, date, datetime

from langchain_core.output_parsers import StrOutputParser

from pipeline.aggregator import normalize_url
from shared import (
    COUNTDOWN_SUFFIX_SEPARATOR,
    THREADS_MAX_POST_CHARS,
    BedrockLanguageModelFactory,
    CollectedItem,
    DigestContent,
    DigestItem,
    DigestPrompt,
    DigestResult,
    GroundingCheckPrompt,
    RankedItem,
    agi_countdown_intro,
    format_collected_item,
    logger,
    parse_json_from_llm_output,
    place_countdown_intro,
    resolve_origin_key,
    retry_async,
    source_tag_and_metrics,
    split_sentences,
    threads_item_overhead_chars,
    threads_meta_line,
)
from shared.config import PipelineConfig


class DigestContentError(ValueError):
    """The editor's output could not be turned into a digest carrying at least one story.
    Retryable: a re-ask usually succeeds, since the cause is a one-off malformed emission."""


class DigestGenerator:

    def __init__(self, config: PipelineConfig, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm_factory = llm_factory
        self.llm = llm_factory.get_model(config.digest_model, stage="digest")
        # Its OWN handle: the grounding pass is a SECOND ~50k-token Sonnet call, and billed under
        # "digest" it was indistinguishable from the generation itself — defeating the only
        # spend-attribution mechanism the repo has (the bill is per MODEL, and several stages share
        # this one). Built eagerly beside the digest model so the stage label cannot drift.
        self.grounding_llm = llm_factory.get_model(config.digest_model, stage="grounding")

    def _truncate(self, text: str, max_tokens: int) -> str:
        return self.llm_factory.truncate_to_tokens(text, max_tokens)

    async def generate(
        self,
        ranked_items: list[RankedItem],
        all_items: list[CollectedItem],
        trends_context: str = "",
        today: date | None = None,
        recent_leads: list[str] | None = None,
        recent_titles: list[str] | None = None,
    ) -> DigestResult:
        if not ranked_items:
            logger.warning("No ranked items to generate digest from")
            return DigestResult(
                digest_text="No notable content collected today.",
                ranked_items=[],
                generated_at=datetime.now(UTC),
            )

        # The ranker over-selects (top_n + buffer); the editor merges same-event items and
        # then emits exactly target_count distinct stories, backfilling from the buffer so a
        # merge never shrinks the digest below the target.
        target_count = min(self.config.top_n, len(ranked_items))
        # A user can pin more URLs than top_n. Both the headline (items[0]) and every pinned item
        # must survive the trim, so raise the target to fit them all rather than silently dropping
        # pins (violating the --pin-url guarantee) or the headline (desyncing lead/visual).
        pin_count = sum(1 for r in ranked_items if r.item.metadata.get("pinned"))
        min_needed = min(len(ranked_items), pin_count + 1)  # +1 reserves the headline slot
        if min_needed > target_count:
            logger.info("Raising digest target %d → %d to fit %d pinned item(s)", target_count, min_needed, pin_count)
            target_count = min_needed
        logger.info(
            "Generating digest from %d candidates → target %d items (model: %s)",
            len(ranked_items),
            target_count,
            self.config.digest_model.value,
        )

        # The countdown gag is code-owned and lands on the lead after generation, so its length is
        # spent before the editor writes a word — compute it here to bound the lead's own budget.
        intro = agi_countdown_intro(
            self.config.agi_countdown_date,
            self.config.agi_countdown_template,
            today or datetime.now(UTC).date(),
            self.config.agi_countdown_after,
        )

        items_text = self._format_ranked_items(ranked_items)
        chain = DigestPrompt.get_prompt() | self.llm | StrOutputParser()
        prompt_vars = {
            "items_text": items_text,
            "trends_context": trends_context or "(No trend data available yet.)",
            "language_rules": self.config.digest_language_rules,
            "audience": self.config.digest_audience_description,
            "voice_guidance": self.config.digest_voice_guidance,
            "target_count": target_count,
            "recent_leads": _format_recent_leads(recent_leads),
            "recent_titles": _format_recent_titles(recent_titles),
            "prose_budget_rule": _PROSE_BUDGET_RULE,
            "lead_budget": self._lead_budget(intro),
        }

        async def _ask_editor() -> tuple[DigestContent, list[CollectedItem]]:
            content = self._parse_content(await chain.ainvoke(prompt_vars))
            # Hard upper-bound: the prompt asks for EXACTLY target_count, but a model can over-emit.
            # Trim deterministically so the digest never exceeds the target (fewer is allowed when
            # the editor genuinely found fewer distinct stories). headline_index is pinned to 1, so
            # the headline is always retained. Pinned URLs are kept even if they'd fall past the
            # cutoff, so a user-pinned story the editor ranked low isn't trimmed out of the digest.
            if len(content.items) > target_count:
                logger.info("Digest emitted %d items; trimming to target %d", len(content.items), target_count)
                content.items = self._trim_keeping_pinned(content.items, target_count, ranked_items)
            # Inside the retry on purpose: an unmatched HEADLINE raises DigestContentError, and a
            # re-ask is exactly the remedy (the lead and the visual are both written about items[0]).
            sources = self._fill_source_metadata(content, ranked_items)
            return content, sources

        # Re-ask on a malformed emission (or a transient Bedrock error) instead of degrading.
        # Exhausting the attempts raises, so a story-less digest is never persisted or posted.
        content, shipped_sources = await retry_async(
            _ask_editor,
            max_retries=self.config.digest_max_retries,
            backoff_sec=self.config.digest_retry_backoff_sec,
            description="Digest content generation",
        )

        if self.config.enable_grounding_check:
            content = await self._verify_grounding(content, shipped_sources, trends_context)

        # Attach the AGI countdown to the lead at generation time, using the digest's own date
        # (the single KST clock for the run) so the day count is consistent with trend stamps and
        # lands on every channel via the stored content — not just one renderer. Its end of the lead
        # is config-driven (agi_countdown_position).
        content.lead = place_countdown_intro(content.lead, intro, self.config.agi_countdown_position)

        digest_text = render_digest_text(content)
        logger.info("Digest generated successfully (%d items, %d characters)", len(content.items), len(digest_text))

        return DigestResult(
            digest_text=digest_text,
            ranked_items=ranked_items,
            content=content,
            generated_at=datetime.now(UTC),
            diversity_breaches=self._audit_shipped_diversity(shipped_sources, ranked_items),
        )

    def _audit_shipped_diversity(self, shipped: list[CollectedItem], ranked_items: list[RankedItem]) -> list[str]:
        """Check the diversity caps against the digest that actually SHIPS.

        The ranker enforces max_per_origin and the guaranteed source slots on the ranked core, but the
        merge-backfill candidates it hands over deliberately ignore both, and nothing verified that
        the editor used a backfill item as a REPLACEMENT rather than an addition — the prompt only
        asks. So a shipped digest can carry two stories from one subreddit/handle while a source that
        HAD a candidate drops out entirely.

        Detection only, by design: reported (ERROR log + DigestResult) so the alerting path can say
        so, never acted on by swapping stories — that would change what ships."""
        breaches: list[str] = []
        cap = self.config.max_per_origin
        origins = Counter(key for key in (resolve_origin_key(item) for item in shipped) if key)
        over = sorted((origin, count) for origin, count in origins.items() if count > cap)
        if over:
            breaches.append(
                f"max_per_origin={cap} exceeded by "
                + ", ".join(f"'{origin}' ({count} stories)" for origin, count in over)
            )
        # Only a source that had a CORE candidate can be said to have been dropped. `ranked_items` is
        # the over-selected list (top_n + digest_candidate_buffer), and its backfill and grace entries
        # are explicitly offered as optional: the prompt tells the editor to use a backfill item as a
        # REPLACEMENT for a merged story, and a grace item is below min_score and only there so its
        # source's guaranteed slot stays fillable. Auditing against the full list therefore alerted on
        # the buffer working as designed — the editor declining exactly what it was told it may
        # decline reached SNS as a 'Ranking Health' ALERT. A collector that returned nothing is a
        # quiet day either way, not a diversity failure.
        shipped_sources = {item.source_type.value for item in shipped}
        available = {r.item.source_type.value for r in ranked_items if not r.backfill and not r.grace}
        dropped = sorted(
            source
            for source, slot in self.config.source_slots.items()
            if slot >= 1 and source in available and source not in shipped_sources
        )
        if dropped:
            breaches.append(
                f"source(s) with a guaranteed slot and a ranked candidate shipped nothing: {', '.join(dropped)}"
            )
        for breach in breaches:
            logger.error("Digest diversity breach (as shipped): %s", breach)
        return breaches

    def _prose_budget(self, meta: str, url: str) -> int:
        """Characters the editor may spend on ONE item's title + body + implication.

        Derived from the real fixed parts of THAT item's Threads post rather than estimated: the URL
        and the source line are code-owned and their lengths are known, so the budget is 500 minus
        this item's own overhead. It used to be the worst case across the whole candidate pool, which
        charged every short-URL item for the longest URL in it. The TITLE is inside the number because
        the editor authors it — the old budget covered body + implication only, so every Korean title
        was spent off-budget and 5 of 95 sampled items lost their closing sentence.
        digest_item_prose_max_chars stays an optional CEILING: 0 means "no channel cap here"."""
        derived = max(0, THREADS_MAX_POST_CHARS - threads_item_overhead_chars(meta, url))
        ceiling = self.config.digest_item_prose_max_chars
        return min(derived, ceiling) if ceiling > 0 else derived

    def _item_prose_budget(self, item: CollectedItem) -> int:
        """The budget for one ranked candidate, off the provenance line its post will really carry."""
        return self._prose_budget(self._threads_meta_line(item), item.url)

    def _lead_budget(self, intro: str) -> int:
        """Characters the editor may spend on the lead. The lead IS the Threads root post, and the
        code-owned countdown gag is appended to it afterwards, so the gag's own length (plus the
        blank line before it in `suffix` position) is not the editor's to spend. An over-long lead
        is trimmed by whole sentences at publish time — silently, until now."""
        reserved = len(intro)
        if intro and self.config.agi_countdown_position == "suffix":
            reserved += len(COUNTDOWN_SUFFIX_SEPARATOR)
        return max(0, THREADS_MAX_POST_CHARS - reserved)

    @classmethod
    def _threads_meta_line(cls, item: CollectedItem) -> str:
        """The item's provenance line exactly as the Threads renderer shows it, so the budget is
        computed off the string that really occupies the post."""
        return threads_meta_line(*cls._source_tag_and_metrics(item))

    def _parse_content(self, raw: str) -> DigestContent:
        """Turn the editor's raw output into a DigestContent, or raise DigestContentError.

        It NEVER degrades to a story-less digest. The old fallback returned
        `DigestContent(lead=raw[:1000], items=[])` on any parse error, which shipped the raw
        fenced JSON as the lead and zero stories — the 2026-08-13 and 2026-08-17 digests each lost
        all five stories that way (the editor emitted a stray `]` after the lead string; downstream
        the visual Lambda then posted a caption-only root and logged success). Raising lets the
        caller re-ask, and a persistent failure surfaces as a failed run instead of a broken post.
        """
        try:
            data = parse_json_from_llm_output(raw)
            if not isinstance(data, dict):
                raise ValueError(f"Expected a JSON object, got {type(data).__name__}")
            # Validate items INDIVIDUALLY so one malformed story (e.g. a missing url/body from an
            # LLM slip) drops only that item — not the whole digest. Whole-object model_validate
            # would raise on the first bad item and collapse every good story to the 0-item
            # fallback (the same silent-empty failure class as the control-char parse bug).
            raw_items = data.get("items", []) or []
            items: list[DigestItem] = []
            for i, raw_item in enumerate(raw_items):
                try:
                    items.append(DigestItem.model_validate(raw_item))
                except Exception:
                    logger.warning("Skipping malformed digest item %d: %r", i, raw_item, exc_info=True)
                    # The lead and the daily visual are BOTH written about items[0] (the headline).
                    # If that first story is the one that fails validation, keeping the rest would
                    # leave the lead/visual describing a story no longer in the digest. Fall back to
                    # minimal content rather than ship a headline/lead/visual mismatch.
                    if i == 0:
                        raise ValueError("Headline item (items[0]) failed validation") from None
            if len(items) < len(raw_items):
                # ERROR, not just the per-item warning above: the digest that follows looks entirely
                # normal, so a dropped story is otherwise invisible — the reader simply gets fewer
                # stories than the editor wrote. A shortfall from MERGING is legitimate and stays a
                # warning in the caller; losing an emitted item to validation is not.
                logger.error(
                    "Digest lost %d of %d emitted stories to validation; the digest ships short",
                    len(raw_items) - len(items),
                    len(raw_items),
                )
            lead = data.get("lead")
            if not isinstance(lead, str) or not lead.strip():
                raise ValueError("Digest content is missing a usable 'lead'")
            # The prompt makes items[0] the headline (lead + image are about it); pin the index
            # to 1 so a stray LLM value can't point the lead and the visual at different stories.
            content = DigestContent(lead=lead, headline_index=1, items=items)
        except Exception as e:
            # Log the offending output: without it the 08-17 root cause was only recoverable by
            # dumping the stored memory snapshot days later.
            logger.warning("Unparseable digest content (%s); raw output: %r", e, raw.strip()[:600])
            raise DigestContentError(f"Could not parse digest content: {e}") from e
        if not content.items:
            raise DigestContentError("Digest content parsed but carries no stories")
        return content

    @staticmethod
    def _trim_keeping_pinned(items: list, target_count: int, ranked_items: list[RankedItem]) -> list:
        """Trim to target_count but never drop the headline (items[0]) nor a user-pinned item.
        items[0] is the headline the lead prose and the daily visual are both written about, so it
        MUST survive the trim — otherwise the lead/visual describe a story no longer in the digest.
        The headline is kept first; the remaining slots preserve every pinned item, walking the
        editor's emitted order and skipping a non-pinned item only when the slots left are all
        needed by pinned items still ahead."""
        if len(items) <= target_count:
            return items[:target_count]
        pinned_urls = {r.item.url for r in ranked_items if r.item.metadata.get("pinned")}
        # Always retain the headline; fill the rest from items[1:] preserving pins.
        kept: list = items[:1]
        rest = items[1:]
        for i, it in enumerate(rest):
            if len(kept) >= target_count:
                break
            remaining_pinned = sum(
                1 for later in rest[i:] if later.url in pinned_urls and later not in kept and later is not it
            )
            slots_left = target_count - len(kept)
            # Skip this non-pinned item only if keeping it would crowd out a pinned item still ahead.
            if it.url not in pinned_urls and slots_left <= remaining_pinned:
                continue
            kept.append(it)
        return kept

    def _fill_source_metadata(self, content: DigestContent, ranked_items: list[RankedItem]) -> list[CollectedItem]:
        """Code owns the source tag/metrics (not the LLM): match each item to its ranked source and
        stamp the backtick tag + emoji metrics the renderers display. Returns the source item behind
        each SURVIVING story, so the grounding check can be scoped to what actually shipped.

        Matched on the aggregator's normalize_url, not an exact string: the editor echoes the URL
        back, and one trailing slash / http→https / dropped utm param was enough to lose the match —
        the story then shipped with no source line at all, on Slack and on Threads. Identical URLs
        take the same path they always did.

        An item that matches NO candidate is a story the editor invented, or whose URL it mangled
        beyond normalization. It used to be tagged with its own host, shipped to the reader, and
        recorded in the published-URL ledger — which then suppressed the REAL article for the whole
        TTL window. It is dropped instead, and an unmatched HEADLINE rejects the emission outright:
        the lead and the daily visual are both written about items[0], so there is nothing to salvage
        and the caller's re-ask is the remedy."""
        by_url = {normalize_url(r.item.url): r.item for r in ranked_items}
        kept: list[DigestItem] = []
        sources: list[CollectedItem] = []
        unmatched: list[str] = []
        for i, item in enumerate(content.items):
            src = by_url.get(normalize_url(item.url))
            if src is None:
                if i == 0:
                    raise DigestContentError(f"Headline item '{item.url}' matches no ranked candidate")
                unmatched.append(item.url)
                continue
            item.source_tag, item.metrics = self._source_tag_and_metrics(src)
            kept.append(item)
            sources.append(src)
        if unmatched:
            # ERROR: the digest that follows looks entirely normal, so a story invented out of
            # nothing (or with a mangled URL) is otherwise invisible until a reader clicks it.
            logger.error(
                "Dropping %d digest item(s) that match no ranked candidate: %s",
                len(unmatched),
                unmatched,
            )
            content.items = kept
        return sources

    async def _verify_grounding(
        self, content: DigestContent, sources_items: list[CollectedItem], trends_context: str = ""
    ) -> DigestContent:
        """Check the digest's specific claims against the source items and surgically revise
        unsupported ones. The trend ammunition (days-running, recurrence counts) is code-derived
        fact from trends.json, so it's passed as a valid source — otherwise grounding would strip
        the very recurrence figures that make the lead sharp. Runs over a plain-text serialization
        of the content; on success the corrected text is re-parsed back into the structured fields.
        Best-effort: any failure keeps the original content.

        `sources_items` is what actually SHIPPED (_fill_source_metadata's match map), not every
        ranked candidate. Joining the whole candidate list meant a number or product name whose only
        support was a dropped backfill candidate read as grounded — a false negative in the single
        pass that exists to catch invented specifics — and carried the buffer's surplus input tokens
        (item_text_max_tokens each) on the critical path."""
        try:
            sources = "\n\n".join(
                f"[{i + 1}] {item.title}\n{self._truncate(item.text, self.config.item_text_max_tokens)}"
                for i, item in enumerate(sources_items)
            )
            if trends_context:
                sources += f"\n\n[TRENDS] Verified trend-tracking history (recurrence facts):\n{trends_context}"
            chain = GroundingCheckPrompt.get_prompt() | self.grounding_llm | StrOutputParser()
            raw = await chain.ainvoke({"digest_text": _grounding_payload(content), "sources": sources})
            data = parse_json_from_llm_output(raw)
            violations = data.get("violations", [])
            corrected = data.get("corrected_digest", "")
            if not violations or not corrected:
                logger.info("Grounding check: no unsupported claims found")
                return content
            for v in violations:
                logger.info("Grounding check revised claim: %s (%s)", v.get("claim", "")[:80], v.get("issue", "")[:80])
            logger.info("Grounding check revised %d unsupported claim(s)", len(violations))
            return _apply_grounding_payload(content, corrected)
        except Exception:
            logger.warning("Grounding check failed; keeping original content", exc_info=True)
            return content

    def _format_ranked_items(self, ranked_items: list[RankedItem]) -> str:
        parts: list[str] = []
        budgets: list[int] = []
        for i, ranked in enumerate(ranked_items):
            item = ranked.item
            tag, metrics = self._source_tag_and_metrics(item)
            fields = [
                ("Score", f"{ranked.score:.2f}"),
                ("Categories", ", ".join(ranked.categories)),
                ("Reasoning", ranked.reasoning),
                ("Title", item.title),
                ("URL", item.url),
                ("Source", item.source_type.value),
                ("Source Detail", " · ".join(p for p in (tag, metrics) if p) or item.source_type.value),
                ("Author", item.author or "Unknown"),
            ]
            # A pinned item (user-requested via --pin-url) must appear in the digest regardless
            # of its score, so flag it for the editor; code also protects it from the trim below.
            if item.metadata.get("pinned"):
                fields.insert(
                    0, ("MUST INCLUDE", "user-pinned — keep this item in the digest, do not drop or merge it away")
                )
            # The ranker enforces the source-mix guarantees on the first top_n candidates; these
            # extras exist so a merge of two same-event items can still be topped up to top_n. Say
            # so per item (a code-owned field, like MUST INCLUDE) instead of asking the prompt to
            # infer which candidates are spare from their position in the list.
            elif ranked.backfill:
                fields.insert(0, ("BACKFILL", "spare candidate — use it to replace a merged item, not in addition"))
            # A grace item is below min_score BY DESIGN: it is the best its source had, and it is here
            # so that source's guaranteed slot stays fillable. Without saying so, its low Score was the
            # only thing the editor could see and it lost to every ordinary candidate — the youtube
            # grace candidate of digest_2026-07-12 is in ranked_items and absent from the shipped five.
            elif ranked.grace:
                fields.insert(
                    0,
                    (
                        "SOURCE COVERAGE",
                        f"best candidate its source ({item.source_type.value}) had — its low score reflects "
                        "the medium, not the story; prefer it over a second story from a source already "
                        "represented",
                    ),
                )
            # This item's OWN prose budget, code-owned like MUST INCLUDE: the fixed parts of a Threads
            # post differ per item, so one worst-case number for the whole pool charged every
            # short-URL item for the longest URL in it — and the lint measures each item against
            # exactly the number stated here.
            budget = self._item_prose_budget(item)
            if budget > 0:
                budgets.append(budget)
                fields.append(("PROSE BUDGET", f"{budget} characters for `title` + `body` + `implication` together"))
            parts.append(
                format_collected_item(
                    item,
                    index=i + 1,
                    max_tokens=self.config.item_text_max_tokens,
                    fields=fields,
                    truncate=self._truncate,
                )
            )
        if budgets:
            logger.info(
                "Item prose budgets: %d-%d chars across %d candidates", min(budgets), max(budgets), len(budgets)
            )
        return "\n".join(parts)

    # The per-SourceType tag/metrics table lives in shared/formatting.py beside the origin key and
    # the origin label it has to agree with — three if/elif chains over the same five SourceTypes
    # each fell through to its own silent default, and only one of them was ever fixed at a time.
    _source_tag_and_metrics = staticmethod(source_tag_and_metrics)


# The clause telling the editor how much prose actually survives to the post. It points at the
# PROSE BUDGET stated with each candidate rather than carrying one number for the whole digest: the
# fixed parts differ per item (a 180-character URL costs what a 20-character one does not), so a
# single worst-case number charged every short-URL item for the longest URL in the pool.
_PROSE_BUDGET_RULE = (
    " `title`, `body` and `implication` TOGETHER must stay under this item's own PROSE BUDGET, stated "
    "with the item — the renderer drops trailing sentences that do not fit, so an over-long body loses "
    "exactly the closing detail you wrote last."
)


# Backstop cap for one recent-lead opening (an unterminated lead has no sentence boundary to cut at).
RECENT_LEAD_PREVIEW_CHARS = 200


def _format_recent_leads(recent_leads: list[str] | None) -> str:
    """Render the last few days' leads as a bulleted block for the anti-repetition prompt
    input. Generic — names no phrase to ban, just 'here are recent openings, differ from them'.

    Only each lead's FIRST SENTENCE is shown. What must differ is the opening ANGLE, and that is the
    opening sentence; the 200-character previews carried a paragraph of prose the editor was never
    asked to compare against. Derived HERE, at format time, so leads already stored as full prose
    keep working — no history migration."""
    leads = [ln.strip() for ln in (recent_leads or []) if ln and ln.strip()]
    if not leads:
        return "(No recent digests — no prior angles to avoid.)"
    return "\n".join(f"- {_first_sentence(ln)}" for ln in leads)


def _first_sentence(text: str) -> str:
    """The text's first sentence, hard-capped as a backstop for prose with no sentence boundary."""
    sentences = split_sentences(text)
    first = (sentences[0] if sentences else text).strip()
    if len(first) <= RECENT_LEAD_PREVIEW_CHARS:
        return first
    return first[:RECENT_LEAD_PREVIEW_CHARS].rstrip() + "…"


def _format_recent_titles(recent_titles: list[str] | None) -> str:
    """The story titles the LAST published digest carried. The editor needs to know what yesterday
    already ran so today isn't a re-run of it; the URL ledger stays the mechanism that actually
    suppresses a repeat, so this is information, not a filter."""
    titles = [t.strip() for t in (recent_titles or []) if t and t.strip()]
    if not titles:
        return "(No recent digest on record.)"
    return "\n".join(f"- {t}" for t in titles)


def render_digest_text(content: DigestContent) -> str:
    """Plain-prose rendering of the structured content — the system-of-record `digest_text`
    fed to the trend classifier, the AgentCore memory snapshot, and the follow-up agent.
    No Slack markup; channel renderers add their own."""
    parts = [content.lead.strip(), ""]
    for item in content.items:
        meta = " · ".join(p for p in (item.source_tag, item.metrics) if p)
        header = f"{item.title}"
        if meta:
            header += f" ({meta})"
        parts.append(header)
        if item.url:
            parts.append(item.url)
        if item.body:
            parts.append(item.body.strip())
        if item.implication:
            parts.append(item.implication.strip())
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def _grounding_payload(content: DigestContent) -> str:
    """Serialize the prose fields (lead + each item's body/implication) as labelled lines for
    the grounding check. Only prose the LLM authored is checked — titles/urls/source tags are
    code-owned and excluded so they can't be altered."""
    lines = [f"LEAD: {content.lead}"]
    for i, item in enumerate(content.items):
        lines.append(f"ITEM {i} BODY: {item.body}")
        if item.implication:
            lines.append(f"ITEM {i} IMPLICATION: {item.implication}")
    return "\n".join(lines)


def _apply_grounding_payload(content: DigestContent, corrected: str) -> DigestContent:
    """Parse the corrected labelled lines back onto the content. Any field whose marker is
    missing keeps its original value; a malformed payload leaves content unchanged."""
    updated = content.model_copy(deep=True)
    current_key: tuple[str, int | None] | None = None
    buffers: dict[tuple[str, int | None], list[str]] = {}
    for line in corrected.splitlines():
        key, _, rest = _match_grounding_marker(line)
        if key is not None:
            current_key = key
            buffers[key] = [rest]
        elif current_key is not None:
            buffers[current_key].append(line)

    if ("LEAD", None) in buffers:
        updated.lead = "\n".join(buffers[("LEAD", None)]).strip()
    for i, item in enumerate(updated.items):
        if ("BODY", i) in buffers:
            item.body = "\n".join(buffers[("BODY", i)]).strip()
        if ("IMPLICATION", i) in buffers:
            item.implication = "\n".join(buffers[("IMPLICATION", i)]).strip()
    return updated


def _match_grounding_marker(line: str) -> tuple[tuple[str, int | None] | None, str, str]:
    if line.startswith("LEAD:"):
        return ("LEAD", None), "LEAD", line[len("LEAD:") :].strip()
    if line.startswith("ITEM "):
        head, sep, rest = line.partition(":")
        if sep:
            tokens = head.split()
            if (
                len(tokens) == 3
                and tokens[0] == "ITEM"
                and tokens[1].isdigit()
                and tokens[2] in ("BODY", "IMPLICATION")
            ):
                return (tokens[2], int(tokens[1])), tokens[2], rest.strip()
    return None, "", ""
