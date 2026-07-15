from __future__ import annotations

from datetime import UTC, date, datetime
from urllib.parse import urlparse

from langchain_core.output_parsers import StrOutputParser

from shared import (
    YOUTUBE_VIEWS_EMOJI,
    BedrockLanguageModelFactory,
    CollectedItem,
    DigestContent,
    DigestItem,
    DigestPrompt,
    DigestResult,
    GroundingCheckPrompt,
    RankedItem,
    SourceType,
    agi_countdown_intro,
    clean_rss_feed_name,
    format_collected_item,
    logger,
    parse_json_from_llm_output,
)
from shared.config import PipelineConfig


class DigestGenerator:

    def __init__(self, config: PipelineConfig, llm_factory: BedrockLanguageModelFactory) -> None:
        self.config = config
        self.llm_factory = llm_factory
        self.llm = llm_factory.get_model(config.digest_model)

    def _truncate(self, text: str, max_tokens: int) -> str:
        return self.llm_factory.truncate_to_tokens(text, max_tokens)

    async def generate(
        self,
        ranked_items: list[RankedItem],
        all_items: list[CollectedItem],
        trends_context: str = "",
        today: date | None = None,
        recent_leads: list[str] | None = None,
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

        items_text = self._format_ranked_items(ranked_items)
        chain = DigestPrompt.get_prompt() | self.llm | StrOutputParser()
        raw = await chain.ainvoke(
            {
                "items_text": items_text,
                "trends_context": trends_context or "(No trend data available yet.)",
                "language_rules": self.config.digest_language_rules,
                "audience": self.config.digest_audience_description,
                "voice_guidance": self.config.digest_voice_guidance,
                "target_count": target_count,
                "recent_leads": _format_recent_leads(recent_leads),
            }
        )
        content = self._parse_content(raw)
        # Hard upper-bound: the prompt asks for EXACTLY target_count, but a model can over-emit.
        # Trim deterministically so the digest never exceeds the target (fewer is allowed when the
        # editor genuinely found fewer distinct stories). headline_index is pinned to 1, so the
        # headline is always retained. Pinned URLs are kept even if they'd fall past the cutoff,
        # so a user-pinned story the editor ranked low isn't trimmed out of the digest.
        if len(content.items) > target_count:
            logger.info("Digest emitted %d items; trimming to target %d", len(content.items), target_count)
            content.items = self._trim_keeping_pinned(content.items, target_count, ranked_items)
        self._fill_source_metadata(content, ranked_items)

        if self.config.enable_grounding_check:
            content = await self._verify_grounding(content, ranked_items, trends_context)

        # Prepend the AGI countdown to the lead at generation time, using the digest's own date
        # (the single KST clock for the run) so the day count is consistent with trend stamps and
        # lands on every channel via the stored content — not just one renderer.
        intro = agi_countdown_intro(
            self.config.agi_countdown_date,
            self.config.agi_countdown_template,
            today or datetime.now(UTC).date(),
            self.config.agi_countdown_after,
        )
        if intro and content.lead and not content.lead.startswith(intro):
            content.lead = intro + content.lead

        digest_text = render_digest_text(content)
        logger.info("Digest generated successfully (%d items, %d characters)", len(content.items), len(digest_text))

        return DigestResult(
            digest_text=digest_text,
            ranked_items=ranked_items,
            content=content,
            generated_at=datetime.now(UTC),
            total_collected=len(all_items),
            total_ranked=len(ranked_items),
        )

    def _parse_content(self, raw: str) -> DigestContent:
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
            lead = data.get("lead")
            if not isinstance(lead, str) or not lead.strip():
                raise ValueError("Digest content is missing a usable 'lead'")
            # The prompt makes items[0] the headline (lead + image are about it); pin the index
            # to 1 so a stray LLM value can't point the lead and the visual at different stories.
            return DigestContent(lead=lead, headline_index=1, items=items)
        except Exception:
            logger.warning("Failed to parse digest content JSON; returning minimal content", exc_info=True)
            return DigestContent(lead=raw.strip()[:1000], headline_index=1, items=[])

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

    def _fill_source_metadata(self, content: DigestContent, ranked_items: list[RankedItem]) -> None:
        """Code owns the source tag/metrics (not the LLM): match each item to its ranked
        source by URL and stamp the backtick tag + emoji metrics the renderers display."""
        by_url = {r.item.url: r.item for r in ranked_items}
        for item in content.items:
            src = by_url.get(item.url)
            if src is None:
                continue
            item.source_tag, item.metrics = self._source_tag_and_metrics(src)

    async def _verify_grounding(
        self, content: DigestContent, ranked_items: list[RankedItem], trends_context: str = ""
    ) -> DigestContent:
        """Check the digest's specific claims against the source items and surgically revise
        unsupported ones. The trend ammunition (days-running, recurrence counts) is code-derived
        fact from trends.json, so it's passed as a valid source — otherwise grounding would strip
        the very recurrence figures that make the lead sharp. Runs over a plain-text serialization
        of the content; on success the corrected text is re-parsed back into the structured fields.
        Best-effort: any failure keeps the original content."""
        try:
            sources = "\n\n".join(
                f"[{i + 1}] {r.item.title}\n{self._truncate(r.item.text, self.config.item_text_max_tokens)}"
                for i, r in enumerate(ranked_items)
            )
            if trends_context:
                sources += f"\n\n[TRENDS] Verified trend-tracking history (recurrence facts):\n{trends_context}"
            chain = GroundingCheckPrompt.get_prompt() | self.llm | StrOutputParser()
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
    def _source_tag_and_metrics(item: CollectedItem) -> tuple[str, str]:
        """Return (source_tag, metrics) for an item: a backtick-wrapped source label and a
        ' · '-joined emoji metric string. Code owns this — the LLM never writes source markup."""
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

        return tag, " · ".join(metrics)


def _format_recent_leads(recent_leads: list[str] | None) -> str:
    """Render the last few days' leads as a bulleted block for the anti-repetition prompt
    input. Generic — names no phrase to ban, just 'here are recent openings, differ from them'."""
    leads = [ln.strip() for ln in (recent_leads or []) if ln and ln.strip()]
    if not leads:
        return "(No recent digests — no prior angles to avoid.)"
    return "\n".join(f"- {ln}" for ln in leads)


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


_GROUNDING_FIELDS = ("LEAD", "BODY", "IMPLICATION")


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
