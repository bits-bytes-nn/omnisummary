from __future__ import annotations

import re
from typing import NamedTuple

from shared import (
    THREADS_MAX_POST_CHARS,
    THREADS_POST_SEPARATOR,
    URL_RE,
    DigestContent,
    logger,
    split_sentences,
    strip_slack_mrkdwn,
    threads_meta_line,
    truncate_at_word,
)

__all__ = [
    "SLACK_MAX_BLOCKS_PER_MESSAGE",
    "SLACK_MAX_SECTION_CHARS",
    "render_agent_blocks",
    "render_research_blocks",
    "render_slack_blocks",
    "render_threads_posts",
    "render_threads_research",
]

# Slack caps a single message at 50 blocks; chunk item blocks across messages under it.
SLACK_MAX_BLOCKS_PER_MESSAGE = 45
# A single Slack section's text field is capped at 3000 chars.
SLACK_MAX_SECTION_CHARS = 2900


def _split_long_paragraph(para: str, max_len: int) -> list[str]:
    """Break an over-length paragraph into <=max_len pieces at sentence then word boundaries,
    never inside a Slack `<url|text>` link span (a raw stride-split would cleave a link across
    two messages, leaving both halves as dead raw text)."""
    pieces: list[str] = []
    current = ""
    for sentence in split_sentences(para) or [para]:
        # A sentence that overflows on its own gets a link-safe hard split; there is nothing to
        # word-trim it down to, because the next unit has to carry the remainder either way. A
        # `truncate_at_word(sentence, max_len)` sat on the fitting side of this guard and was
        # discarded exactly when it was produced, since this same comparison decides both.
        units = [sentence] if len(sentence) <= max_len else _hard_split_link_safe(sentence, max_len)
        for u in units:
            candidate = f"{current} {u}".strip() if current else u
            if len(candidate) > max_len and current:
                pieces.append(current)
                current = u
            else:
                current = candidate
    if current:
        pieces.append(current)
    return pieces


def _hard_split_link_safe(text: str, max_len: int) -> list[str]:
    """Last-resort hard split that never cuts INSIDE a `<...>` link span. When a cut would land
    inside a span, emit the text BEFORE the link as its own (<=max_len) piece and put the whole
    link span on the next piece. A link span longer than max_len can't be split without breaking
    the link, so it becomes one over-cap piece on its own — the least-bad option, and far better
    than merging it with preceding text (which the old 'extend end to the closing >' did, blowing
    the cap by the length of the preceding run too)."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_len, n)
        if end < n:  # a real cut point (not the tail)
            open_idx = text.rfind("<", i, end)
            close_idx = text.rfind(">", i, end)
            if open_idx > close_idx:  # cut lands inside a link span
                if open_idx > i:
                    # Emit the text before the link first (guaranteed <=max_len), then restart at
                    # the link so it lands on its own piece.
                    out.append(text[i:open_idx])
                    i = open_idx
                    continue
                # The link starts at i and overflows the window — keep the whole span intact on one
                # piece (can't split a link); this single piece may exceed max_len by necessity.
                next_close = text.find(">", end)
                end = next_close + 1 if next_close != -1 else n
        out.append(text[i:end])
        i = end
    return out


def render_agent_blocks(text: str) -> list[list[dict]]:
    """Wrap a free-form agent mrkdwn reply in Block Kit section blocks. The agent's output
    has no fixed structure, so this just paragraph-packs the text into <=3000-char sections
    (keeping the agent's own *bold*/`code`/<links>) and chunks under the per-message block
    cap. A generic wrapper — it does not parse or restructure the content."""
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    sections: list[str] = []
    current = ""
    for para in paragraphs:
        if len(para) > SLACK_MAX_SECTION_CHARS:
            if current:
                sections.append(current)
                current = ""
            sections.extend(_split_long_paragraph(para, SLACK_MAX_SECTION_CHARS))
        elif len(current) + len(para) + 2 > SLACK_MAX_SECTION_CHARS:
            sections.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        sections.append(current)

    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": s}} for s in sections]
    # Empty/whitespace input → no chunks (callers post nothing); never [[]], which would send an
    # invalid empty blocks=[] to Slack.
    return _chunk_blocks(blocks) if blocks else []


# A numbered section heading the agent emits on its own line, e.g. "*1. 벤치마크 성적표*".
_NUMBERED_HEADING = re.compile(r"^\*\d+\.\s")


def render_research_blocks(report: str, *, header: str) -> list[list[dict]]:
    """Render a deep-research report as Block Kit with the daily-digest look: a header block,
    then the report's prose with a divider before each numbered section heading ("*N. ...*") so
    it reads as cleanly sectioned rather than one wall of text. Paragraph-packs prose into
    <=SLACK_MAX_SECTION_CHARS sections and chunks under the per-message block cap."""
    # Empty/whitespace report → nothing to post (never a lone header band). Mirrors
    # render_agent_blocks / render_threads_research, which also no-op on empty input.
    if not report.strip():
        return []

    header_block = {"type": "header", "text": {"type": "plain_text", "text": header, "emoji": True}}
    blocks: list[dict] = [header_block]

    paragraphs = [p for p in report.split("\n\n") if p.strip()]
    current = ""

    def flush() -> None:
        nonlocal current
        if current.strip():
            blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": current}})
        current = ""

    for para in paragraphs:
        is_heading = bool(_NUMBERED_HEADING.match(para.strip()))
        if is_heading:
            flush()
            # Suppress a divider directly under the header (when the report opens with a numbered
            # heading) — header → divider → section reads as an empty band.
            if blocks[-1]["type"] != "header":
                blocks.append({"type": "divider"})
        if len(para) > SLACK_MAX_SECTION_CHARS:
            flush()
            for piece in _split_long_paragraph(para, SLACK_MAX_SECTION_CHARS):
                blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": piece}})
        elif is_heading:
            current = para
        elif len(current) + len(para) + 2 > SLACK_MAX_SECTION_CHARS:
            flush()
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    flush()

    return _chunk_blocks(blocks)


def _mrkdwn_sections(text: str, *, wrap: str = "{}") -> list[dict]:
    """One or more Slack `section` blocks for a mrkdwn string, splitting on the 3000-char section
    cap so an unusually long lead/implication can't get the whole message rejected as
    invalid_blocks. `wrap` applies emphasis to EACH piece (e.g. "_{}_" for the italic implication).
    The split budget accounts for the wrapper so a wrapped piece still fits."""
    body = text.strip()
    if not body:
        return []
    budget = SLACK_MAX_SECTION_CHARS - (len(wrap) - 2)
    pieces = [body] if len(body) <= budget else _split_long_paragraph(body, budget)
    return [{"type": "section", "text": {"type": "mrkdwn", "text": wrap.format(p)}} for p in pieces]


def _item_blocks(item, *, with_divider: bool) -> list[dict]:
    """Block Kit blocks for one DigestItem: title link, source/metrics context,
    body as a rich_text quote (the gray vertical bar), then the implication."""
    blocks: list[dict] = []
    if with_divider:
        blocks.append({"type": "divider"})

    blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"*<{item.url}|{item.title}>*"}})

    meta = " · ".join(p for p in (item.source_tag, item.metrics) if p)
    if meta:
        blocks.append({"type": "context", "elements": [{"type": "mrkdwn", "text": meta}]})

    if item.body:
        # Split an over-length body so no single rich_text_quote text element exceeds the section
        # cap (which would make Slack reject the whole message as invalid_blocks).
        body = item.body.strip()
        pieces = (
            [body] if len(body) <= SLACK_MAX_SECTION_CHARS else _split_long_paragraph(body, SLACK_MAX_SECTION_CHARS)
        )
        # rich_text_quote text elements render inline (no implicit separator), so join split pieces
        # with an explicit newline element to avoid merging words across a boundary.
        quote_elements: list[dict] = []
        for idx, p in enumerate(pieces):
            if idx:
                quote_elements.append({"type": "text", "text": "\n"})
            quote_elements.append({"type": "text", "text": p})
        blocks.append({"type": "rich_text", "elements": [{"type": "rich_text_quote", "elements": quote_elements}]})

    blocks.extend(_mrkdwn_sections(item.implication, wrap="_{}_"))
    return blocks


def render_slack_blocks(
    content: DigestContent, *, header: str, image_url: str = "", image_alt: str = ""
) -> list[list[dict]]:
    """Render DigestContent as Slack Block Kit, split into <=SLACK_MAX_BLOCKS_PER_MESSAGE
    chunks. Returns a list of block-lists, one per chat_postMessage call."""
    blocks: list[dict] = [{"type": "header", "text": {"type": "plain_text", "text": header, "emoji": True}}]
    blocks.extend(_mrkdwn_sections(content.lead))
    if image_url:
        blocks.append({"type": "image", "image_url": image_url, "alt_text": image_alt or "daily visual"})

    for item in content.items:
        blocks.extend(_item_blocks(item, with_divider=True))

    return _chunk_blocks(blocks)


def _chunk_blocks(blocks: list[dict]) -> list[list[dict]]:
    chunks: list[list[dict]] = []
    for i in range(0, len(blocks), SLACK_MAX_BLOCKS_PER_MESSAGE):
        chunks.append(blocks[i : i + SLACK_MAX_BLOCKS_PER_MESSAGE])
    return chunks or [[]]


def _fit_one_post(
    title: str, meta: str, body: str, implication: str, url: str, max_len: int = THREADS_MAX_POST_CHARS
) -> str:
    """Build ONE Threads post for an item that fits within max_len. Title, the source line and the
    URL are always kept; the implication (the voice line) is preserved over body — body sentences
    are dropped from the end first, and the implication only goes if the fixed parts still overflow.
    Nothing is cut mid-sentence and the link is never split. Each item maps to exactly one reply.

    `meta` is the "r/LocalLLaMA · ▶️ 775" provenance line the pipeline already computes and Slack
    already shows. Threads used to discard it, so a reader could not tell a Reddit thread from an
    arXiv paper without opening the link. It is treated as fixed: knowing the source is worth more
    than one more clause of body.

    It rides in PARENTHESES on the title line rather than as a block of its own. Slack renders the
    tag as inline code, which marks it as metadata; Threads renders no markup, so the backticks are
    stripped and a publication name ("Simon Willison's Weblog") was left as a bare noun phrase
    sitting on its own line, reading like a stray fragment. Parentheses are a self-evident
    attribution marker in Korean and survive markup-stripping, they work for all five source shapes
    without a label word, and merging the two blocks saves a separator — net +1 character against
    the old form. Slack keeps its context block: it was never the channel that read wrong."""
    titled = f"{title.strip()} ({meta.strip()})" if title.strip() and meta.strip() else (title.strip() or meta.strip())
    fixed = [titled] if titled else []
    tail = [url.strip()] if url.strip() else []
    impl = implication.strip()

    def assemble(prose: str, *, keep_impl: bool = True) -> str:
        # The implication is its OWN block, not the body's last sentence: it is the voice line the
        # item closes on, and run into the body as one paragraph it read as more body — the beat
        # that makes each item land was invisible in the post.
        blocks = list(fixed)
        if prose:
            blocks.append(prose)
        if keep_impl and impl:
            blocks.append(impl)
        return THREADS_POST_SEPARATOR.join(blocks + tail)

    body_sents = split_sentences(body)
    # Drop body sentences from the end while keeping the implication block.
    while body_sents:
        candidate = assemble(" ".join(body_sents))
        if len(candidate) <= max_len:
            return candidate
        body_sents.pop()
    # No body sentence fits alongside the implication. Keep the implication alone if it fits.
    if impl and len(assemble("")) <= max_len:
        return assemble("")

    # Even the implication won't fit. Word-trim the body into the remaining room (never drop
    # it to bare title+URL), or word-trim the title if title+URL alone overflow.
    bare = assemble("", keep_impl=False)
    room = max_len - len(bare) - len(THREADS_POST_SEPARATOR)
    if room > 0 and body.strip():
        return assemble(truncate_at_word(body, room), keep_impl=False)
    if len(bare) <= max_len:
        return bare
    room = max_len - (len(url.strip()) + len(THREADS_POST_SEPARATOR) if url.strip() else 0)
    return THREADS_POST_SEPARATOR.join([p for p in (truncate_at_word(title, max(0, room)), url.strip()) if p])


def _item_post_overflows(title: str, meta: str, body: str, implication: str, url: str) -> bool:
    """True when the item's FULL prose cannot fit one post — i.e. _fit_one_post had to drop
    something. Mirrors its assembly with nothing trimmed (the source line in parentheses on the
    title, the implication as its own block, so the separator count matches); used for counts-only
    trim reporting."""
    titled = f"{title.strip()} ({meta.strip()})" if title.strip() and meta.strip() else (title.strip() or meta.strip())
    blocks = [p for p in (titled, body.strip(), implication.strip()) if p]
    tail = [url.strip()] if url.strip() else []
    assembled = THREADS_POST_SEPARATOR.join(blocks + tail)
    return len(assembled) > THREADS_MAX_POST_CHARS


def _fit_lead(lead: str, countdown: str = "") -> str:
    """Fit the root text (the digest lead) into one Threads post.

    An over-long lead loses the CODE-OWNED countdown gag first, and only then the editor's prose:
    the gag is the same fixed template every day, while the sentence it was crowding out is the day's
    actual argument. It is dropped only when the lead's final line can be IDENTIFIED as that gag by
    comparing it to the countdown string the caller passes in — a blind "drop the last line" would
    delete real prose whenever the gag sits at the front (agi_countdown_position="prefix") or is
    disabled. Otherwise whole sentences are kept from the front, so a prefix gag survives.

    Logged at WARNING because a trimmed lead means the editor's prose budget was overrun."""
    if len(lead) <= THREADS_MAX_POST_CHARS:
        return lead
    logger.warning("Threads lead is %d chars (cap %d); trimming prose", len(lead), THREADS_MAX_POST_CHARS)
    head, sep, last_line = lead.rpartition("\n")
    gag = countdown.strip()
    if sep and gag and last_line.strip() == gag:
        logger.warning("Dropping the countdown line from the Threads lead to keep the editor's prose")
        return _pack_sentences(head.strip(), THREADS_MAX_POST_CHARS)
    return _pack_sentences(lead, THREADS_MAX_POST_CHARS)


def _pack_sentences(text: str, max_len: int) -> str:
    """Keep whole leading sentences up to max_len; word-trim when even the first one overflows."""
    kept = split_sentences(text)
    while kept and len(" ".join(kept)) > max_len:
        kept.pop()
    return " ".join(kept) if kept else truncate_at_word(text, max_len)


def render_threads_posts(content: DigestContent, countdown: str = "") -> tuple[str, list[str]]:
    """Render DigestContent for Threads: a root text (the lead) and a reply chain with EXACTLY ONE
    reply per item (title + source line + body + implication + URL). Each reply is trimmed to fit
    Threads' 500-char cap at a clean sentence boundary — never mid-word — keeping the title, the
    source line and the URL. No Slack markup (Threads renders none).

    `countdown` is the code-owned AGI-countdown gag as the pipeline computed it, so an over-long
    lead can drop THAT rather than the day's argument. Empty (the default) simply trims prose."""
    lead = _fit_lead(content.lead.strip(), countdown)
    replies: list[str] = []
    trimmed = 0
    for item in content.items:
        meta = _item_meta(item)
        implication = item.implication or ""
        replies.append(_fit_one_post(item.title, meta, item.body, implication, item.url))
        if _item_post_overflows(item.title, meta, item.body, implication, item.url):
            trimmed += 1
    if trimmed:
        # Counts only, never the text: a trimmed item means the editor's prose budget was overrun,
        # and 5 of 95 sampled items silently lost their closing sentence (the concrete figures).
        logger.warning(
            "Threads: %d of %d item posts lost prose to the %d-char cap", trimmed, len(replies), THREADS_MAX_POST_CHARS
        )
    return lead, replies


def _item_meta(item) -> str:
    """The provenance line, composed by the shared helper the pipeline also budgets against."""
    return threads_meta_line(item.source_tag, item.metrics)


def _trim_long_sentence(sentence: str, max_len: int) -> str:
    """Word-trim an over-length sentence to fit max_len, but PRESERVE a trailing citation URL
    (research sentences often end in '... (https://...)'). Reserve room for the last URL, trim
    the prose before it, then re-append the URL — so a hard-trim never drops the citation."""
    urls = URL_RE.findall(sentence)
    if not urls:
        return truncate_at_word(sentence, max_len)
    tail = urls[-1].rstrip(").,")
    if len(tail) >= max_len:  # the URL alone overflows — nothing useful to keep but the URL
        return tail[:max_len]
    prose = sentence[: sentence.rfind(tail)].rstrip(" (")
    trimmed = truncate_at_word(prose, max_len - len(tail) - 1)
    return f"{trimmed} {tail}".strip()


# The agent separates Threads posts with a line containing only this delimiter, so post
# boundaries are the AGENT's choice (number + heading + body stay in ONE post) rather than
# the renderer guessing from blank lines.
_THREADS_POST_DELIMITER = re.compile(r"\n\s*---\s*\n")
# A delimiter line at the very START of the report (no preceding newline) the split regex can't
# see — strip it so a leading "---" never contaminates the first post as literal text.
_THREADS_LEADING_DELIMITER = re.compile(r"^\s*---\s*\n")
# The "N/M" index at the head of a post. The RENDERER owns it, because max_posts is applied AFTER the
# model has written its numbers: a capped report went out publicly as "1/8 ... 6/8" and stopped
# mid-argument, telling every reader two posts were missing. Any index the model still writes is
# stripped and replaced from the FINAL post count.
_THREADS_POST_INDEX = re.compile(r"^\s*\d+\s*/\s*\d+[.:)]?\s*")
# Separator between the index and the post's own 소제목.
_THREADS_INDEX_SEPARATOR = "  "


def _renumber_threads_posts(posts: list[str]) -> list[str]:
    """Re-prefix each post with 'N/M' derived from the FINAL list length, replacing whatever index the
    model wrote. Applied only to agent-delimited posts: the sentence-packing fallback fills each post
    to the 500-char cap, so prepending an index there would push every post over it and cost each one
    a trailing sentence — and that fallback exists for output that predates the numbered format
    anyway."""
    total = len(posts)
    renumbered: list[str] = []
    for index, post in enumerate(posts, start=1):
        head, separator, rest = post.partition("\n")
        subheading = _THREADS_POST_INDEX.sub("", head).strip()
        prefix = f"{index}/{total}"
        head = f"{prefix}{_THREADS_INDEX_SEPARATOR}{subheading}" if subheading else prefix
        renumbered.append(head + separator + rest)
    return renumbered


def _pack_by_sentence(text: str) -> list[str]:
    """Fallback packer when the agent gave no explicit post delimiters: greedily pack sentences
    into <=500-char posts at sentence boundaries (never mid-word)."""
    posts: list[str] = []
    current = ""
    for section in [s.strip() for s in text.split("\n\n") if s.strip()]:
        for sentence in split_sentences(section) or [section]:
            sentence = sentence.strip()
            if not sentence:
                continue
            if len(sentence) > THREADS_MAX_POST_CHARS:
                if current:
                    posts.append(current.strip())
                    current = ""
                posts.append(_trim_long_sentence(sentence, THREADS_MAX_POST_CHARS))
                continue
            candidate = f"{current} {sentence}".strip() if current else sentence
            if len(candidate) > THREADS_MAX_POST_CHARS and current:
                posts.append(current.strip())
                current = sentence
            else:
                current = candidate
        if current:
            posts.append(current.strip())
            current = ""
    return posts


def _trim_oversize_post(post: str) -> str:
    """Safety net: a single agent-delimited post over 500 chars is TRIMMED down to ONE post, not
    fanned out into several. Fanning out (the old behavior) flushed on the blank line under the
    heading, orphaning the 'N/M 소제목' line into its own post and scattering the body across
    unnumbered posts — the choppy mess this guards against. Instead: keep the first line (the
    'N/M 소제목' heading) and its blank line, then drop trailing body sentences from the end until
    the whole post fits, preserving a trailing citation URL on the last kept sentence."""
    if len(post) <= THREADS_MAX_POST_CHARS:
        return post
    logger.info("Threads research post exceeds %d chars, trimming to fit", THREADS_MAX_POST_CHARS)
    head, _, body = post.partition("\n")
    heading = head.strip()
    body = body.strip()
    room = THREADS_MAX_POST_CHARS - len(heading) - 2  # reserve "heading\n\n"
    if not body or room <= 0:
        # No heading/body split (one long line), or the heading alone already fills the post so
        # there's no body room to preserve — trim the whole post as one run (keeps a trailing URL).
        return _trim_long_sentence(post, THREADS_MAX_POST_CHARS)
    # A research post ends in a citation URL; reserve room for it so trimming the prose from the
    # end never drops the source. Keep leading sentences that fit, then re-append the citation.
    body_urls = URL_RE.findall(body)
    citation = body_urls[-1].rstrip(").,") if body_urls else ""
    prose_room = room - (len(citation) + 1) if citation else room
    sentences = split_sentences(body) or [body]
    kept: list[str] = []
    for sentence in sentences:
        # Don't double-count the citation: a sentence that is just the trailing URL is folded in
        # via `citation` below, not packed here.
        if citation and sentence.strip().rstrip(").,") == citation:
            continue
        candidate = " ".join(kept + [sentence])
        if len(candidate) > prose_room and kept:
            break
        kept.append(sentence)
    trimmed = " ".join(kept).strip()
    if not trimmed or len(trimmed) > max(0, prose_room):
        # Even the first body sentence overflows the room — word-trim it to fit the prose room.
        trimmed = truncate_at_word(sentences[0], max(0, prose_room))
    if citation and citation not in trimmed:
        trimmed = f"{trimmed} {citation}".strip()
    result = f"{heading}\n\n{trimmed}".strip()
    # Final guard: the post is GUARANTEED <=500 chars regardless of heading/body pathology, so the
    # downstream API truncation in post_to_threads never has to blind-cut (and dry-run matches prod).
    return result if len(result) <= THREADS_MAX_POST_CHARS else _trim_long_sentence(post, THREADS_MAX_POST_CHARS)


class ThreadsResearchRender(NamedTuple):
    """The rendered report plus what the rendering COST: `dropped` posts that exceeded the cap and
    were discarded, `trimmed` posts whose tail sentences were cut to fit 500 chars. Both used to be
    log-only, so the agent reported "Delivered the report" for a report the reader never saw in
    full. Unpacks as a 4-tuple, so every call site states which parts it wants."""

    root: str
    replies: list[str]
    dropped: int = 0
    trimmed: int = 0

    @property
    def rendered(self) -> int:
        return (1 if self.root else 0) + len(self.replies)


def render_threads_research(report: str, *, max_posts: int = 0) -> ThreadsResearchRender:
    """Render a Threads research report into a root post + flat reply chain, each <=500 chars.
    Slack mrkdwn is stripped (Threads renders none). The agent marks its own post boundaries with
    a line containing only '---', so a post's number + heading + body stay together; the renderer
    honors those boundaries and only re-splits a post that overflows 500 chars. If no delimiters
    are present (older output), it falls back to sentence packing.

    `max_posts` (>0) hard-caps the total post count (root + replies) so a too-long report can't
    fan out into dozens of public posts; excess posts are dropped. Returns the render plus the
    dropped/trimmed counts, so the caller can report an incomplete report as incomplete."""
    plain = strip_slack_mrkdwn(report).strip()
    # A leading "---" (delimiter as the report's first line) isn't matched by the split regex,
    # which requires a preceding newline; drop it so it can't ride into the first post as text.
    plain = _THREADS_LEADING_DELIMITER.sub("", plain).strip()

    delimited = bool(_THREADS_POST_DELIMITER.search(plain))
    # Each agent-delimited block is exactly ONE post: the heading and body stay together.
    raw_posts = [p.strip() for p in _THREADS_POST_DELIMITER.split(plain) if p.strip()] if delimited else []
    if not delimited:
        raw_posts = _pack_by_sentence(plain)

    if not raw_posts:
        # Empty/whitespace report → no post (caller skips delivery). Returning ("", []) here would
        # make post_to_threads create an empty TEXT container, which Meta's API rejects with a 400.
        return ThreadsResearchRender("", [])
    dropped = 0
    if max_posts > 0 and len(raw_posts) > max_posts:
        dropped = len(raw_posts) - max_posts
        logger.warning("Threads research: dropping %d post(s) over the cap of %d", dropped, max_posts)
        raw_posts = raw_posts[:max_posts]
    # Number AFTER the cap, so the published indices describe the thread the reader actually gets.
    if delimited:
        raw_posts = _renumber_threads_posts(raw_posts)
    # Trim LAST, so the index the renderer just added is inside the 500-char guarantee.
    posts = [_trim_oversize_post(p) for p in raw_posts]
    trimmed = sum(1 for before, after in zip(raw_posts, posts, strict=True) if before != after)
    return ThreadsResearchRender(posts[0], posts[1:], dropped, trimmed)
