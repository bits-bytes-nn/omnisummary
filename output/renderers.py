from __future__ import annotations

import re

from shared import DigestContent, logger

# Slack caps a single message at 50 blocks; chunk item blocks across messages under it.
SLACK_MAX_BLOCKS_PER_MESSAGE = 45
# A single Slack section's text field is capped at 3000 chars.
SLACK_MAX_SECTION_CHARS = 2900
# Threads caps each post at 500 characters.
THREADS_MAX_POST_CHARS = 500


def _split_long_paragraph(para: str, max_len: int) -> list[str]:
    """Break an over-length paragraph into <=max_len pieces at sentence then word boundaries,
    never inside a Slack `<url|text>` link span (a raw stride-split would cleave a link across
    two messages, leaving both halves as dead raw text)."""
    pieces: list[str] = []
    current = ""
    for sentence in _sentences(para) or [para]:
        unit = sentence if len(sentence) <= max_len else _truncate_at_word(sentence, max_len)
        # If the sentence itself overflows even after word-trim is impossible without a space,
        # fall back to a link-safe hard split of the raw sentence.
        units = [unit] if len(sentence) <= max_len else _hard_split_link_safe(sentence, max_len)
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
    """Last-resort hard split that never cuts inside a `<...>` link span: extend a cut point
    forward to the closing `>` if it would land inside an unbalanced `<`."""
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        end = min(i + max_len, n)
        open_idx = text.rfind("<", i, end)
        close_idx = text.rfind(">", i, end)
        if open_idx > close_idx:  # cut lands inside a link span — extend to its closing '>'
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


_SENTENCE_END = ("다.", "다!", "다?", ". ", "。", "! ", "? ", "…")


def _sentences(text: str) -> list[str]:
    """Split prose into sentences without losing characters, breaking only AFTER a
    sentence-ending boundary (Korean '다.' / '?' / '!' or '. '). Whitespace-only tails
    are dropped. Used so a post is trimmed at a clean sentence, never mid-word."""
    out: list[str] = []
    start = 0
    i = 0
    n = len(text)
    while i < n:
        matched = next((e for e in _SENTENCE_END if text.startswith(e, i)), None)
        if matched:
            end = i + len(matched)
            out.append(text[start:end].strip())
            start = end
            i = end
        else:
            i += 1
    if text[start:].strip():
        out.append(text[start:].strip())
    return [s for s in out if s]


def _truncate_at_word(text: str, max_len: int) -> str:
    """Trim text to <=max_len on a whitespace boundary (never mid-word); if there's no space
    in range, fall back to a hard character cut. Used only when prose has no sentence boundary."""
    text = text.strip()
    if len(text) <= max_len:
        return text
    window = text[:max_len]
    cut = window.rfind(" ")
    return (window[:cut] if cut > 0 else window).rstrip()


def _fit_one_post(title: str, body: str, implication: str, url: str, max_len: int = THREADS_MAX_POST_CHARS) -> str:
    """Build ONE Threads post for an item that fits within max_len. Title and URL are always
    kept; the implication (the voice line) is preserved over body — body sentences are dropped
    from the end first, and the implication only goes if title+implication+URL still overflow.
    Nothing is cut mid-sentence and the link is never split. Each item maps to exactly one reply."""
    fixed = [p for p in (title.strip(),) if p]
    tail = [url.strip()] if url.strip() else []
    impl = implication.strip()

    def assemble(prose: str) -> str:
        return "\n\n".join(fixed + ([prose] if prose else []) + tail)

    body_sents = _sentences(body)
    # Drop body sentences from the end while keeping the implication appended.
    while body_sents:
        prose = " ".join(body_sents + ([impl] if impl else []))
        if len(assemble(prose)) <= max_len:
            return assemble(prose)
        body_sents.pop()
    # No body sentence fits alongside the implication. Keep the implication alone if it fits.
    if impl and len(assemble(impl)) <= max_len:
        return assemble(impl)

    # Even the implication won't fit. Word-trim the body into the remaining room (never drop
    # it to bare title+URL), or word-trim the title if title+URL alone overflow.
    room = max_len - len(assemble("")) - 2
    if room > 0 and body.strip():
        return assemble(_truncate_at_word(body, room))
    if len(assemble("")) <= max_len:
        return assemble("")
    room = max_len - (len(url.strip()) + 2 if url.strip() else 0)
    return "\n\n".join([p for p in (_truncate_at_word(title, max(0, room)), url.strip()) if p])


def render_threads_posts(content: DigestContent) -> tuple[str, list[str]]:
    """Render DigestContent for Threads: a root text (the lead) and a reply chain with
    EXACTLY ONE reply per item (title + body + implication + URL). Each reply is trimmed to
    fit Threads' 500-char cap at a clean sentence boundary — never mid-word — keeping the
    title and URL. No Slack markup (Threads renders none)."""
    lead = content.lead.strip()
    if len(lead) > THREADS_MAX_POST_CHARS:
        kept = _sentences(lead)
        while kept and len(" ".join(kept)) > THREADS_MAX_POST_CHARS:
            kept.pop()
        # No sentence boundary in range → word-trim, never a mid-word slice.
        lead = " ".join(kept) if kept else _truncate_at_word(lead, THREADS_MAX_POST_CHARS)

    replies = [_fit_one_post(item.title, item.body, item.implication or "", item.url) for item in content.items]
    return lead, replies


_URL_RE = re.compile(r"https?://\S+")


def _strip_slack_mrkdwn(text: str) -> str:
    """Convert Slack mrkdwn to plain text for Threads (which renders no markup): turn
    <url|label> into 'label (url)', drop *bold*/_italic_/`code` markers, and remove
    leading bullet/heading glyphs. URLs are protected from the marker strip — they
    legitimately contain '_'/'*' (arxiv, github, query params), so stripping those
    characters globally would silently break the links. Whitespace structure is preserved."""
    text = re.sub(r"<([^|>]+)\|([^>]+)>", r"\2 (\1)", text)
    text = re.sub(r"<([^>]+)>", r"\1", text)

    # Stash URLs so the [*_`] strip below can't corrupt them, then restore verbatim.
    urls: list[str] = []

    def _stash(match: re.Match) -> str:
        urls.append(match.group(0))
        return f"\x00{len(urls) - 1}\x00"

    text = _URL_RE.sub(_stash, text)
    # Strip leading bullet/heading glyphs FIRST (so a "* 항목" or "- 항목" bullet is removed as a
    # unit), THEN drop inline *bold*/_italic_/`code` markers.
    out_lines = [re.sub(r"^\s*(?:[-*•]\s+|#{1,6}\s+)", "", line) for line in text.split("\n")]
    text = re.sub(r"[*_`]", "", "\n".join(out_lines))
    return re.sub(r"\x00(\d+)\x00", lambda m: urls[int(m.group(1))], text)


def _trim_long_sentence(sentence: str, max_len: int) -> str:
    """Word-trim an over-length sentence to fit max_len, but PRESERVE a trailing citation URL
    (research sentences often end in '... (https://...)'). Reserve room for the last URL, trim
    the prose before it, then re-append the URL — so a hard-trim never drops the citation."""
    urls = _URL_RE.findall(sentence)
    if not urls:
        return _truncate_at_word(sentence, max_len)
    tail = urls[-1].rstrip(").,")
    if len(tail) >= max_len:  # the URL alone overflows — nothing useful to keep but the URL
        return tail[:max_len]
    prose = sentence[: sentence.rfind(tail)].rstrip(" (")
    trimmed = _truncate_at_word(prose, max_len - len(tail) - 1)
    return f"{trimmed} {tail}".strip()


# The agent separates Threads posts with a line containing only this delimiter, so post
# boundaries are the AGENT's choice (number + heading + body stay in ONE post) rather than
# the renderer guessing from blank lines.
_THREADS_POST_DELIMITER = re.compile(r"\n\s*---\s*\n")
# A delimiter line at the very START of the report (no preceding newline) the split regex can't
# see — strip it so a leading "---" never contaminates the first post as literal text.
_THREADS_LEADING_DELIMITER = re.compile(r"^\s*---\s*\n")


def _pack_by_sentence(text: str) -> list[str]:
    """Fallback packer when the agent gave no explicit post delimiters: greedily pack sentences
    into <=500-char posts at sentence boundaries (never mid-word)."""
    posts: list[str] = []
    current = ""
    for section in [s.strip() for s in text.split("\n\n") if s.strip()]:
        for sentence in _sentences(section) or [section]:
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
    body_urls = _URL_RE.findall(body)
    citation = body_urls[-1].rstrip(").,") if body_urls else ""
    prose_room = room - (len(citation) + 1) if citation else room
    sentences = _sentences(body) or [body]
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
        trimmed = _truncate_at_word(sentences[0], max(0, prose_room))
    if citation and citation not in trimmed:
        trimmed = f"{trimmed} {citation}".strip()
    result = f"{heading}\n\n{trimmed}".strip()
    # Final guard: the post is GUARANTEED <=500 chars regardless of heading/body pathology, so the
    # downstream API truncation in post_to_threads never has to blind-cut (and dry-run matches prod).
    return result if len(result) <= THREADS_MAX_POST_CHARS else _trim_long_sentence(post, THREADS_MAX_POST_CHARS)


def render_threads_research(report: str, *, max_posts: int = 0) -> tuple[str, list[str]]:
    """Render a Threads research report into a root post + flat reply chain, each <=500 chars.
    Slack mrkdwn is stripped (Threads renders none). The agent marks its own post boundaries with
    a line containing only '---', so a post's number + heading + body stay together; the renderer
    honors those boundaries and only re-splits a post that overflows 500 chars. If no delimiters
    are present (older output), it falls back to sentence packing.

    `max_posts` (>0) hard-caps the total post count (root + replies) so a too-long report can't
    fan out into dozens of public posts; excess posts are dropped. Returns (root_text, replies)."""
    plain = _strip_slack_mrkdwn(report).strip()
    # A leading "---" (delimiter as the report's first line) isn't matched by the split regex,
    # which requires a preceding newline; drop it so it can't ride into the first post as text.
    plain = _THREADS_LEADING_DELIMITER.sub("", plain).strip()

    if _THREADS_POST_DELIMITER.search(plain):
        raw_posts = [p.strip() for p in _THREADS_POST_DELIMITER.split(plain) if p.strip()]
        # Each agent-delimited block is exactly ONE post: keep the heading + body together and only
        # trim (never fan out) when it overflows, so the 'N/M 소제목' line never orphans.
        posts = [_trim_oversize_post(p) for p in raw_posts]
    else:
        posts = _pack_by_sentence(plain)

    if not posts:
        # Empty/whitespace report → no post (caller skips delivery). Returning ("", []) here would
        # make post_to_threads create an empty TEXT container, which Meta's API rejects with a 400.
        return "", []
    if max_posts > 0 and len(posts) > max_posts:
        logger.info("Threads research: %d posts exceed cap %d, truncating", len(posts), max_posts)
        posts = posts[:max_posts]
    return posts[0], posts[1:]
