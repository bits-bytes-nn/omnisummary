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
    strip_markup,
    threads_meta_line,
    truncate_at_word,
)

# Marks each of the other stories' titles in the root post.
STORY_INDEX_BULLET = "· "

__all__ = [
    "render_threads_posts",
    "render_threads_research",
]


def _fit_one_post(
    title: str, meta: str, body: str, implication: str, url: str, max_len: int = THREADS_MAX_POST_CHARS
) -> str:
    """Build ONE Threads post for an item that fits within max_len. Title, the source line and the
    URL are always kept; the implication (the voice line) is preserved over body — body sentences
    are dropped from the end first, and the implication only goes if the fixed parts still overflow.
    Nothing is cut mid-sentence and the link is never split. Each item maps to exactly one reply.

    `meta` is the "r/LocalLLaMA · 조회 775" provenance line the pipeline already computes. Without
    it a reader could not tell a Reddit thread from an arXiv paper without opening the link, so it
    is treated as fixed: knowing the source is worth more than one more clause of body.

    It rides in PARENTHESES on the title line rather than as a block of its own: on its own line a
    publication name ("Simon Willison's Weblog") read like a stray fragment. Parentheses are a
    self-evident attribution marker in Korean, they work for all five source shapes without a label
    word, and merging the two blocks saves a separator."""
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


def _with_story_index(lead: str, titles: list[str], countdown: str = "") -> str:
    """The root text with the day's other story titles listed under the lead's prose (above a suffix
    countdown sign-off). The story replies are read by 2-5% of the root's viewers, so without this a
    feed reader never learned what else the digest carried. Titles that do not fit the cap are
    dropped from the end; the lead itself is never trimmed for the list."""
    head, sep, last_line = lead.rpartition("\n")
    gag = countdown.strip()
    sign_off = last_line.strip() if sep and gag and last_line.strip() == gag else ""
    prose = head.strip() if sign_off else lead
    lines = [f"{STORY_INDEX_BULLET}{title.strip()}" for title in titles if title.strip()]
    for kept in range(len(lines), 0, -1):
        root = THREADS_POST_SEPARATOR.join(p for p in (prose, "\n".join(lines[:kept]), sign_off) if p)
        if len(root) <= THREADS_MAX_POST_CHARS:
            if kept < len(lines):
                logger.info(
                    "Threads root lists %d of %d other stories (cap %d)", kept, len(lines), THREADS_MAX_POST_CHARS
                )
            return root
    return lead


def render_threads_posts(content: DigestContent, countdown: str = "") -> tuple[str, list[str]]:
    """Render DigestContent for Threads: a root text (the lead, then the other stories' titles) and a
    reply chain with EXACTLY ONE reply per item (title + source line + body + implication + URL). Each
    reply is trimmed to fit Threads' 500-char cap at a clean sentence boundary — never mid-word —
    keeping the title, the source line and the URL. No markup (Threads renders none).

    `countdown` is the code-owned AGI-countdown gag as the pipeline computed it, so an over-long
    lead can drop THAT rather than the day's argument. Empty (the default) simply trims prose."""
    lead = _with_story_index(
        _fit_lead(content.lead.strip(), countdown), [item.title for item in content.items[1:]], countdown
    )
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
    Markup is stripped (Threads renders none). The agent marks its own post boundaries with
    a line containing only '---', so a post's number + heading + body stay together; the renderer
    honors those boundaries and only re-splits a post that overflows 500 chars. If no delimiters
    are present (older output), it falls back to sentence packing.

    `max_posts` (>0) hard-caps the total post count (root + replies) so a too-long report can't
    fan out into dozens of public posts; excess posts are dropped. Returns the render plus the
    dropped/trimmed counts, so the caller can report an incomplete report as incomplete."""
    plain = strip_markup(report).strip()
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
