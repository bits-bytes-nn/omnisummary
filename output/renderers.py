from __future__ import annotations

from shared import DigestContent

# Slack caps a single message at 50 blocks; chunk item blocks across messages under it.
SLACK_MAX_BLOCKS_PER_MESSAGE = 45
# A single Slack section's text field is capped at 3000 chars.
SLACK_MAX_SECTION_CHARS = 2900
# Threads caps each post at 500 characters.
THREADS_MAX_POST_CHARS = 500


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
            for i in range(0, len(para), SLACK_MAX_SECTION_CHARS):
                sections.append(para[i : i + SLACK_MAX_SECTION_CHARS])
        elif len(current) + len(para) + 2 > SLACK_MAX_SECTION_CHARS:
            sections.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        sections.append(current)

    blocks = [{"type": "section", "text": {"type": "mrkdwn", "text": s}} for s in sections]
    return _chunk_blocks(blocks) if blocks else [[]]


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
        blocks.append(
            {
                "type": "rich_text",
                "elements": [
                    {
                        "type": "rich_text_quote",
                        "elements": [{"type": "text", "text": item.body}],
                    }
                ],
            }
        )

    if item.implication:
        blocks.append({"type": "section", "text": {"type": "mrkdwn", "text": f"_{item.implication}_"}})
    return blocks


def render_slack_blocks(
    content: DigestContent, *, header: str, image_url: str = "", image_alt: str = ""
) -> list[list[dict]]:
    """Render DigestContent as Slack Block Kit, split into <=SLACK_MAX_BLOCKS_PER_MESSAGE
    chunks. Returns a list of block-lists, one per chat_postMessage call."""
    blocks: list[dict] = [
        {"type": "header", "text": {"type": "plain_text", "text": header, "emoji": True}},
        {"type": "section", "text": {"type": "mrkdwn", "text": content.lead}},
    ]
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
    """Build ONE Threads post for an item that fits within max_len — title and URL are always
    kept; body/implication sentences are dropped from the end until it fits, never cut
    mid-sentence and the link is never split. If even one sentence won't fit, the body is
    word-trimmed (not dropped whole). Each item maps to exactly one reply."""
    fixed = [p for p in (title.strip(),) if p]
    tail = [url.strip()] if url.strip() else []

    def assemble(body_text: str) -> str:
        return "\n\n".join(fixed + ([body_text] if body_text else []) + tail)

    sentences = _sentences(body) + (_sentences(implication) if implication else [])
    while sentences and len(assemble(" ".join(sentences))) > max_len:
        sentences.pop()
    if sentences:
        return assemble(" ".join(sentences))

    # No whole sentence fits. Keep title + URL, and fill remaining room with a word-trimmed
    # slice of the body rather than dropping it entirely.
    room = max_len - len(assemble("")) - 2
    if room > 0 and body.strip():
        return assemble(_truncate_at_word(body, room))
    post = assemble("")
    if len(post) <= max_len:
        return post
    # Title + URL alone overflow (rare): word-trim the title, keep the URL intact.
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
