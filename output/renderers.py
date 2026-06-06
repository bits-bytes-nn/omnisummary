from __future__ import annotations

from shared import DigestContent

# Slack caps a single message at 50 blocks; chunk item blocks across messages under it.
SLACK_MAX_BLOCKS_PER_MESSAGE = 45
# Threads caps each post at 500 characters.
THREADS_MAX_POST_CHARS = 500


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


def _split_to_limit(text: str, max_len: int = THREADS_MAX_POST_CHARS) -> list[str]:
    """Split plain text into <=max_len posts on sentence/space boundaries when possible."""
    text = text.strip()
    if len(text) <= max_len:
        return [text] if text else []
    chunks: list[str] = []
    remaining = text
    while len(remaining) > max_len:
        window = remaining[:max_len]
        cut = max(window.rfind(". "), window.rfind("。"), window.rfind("\n"), window.rfind(" "))
        if cut <= 0:
            cut = max_len
        else:
            cut += 1
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    if remaining:
        chunks.append(remaining)
    return chunks


def render_threads_posts(content: DigestContent) -> tuple[str, list[str]]:
    """Render DigestContent for Threads: a root text (the lead, capped to one post) and a
    reply chain — one reply per item (title + body + implication + URL), each plain text and
    split if it exceeds the 500-char cap. No Slack markup (Threads renders none)."""
    root_chunks = _split_to_limit(content.lead)
    root = root_chunks[0] if root_chunks else content.lead[:THREADS_MAX_POST_CHARS]

    replies: list[str] = []
    # If the lead overflowed one post, carry the remainder as the first reply(ies).
    replies.extend(root_chunks[1:])

    for item in content.items:
        parts = [item.title, item.body]
        if item.implication:
            parts.append(item.implication)
        if item.url:
            parts.append(item.url)
        post = "\n\n".join(p for p in parts if p)
        replies.extend(_split_to_limit(post))
    return root, replies
