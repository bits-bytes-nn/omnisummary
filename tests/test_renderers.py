from output.renderers import (
    SLACK_MAX_BLOCKS_PER_MESSAGE,
    SLACK_MAX_SECTION_CHARS,
    THREADS_MAX_POST_CHARS,
    render_agent_blocks,
    render_slack_blocks,
    render_threads_posts,
)
from shared.models import DigestContent, DigestItem


def _content(n_items: int = 2, lead: str = "오늘의 리드.") -> DigestContent:
    return DigestContent(
        lead=lead,
        headline_index=1,
        items=[
            DigestItem(
                title=f"스토리 {i}",
                url=f"http://e.com/{i}",
                source_tag=f"`src{i}`",
                metrics="👍 +10" if i == 1 else "",
                body=f"본문 {i} 입니다.",
                implication=f"시사점 {i}.",
            )
            for i in range(1, n_items + 1)
        ],
    )


class TestSlackBlocks:
    def test_header_lead_and_item_blocks(self):
        chunks = render_slack_blocks(_content(2), header="HDR")
        blocks = chunks[0]
        assert blocks[0]["type"] == "header"
        assert blocks[1]["type"] == "section"  # lead
        types = [b["type"] for b in blocks]
        assert "divider" in types
        assert any(b.get("type") == "rich_text" for b in blocks)  # body quote

    def test_image_block_added_when_url_present(self):
        chunks = render_slack_blocks(_content(1), header="HDR", image_url="https://img/x.png")
        assert any(b["type"] == "image" and b["image_url"] == "https://img/x.png" for b in chunks[0])

    def test_title_links_and_implication_italic(self):
        chunks = render_slack_blocks(_content(1), header="HDR")
        texts = [b["text"]["text"] for b in chunks[0] if b["type"] == "section"]
        assert any("<http://e.com/1|스토리 1>" in t for t in texts)
        assert any(t.startswith("_") and t.endswith("_") for t in texts)

    def test_chunks_under_block_limit(self):
        chunks = render_slack_blocks(_content(40), header="HDR")
        assert all(len(c) <= SLACK_MAX_BLOCKS_PER_MESSAGE for c in chunks)
        assert len(chunks) > 1


class TestThreadsPosts:
    def test_exactly_one_reply_per_item(self):
        root, replies = render_threads_posts(_content(3))
        assert root == "오늘의 리드."
        assert len(replies) == 3  # one reply per item, never more
        assert "스토리 1" in replies[0] and "http://e.com/1" in replies[0]

    def test_long_lead_truncated_not_carried(self):
        # An overflowing lead is trimmed to one post; it does NOT spill into the reply chain.
        root, replies = render_threads_posts(_content(2, lead="첫 문장이다. " + "가나다라마바사 " * 100))
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert len(replies) == 2  # still one per item, no overflow replies

    def test_long_item_fits_one_post_at_sentence_boundary(self):
        # A body far over the cap must still be ONE reply, trimmed at a sentence end (not
        # mid-word), with the title and URL preserved.
        long_body = "이것은 한 문장이다. " * 80  # ~1000+ chars
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[DigestItem(title="긴 스토리", url="http://e.com/long", body=long_body, implication="시사점이다.")],
        )
        _, replies = render_threads_posts(content)
        assert len(replies) == 1
        post = replies[0]
        assert len(post) <= THREADS_MAX_POST_CHARS
        assert "긴 스토리" in post  # title kept
        assert "http://e.com/long" in post  # url kept and intact
        # body portion ends at a clean sentence boundary (no mid-word cut)
        body_line = [ln for ln in post.split("\n\n") if "문장" in ln][0]
        assert body_line.rstrip().endswith("다.")

    def test_all_posts_within_cap(self):
        root, replies = render_threads_posts(_content(3, lead="나" * 700))
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)

    def test_unterminated_body_word_trimmed_not_dropped(self):
        # A long body with NO sentence boundary must be word-trimmed into the post (keeping
        # title + URL), not dropped entirely down to title+URL.
        body = "가나다 " * 300  # ~1200 chars, no sentence-ending punctuation
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[DigestItem(title="스토리", url="http://e.com/x", body=body)],
        )
        _, replies = render_threads_posts(content)
        assert len(replies) == 1
        post = replies[0]
        assert len(post) <= THREADS_MAX_POST_CHARS
        assert "스토리" in post and "http://e.com/x" in post
        assert "가나다" in post  # body present, not dropped
        assert not post.split("\n\n")[1].endswith("가나")  # trimmed on a word boundary

    def test_unterminated_long_lead_word_trimmed_not_mid_word(self):
        root, _ = render_threads_posts(_content(1, lead="가나다라 " * 200))  # no sentence end
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert not root.endswith("가나")  # cut on a space, never mid-word


class TestAgentBlocks:
    def test_wraps_text_in_section(self):
        chunks = render_agent_blocks("안녕하세요 *굵게* 답변입니다.")
        assert chunks[0][0]["type"] == "section"
        assert chunks[0][0]["text"]["text"] == "안녕하세요 *굵게* 답변입니다."

    def test_empty_text_returns_empty_chunk(self):
        assert render_agent_blocks("") == [[]]

    def test_long_text_split_into_section_sized_blocks(self):
        text = "\n\n".join(["가" * 2000, "나" * 2000, "다" * 2000])
        chunks = render_agent_blocks(text)
        all_blocks = [b for c in chunks for b in c]
        assert all(len(b["text"]["text"]) <= SLACK_MAX_SECTION_CHARS for b in all_blocks)

    def test_single_oversized_paragraph_is_hard_split(self):
        chunks = render_agent_blocks("x" * (SLACK_MAX_SECTION_CHARS * 2 + 50))
        all_blocks = [b for c in chunks for b in c]
        assert all(len(b["text"]["text"]) <= SLACK_MAX_SECTION_CHARS for b in all_blocks)
        assert len(all_blocks) >= 3
