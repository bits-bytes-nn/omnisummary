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
    def test_root_is_lead_replies_per_item(self):
        root, replies = render_threads_posts(_content(2))
        assert root == "오늘의 리드."
        assert len(replies) == 2
        assert "스토리 1" in replies[0] and "http://e.com/1" in replies[0]

    def test_long_lead_overflow_carried_to_replies(self):
        root, replies = render_threads_posts(_content(1, lead="가" * 1200))
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert len(replies) >= 2  # overflow + the one item

    def test_all_posts_within_cap(self):
        root, replies = render_threads_posts(_content(3, lead="나" * 700))
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)


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
