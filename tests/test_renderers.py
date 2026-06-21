from output.renderers import (
    SLACK_MAX_BLOCKS_PER_MESSAGE,
    SLACK_MAX_SECTION_CHARS,
    THREADS_MAX_POST_CHARS,
    _strip_slack_mrkdwn,
    render_agent_blocks,
    render_research_blocks,
    render_slack_blocks,
    render_threads_posts,
    render_threads_research,
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

    def test_implication_preserved_over_body_when_trimming(self):
        # When body + implication don't both fit, body sentences are dropped first; the
        # implication (the voice line) must survive.
        long_body = "이것은 긴 본문 문장이다. " * 40  # ~600+ chars
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[
                DigestItem(title="스토리", url="http://e.com/x", body=long_body, implication="이것이 핵심 시사점이다.")
            ],
        )
        _, replies = render_threads_posts(content)
        assert len(replies) == 1
        post = replies[0]
        assert len(post) <= THREADS_MAX_POST_CHARS
        assert "이것이 핵심 시사점이다." in post  # implication kept even as body is trimmed away

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

    def test_empty_text_returns_no_chunks(self):
        # Empty input → no chunks so callers post nothing (never blocks=[], which Slack rejects).
        assert render_agent_blocks("") == []
        assert render_agent_blocks("   \n\n  ") == []

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

    def test_oversize_paragraph_never_splits_a_link(self):
        # A long paragraph with a Slack link straddling the section boundary must not be cleaved:
        # no produced block may contain a '<' without its matching '>'.
        link = "<https://example.com/very/long/path/with_underscores|클릭하세요>"
        para = "가" * (SLACK_MAX_SECTION_CHARS - 10) + link + "나" * 200
        chunks = render_agent_blocks(para)
        for c in chunks:
            for b in c:
                t = b["text"]["text"]
                assert t.count("<") == t.count(">")  # balanced — no half-link


class TestResearchBlocks:
    def test_first_block_is_header(self):
        chunks = render_research_blocks("본문이다.", header=":satellite: Deep Research")
        assert chunks[0][0]["type"] == "header"
        assert chunks[0][0]["text"]["text"] == ":satellite: Deep Research"

    def test_divider_before_numbered_heading(self):
        report = "도입부 문장이다.\n\n*1. 첫 섹션*\n\n첫 섹션 본문이다.\n\n*2. 둘째 섹션*\n\n둘째 본문이다."
        blocks = [b for c in render_research_blocks(report, header="H") for b in c]
        types = [b["type"] for b in blocks]
        # header first, then a divider precedes each "*N. ...*" heading
        assert types[0] == "header"
        assert types.count("divider") == 2

    def test_sections_under_char_cap(self):
        report = "\n\n".join(["가" * 2500, "*1. 섹션*", "나" * 2500])
        blocks = [b for c in render_research_blocks(report, header="H") for b in c]
        for b in blocks:
            if b["type"] == "section":
                assert len(b["text"]["text"]) <= SLACK_MAX_SECTION_CHARS

    def test_empty_report_still_has_header(self):
        chunks = render_research_blocks("   ", header="H")
        assert chunks[0][0]["type"] == "header"


class TestStripSlackMrkdwn:
    def test_link_becomes_label_and_url(self):
        assert _strip_slack_mrkdwn("<https://x.com|엑스>") == "엑스 (https://x.com)"

    def test_bare_angle_link(self):
        assert _strip_slack_mrkdwn("<https://x.com>") == "https://x.com"

    def test_drops_bold_italic_code(self):
        assert _strip_slack_mrkdwn("*굵게* _기울임_ `코드`") == "굵게 기울임 코드"

    def test_strips_leading_bullets_and_headings(self):
        assert _strip_slack_mrkdwn("- 항목\n## 제목") == "항목\n제목"

    def test_preserves_underscores_in_linked_url(self):
        # Regression: the [*_`] strip must not corrupt URLs (arxiv/github/query params use '_').
        assert _strip_slack_mrkdwn("<https://arxiv.org/abs/2_3_4|논문>") == "논문 (https://arxiv.org/abs/2_3_4)"

    def test_preserves_underscores_in_bare_url(self):
        out = _strip_slack_mrkdwn("참고 https://github.com/a_b/c_d *굵게*")
        assert "https://github.com/a_b/c_d" in out
        assert "굵게" in out and "*" not in out


class TestThreadsResearch:
    def test_root_and_replies_under_cap(self):
        report = "\n\n".join(["문장 하나다. " * 20 for _ in range(4)])
        root, replies = render_threads_research(report)
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)
        assert len(replies) >= 1

    def test_strips_markdown_for_threads(self):
        root, replies = render_threads_research("리드 *굵게* 본문이다. <https://x.com|링크> 참고하라.")
        joined = root + " " + " ".join(replies)
        assert "*" not in joined
        assert "<https" not in joined
        assert "링크 (https://x.com)" in joined

    def test_max_posts_caps_total_count(self):
        # A long report must not fan out past the cap (root + replies <= max_posts).
        report = "\n\n".join(f"섹션 {i} 문장이다. " * 8 for i in range(30))
        root, replies = render_threads_research(report, max_posts=8)
        assert 1 + len(replies) <= 8
        # no Slack pointer — the Threads post stands on its own
        assert "Slack" not in (root + " ".join(replies))

    def test_no_cap_when_max_posts_zero(self):
        report = "\n\n".join(f"섹션 {i} 문장이다. " * 8 for i in range(30))
        _, replies = render_threads_research(report, max_posts=0)
        assert 1 + len(replies) > 8  # uncapped

    def test_oversize_sentence_word_trimmed(self):
        report = "단어 " * 400  # one giant "sentence" with no terminator, >500 chars
        root, replies = render_threads_research(report)
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)

    def test_short_report_single_root_no_replies(self):
        root, replies = render_threads_research("짧은 보고서다.")
        assert root == "짧은 보고서다."
        assert replies == []

    def test_long_sections_stay_separate(self):
        # Two substantial sections (each above the root-merge threshold) stay as separate posts.
        a = "첫째 문단이다. " * 10
        b = "둘째 문단이다. " * 10
        root, replies = render_threads_research(f"{a}\n\n{b}")
        assert root.startswith("첫째")
        assert any(r.startswith("둘째") for r in replies)

    def test_agent_delimiters_define_posts(self):
        # When the agent marks boundaries with '---', each block is ONE post — number + heading +
        # body stay together, not split on the internal blank line.
        report = "1/2 첫 포스트 본문이다.\n\n부연 설명이다.\n---\n2/2 둘째 포스트 본문이다."
        root, replies = render_threads_research(report, max_posts=8)
        assert root.startswith("1/2")
        assert "부연 설명이다." in root  # stayed in the same post despite the blank line
        assert len(replies) == 1
        assert replies[0].startswith("2/2")

    def test_oversize_delimited_post_is_trimmed_to_one(self):
        # An agent-delimited post over 500 chars is TRIMMED to one post (heading kept), not fanned
        # out — so the 'N/M 소제목' line never orphans and the post count matches the agent's.
        heading = "3/4 마스터카드의 전략"
        big = heading + "\n\n" + "문장이다. " * 120  # >500 chars, multiple body sentences
        root, replies = render_threads_research(f"{big}\n---\n4/4 끝이다.", max_posts=8)
        assert root.startswith(heading)  # heading stays attached to its body, not orphaned
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert len(replies) == 1  # exactly the agent's 2 posts → 1 root + 1 reply, no fan-out
        assert replies[0].startswith("4/4")
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)

    def test_long_sentence_preserves_trailing_url(self):
        # Regression: an over-length sentence ending in a citation URL must keep the URL.
        long = "이것은 매우 긴 문장이다 " * 30 + "출처는 https://arxiv.org/abs/2401.00001 이다"
        root, replies = render_threads_research(long)
        joined = root + " " + " ".join(replies)
        assert "https://arxiv.org/abs/2401.00001" in joined
        assert len(root) <= THREADS_MAX_POST_CHARS

    def test_oversize_delimited_post_with_huge_heading_stays_under_cap(self):
        # Regression: a delimited post whose HEADING line alone exceeds 500 chars must still be
        # trimmed to <=500 (the heading-only-overflow branch), never returned over-cap.
        huge_heading = "가" * 600
        root, replies = render_threads_research(f"{huge_heading}\n\n짧은 본문이다.\n---\n2/2 다음이다.")
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)

    def test_oversize_delimited_post_keeps_trailing_citation(self):
        # The delimited oversize path (_trim_oversize_post) must preserve a citation URL that sits
        # on the LAST body sentence even as earlier sentences are kept and the post is trimmed.
        url = "https://arxiv.org/abs/2406.12345"
        body = "이것은 본문 문장이다. " * 40 + f"핵심 출처는 {url} 이다."
        root, replies = render_threads_research(f"1/2 긴 섹션\n\n{body}\n---\n2/2 짧은 마무리다.")
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert url in root  # citation on the trailing sentence survives the trim

    def test_leading_delimiter_does_not_contaminate_first_post(self):
        # A "---" as the report's very first line must be stripped, not baked into the root post.
        root, replies = render_threads_research("---\n1/2 첫 포스트다.\n\n본문이다.\n---\n2/2 둘째다.")
        assert not root.startswith("---")
        assert root.startswith("1/2")
        assert len(replies) == 1

    def test_empty_report_returns_empty_root_no_replies(self):
        # An empty/whitespace report must yield ("", []) so the caller skips the Threads API
        # (an empty TEXT container 400s). Previously this returned a stray empty root.
        root, replies = render_threads_research("   \n\n  ")
        assert root == ""
        assert replies == []
