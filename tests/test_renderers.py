from output.renderers import (
    _fit_one_post,
    render_threads_posts,
    render_threads_research,
)
from shared.constants import THREADS_MAX_POST_CHARS, SourceType
from shared.formatting import source_tag_and_metrics, strip_markup, threads_item_overhead_chars
from shared.models import CollectedItem, DigestContent, DigestItem


def _content(n_items: int = 2, lead: str = "오늘의 리드.") -> DigestContent:
    return DigestContent(
        lead=lead,
        headline_index=1,
        items=[
            DigestItem(
                title=f"스토리 {i}",
                url=f"http://e.com/{i}",
                source_tag=f"src{i}",
                metrics="👍 +10" if i == 1 else "",
                body=f"본문 {i} 입니다.",
                implication=f"시사점 {i}.",
            )
            for i in range(1, n_items + 1)
        ],
    )


class TestThreadsPosts:
    def test_exactly_one_reply_per_item(self):
        root, replies = render_threads_posts(_content(3))
        assert root == "오늘의 리드.\n\n· 스토리 2\n· 스토리 3"
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

    def test_source_line_never_publishes_an_emoji_shortcode(self):
        # Regression (2026-08-18, live post): metrics carried the shortcode `:arrow_forward:`, and the
        # markup strip removes `_`, so Threads published a bare ":arrowforward:". Metric emoji must
        # be real Unicode.
        from shared.formatting import YOUTUBE_VIEWS_LABEL

        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[
                DigestItem(
                    title="스토리",
                    url="http://e.com/x",
                    source_tag="YouTube",
                    metrics=f"{YOUTUBE_VIEWS_LABEL} 14,235",
                    body="본문이다.",
                )
            ],
        )
        _, replies = render_threads_posts(content)
        assert f"YouTube · {YOUTUBE_VIEWS_LABEL} 14,235" in replies[0]
        assert ":" not in replies[0].split("\n\n")[1]

    def test_reply_carries_the_source_line(self):
        # The Threads reply used to discard `source_tag`/`metrics` — a reader couldn't tell a Reddit
        # thread from an arXiv paper without opening the link.
        _, replies = render_threads_posts(_content(1))
        post = replies[0]
        assert "src1 · 👍 +10" in post
        assert "`" not in post

    def test_source_line_is_kept_even_when_the_body_must_be_trimmed(self):
        # Provenance is fixed, like the title and the URL: knowing the source is worth more than
        # one more clause of body.
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[
                DigestItem(
                    title="스토리",
                    url="http://e.com/x",
                    source_tag="r/LocalLLaMA",
                    metrics="👍 +44",
                    body="이것은 긴 본문 문장이다. " * 40,
                    implication="핵심 시사점이다.",
                )
            ],
        )
        _, replies = render_threads_posts(content)
        assert len(replies[0]) <= THREADS_MAX_POST_CHARS
        assert "r/LocalLLaMA · 👍 +44" in replies[0]

    def test_missing_source_metadata_adds_no_blank_line(self):
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[DigestItem(title="스토리", url="http://e.com/x", body="본문이다.", implication="시사점이다.")],
        )
        _, replies = render_threads_posts(content)
        # The implication is its OWN block: run into the body as one paragraph it read as more body,
        # so the closing beat each item lands on was invisible in the post.
        assert replies[0].split("\n\n") == ["스토리", "본문이다.", "시사점이다.", "http://e.com/x"]

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

    def test_only_the_implication_survives_when_no_body_sentence_fits(self):
        # Every body sentence is over-long on its own, so the loop empties: the implication block
        # alone is kept alongside the fixed parts (this branch had no coverage).
        body = "이것은 아주 긴 한 문장이다 " * 40 + "끝난다."
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[DigestItem(title="스토리", url="http://e.com/x", body=body, implication="시사점이다.")],
        )
        _, replies = render_threads_posts(content)
        post = replies[0]
        assert len(post) <= THREADS_MAX_POST_CHARS
        assert post.split("\n\n") == ["스토리", "시사점이다.", "http://e.com/x"]

    def test_body_is_word_trimmed_and_the_implication_dropped_when_neither_fits(self):
        # The implication itself overflows, so it goes and the body is word-trimmed into the room
        # that is left — never down to a bare title + URL.
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[
                DigestItem(
                    title="스토리",
                    url="http://e.com/x",
                    body="가나다 " * 300,
                    implication="시사점 " * 200,
                )
            ],
        )
        _, replies = render_threads_posts(content)
        post = replies[0]
        assert len(post) <= THREADS_MAX_POST_CHARS
        assert "시사점" not in post
        assert "가나다" in post and "http://e.com/x" in post

    def test_an_over_long_title_is_trimmed_as_the_last_resort(self):
        # Title + URL alone overflow: the title is word-trimmed and the URL kept intact, so the
        # reader can still reach the story.
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[DigestItem(title="아주 긴 제목 " * 80, url="http://e.com/x", body="", implication="")],
        )
        _, replies = render_threads_posts(content)
        post = replies[0]
        assert len(post) <= THREADS_MAX_POST_CHARS
        assert post.endswith("http://e.com/x")
        assert not post.split("\n\n")[0].endswith("아주 긴 제")  # word boundary, never mid-word

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

    def test_lead_trim_keeps_leading_sentences_so_the_gag_survives(self):
        gag = "AGI 등장 870일 전이다."
        lead = f"{gag} 첫 문장이다. " + "가나다라마바사 이것은 문장이다. " * 40 + "\n\n마지막 논평이다."
        root, _ = render_threads_posts(_content(1, lead=lead))
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert root.startswith(gag)

    def test_lead_trim_warns(self):
        # A trimmed lead means the editor overran its prose budget: visible, not silent.
        from unittest.mock import patch

        with patch("output.renderers.logger") as log:
            render_threads_posts(_content(1, lead="긴 문장이다. " * 100))
        assert log.warning.called

    def test_short_lead_is_untouched(self):
        from unittest.mock import patch

        with patch("output.renderers.logger") as log:
            root, _ = render_threads_posts(_content(1, lead="짧은 리드다.\n\nAGI 등장 870일 전이다."))
        assert root == "짧은 리드다.\n\nAGI 등장 870일 전이다."
        log.warning.assert_not_called()


class TestRootStoryIndex:
    def test_the_root_lists_the_other_stories_but_not_the_headline(self):
        root, _ = render_threads_posts(_content(5))
        lines = root.split("\n\n")[1].split("\n")
        assert lines == ["· 스토리 2", "· 스토리 3", "· 스토리 4", "· 스토리 5"]
        assert "스토리 1" not in root

    def test_titles_that_do_not_fit_drop_from_the_end_and_the_lead_is_kept(self):
        lead = "리드 문장이다. " * 53
        root, _ = render_threads_posts(_content(5, lead=lead.strip()))
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert root.startswith(lead.strip())
        assert "· 스토리 2" in root and "· 스토리 5" not in root

    def test_a_single_story_digest_has_no_list(self):
        root, _ = render_threads_posts(_content(1))
        assert root == "오늘의 리드."


class TestStripMarkup:
    def test_link_becomes_label_and_url(self):
        assert strip_markup("[엑스](https://x.com)") == "엑스 (https://x.com)"

    def test_image_link_keeps_only_the_url_when_the_alt_is_empty(self):
        assert strip_markup("![](https://x.com/a.png)") == "(https://x.com/a.png)"

    def test_bare_angle_link(self):
        assert strip_markup("<https://x.com>") == "https://x.com"

    def test_link_with_balanced_parens_in_the_url_is_kept_whole(self):
        out = strip_markup("[Foo](https://en.wikipedia.org/wiki/Foo_(bar)) 참고")
        assert out == "Foo (https://en.wikipedia.org/wiki/Foo_(bar)) 참고"

    def test_drops_bold_italic_code(self):
        assert strip_markup("**굵게** _기울임_ `코드`") == "굵게 기울임 코드"

    def test_strips_leading_bullets_and_headings(self):
        assert strip_markup("- 항목\n## 제목") == "항목\n제목"

    def test_preserves_underscores_in_linked_url(self):
        # Regression: the [*_`] strip must not corrupt URLs (arxiv/github/query params use '_').
        assert strip_markup("[논문](https://arxiv.org/abs/2_3_4)") == "논문 (https://arxiv.org/abs/2_3_4)"

    def test_preserves_underscores_in_bare_url(self):
        out = strip_markup("참고 https://github.com/a_b/c_d **굵게**")
        assert "https://github.com/a_b/c_d" in out
        assert "굵게" in out and "*" not in out


class TestThreadsResearch:
    def test_root_and_replies_under_cap(self):
        report = "\n\n".join(["문장 하나다. " * 20 for _ in range(4)])
        root, replies, *_ = render_threads_research(report)
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)
        assert len(replies) >= 1

    def test_strips_markdown_for_threads(self):
        root, replies, *_ = render_threads_research("리드 **굵게** 본문이다. [링크](https://x.com) 참고하라.")
        joined = root + " " + " ".join(replies)
        assert "*" not in joined
        assert "](" not in joined
        assert "링크 (https://x.com)" in joined

    def test_reports_dropped_and_trimmed_counts(self):
        # The renderer used to drop/trim silently, so the agent reported a truncated report as
        # "Delivered the report". The counts now ride back with the render.
        report = "\n---\n".join(f"{i}/30 소제목\n\n섹션 {i} 문장이다." for i in range(30))
        render = render_threads_research(report, max_posts=8)
        assert render.rendered == 8
        assert render.dropped == 22
        assert render.trimmed == 0

        oversize = "1/2 소제목\n\n" + "이것은 아주 긴 문장이다. " * 60 + "\n---\n2/2 짧다."
        render = render_threads_research(oversize, max_posts=8)
        assert render.trimmed == 1
        assert render.dropped == 0

    def test_complete_render_reports_nothing_lost(self):
        render = render_threads_research("1/2 소제목\n\n본문이다.\n---\n2/2 마무리다.", max_posts=8)
        assert (render.dropped, render.trimmed) == (0, 0)
        assert render.rendered == 2

    def test_max_posts_caps_total_count(self):
        # A long report must not fan out past the cap (root + replies <= max_posts).
        report = "\n\n".join(f"섹션 {i} 문장이다. " * 8 for i in range(30))
        root, replies, *_ = render_threads_research(report, max_posts=8)
        assert 1 + len(replies) <= 8

    def test_no_cap_when_max_posts_zero(self):
        report = "\n\n".join(f"섹션 {i} 문장이다. " * 8 for i in range(30))
        _, replies, *_ = render_threads_research(report, max_posts=0)
        assert 1 + len(replies) > 8  # uncapped

    def test_oversize_sentence_word_trimmed(self):
        report = "단어 " * 400  # one giant "sentence" with no terminator, >500 chars
        root, replies, *_ = render_threads_research(report)
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)

    def test_short_report_single_root_no_replies(self):
        root, replies, *_ = render_threads_research("짧은 보고서다.")
        assert root == "짧은 보고서다."
        assert replies == []

    def test_long_sections_stay_separate(self):
        # Two substantial sections (each above the root-merge threshold) stay as separate posts.
        a = "첫째 문단이다. " * 10
        b = "둘째 문단이다. " * 10
        root, replies, *_ = render_threads_research(f"{a}\n\n{b}")
        assert root.startswith("첫째")
        assert any(r.startswith("둘째") for r in replies)

    def test_agent_delimiters_define_posts(self):
        # When the agent marks boundaries with '---', each block is ONE post — number + heading +
        # body stay together, not split on the internal blank line.
        report = "1/2 첫 포스트 본문이다.\n\n부연 설명이다.\n---\n2/2 둘째 포스트 본문이다."
        root, replies, *_ = render_threads_research(report, max_posts=8)
        assert root.startswith("1/2")
        assert "부연 설명이다." in root  # stayed in the same post despite the blank line
        assert len(replies) == 1
        assert replies[0].startswith("2/2")

    def test_oversize_delimited_post_is_trimmed_to_one(self):
        # An agent-delimited post over 500 chars is TRIMMED to one post (heading kept), not fanned
        # out — so the 'N/M 소제목' line never orphans and the post count matches the agent's. The
        # index itself is the renderer's, computed from the final count, so "3/4" becomes "1/2".
        subheading = "마스터카드의 전략"
        big = f"3/4 {subheading}" + "\n\n" + "문장이다. " * 120  # >500 chars, multiple body sentences
        root, replies, *_ = render_threads_research(f"{big}\n---\n4/4 끝이다.", max_posts=8)
        assert root.startswith(f"1/2  {subheading}")  # heading stays attached to its body
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert len(replies) == 1  # exactly the agent's 2 posts → 1 root + 1 reply, no fan-out
        assert replies[0].startswith("2/2  끝이다.")
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)

    def test_long_sentence_preserves_trailing_url(self):
        # Regression: an over-length sentence ending in a citation URL must keep the URL.
        long = "이것은 매우 긴 문장이다 " * 30 + "출처는 https://arxiv.org/abs/2401.00001 이다"
        root, replies, *_ = render_threads_research(long)
        joined = root + " " + " ".join(replies)
        assert "https://arxiv.org/abs/2401.00001" in joined
        assert len(root) <= THREADS_MAX_POST_CHARS

    def test_oversize_delimited_post_with_huge_heading_stays_under_cap(self):
        # Regression: a delimited post whose HEADING line alone exceeds 500 chars must still be
        # trimmed to <=500 (the heading-only-overflow branch), never returned over-cap.
        huge_heading = "가" * 600
        root, replies, *_ = render_threads_research(f"{huge_heading}\n\n짧은 본문이다.\n---\n2/2 다음이다.")
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert all(len(r) <= THREADS_MAX_POST_CHARS for r in replies)

    def test_oversize_delimited_post_keeps_trailing_citation(self):
        # The delimited oversize path (_trim_oversize_post) must preserve a citation URL that sits
        # on the LAST body sentence even as earlier sentences are kept and the post is trimmed.
        url = "https://arxiv.org/abs/2406.12345"
        body = "이것은 본문 문장이다. " * 40 + f"핵심 출처는 {url} 이다."
        root, replies, *_ = render_threads_research(f"1/2 긴 섹션\n\n{body}\n---\n2/2 짧은 마무리다.")
        assert len(root) <= THREADS_MAX_POST_CHARS
        assert url in root  # citation on the trailing sentence survives the trim

    def test_leading_delimiter_does_not_contaminate_first_post(self):
        # A "---" as the report's very first line must be stripped, not baked into the root post.
        root, replies, *_ = render_threads_research("---\n1/2 첫 포스트다.\n\n본문이다.\n---\n2/2 둘째다.")
        assert not root.startswith("---")
        assert root.startswith("1/2")
        assert len(replies) == 1

    def test_empty_report_returns_empty_root_no_replies(self):
        # An empty/whitespace report must yield ("", []) so the caller skips the Threads API
        # (an empty TEXT container 400s). Previously this returned a stray empty root.
        root, replies, *_ = render_threads_research("   \n\n  ")
        assert root == ""
        assert replies == []


class TestThreadsSourceAttribution:
    """On Threads the source tag is the ONLY thing telling a Reddit thread from an arXiv paper, and
    it read wrong for two separate reasons. Both shipped in the 2026-08-18 digest."""

    def test_the_source_rides_in_parentheses_on_the_title_line(self):
        # It was a block of its own, so a publication name was left as a bare noun phrase alone on a
        # line, indistinguishable from a stray fragment.
        post = _fit_one_post("제목이다", "Simon Willison's Weblog", "본문이다.", "시사점이다.", "http://e.com/1")
        assert post.splitlines()[0] == "제목이다 (Simon Willison's Weblog)"
        assert "\n\nSimon Willison's Weblog\n\n" not in post

    def test_a_title_with_no_source_is_not_left_with_empty_parentheses(self):
        assert _fit_one_post("제목이다", "", "본문이다.", "", "").splitlines()[0] == "제목이다"

    def test_the_youtube_tag_names_the_channel_not_the_platform(self):
        # It was the literal "YouTube" for every video, so the reader learned nothing and the editor
        # compensated by appending the speaker to the TITLE instead — the source of the em-dash in
        # "AI를 키우는 건 ... — Ryan Greenblatt". `author` is the channelTitle the collector stores.
        item = CollectedItem(
            source_type=SourceType.YOUTUBE,
            title="t",
            url="https://www.youtube.com/watch?v=x",
            author="AI Engineer",
            metadata={"view_count": 775, "channel_url": "https://youtube.com/@x"},
        )
        tag, metrics = source_tag_and_metrics(item)
        assert tag == "AI Engineer"
        assert metrics == "조회 775"

    def test_a_youtube_item_with_no_author_keeps_the_platform_as_the_floor(self):
        # The RSS-fallback collection path stores no author.
        item = CollectedItem(
            source_type=SourceType.YOUTUBE, title="t", url="https://youtu.be/x", metadata={"channel_url": "u"}
        )
        assert source_tag_and_metrics(item)[0] == "YouTube"


class TestThreadsItemOverhead:
    """The per-item prose budget the editor is told about is derived from THIS number, so it has to
    match what the assembled post really spends on the parts code owns."""

    def test_counts_the_inline_source_line_url_and_separators(self):
        # "title (meta)" \n\n body \n\n implication \n\n url → 3 separators (6 chars) + url + the
        # meta and the " ()" that wraps it onto the title line, which costs no separator of its own.
        meta, url = "src · 조회 775", "http://e.com/1"
        assert threads_item_overhead_chars(meta, url) == len(meta) + 3 + len(url) + 6

    def test_no_source_line_costs_neither_the_text_nor_the_parentheses(self):
        assert threads_item_overhead_chars("", "http://e.com/1") == len("http://e.com/1") + 6

    def test_matches_the_real_assembly(self):
        # An item whose title+body+implication exactly fill the derived budget must not be trimmed.
        meta, url = "src1 · 👍 +10", "http://e.com/1"
        budget = THREADS_MAX_POST_CHARS - threads_item_overhead_chars(meta, url)
        title, implication = "제목", "시사점이다."
        body = "가" * (budget - len(title) - len(implication))
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[
                DigestItem(
                    title=title, url=url, source_tag="src1", metrics="👍 +10", body=body, implication=implication
                )
            ],
        )
        _, replies = render_threads_posts(content)
        assert len(replies[0]) == THREADS_MAX_POST_CHARS
        assert body in replies[0] and implication in replies[0]

    def test_trimmed_items_are_reported_by_count(self):
        from unittest.mock import patch

        # Counts only, never the text: a trimmed item means the prose budget was overrun.
        content = DigestContent(
            lead="리드.",
            headline_index=1,
            items=[
                DigestItem(title="스토리", url="http://e.com/x", body="긴 본문이다. " * 80, implication="시사점이다."),
                DigestItem(title="짧은 스토리", url="http://e.com/y", body="짧다.", implication="끝."),
            ],
        )
        with patch("output.renderers.logger") as log:
            render_threads_posts(content)
        args = log.warning.call_args.args
        assert args[1] == 1 and args[2] == 2
        assert not any("긴 본문이다" in str(a) for a in args)

    def test_nothing_reported_when_everything_fits(self):
        from unittest.mock import patch

        with patch("output.renderers.logger") as log:
            render_threads_posts(_content(3))
        log.warning.assert_not_called()


class TestThreadsResearchNumbering:
    """max_posts is applied AFTER the model has already written its "N/M" headings, so a capped report
    went out publicly as "1/8 ... 6/8" and stopped mid-argument — telling every reader two posts were
    missing. The renderer owns the index: it strips whatever the model wrote and re-derives it from
    the final post count."""

    def _report(self, count: int) -> str:
        return "\n---\n".join(f"{i}/{count} 소제목 {i}\n\n본문 {i}이다." for i in range(1, count + 1))

    def test_a_capped_report_is_renumbered_to_what_ships(self):
        root, replies, dropped, _ = render_threads_research(self._report(8), max_posts=3)
        assert dropped == 5
        assert root.startswith("1/3  소제목 1")
        assert [r.split("\n", 1)[0] for r in replies] == ["2/3  소제목 2", "3/3  소제목 3"]

    def test_an_uncapped_report_keeps_its_own_count(self):
        root, replies, dropped, _ = render_threads_research(self._report(3), max_posts=10)
        assert dropped == 0
        assert root.startswith("1/3  소제목 1")
        assert replies[-1].startswith("3/3  소제목 3")

    def test_an_unnumbered_report_still_gets_indices(self):
        # The prompt no longer asks the model to number Threads posts at all.
        report = "도입 소제목\n\n본문이다.\n---\n결론 소제목\n\n마무리다."
        root, replies, *_ = render_threads_research(report, max_posts=10)
        assert root.startswith("1/2  도입 소제목")
        assert replies[0].startswith("2/2  결론 소제목")

    def test_numbered_posts_stay_within_the_channel_cap(self):
        # The index is added BEFORE the oversize trim, so it can never push a post past 500 chars.
        report = "\n---\n".join(f"소제목 {i}\n\n" + "문장이다. " * 90 for i in range(1, 4))
        root, replies, *_ = render_threads_research(report, max_posts=10)
        assert all(len(post) <= 500 for post in [root, *replies])
        assert root.startswith("1/3  소제목 1")


class TestThreadsItemFit:
    def test_body_sentences_go_from_the_end_and_the_first_one_stays(self):
        from output.renderers import _fit_one_post

        body = "첫 문장은 핵심이다. " + " ".join(f"덧붙인 문장 {i}번이다." for i in range(60))
        post = _fit_one_post("제목", "src", body, "시사점이다.", "https://e.com/x")
        assert len(post) <= THREADS_MAX_POST_CHARS
        assert "첫 문장은 핵심이다." in post and "덧붙인 문장 59번이다." not in post

    def test_a_hard_cut_body_still_ends_with_the_whole_url(self):
        # With no word boundary the body is cut exactly to the room; a room that forgot the separator
        # overran 500 by two, and the publisher's own cut then chopped the link's last characters.
        from output.renderers import _fit_one_post

        url = "https://e.com/a-long-path-that-must-survive"
        post = _fit_one_post("제목", "src", "가" * 1000, "시" * 600, url)
        assert len(post) <= THREADS_MAX_POST_CHARS
        assert post.endswith(url)
