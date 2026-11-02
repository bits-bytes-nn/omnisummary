"""Generator + renderers: hidden-link research items, the 500-char cap with footer and countdown, the countdown."""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

from application.digest_generator import DigestGenerator, render_digest_text
from domain.formatting import countdown_intro, place_countdown_intro
from domain.models import CollectedItem, DigestContent, DigestItem, RankedItem
from domain.renderers import render_threads_posts
from shared import THREADS_MAX_POST_CHARS, THREADS_POST_SEPARATOR
from shared.config import PipelineConfig
from shared.constants import SourceType
from tests.edge.edge_harness import ScriptedModels, editor_picking, no_violations

FOOTER = "투자 권유가 아닙니다. 투자 판단의 책임은 본인에게 있습니다."
TEMPLATE = "보편적 고소득 시대까지 {days}일 남았다. "
AFTER = "보편적 고소득 시대 예정일 D+{days}일째, 아직이다. "
RESEARCH_URL = "https://stock.naver.com/research/company/101"


def _research() -> CollectedItem:
    return CollectedItem(
        item_id="naver-research-company-현대모비스-2026-11-02",
        source_type=SourceType.RESEARCH,
        title="현대모비스: 증권사 3곳 리포트",
        url=RESEARCH_URL,
        text="A증권 Buy 550,000원",
        author="A증권 외 2곳",
        published_at=datetime(2026, 11, 2, 23, 59, 59, tzinfo=ZoneInfo("Asia/Seoul")),
        metadata={"hide_link": True, "broker": "A증권", "research_category": "company", "stock": "현대모비스"},
    )


def _news(k) -> CollectedItem:
    return CollectedItem(
        item_id=f"n{k}", source_type=SourceType.WEB, title=f"뉴스 {k}", url=f"https://n{k}.example/a", text="본문"
    )


def _generator(urls, **cfg):
    config = PipelineConfig(digest_retry_backoff_sec=0, **cfg)
    models = ScriptedModels({"digest": editor_picking(urls), "grounding": no_violations})
    return DigestGenerator(config, models), models


async def test_hidden_link_research_item_as_the_headline_and_a_reply():
    ranked = [RankedItem(item=_research(), score=0.9), RankedItem(item=_news(1), score=0.8)]
    gen, _ = _generator([RESEARCH_URL, "https://n1.example/a"], threads_footer=FOOTER)
    digest = await gen.generate(ranked, today=date(2026, 11, 2))
    head = digest.content.items[0]
    assert head.show_link is False and head.source_tag == "A증권 외 2곳 리서치"
    root, replies = render_threads_posts(digest.content, FOOTER)
    assert RESEARCH_URL not in root and RESEARCH_URL not in replies[0]
    assert "https://n1.example/a" in replies[1]
    assert root.endswith(FOOTER)


def test_render_digest_text_still_carries_the_hidden_link():
    """digest_text (snapshot, recall_cli, research agent's recall_digest) prints every URL, hidden or not."""
    content = DigestContent(lead="리드.", items=[DigestItem(title="t", url=RESEARCH_URL, body="b", show_link=False)])
    assert RESEARCH_URL not in render_digest_text(content)


def _intro(today):
    return countdown_intro("2035-11-19", TEMPLATE, today, AFTER)


def test_root_at_exactly_the_cap_with_footer_and_countdown():
    today = date(2026, 11, 2)
    intro = _intro(today)
    room = THREADS_MAX_POST_CHARS - len(THREADS_POST_SEPARATOR) - len(FOOTER) - len(intro)
    sentence = "가" * 48 + "다. "
    body = (sentence * 20)[: room - 2].rstrip() + "다."
    lead = place_countdown_intro(body, intro)
    assert len(lead) == THREADS_MAX_POST_CHARS - len(THREADS_POST_SEPARATOR) - len(FOOTER)
    content = DigestContent(
        lead=lead,
        items=[DigestItem(title="t0", url="https://a/0", body="b")]
        + [DigestItem(title=f"다른 제목 {k}", url=f"https://a/{k}", body="b") for k in range(1, 5)],
    )
    root, _ = render_threads_posts(content, FOOTER)
    assert len(root) == THREADS_MAX_POST_CHARS
    assert root.startswith(intro) and root.endswith(FOOTER)
    # One char over: the lead loses whole sentences, the countdown and the footer survive.
    content.lead = lead + "가"
    root, _ = render_threads_posts(content, FOOTER)
    assert len(root) <= THREADS_MAX_POST_CHARS and root.startswith(intro) and root.endswith(FOOTER)


def test_a_lead_that_is_one_unbroken_sentence_over_the_cap_keeps_the_countdown_and_footer():
    intro = _intro(date(2026, 11, 2))
    lead = place_countdown_intro("가" * 600, intro)
    root, _ = render_threads_posts(DigestContent(lead=lead, items=[DigestItem(title="t", url="u", body="b")]), FOOTER)
    assert len(root) <= THREADS_MAX_POST_CHARS and root.startswith(intro.strip()) and root.endswith(FOOTER)


def test_reply_at_exactly_500_is_kept_whole_and_501_trims():
    meta_tag = "A증권 리서치"
    title = "제목"
    url = "https://n.example/a"
    impl = "시사점이다."
    fixed = len(f"{title} ({meta_tag})") + len(impl) + len(url) + 3 * len(THREADS_POST_SEPARATOR)
    body = "나" * (THREADS_MAX_POST_CHARS - fixed - 2) + "다."
    item = DigestItem(title=title, url=url, body=body, implication=impl, source_tag=meta_tag)
    _, [reply] = render_threads_posts(DigestContent(lead="l", items=[item]))
    assert len(reply) == THREADS_MAX_POST_CHARS and body in reply
    item.body = body + "x"
    _, [reply] = render_threads_posts(DigestContent(lead="l", items=[item]))
    assert len(reply) <= THREADS_MAX_POST_CHARS and url in reply and impl in reply


async def test_generator_lead_budget_counts_the_footer_and_intro():
    gen, models = _generator(
        ["https://n1.example/a"],
        threads_footer=FOOTER,
        countdown_date="2035-11-19",
        countdown_template=TEMPLATE,
        countdown_after=AFTER,
    )
    await gen.generate([RankedItem(item=_news(1), score=0.9)], today=date(2026, 11, 2))
    intro = _intro(date(2026, 11, 2))
    expected = THREADS_MAX_POST_CHARS - len(intro) - len(THREADS_POST_SEPARATOR) - len(FOOTER)
    assert str(expected) in models.prompts["digest"][0]


def test_countdown_the_day_before_on_d_day_and_after():
    assert _intro(date(2035, 11, 18)) == "보편적 고소득 시대까지 1일 남았다. "
    # On D-day the gag switches to the count-up (D+0), as countdown_intro documents.
    assert _intro(date(2035, 11, 19)) == "보편적 고소득 시대 예정일 D+0일째, 아직이다. "
    assert _intro(date(2035, 11, 20)) == "보편적 고소득 시대 예정일 D+1일째, 아직이다. "
    assert countdown_intro("2035-11-19", TEMPLATE, date(2035, 11, 19), "") == ""


def test_countdown_with_a_bad_template_degrades():
    assert countdown_intro("2035-11-19", "{day}일", date(2026, 1, 1)) == ""
    assert countdown_intro("not-a-date", TEMPLATE, date(2026, 1, 1)) == ""


async def test_hidden_link_item_budget_is_charged_for_a_url_it_never_shows():
    gen, _ = _generator([RESEARCH_URL])
    hidden = gen._item_prose_budget(_research())
    shown_item = _research().model_copy(update={"metadata": {}})
    assert hidden > gen._item_prose_budget(shown_item), "a hidden link should not cost prose budget"
