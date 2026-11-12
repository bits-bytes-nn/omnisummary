"""ContentAggregator: tracking params across sources, HTML-entity titles, news vs research note titles."""

from __future__ import annotations

from domain.aggregator import ContentAggregator
from domain.models import CollectedItem
from shared.constants import SourceType


def _it(source, title, url, text="", **meta):
    return CollectedItem(source_type=source, title=title, url=url, text=text, metadata=meta)


def test_same_url_with_and_without_utm_from_two_sources():
    a = _it(SourceType.RSS, "A", "https://ex.com/post?utm_source=rss&utm_medium=feed", "short")
    b = _it(SourceType.WEB, "B", "https://www.ex.com/post/", "a much longer body text")
    [kept] = ContentAggregator().aggregate([a, b])
    assert kept.text == "a much longer body text"
    assert "utm_" not in kept.url


def test_utm_survivor_is_the_first_but_shipped_url_has_no_tracking():
    a = _it(SourceType.RSS, "A", "https://ex.com/post?utm_source=rss&id=7", "a much longer body text")
    b = _it(SourceType.WEB, "B", "https://ex.com/post?id=7", "short")
    [kept] = ContentAggregator().aggregate([a, b])
    assert kept.url == "https://ex.com/post?id=7"


def test_html_entity_titles_dedup():
    a = _it(SourceType.RSS, "OpenAI &amp; Microsoft&#39;s new deal", "https://a.com/1", "x")
    b = _it(SourceType.WEB, "OpenAI & Microsoft's new deal", "https://b.com/2", "xx")
    c = _it(SourceType.X, "OpenAI &amp;amp; Microsoft&#x27;s new deal", "https://x.com/u/status/1", "x")
    kept = ContentAggregator().aggregate([a, b, c])
    assert len(kept) == 1, [k.title for k in kept]


def test_news_item_and_research_note_with_the_same_title():
    note = _it(
        SourceType.RESEARCH,
        "삼성전자: 3Q 실적 리뷰",
        "https://stock.naver.com/research/company/1",
        "브로커 노트 본문" * 5,
        hide_link=True,
        broker="A증권",
        stock="삼성전자",
        research_category="company",
    )
    news = _it(SourceType.WEB, "삼성전자: 3Q 실적 리뷰", "https://news.example/a", "짧음")
    [kept] = ContentAggregator().aggregate([news, note])
    # The longer research note wins: its hidden link must stay hidden.
    assert kept.source_type == SourceType.RESEARCH and kept.show_link is False

    note2 = note.model_copy(deep=True)
    news2 = _it(SourceType.WEB, "삼성전자: 3Q 실적 리뷰", "https://news.example/a", "긴 기사 본문" * 50)
    [kept] = ContentAggregator().aggregate([note2, news2])
    assert kept.source_type == SourceType.WEB and kept.show_link is True
    # The research note's broker/stock metadata rides onto the news item.
    leaked = {k for k in ("broker", "stock", "research_category") if k in kept.metadata}
    assert not leaked, f"research metadata copied onto a news item: {leaked}"
