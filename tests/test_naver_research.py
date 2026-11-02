"""Naver 증권 research: the broker's note becomes a dated, attributed item whose page the post does not link."""

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from adapters.collectors import naver_research
from adapters.collectors.naver_research import NaverResearchCollector
from domain.models import DigestContent, DigestItem
from domain.renderers import render_threads_posts
from shared.config import NaverResearchCollectorConfig

SEOUL = ZoneInfo("Asia/Seoul")
COMPANY = {
    "nid": "97402",
    "title": "26.3Q Review: 4분기에는 더 좋은 그림",
    "content": "<p><strong>매출액 15.7조원</strong></p><p>영업이익 9,141억원</p>",
    "brokerName": "iM증권",
    "writeDate": "2026-11-01",
    "itemName": "현대모비스",
    "opinionText": "Buy",
    "goalPrice": "580000",
}


def _collector(monkeypatch, pages: dict[str, list[dict]], **cfg) -> tuple[NaverResearchCollector, list]:
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        category = request.url.path.rsplit("/", 1)[-1]
        rows = pages.get(category, [])
        index = int(request.url.params["index"])
        chunk = rows[index * 2 : index * 2 + 2]
        return httpx.Response(200, content=json.dumps({"items": chunk, "hasNext": len(rows) > index * 2 + 2}))

    real = httpx.AsyncClient
    monkeypatch.setattr(
        naver_research.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(answer), **kw)
    )
    config = NaverResearchCollectorConfig(
        enabled=True,
        lookback_hours=42,
        page_delay_sec=0,
        reference_time=datetime(2026, 11, 3, tzinfo=SEOUL),
        **cfg,
    )
    return NaverResearchCollector(config), seen


@pytest.mark.asyncio
async def test_a_company_note_is_attributed_dated_and_unlinked(monkeypatch):
    collector, seen = _collector(monkeypatch, {"company": [COMPANY]}, categories=["company"])
    [item] = await collector.collect()
    assert item.title == "현대모비스: 26.3Q Review: 4분기에는 더 좋은 그림"
    assert item.text.startswith("iM증권 · 현대모비스 · 투자의견 Buy · 목표주가 580,000원 · 2026-11-01\n")
    assert "영업이익 9,141억원" in item.text and "<p>" not in item.text
    assert item.source_type.value == "research" and item.author == "iM증권" and item.metadata["hide_link"] is True
    # The window's dates go upstream; the reference is midnight at the end of 11-02.
    assert seen[0].url.params["startDate"] == "2026-11-01" and seen[0].url.params["endDate"] == "2026-11-02"


@pytest.mark.asyncio
async def test_pages_are_followed_and_out_of_window_or_broken_rows_dropped(monkeypatch):
    rows = [
        {**COMPANY, "nid": str(n), "writeDate": day}
        for n, day in enumerate(["2026-11-02", "2026-11-02", "2026-11-01", "2026-10-30", "bad"])
    ]
    collector, seen = _collector(monkeypatch, {"company": rows}, categories=["company"])
    items = await collector.collect()
    assert len(seen) == 3  # two rows a page, hasNext until the last
    assert sorted(i.item_id for i in items) == [
        "naver-research-company-0",
        "naver-research-company-1",
        "naver-research-company-2",
    ]


def test_the_post_names_the_broker_but_does_not_link_the_page():
    item = DigestItem(
        title="현대모비스 4분기 개선 전망",
        url="https://stock.naver.com/research/company/97402",
        source_tag="iM증권 리서치",
        body="iM증권은 현대모비스의 4분기를 더 좋게 봤다.",
        show_link=False,
    )
    _, [reply] = render_threads_posts(DigestContent(lead="리드.", items=[item]))
    assert "iM증권 리서치" in reply and "stock.naver.com" not in reply
    linked = item.model_copy(update={"show_link": True})
    assert "stock.naver.com" in render_threads_posts(DigestContent(lead="리드.", items=[linked]))[1][0]


def test_the_generator_carries_the_hidden_link_from_the_source():
    from unittest.mock import MagicMock

    from application.digest_generator import DigestGenerator
    from domain.models import CollectedItem, RankedItem
    from shared.config import PipelineConfig
    from shared.constants import SourceType

    src = CollectedItem(
        item_id="n",
        source_type=SourceType.RESEARCH,
        title="t",
        url="https://stock.naver.com/research/company/1",
        text="x",
        author="iM증권",
        metadata={"hide_link": True},
    )
    content = DigestContent(lead="리드.", items=[DigestItem(title="t", url=src.url, body="b")])
    DigestGenerator(PipelineConfig(), MagicMock())._fill_source_metadata(content, [RankedItem(item=src, score=0.9)])
    assert content.items[0].show_link is False and content.items[0].source_tag == "iM증권 리서치"


@pytest.mark.asyncio
async def test_a_changed_date_format_fails_the_category_instead_of_reading_empty(monkeypatch):
    rows = [{**COMPANY, "writeDate": "20261101"}]
    collector, _ = _collector(monkeypatch, {"company": rows, "industry": [COMPANY]}, categories=["company", "industry"])
    items = await collector.collect()
    assert [i.metadata["research_category"] for i in items] == ["industry"]
    assert collector.run_meta  # one of two categories failed


@pytest.mark.asyncio
async def test_a_transient_status_is_retried(monkeypatch):
    calls = []

    def answer(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(503)
        return httpx.Response(200, content=json.dumps({"items": [COMPANY], "hasNext": False}))

    real = httpx.AsyncClient
    monkeypatch.setattr(
        naver_research.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(answer), **kw)
    )
    config = NaverResearchCollectorConfig(
        enabled=True,
        lookback_hours=42,
        page_delay_sec=0,
        retry_backoff_sec=0,
        categories=["company"],
        reference_time=datetime(2026, 11, 3, tzinfo=SEOUL),
    )
    assert len(await NaverResearchCollector(config).collect()) == 1 and len(calls) == 2


def test_a_news_item_that_wins_a_title_tie_keeps_its_own_link():
    from domain.aggregator import ContentAggregator
    from domain.models import CollectedItem
    from shared.constants import SourceType

    news = CollectedItem(
        item_id="a", source_type=SourceType.RSS, title="같은 제목", url="https://e.com/a", text="긴 본문 " * 50
    )
    note = CollectedItem(
        item_id="b",
        source_type=SourceType.RESEARCH,
        title="같은 제목",
        url="https://stock.naver.com/research/company/1",
        text="짧다",
        author="iM증권",
        metadata={"hide_link": True},
    )
    [kept] = ContentAggregator().aggregate([note, news])
    assert kept.url == "https://e.com/a" and "hide_link" not in kept.metadata
