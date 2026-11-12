"""Naver 증권 research: the broker's note becomes a dated, attributed item whose page the post does not link."""

import json
from datetime import datetime
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import httpx
import pytest

from adapters.collectors import naver_research
from adapters.collectors.naver_research import NaverResearchCollector
from domain.models import CollectedItem, DigestContent, DigestItem
from domain.renderers import render_threads_posts
from shared.config import NaverResearchCollectorConfig
from shared.constants import SourceType

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
        **cfg,
    )
    return NaverResearchCollector(config, reference_time=datetime(2026, 11, 3, tzinfo=SEOUL)), seen


@pytest.mark.asyncio
async def test_a_company_note_is_attributed_dated_and_unlinked(monkeypatch):
    collector, seen = _collector(monkeypatch, {"company": [COMPANY]}, categories=["company"])
    [item] = await collector.collect()
    assert item.title == "현대모비스: 26.3Q Review: 4분기에는 더 좋은 그림"
    assert item.text.startswith("iM증권 · 현대모비스 · 투자의견 Buy · 목표주가 580,000원 · 2026-11-01\n")
    assert "영업이익 9,141억원" in item.text and "<p>" not in item.text
    assert item.source_type.value == "research" and item.author == "iM증권" and item.show_link is False
    # The window's dates go upstream; the reference is midnight at the end of 11-02.
    assert seen[0].url.params["startDate"] == "2026-11-01" and seen[0].url.params["endDate"] == "2026-11-02"


@pytest.mark.asyncio
async def test_pages_are_followed_and_out_of_window_or_broken_rows_dropped(monkeypatch):
    rows = [
        {**COMPANY, "nid": str(n), "writeDate": day, "itemName": f"종목{n}"}
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
        show_link=False,
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
    )
    assert (
        len(await NaverResearchCollector(config, reference_time=datetime(2026, 11, 3, tzinfo=SEOUL)).collect()) == 1
        and len(calls) == 2
    )


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
        show_link=False,
    )
    [kept] = ContentAggregator().aggregate([note, news])
    assert kept.url == "https://e.com/a" and kept.show_link is True


@pytest.mark.asyncio
async def test_several_brokers_on_one_stock_and_day_become_one_consensus_item(monkeypatch):
    rows = [
        {**COMPANY, "nid": "1", "brokerName": "iM증권", "goalPrice": "580000"},
        {**COMPANY, "nid": "2", "brokerName": "KB증권", "opinionText": "매수", "goalPrice": "600000"},
        {**COMPANY, "nid": "3", "brokerName": "한국IR협의회", "opinionText": "", "goalPrice": "0"},
        {**COMPANY, "nid": "4", "itemName": "기아"},
    ]
    collector, _ = _collector(monkeypatch, {"company": rows}, categories=["company"])
    items = await collector.collect()
    assert len(items) == 2
    merged = next(i for i in items if i.metadata.get("notes") == 3)
    assert merged.title == "현대모비스: 증권사 3곳 리포트, 목표주가 580,000~600,000원"
    assert "iM증권 Buy 580,000원, KB증권 매수 600,000원, 한국IR협의회 의견 없음" in merged.text
    assert merged.author == "iM증권 외 2곳" and merged.show_link is False


@pytest.mark.asyncio
async def test_the_merged_item_keeps_the_first_notes_url_whatever_the_api_order(monkeypatch):
    rows = [
        {**COMPANY, "nid": "300", "brokerName": "KB증권"},
        {**COMPANY, "nid": "100", "brokerName": "iM증권"},
        {**COMPANY, "nid": "200", "brokerName": "iM증권"},  # the same broker twice
    ]
    collector, _ = _collector(monkeypatch, {"company": rows}, categories=["company"])
    [merged] = await collector.collect()
    # What an earlier run that saw only note 100 shipped, so the URL ledger still matches.
    assert merged.url.endswith("/company/100")
    assert merged.title.startswith("현대모비스: 증권사 2곳 리포트") and merged.author == "iM증권 외 1곳"


def test_a_single_notes_origin_is_its_broker_and_a_consensus_has_its_own():
    from domain.formatting import resolve_origin_key
    from domain.models import CollectedItem
    from shared.constants import SourceType

    item = CollectedItem(
        item_id="m",
        source_type=SourceType.RESEARCH,
        title="t",
        url="https://e.com/1",
        author="iM증권 외 2곳",
        metadata={"broker": "iM증권"},
    )
    assert resolve_origin_key(item) == "iM증권"
    merged = item.model_copy(update={"metadata": {"broker": "iM증권", "stock": "기아", "notes": 8}})
    assert resolve_origin_key(merged) == "consensus:기아"


@pytest.mark.asyncio
async def test_rows_that_all_lack_a_broker_fail_the_category(monkeypatch):
    rows = [{**COMPANY, "brokerName": ""}]
    collector, _ = _collector(monkeypatch, {"company": rows, "industry": [COMPANY]}, categories=["company", "industry"])
    items = await collector.collect()
    assert [i.metadata["research_category"] for i in items] == ["industry"]
    from adapters.collectors.base import PARK_META_ACCOUNTS_FAILED

    assert collector.run_meta[PARK_META_ACCOUNTS_FAILED] == 1  # a failure, not a quiet empty


class TestNoteText:
    @staticmethod
    def _note(**metadata) -> CollectedItem:
        return CollectedItem(
            item_id="naver-research-invest-1",
            source_type=SourceType.RESEARCH,
            title="한화투자증권: 11월 전략",
            url="https://stock.naver.com/research/invest/1",
            metadata={"research_category": "invest", "nid": 1, **metadata},
        )

    @staticmethod
    def _transport(detail: dict, pdf_status: int = 200):
        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.host == "stock.naver.com":
                assert request.url.path.endswith("/researches/v2/invest/1")
                return httpx.Response(200, json=detail)
            return httpx.Response(pdf_status, content=b"%PDF-1.4 fake")

        return httpx.MockTransport(handle)

    @pytest.mark.asyncio
    async def test_the_attached_pdf_is_read(self, monkeypatch):
        transport = self._transport({"attachUrl": "https://stock.pstatic.net/stock-research/invest/1.pdf"})
        real = httpx.AsyncClient
        monkeypatch.setattr(naver_research.httpx, "AsyncClient", lambda **kw: real(transport=transport))
        monkeypatch.setattr(naver_research, "_pdf_text", lambda data, cap: "본문" if data.startswith(b"%PDF") else "")
        assert await naver_research.note_text(self._note(), timeout=5, max_chars=1000) == "본문"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(("detail", "pdf_status"), [({}, 200), ({"attachUrl": "https://x/1.pdf"}, 403)])
    async def test_no_attachment_or_a_refusal_reads_as_nothing(self, monkeypatch, detail, pdf_status):
        transport = self._transport(detail, pdf_status)
        real = httpx.AsyncClient
        monkeypatch.setattr(naver_research.httpx, "AsyncClient", lambda **kw: real(transport=transport))
        assert await naver_research.note_text(self._note(), timeout=5, max_chars=1000) == ""

    @pytest.mark.asyncio
    async def test_a_parse_past_the_timeout_reads_as_nothing(self, monkeypatch):
        import time as _time

        transport = self._transport({"attachUrl": "https://stock.pstatic.net/stock-research/invest/1.pdf"})
        real = httpx.AsyncClient
        monkeypatch.setattr(naver_research.httpx, "AsyncClient", lambda **kw: real(transport=transport))
        monkeypatch.setattr(naver_research, "_pdf_text", lambda data, cap: _time.sleep(0.5) or "late")
        assert await naver_research.note_text(self._note(), timeout=0.05, max_chars=1000) == ""


def test_pdf_text_stops_at_its_character_budget(monkeypatch):
    pages = [MagicMock(extract_text=MagicMock(return_value="가" * 600)) for _ in range(50)]
    monkeypatch.setattr(naver_research, "PdfReader", lambda stream: MagicMock(pages=pages))
    assert len(naver_research._pdf_text(b"", 1000)) == 1000
    assert sum(p.extract_text.called for p in pages) == 2


def _serve(monkeypatch, pdf: bytes, seen: list | None = None):
    def handle(request):
        if seen is not None:
            seen.append(request.url)
        if request.url.host == "stock.naver.com":
            return httpx.Response(200, json={"attachUrl": "https://stock.pstatic.net/r/1.pdf"})
        return httpx.Response(200, content=pdf)

    real = httpx.AsyncClient
    monkeypatch.setattr(naver_research.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handle)))


@pytest.mark.asyncio
async def test_an_unparseable_pdf_reads_as_nothing(monkeypatch):
    _serve(monkeypatch, b"%PDF-1.4 truncated")
    monkeypatch.setattr(naver_research, "_pdf_text", MagicMock(side_effect=Exception("PdfReadError")))
    assert await naver_research.note_text(TestNoteText._note(), timeout=5, max_chars=1000) == ""


@pytest.mark.asyncio
async def test_a_pdf_over_the_cap_is_not_parsed(monkeypatch):
    monkeypatch.setattr(naver_research, "_NOTE_MAX_BYTES", 10)
    _serve(monkeypatch, b"%PDF" + b"x" * 100)
    parse = MagicMock(return_value="text")
    monkeypatch.setattr(naver_research, "_pdf_text", parse)
    assert await naver_research.note_text(TestNoteText._note(), timeout=5, max_chars=1000) == ""
    parse.assert_not_called()


@pytest.mark.asyncio
async def test_a_consensus_item_fetches_nothing(monkeypatch):
    seen: list = []
    _serve(monkeypatch, b"%PDF", seen)
    monkeypatch.setattr(naver_research, "_pdf_text", lambda data, cap: "one broker's note")
    assert await naver_research.note_text(TestNoteText._note(notes=3), timeout=5, max_chars=1000) == ""
    assert seen == []
