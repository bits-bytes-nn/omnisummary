"""Naver research collector against httpx.MockTransport."""

from __future__ import annotations

import json
from datetime import datetime
from zoneinfo import ZoneInfo

import httpx
import pytest

from adapters.collectors import naver_research
from adapters.collectors.naver_research import NaverResearchCollector
from application.digest_pipeline import run_collectors_with_health
from domain.formatting import source_tag_and_metrics
from shared.config import Config, NaverResearchCollectorConfig

SEOUL = ZoneInfo("Asia/Seoul")
REF = datetime(2026, 11, 3, tzinfo=SEOUL)
_REAL_CLIENT = httpx.AsyncClient


def _row(nid, broker="iM증권", stock="현대모비스", day="2026-11-02", goal="580000", **kw):
    row = {
        "nid": str(nid),
        "title": f"리포트 {nid}",
        "content": "<p>요약</p>",
        "brokerName": broker,
        "writeDate": day,
        "itemName": stock,
        "opinionText": "Buy",
        "goalPrice": goal,
    }
    row.update(kw)
    return {k: v for k, v in row.items() if v is not None}


def _collector(monkeypatch, answer, **cfg):
    seen: list[httpx.Request] = []

    def handler(request):
        seen.append(request)
        return answer(request, len(seen))

    monkeypatch.setattr(
        naver_research.httpx, "AsyncClient", lambda **kw: _REAL_CLIENT(transport=httpx.MockTransport(handler), **kw)
    )
    base = {"enabled": True, "lookback_hours": 42, "page_delay_sec": 0, "retry_backoff_sec": 0, "reference_time": REF}
    base.update(cfg)
    return NaverResearchCollector(NaverResearchCollectorConfig(**base)), seen


def _ok(rows, has_next=False):
    return httpx.Response(200, content=json.dumps({"items": rows, "hasNext": has_next}))


async def test_403_on_every_category_fails_the_source_without_retries(monkeypatch):
    collector, seen = _collector(monkeypatch, lambda r, n: httpx.Response(403), categories=["company", "industry"])
    _, health = await run_collectors_with_health(Config(), [("naver_research", collector)])
    assert health.sources[0].status.value == "failed"
    assert len(seen) == 2


async def test_403_on_one_category_reads_degraded(monkeypatch):
    def answer(r, n):
        return httpx.Response(403) if r.url.path.endswith("/industry") else _ok([_row(1)])

    collector, _ = _collector(monkeypatch, answer, categories=["company", "industry"], error_rate_threshold=15.0)
    items, health = await run_collectors_with_health(Config(), [("naver_research", collector)])
    assert len(items) == 1 and health.sources[0].status.value == "degraded"


async def test_429_then_200_is_retried(monkeypatch):
    collector, seen = _collector(
        monkeypatch, lambda r, n: httpx.Response(429) if n == 1 else _ok([_row(1)]), categories=["company"]
    )
    items = await collector.collect()
    assert len(items) == 1 and len(seen) == 2


async def test_an_empty_sunday_reads_empty_not_failed(monkeypatch):
    collector, _ = _collector(monkeypatch, lambda r, n: _ok([]))
    items, health = await run_collectors_with_health(Config(), [("naver_research", collector)])
    assert items == [] and health.sources[0].status.value == "empty"


@pytest.mark.parametrize("payload", [{"hasNext": False}, {"items": None}, [], {"items": {"a": 1}}, "null"])
async def test_a_payload_without_items_fails(monkeypatch, payload):
    collector, _ = _collector(
        monkeypatch, lambda r, n: httpx.Response(200, content=json.dumps(payload)), categories=["company"]
    )
    with pytest.raises(RuntimeError, match="All 1 collector tasks failed"):
        await collector.collect()


async def test_a_non_json_200_fails(monkeypatch):
    collector, _ = _collector(
        monkeypatch, lambda r, n: httpx.Response(200, content=b"<html>blocked</html>"), categories=["company"]
    )
    _, health = await run_collectors_with_health(Config(), [("naver_research", collector)])
    assert health.sources[0].status.value == "failed"


async def test_max_pages_with_has_next_always_true(monkeypatch):
    def answer(r, n):
        index = int(r.url.params["index"])
        return _ok([_row(1000 + index, stock=f"종목{index}")], has_next=True)

    collector, seen = _collector(monkeypatch, answer, categories=["company"], max_pages=10)
    items = await collector.collect()
    assert len(seen) == 10 and len(items) == 10


async def test_a_merged_stock_day_is_identical_whatever_the_arrival_order(monkeypatch):
    rows = [_row(105, "B증권", goal="600,000"), _row(101, "A증권", goal="550000"), _row(109, "C증권", goal=None)]

    async def run(order):
        collector, _ = _collector(monkeypatch, lambda r, n: _ok(order), categories=["company"])
        [item] = await collector.collect()
        return item

    first = await run(rows)
    second = await run(list(reversed(rows)))
    assert first.url == second.url == "https://stock.naver.com/research/company/101"
    assert first.item_id == second.item_id
    assert (first.title, first.text, first.author) == (second.title, second.text, second.author)
    assert first.title == "현대모비스: 증권사 3곳 리포트, 목표주가 550,000~600,000원"
    assert first.author == "A증권 외 2곳"


async def test_a_merged_day_from_one_broker_counts_one(monkeypatch):
    rows = [_row(1, "A증권"), _row(2, "A증권")]
    collector, _ = _collector(monkeypatch, lambda r, n: _ok(rows), categories=["company"])
    [item] = await collector.collect()
    assert item.title.startswith("현대모비스: 증권사 1곳") and item.author == "A증권"


async def test_a_row_missing_broker_name_is_dropped(monkeypatch):
    # No broker, no attribution: the domain rules name every analyst view's source.
    rows = [_row(1, broker=None), _row(2, "B증권", stock="기아")]
    collector, _ = _collector(monkeypatch, lambda r, n: _ok(rows), categories=["company"])
    assert [i.author for i in await collector.collect()] == ["B증권"]


async def test_merged_group_whose_earliest_note_has_no_broker(monkeypatch):
    rows = [_row(1, broker=None), _row(2, "B증권")]
    collector, _ = _collector(monkeypatch, lambda r, n: _ok(rows), categories=["company"])
    [item] = await collector.collect()
    tag, _ = source_tag_and_metrics(item)
    # The reader-facing source line must not start with a dangling " 외 1곳".
    assert not item.author.startswith(" "), repr(item.author)
    assert not tag.startswith(" "), repr(tag)


async def test_unnamed_brokers_never_reach_a_merged_count(monkeypatch):
    rows = [_row(1, broker=None), _row(2, broker=None), _row(3, "B증권")]
    collector, _ = _collector(monkeypatch, lambda r, n: _ok(rows), categories=["company"])
    [item] = await collector.collect()
    assert item.author == "B증권" and "증권사" not in item.title


async def test_html_heavy_content_strips_scripts_and_styles(monkeypatch):
    html = (
        "<style>.x{color:red}</style><script>var tracking = 1;</script>"
        "<table><tr><td>매출</td><td>15.7조</td></tr></table>&nbsp;&amp;<br/>영업이익<!-- c -->"
    )
    collector, _ = _collector(monkeypatch, lambda r, n: _ok([_row(1, content=html)]), categories=["company"])
    [item] = await collector.collect()
    assert "15.7조" in item.text and "&amp;" not in item.text and "<td>" not in item.text
    assert "var tracking" not in item.text and "color:red" not in item.text, item.text


async def test_rows_without_any_readable_date_fail_the_category(monkeypatch):
    collector, _ = _collector(monkeypatch, lambda r, n: _ok([_row(1, day="2026.11.02")]), categories=["company"])
    with pytest.raises(RuntimeError):
        await collector.collect()
