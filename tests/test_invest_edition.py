"""The investing edition: one codebase, its own config, state prefix and memory sessions, and two tools."""

import json

import httpx
import pytest

from application import research
from application.research import DeliveryContext, ResearchPorts, request_context
from shared import Config
from shared.config import EDITION_ENV
from tests.fakes import FakeSources


def test_the_edition_picks_its_config_file(monkeypatch):
    monkeypatch.setenv(EDITION_ENV, "invest")
    config = Config.load()
    assert config.edition == "invest"
    assert config.agent.market_quotes and config.agent.report_search_domains


def test_a_config_that_declares_another_edition_is_refused(monkeypatch, tmp_path):
    import shared.config as cfg

    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "invest.yaml").write_text('edition: "other"\n')
    monkeypatch.setattr(cfg, "__file__", str(tmp_path / "shared" / "config.py"))
    monkeypatch.setenv(EDITION_ENV, "invest")
    with pytest.raises(ValueError, match="declares edition"):
        Config.load()


def test_memory_sessions_are_scoped_by_edition():
    from unittest.mock import MagicMock

    from adapters.memory import AgentCoreMemoryStore

    session = MagicMock()
    assert AgentCoreMemoryStore("m", session=session).session_prefix == "digest"
    assert AgentCoreMemoryStore("m", session=session, edition="invest").session_prefix == "digest-invest"


def test_the_investing_tools_are_bound_only_when_configured(monkeypatch):
    from unittest.mock import MagicMock

    from adapters import research_agent

    captured = {}
    monkeypatch.setattr(
        research_agent, "Agent", lambda **kw: captured.update(kw) or type("A", (), {"tool_names": []})()
    )
    monkeypatch.setattr(research_agent.BedrockCrossRegionModelHelper, "get_cross_region_model_id", lambda *a: "m")
    monkeypatch.setattr(research_agent, "BedrockModel", lambda **kw: object())
    monkeypatch.setattr(research_agent.boto3, "Session", lambda **kw: MagicMock())
    names = lambda: {getattr(t, "tool_name", "") for t in captured["tools"]}  # noqa: E731

    research_agent.create_research_agent(Config())
    assert not names() & {"market_quote", "analyst_ratings", "search_reports"}
    assert "AI/ML topics" in captured["system_prompt"] and "<domain_rules>" not in captured["system_prompt"]

    config = Config(
        pipeline={"domain": "investing", "domain_rules": "No calls."},
        agent={"market_quotes": True, "report_search_domains": ["x.com"]},
    )
    research_agent.create_research_agent(config)
    assert {"market_quote", "analyst_ratings", "search_reports"} <= names()
    assert "investing topics" in captured["system_prompt"] and "No calls." in captured["system_prompt"]


@pytest.mark.asyncio
async def test_a_quote_is_a_citable_source_with_its_numbers():
    block = "- Apple Inc. (AAPL)\n  URL: https://finance.yahoo.com/quote/AAPL\n  Last close: 344.60 USD on 2026-10-30"
    sources = FakeSources(quote=block)
    delivery = DeliveryContext(ports=ResearchPorts(sources=sources))
    with request_context(delivery):
        out = await research.market_quote("AAPL, AAPL nvda")
    assert out.count("Apple Inc.") == 2  # AAPL once, and nvda answers the same fake block
    assert sources.market_quote.await_count == 2
    assert any("finance.yahoo.com/quote/aapl" in key.lower() for key in delivery.seen_urls)
    assert any("344.60" in text for text in delivery.source_texts.values())


@pytest.mark.asyncio
async def test_the_quote_parser_reads_the_chart_payload(monkeypatch):
    from adapters import research_backends

    payload = {
        "chart": {
            "result": [
                {
                    "meta": {"longName": "Samsung Electronics", "currency": "KRW", "exchangeName": "KSC"},
                    "timestamp": [1761436800, 1761782400],
                    "indicators": {"quote": [{"close": [213000.0, 225000.0]}]},
                }
            ]
        }
    }
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=json.dumps(payload)))
    real = httpx.AsyncClient
    monkeypatch.setattr(research_backends.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    block = await research_backends.yahoo_quote("005930.ks")
    assert "Samsung Electronics (005930.KS)" in block
    assert "225,000 KRW" in block and "+5.63%" in block
    assert "Period high / low: 225,000 on" in block
    assert await research_backends.yahoo_quote("bad symbol!") == ""


def test_the_footer_closes_the_root_and_the_research_thread_within_the_cap():
    from domain.models import DigestContent, DigestItem
    from domain.renderers import render_threads_posts, render_threads_research
    from shared import THREADS_MAX_POST_CHARS

    footer = "투자 권유가 아닙니다. 투자 판단의 책임은 본인에게 있습니다."
    items = [DigestItem(title=f"기사 {n} " + "가" * 60, url=f"https://e.com/{n}", body="본문.") for n in range(5)]
    root, _ = render_threads_posts(DigestContent(lead="리드다. " * 80, items=items), footer)
    assert root.endswith(footer) and len(root) <= THREADS_MAX_POST_CHARS
    assert render_threads_posts(DigestContent(lead="리드.", items=items))[0].count(footer) == 0

    render = render_threads_research("제목\n\n첫 글.\n---\n결론\n\n마지막 글.", max_posts=10, footer=footer)
    assert render.replies[-1].endswith(footer) and footer not in render.root


@pytest.mark.asyncio
async def test_the_quote_uses_exchange_dates_labels_a_live_session_and_survives_odd_payloads(monkeypatch):
    from adapters import research_backends

    payloads = []
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=json.dumps(payloads.pop(0))))
    real = httpx.AsyncClient
    monkeypatch.setattr(research_backends.httpx, "AsyncClient", lambda **kw: real(transport=transport, **kw))
    # 2026-10-30 15:00 UTC is 2026-10-31 in Seoul; the session is open right now.
    import time

    stamp, now = 1793372400, int(time.time())
    payloads.append(
        {
            "chart": {
                "result": [
                    {
                        "meta": {
                            "symbol": "X",
                            "currency": "KRW",
                            "exchangeTimezoneName": "Asia/Seoul",
                            "currentTradingPeriod": {"regular": {"start": now - 60, "end": now + 3600}},
                        },
                        "timestamp": [stamp, now - 60],
                        "indicators": {"quote": [{"close": [100.0, 101.0]}]},
                    }
                ]
            }
        }
    )
    block = await research_backends.yahoo_quote("X.KS")
    assert "Latest price (session open): 101.00 KRW on" in block
    assert "2026-10-31 100.00" in block  # 15:00 UTC on 10-30 is 10-31 in Seoul
    odd_payloads = (
        [],
        {"chart": None},
        {"chart": {"result": {"x": 1}}},
        {"chart": {"result": [{"timestamp": [1e20], "indicators": {"quote": [{"close": [1.0]}]}}]}},
        {"chart": {"result": [{"indicators": {"quote": [{"close": "x"}]}}]}},
    )
    for odd in odd_payloads:
        payloads.append(odd)
        assert await research_backends.yahoo_quote("AAPL") == ""
    assert await research_backends.yahoo_quote("삼성") == ""


@pytest.mark.asyncio
async def test_analyst_ratings_are_a_citable_source():
    block = "- Analyst ratings for NVDA (Yahoo Finance, as of 2026-11-01)\n  URL: https://finance.yahoo.com/quote/NVDA/analysis\n  Analysts: 58"
    sources = FakeSources(quote=block)
    delivery = DeliveryContext(ports=ResearchPorts(sources=sources))
    with request_context(delivery):
        out = await research.analyst_ratings("NVDA, nvda TSLA")
    assert sources.analyst_ratings.await_count == 2
    assert out.count("Analysts: 58") == 2
    assert any("finance.yahoo.com/quote/nvda/analysis" in key.lower() for key in delivery.seen_urls)


def test_the_ratings_block_dates_recent_actions_and_drops_old_ones():
    from adapters.research_backends import _format_ratings

    now = 1793000000  # 2026-10-26 UTC
    result = {
        "financialData": {
            "financialCurrency": "USD",
            "numberOfAnalystOpinions": {"raw": 40},
            "targetMeanPrice": {"raw": 411.8},
            "targetHighPrice": {"raw": 600.0},
            "targetLowPrice": {"raw": 120.0},
        },
        "recommendationTrend": {
            "trend": [{"period": "0m", "strongBuy": 5, "buy": 15, "hold": 22, "sell": 2, "strongSell": 0}]
        },
        "upgradeDowngradeHistory": {
            "history": [
                {
                    "epochGradeDate": now - 86400,
                    "firm": "Barclays",
                    "fromGrade": "Equal-Weight",
                    "toGrade": "Overweight",
                    "action": "up",
                    "currentPriceTarget": 450.0,
                    "priorPriceTarget": 370.0,
                },
                {"epochGradeDate": now - 200 * 86400, "firm": "Old Firm", "toGrade": "Sell", "action": "init"},
                {"epochGradeDate": "bad", "firm": "Broken"},
            ]
        },
    }
    block = _format_ratings("TSLA", result, now)
    assert "URL: https://finance.yahoo.com/quote/TSLA/analysis" in block
    assert "Analysts: 40" in block and "mean 411.80 USD (high 600.00, low 120.00)" in block
    assert "strong buy 5, buy 15, hold 22, sell 2, strong sell 0" in block
    assert "2026-10-25 Barclays: Equal-Weight -> Overweight, target 370.00 -> 450.00 (up)" in block
    assert "Old Firm" not in block and "Broken" not in block
    assert _format_ratings("X", {}, now) == ""
