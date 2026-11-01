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
    assert not names() & {"market_quote", "search_reports"}
    assert "AI/ML topics" in captured["system_prompt"] and "<domain_rules>" not in captured["system_prompt"]

    config = Config(
        pipeline={"domain": "investing"},
        agent={"market_quotes": True, "report_search_domains": ["x.com"], "domain_rules": "No calls."},
    )
    research_agent.create_research_agent(config)
    assert {"market_quote", "search_reports"} <= names()
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
