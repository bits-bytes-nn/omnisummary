"""cost_report reads spend back from the emitters' usage lines: a change to one line's format would
silently drop that spend from the report."""

import base64
import importlib.util
import re
import sys
from pathlib import Path
from unittest.mock import MagicMock
from zoneinfo import ZoneInfo

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "cost_report.py"


@pytest.fixture(scope="module")
def report():
    spec = importlib.util.spec_from_file_location("cost_report", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["cost_report"] = module
    spec.loader.exec_module(module)
    return module


def _python(pattern: str) -> re.Pattern:
    """A Logs Insights named group `(?<x>...)` in Python's spelling."""
    return re.compile(pattern.replace("(?<", "(?P<"))


def _fields(report, line: str) -> dict:
    fields = {}
    for pattern in (report.USAGE_PATTERN, report.USD_PATTERN, report.CREDITS_PATTERN):
        if match := _python(pattern).search(line):
            fields.update(match.groupdict())
    return fields


def _captured(monkeypatch, module, emit) -> list[str]:
    logged: list[str] = []
    monkeypatch.setattr(module.logger, "info", lambda msg, *a: logged.append(msg % a))
    emit()
    return [line for line in logged if " usage stage=" in line]


def test_the_llm_line_parses(report, monkeypatch):
    from adapters import bedrock

    handler = bedrock._TokenUsageLogger("ranking", "anthropic.claude-opus-5-5", bedrock.OPUS_5_5_PRICE)
    message = MagicMock(usage_metadata={"input_tokens": 1_000_000, "output_tokens": 100_000})
    [line] = _captured(
        monkeypatch, bedrock, lambda: handler.on_llm_end(MagicMock(generations=[[MagicMock(message=message)]]))
    )
    fields = _fields(report, line)
    assert fields["s_kind"] == "LLM" and fields["s_stage"] == "ranking"
    assert float(fields["s_usd"]) == pytest.approx(6.0)


def test_the_image_line_parses(report, monkeypatch):
    from adapters import openai_images

    response = MagicMock(data=[MagicMock(b64_json=base64.b64encode(b"P").decode())], quality="high")
    response.usage = MagicMock(input_tokens=1000, output_tokens=10_000, input_tokens_details=None)
    client = MagicMock()
    client.images.generate.return_value = response
    renderer = openai_images.OpenAIImageRenderer(
        image_model="gpt-image-2.5-sunburst", image_quality="high", timeout_sec=1, max_retries=0
    )
    renderer._client = lambda: client
    [line] = _captured(monkeypatch, openai_images, lambda: renderer.render("draw", "1024x1024", [], "visual-render"))
    fields = _fields(report, line)
    assert fields["s_kind"] == "Image" and fields["s_stage"] == "visual-render"
    assert float(fields["s_usd"]) == pytest.approx(0.305)


def test_the_search_line_parses_and_an_unreported_one_reads_unpriced(report, monkeypatch):
    from adapters import research_backends

    lines = _captured(
        monkeypatch,
        research_backends,
        lambda: (
            research_backends.record_tavily_usage("web-search", "search", {"usage": {"credits": 2}}),
            research_backends.record_tavily_usage("page-fill", "extract", {}),
        ),
    )
    priced, unreported = (_fields(report, line) for line in lines)
    assert priced == {"s_kind": "Search", "s_stage": "web-search", "s_credits": "2"}
    assert unreported == {"s_kind": "Search", "s_stage": "page-fill"}


def test_the_query_carries_the_tested_patterns(report):
    query = report.QUERY.format(limit=report.ROW_LIMIT)
    for pattern in (report.USAGE_PATTERN, report.USD_PATTERN, report.CREDITS_PATTERN):
        assert f"/{pattern}/" in query
    assert f"limit {report.ROW_LIMIT}" in query


def test_rows_are_tallied_per_local_day_and_function_with_the_unpriced_counted(report):
    prefix = "/aws/lambda/omnisummary-dev-"

    def row(stamp: str, function: str, **fields: str) -> list[dict]:
        cells = {"@timestamp": stamp, "@log": f"123456789012:{prefix}{function}", **fields}
        return [{"field": k, "value": v} for k, v in cells.items()]

    rows = [
        row("2026-11-10 10:00:00.000", "digest", s_kind="LLM", s_stage="ranking", s_usd="1.5"),
        # 16:00 UTC is the next day in Seoul.
        row("2026-11-10 16:00:00.000", "digest", s_kind="LLM", s_stage="digest", s_usd="0.25"),
        row("2026-11-10 10:05:00.000", "digest", s_kind="Search", s_stage="web-search", s_credits="3"),
        row("2026-11-10 10:06:00.000", "visual", s_kind="LLM", s_stage="visual-editor"),
    ]
    usd, credits, unpriced = report.tally(rows, prefix=prefix, tz=ZoneInfo("Asia/Seoul"), by="function")
    assert dict(usd) == {("2026-11-10", "digest"): 1.5, ("2026-11-11", "digest"): 0.25}
    assert dict(credits) == {("2026-11-10", "digest"): 3.0}
    assert dict(unpriced) == {("2026-11-10", "visual"): 1}
    usd, _, _ = report.tally(rows, prefix=prefix, tz=ZoneInfo("Asia/Seoul"), by="stage")
    assert dict(usd) == {("2026-11-10", "llm:ranking"): 1.5, ("2026-11-11", "llm:digest"): 0.25}
