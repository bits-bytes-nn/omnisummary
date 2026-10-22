import ast
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import research_cli
from output.delivery import current_delivery_context

REPO_ROOT = Path(__file__).resolve().parent.parent


def _metrics() -> SimpleNamespace:
    return SimpleNamespace(
        accumulated_usage={"inputTokens": 1200, "outputTokens": 340, "cacheReadInputTokens": 800},
        tool_metrics={"web_search": SimpleNamespace(call_count=3)},
        cycle_count=4,
    )


class _Result:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.text = text
        self.stop_reason = stop_reason
        self.metrics = _metrics()

    def __str__(self) -> str:
        return self.text


def _run(
    argv,
    *,
    submit: str | None = "# 제목\n\n본문 https://real.example/a",
    final="done",
    stop="end_turn",
    raises=None,
    raises_after_submit=None,
    tavily_key="key",
):
    """Run the CLI with a fake agent. `submit` is what the agent hands to deliver_report("markdown");
    None means it answered without submitting. `raises_after_submit` fails the run after delivery."""
    captured: dict = {}

    def fake_create(*_):
        def call(prompt, **kwargs):
            captured["prompt"] = prompt
            captured["kwargs"] = kwargs
            if raises:
                raise raises
            delivery = current_delivery_context()
            captured["delivery"] = delivery
            delivery.seen_urls.add("real.example/a")
            if submit is not None:
                delivery.report = submit
                delivery.delivered_channels.add("markdown")
            if raises_after_submit:
                delivery.delivered_channels.add("threads")
                raise raises_after_submit
            return _Result(final, stop)

        return call

    with (
        patch.object(research_cli, "create_research_agent", side_effect=fake_create) as create,
        patch.object(research_cli, "resolve_secret", return_value=tavily_key),
    ):
        result, code = research_cli.run(argv)
    captured["create"] = create
    return result, code, captured


class TestInputAndPublishing:
    def test_an_empty_topic_is_rejected_without_running_the_agent(self, tmp_path):
        result, code, cap = _run(["   ", "--output-dir", str(tmp_path)])
        assert code == research_cli.EXIT_REJECTED
        assert result["status"] == "rejected"
        cap["create"].assert_not_called()

    def test_nothing_is_published_without_the_flag(self, tmp_path):
        _, _, cap = _run(["주제", "--output-dir", str(tmp_path)])
        assert cap["delivery"].publish_threads is False
        assert "[DELIVERY] Threads publishing: NOT allowed." in cap["prompt"]

    def test_the_flag_travels_as_data(self, tmp_path):
        # The agent is TOLD whether it may publish; the tool enforces it from the context, so a topic
        # whose subject is Threads cannot publish on its own.
        _, _, cap = _run(["쓰레드 알고리즘 변화", "--publish", "threads", "--output-dir", str(tmp_path)])
        assert cap["delivery"].publish_threads is True
        assert cap["prompt"].startswith("쓰레드 알고리즘 변화")
        assert "[DELIVERY] Threads publishing: ALLOWED." in cap["prompt"]

    def test_an_unknown_publish_target_is_an_argparse_rejection(self):
        with pytest.raises(SystemExit) as exc:
            research_cli.run(["주제", "--publish", "slack"])
        assert exc.value.code == research_cli.EXIT_REJECTED


class TestResult:
    def test_a_submitted_report_is_saved_and_described(self, tmp_path):
        result, code, _ = _run(["주제", "--output-dir", str(tmp_path)])
        assert code == research_cli.EXIT_OK
        assert result["status"] == "succeeded"
        assert result["title"] == "제목"
        assert Path(result["markdown_path"]).read_text(encoding="utf-8") == result["markdown"]
        assert Path(result["markdown_path"]).parent == tmp_path
        assert result["sources"] == ["https://real.example/a"]
        assert result["threads"] is None
        assert result["warnings"] == []
        assert result["usage"]["input_tokens"] == 1200
        assert result["usage"]["tool_calls"] == {"web_search": 3}

    def test_an_unsubmitted_answer_is_used_and_flagged(self, tmp_path):
        result, code, _ = _run(
            ["주제", "--output-dir", str(tmp_path)],
            submit=None,
            final="# 답\n\n근거 https://real.example/a 와 https://invented.example/x",
        )
        assert code == research_cli.EXIT_OK
        assert result["markdown"].startswith("# 답")
        assert any("unchecked" in w for w in result["warnings"])
        assert any("https://invented.example/x" in w for w in result["warnings"])
        assert not any("https://real.example/a" in w for w in result["warnings"])

    def test_a_budget_stop_is_a_warning(self, tmp_path):
        result, code, _ = _run(["주제", "--output-dir", str(tmp_path)], stop="limit_turns")
        assert code == research_cli.EXIT_OK
        assert any("budget cap" in w for w in result["warnings"])

    def test_an_agent_error_fails_with_the_error(self, tmp_path):
        result, code, _ = _run(["주제", "--output-dir", str(tmp_path)], raises=RuntimeError("bedrock down"))
        assert code == research_cli.EXIT_FAILED
        assert result["status"] == "failed"
        assert "bedrock down" in result["error"]

    def test_a_failure_after_publishing_still_reports_the_thread_and_the_report(self, tmp_path):
        # A thread that went out before the failure is live: a caller that can't see it re-runs.
        result, code, _ = _run(
            ["주제", "--publish", "threads", "--output-dir", str(tmp_path)],
            raises_after_submit=RuntimeError("throttled"),
        )
        assert code == research_cli.EXIT_FAILED
        assert "throttled" in result["error"]
        assert result["threads"]["published"] is True
        assert Path(result["markdown_path"]).exists()

    def test_a_budget_stop_without_a_submitted_report_fails_but_keeps_the_fragment(self, tmp_path):
        result, code, _ = _run(
            ["주제", "--output-dir", str(tmp_path)], submit=None, final="이제 검색하겠습니다", stop="limit_turns"
        )
        assert code == research_cli.EXIT_FAILED
        assert "budget cap" in result["error"]
        assert result["markdown"] == "이제 검색하겠습니다"

    def test_a_missing_tavily_key_fails_before_the_agent_runs(self, tmp_path):
        result, code, captured = _run(["주제", "--output-dir", str(tmp_path)], tavily_key="")
        assert code == research_cli.EXIT_FAILED
        assert "TAVILY_API_KEY" in result["error"]
        assert not captured["create"].called

    def test_no_report_at_all_fails(self, tmp_path):
        result, code, _ = _run(["주제", "--output-dir", str(tmp_path)], submit=None, final="   ")
        assert code == research_cli.EXIT_FAILED
        assert result["markdown_path"] is None

    def test_a_requested_publish_that_did_not_land_fails_but_keeps_the_report(self, tmp_path):
        result, code, _ = _run(["주제", "--publish", "threads", "--output-dir", str(tmp_path)])
        assert code == research_cli.EXIT_FAILED
        assert result["threads"]["published"] is False
        assert Path(result["markdown_path"]).exists()

    def test_main_prints_exactly_one_json_object(self, tmp_path, capsys):
        with patch.object(research_cli, "run", return_value=({"status": "succeeded"}, 0)):
            with pytest.raises(SystemExit) as exc:
                research_cli.main()
        assert exc.value.code == 0
        assert json.loads(capsys.readouterr().out) == {"status": "succeeded"}


class TestEveryInvocationIsBudgeted:
    """A loop that never converges is unbounded in both cost and wall time, so every invocation of the
    agent carries the shared limits."""

    def test_the_cli_passes_the_shared_limits(self, tmp_path):
        from agent.research_agent import research_run_limits

        _, _, cap = _run(["주제", "--output-dir", str(tmp_path)])
        assert cap["kwargs"]["limits"] == research_run_limits()

    def test_every_agent_invocation_carries_limits(self):
        # Source-level (over the AST, so a docstring mentioning a bare call doesn't count): an
        # unbounded invocation is not an error the SDK reports, it just runs until it converges.
        tree = ast.parse((REPO_ROOT / "research_cli.py").read_text(encoding="utf-8"))
        calls = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "agent"
        ]
        assert calls
        for call in calls:
            assert "limits" in {kw.arg for kw in call.keywords}, f"line {call.lineno} invokes the agent without limits"
