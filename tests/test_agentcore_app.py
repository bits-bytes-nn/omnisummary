import json
import sys
import types
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

# The BedrockAgentCoreApp runtime SDK isn't installed in the test/lint environment, so
# stub the module before importing agent_runtime.app. The stub's @entrypoint decorator
# returns the function unchanged, exercising the real invoke() logic below.
if "bedrock_agentcore" not in sys.modules:
    _runtime_mod = types.ModuleType("bedrock_agentcore.runtime")

    class _StubApp:
        def entrypoint(self, func):
            return func

        def run(self):  # pragma: no cover - only used by __main__
            return None

    _runtime_mod.BedrockAgentCoreApp = _StubApp
    _pkg = types.ModuleType("bedrock_agentcore")
    _pkg.runtime = _runtime_mod
    sys.modules["bedrock_agentcore"] = _pkg
    sys.modules["bedrock_agentcore.runtime"] = _runtime_mod

from agent_runtime import app as app_module  # noqa: E402
from output.delivery import _request_delivery  # noqa: E402
from shared.config import get_config  # noqa: E402
from shared.logger import get_correlation_id  # noqa: E402


class TestSendSlackMessage:
    """The token comes from shared.resolve_secret — the ONE env-then-SSM ladder (which also treats
    the put_secrets placeholder as unset and resolves the region from the environment/config instead
    of a baked-in ap-northeast-2). This module used to carry a 15-line copy of it."""

    def test_uses_env_token_without_ssm(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env")
        client = MagicMock()
        boto = MagicMock()
        with patch.object(app_module, "WebClient", return_value=client) as web_client:
            with patch("shared.utils.boto3.client", boto):
                app_module._send_slack_message("C1", "hello", "")
        web_client.assert_called_once_with(token="xoxb-env")
        boto.assert_not_called()  # env token short-circuits SSM
        client.chat_postMessage.assert_called_once()
        kwargs = client.chat_postMessage.call_args.kwargs
        assert kwargs["channel"] == "C1"
        assert "thread_ts" not in kwargs

    def test_falls_back_to_ssm_when_env_missing(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        monkeypatch.setenv("PROJECT_NAME", "proj")
        monkeypatch.setenv("STAGE", "prod")
        ssm = MagicMock()
        ssm.get_parameter.return_value = {"Parameter": {"Value": "xoxb-ssm"}}
        client = MagicMock()
        with patch.object(app_module, "WebClient", return_value=client) as web_client:
            with patch("shared.utils.boto3.client", return_value=ssm):
                app_module._send_slack_message("C2", "hi", "ts-1")
        web_client.assert_called_once_with(token="xoxb-ssm")
        assert ssm.get_parameter.call_args.kwargs["Name"] == "/proj/prod/slack-bot-token"
        assert client.chat_postMessage.call_args.kwargs["thread_ts"] == "ts-1"

    def test_returns_silently_when_ssm_lookup_fails(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        ssm = MagicMock()
        ssm.get_parameter.side_effect = RuntimeError("access denied")
        with patch.object(app_module, "WebClient") as web_client:
            with patch("shared.utils.boto3.client", return_value=ssm):
                app_module._send_slack_message("C3", "text", "")
        web_client.assert_not_called()  # no token -> no Slack client constructed

    def test_the_ssm_placeholder_is_treated_as_no_token(self, monkeypatch):
        # The stack creates every parameter holding a placeholder; a deploy whose put_secrets step
        # was skipped must not send that literal to Slack as a bearer token.
        from shared.constants import SSM_PLACEHOLDER

        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        ssm = MagicMock()
        ssm.get_parameter.return_value = {"Parameter": {"Value": SSM_PLACEHOLDER}}
        with patch.object(app_module, "WebClient") as web_client:
            with patch("shared.utils.boto3.client", return_value=ssm):
                app_module._send_slack_message("C4", "text", "")
        web_client.assert_not_called()

    def test_splits_long_messages_into_chunks(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb")
        client = MagicMock()
        with patch.object(app_module, "WebClient", return_value=client):
            with patch.object(app_module, "render_agent_blocks", return_value=[["b1"], ["b2"], ["b3"]]):
                app_module._send_slack_message("C", "long", "")
        assert client.chat_postMessage.call_count == 3
        assert client.chat_postMessage.call_args_list[0].kwargs["blocks"] == ["b1"]


class TestInvoke:
    def _agent(self, response: str) -> MagicMock:
        agent = MagicMock()
        agent.return_value = response
        return agent

    def test_binds_delivery_context_and_falls_back_when_undelivered(self):
        # The agent answered but never called deliver_report, so the runtime posts the fallback.
        agent = self._agent("research answer")
        captured: dict[str, object] = {}

        def fake_agent_call(prompt, **kwargs):
            captured["delivery"] = _request_delivery.get()
            return "research answer"

        agent.side_effect = fake_agent_call

        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message") as send:
                result = app_module.invoke(
                    {"prompt": "research X", "channel_id": "C9", "thread_ts": "t1", "correlation_id": "corr-xyz"}
                )

        assert result == "research answer"
        assert captured["delivery"].channel_id == "C9"
        assert captured["delivery"].thread_ts == "t1"
        send.assert_called_once_with("C9", "research answer", "t1")
        # contextvar is reset once the request finishes
        assert _request_delivery.get() is None

    def test_no_fallback_when_slack_already_delivered(self):
        # When the agent delivered to Slack via the tool, the runtime must NOT double-post.
        def fake_agent_call(prompt, **kwargs):
            _request_delivery.get().delivered_channels.add("slack")
            return "delivered already"

        agent = MagicMock(side_effect=fake_agent_call)
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message") as send:
                result = app_module.invoke({"prompt": "p", "channel_id": "C"})
        assert result == "delivered already"
        send.assert_not_called()

    def test_propagates_correlation_id(self):
        agent = self._agent("ok")
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message"):
                app_module.invoke({"prompt": "p", "channel_id": "C", "correlation_id": "fixed-corr"})
        assert get_correlation_id() == "fixed-corr"

    def test_does_not_post_when_no_channel(self):
        agent = self._agent("answer")
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message") as send:
                result = app_module.invoke({"prompt": "p", "channel_id": ""})
        assert result == "answer"
        send.assert_not_called()

    def test_agent_exception_is_caught_and_posts_fallback(self):
        agent = MagicMock()
        agent.side_effect = RuntimeError("boom")
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message") as send:
                with patch.object(app_module, "_emit_agent_error_metric") as emit:
                    result = app_module.invoke({"prompt": "p", "channel_id": "C"})
        # The raw exception must NOT leak into the user-facing response (model IDs, ARNs, etc.).
        assert "boom" not in result
        assert "failed" in result.lower()
        send.assert_called_once()
        emit.assert_called_once()  # the EMF error metric is the only alarmable signal

    def test_no_slack_fallback_when_threads_only_delivered(self):
        # A Threads-only request that succeeded on Threads must NOT also dump the report into
        # Slack — the fallback fires only when NOTHING was delivered.
        def fake_agent_call(prompt, **kwargs):
            _request_delivery.get().delivered_channels.add("threads")
            return "report text"

        agent = MagicMock(side_effect=fake_agent_call)
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message") as send:
                app_module.invoke({"prompt": "p", "channel_id": "C"})
        send.assert_not_called()

    def test_slack_fallback_fires_when_nothing_delivered(self):
        # The agent never called deliver_report → fallback posts the report to Slack so the user
        # always gets something.
        agent = self._agent("report text")
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message") as send:
                app_module.invoke({"prompt": "p", "channel_id": "C"})
        send.assert_called_once()

    def test_failed_fallback_post_emits_the_error_metric(self):
        # This is the last-resort path: if it raises, NOTHING reached the user on any channel. The
        # entrypoint still returns text, so without the metric the lost report is invisible.
        agent = self._agent("report text")
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message", side_effect=RuntimeError("rate limited")):
                with patch.object(app_module, "_emit_agent_error_metric") as emit:
                    result = app_module.invoke({"prompt": "p", "channel_id": "C"})
        assert result == "report text"  # still answered, never raised
        emit.assert_called_once()


class TestRunLimits:
    """The tool loop re-sends the whole conversation each cycle, so the one internet-triggered path
    needs hard per-invocation caps — research_breadth/research_max_iterations are prompt guidance."""

    def test_limits_are_passed_from_config(self):
        agent = MagicMock(return_value="ok")
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message"):
                app_module.invoke({"prompt": "p", "channel_id": "C"})
        limits = agent.call_args.kwargs["limits"]
        expected = get_config().agent
        assert limits["turns"] == expected.research_max_turns
        assert limits["total_tokens"] == expected.research_max_total_tokens
        assert limits["output_tokens"] == expected.research_max_output_tokens

    @staticmethod
    def _capped_result(stop_reason: str) -> MagicMock:
        result = MagicMock()
        result.__str__ = lambda _self: "partial report"  # type: ignore[assignment]
        result.stop_reason = stop_reason
        result.metrics = None
        return result

    def test_capped_run_tells_the_user_the_report_is_partial(self):
        agent = MagicMock(return_value=self._capped_result("limit_turns"))
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message") as send:
                result = app_module.invoke({"prompt": "p", "channel_id": "C"})
        assert result.startswith("partial report")
        assert "cut short" in result
        assert "cut short" in send.call_args.args[1]

    def test_capped_run_emits_the_emf_counter(self, capsys):
        agent = MagicMock(return_value=self._capped_result("limit_total_tokens"))
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message"):
                app_module.invoke({"prompt": "p", "channel_id": "C"})
        records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        emf = next(r for r in records if "AgentLimitStops" in r)
        assert emf["AgentLimitStops"] == 1
        assert emf["StopReason"] == "limit_total_tokens"

    def test_a_normal_stop_reason_adds_no_notice(self, capsys):
        agent = MagicMock(return_value=self._capped_result("end_turn"))
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message"):
                result = app_module.invoke({"prompt": "p", "channel_id": "C"})
        assert result == "partial report"
        assert "AgentLimitStops" not in capsys.readouterr().out


class TestRunMetrics:
    """The entrypoint did `str(agent(prompt))` and dropped the AgentResult, so the most expensive
    component in the system was the only stage whose token spend nothing recorded."""

    @staticmethod
    def _result(text: str) -> MagicMock:
        result = MagicMock()
        result.__str__ = lambda _self: text  # type: ignore[assignment]
        result.metrics.accumulated_usage = {
            "inputTokens": 1200,
            "outputTokens": 340,
            "cacheReadInputTokens": 900,
            "cacheWriteInputTokens": 10,
        }
        result.metrics.cycle_count = 4
        result.metrics.tool_metrics = {
            "web_search": MagicMock(call_count=3),
            "search_papers": MagicMock(call_count=1),
        }
        return result

    def test_usage_is_logged_in_the_same_shape_as_every_pipeline_stage(self, capsys):
        agent = MagicMock(return_value=self._result("report text"))
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message"):
                with patch.object(app_module, "logger") as log:
                    result = app_module.invoke({"prompt": "p", "channel_id": "C"})
        assert result == "report text"
        line = next(c for c in log.info.call_args_list if "LLM usage stage=research" in str(c.args[0]))
        assert line.args[1:5] == (1200, 340, 900, 10)
        assert line.args[5] == 4  # cycles
        assert line.args[6] == {"web_search": 3, "search_papers": 1}

    def test_usage_is_emitted_as_emf(self, capsys):
        agent = MagicMock(return_value=self._result("report text"))
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message"):
                app_module.invoke({"prompt": "p", "channel_id": "C"})
        records = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]
        emf = next(r for r in records if "AgentInputTokens" in r)
        assert emf["AgentInputTokens"] == 1200 and emf["AgentOutputTokens"] == 340
        assert emf["AgentCycles"] == 4 and emf["AgentToolCalls"] == 4
        # EMF reads Timestamp as epoch-UTC ms; a naive local clock files every point at the wrong time.
        assert abs(emf["_aws"]["Timestamp"] - int(datetime.now(UTC).timestamp() * 1000)) < 60_000

    def test_a_result_without_metrics_is_not_fatal(self):
        # Telemetry must never break a completed run (an older SDK, a stubbed agent in tests).
        agent = MagicMock(return_value="plain string, no metrics")
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message"):
                assert app_module.invoke({"prompt": "p", "channel_id": "C"}) == "plain string, no metrics"


@pytest.fixture(autouse=True)
def _reset_request_context():
    # Defensive: ensure no test leaks request-scoped contextvars into the next.
    yield
    _request_delivery.set(None)
