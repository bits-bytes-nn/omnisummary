import sys
import types
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
from shared.logger import get_correlation_id  # noqa: E402


class TestSendSlackMessage:
    def test_uses_env_token_without_ssm(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env")
        client = MagicMock()
        boto = MagicMock()
        with patch.object(app_module, "WebClient", return_value=client) as web_client:
            with patch.object(app_module.boto3, "client", boto):
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
            with patch.object(app_module.boto3, "client", return_value=ssm):
                app_module._send_slack_message("C2", "hi", "ts-1")
        web_client.assert_called_once_with(token="xoxb-ssm")
        assert ssm.get_parameter.call_args.kwargs["Name"] == "/proj/prod/slack-bot-token"
        assert client.chat_postMessage.call_args.kwargs["thread_ts"] == "ts-1"

    def test_returns_silently_when_ssm_lookup_fails(self, monkeypatch):
        monkeypatch.delenv("SLACK_BOT_TOKEN", raising=False)
        ssm = MagicMock()
        ssm.get_parameter.side_effect = RuntimeError("access denied")
        with patch.object(app_module, "WebClient") as web_client:
            with patch.object(app_module.boto3, "client", return_value=ssm):
                app_module._send_slack_message("C3", "text", "")
        web_client.assert_not_called()  # no token -> no Slack client constructed

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

        def fake_agent_call(prompt):
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
        def fake_agent_call(prompt):
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
        assert "Error processing request" in result
        assert "boom" in result
        send.assert_called_once()
        emit.assert_called_once()  # the EMF error metric is the only alarmable signal

    def test_slack_fallback_fires_when_only_threads_delivered(self):
        # Threads succeeded but Slack never did → Slack is the always-available channel, so the
        # runtime must still post the report text there.
        def fake_agent_call(prompt):
            _request_delivery.get().delivered_channels.add("threads")
            return "report text"

        agent = MagicMock(side_effect=fake_agent_call)
        with patch.object(app_module, "create_research_agent", return_value=agent):
            with patch.object(app_module, "_send_slack_message") as send:
                app_module.invoke({"prompt": "p", "channel_id": "C"})
        send.assert_called_once()


@pytest.fixture(autouse=True)
def _reset_request_context():
    # Defensive: ensure no test leaks request-scoped contextvars into the next.
    yield
    _request_delivery.set(None)
