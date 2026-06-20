import ast
import hashlib
import hmac
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from lambda_handlers import slack_event_handler as h

SIGNING_SECRET = "test-signing-secret"


def test_handler_has_no_sibling_package_imports():
    # This handler ships as a standalone zip containing ONLY lambda_handlers/, so importing any
    # sibling package (shared, agent, output, pipeline, ...) fails at cold start with
    # ImportModuleError. Guard against reintroducing such an import (regression: 'No module named
    # shared'). The test env CAN import shared, so a runtime import test wouldn't catch it — scan
    # the source instead.
    forbidden = {"shared", "agent", "output", "pipeline", "collectors", "agent_runtime", "infrastructure"}
    src = Path(h.__file__).read_text()
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".")[0] in forbidden:
                bad.append(node.module)
        elif isinstance(node, ast.Import):
            bad += [n.name for n in node.names if n.name.split(".")[0] in forbidden]
    assert not bad, f"slack_event_handler must not import sibling packages (not in its zip): {bad}"


def _signed_headers(body: str, secret: str = SIGNING_SECRET, ts: str | None = None) -> dict[str, str]:
    ts = ts or str(int(time.time()))
    sig = "v0=" + hmac.new(secret.encode(), f"v0:{ts}:{body}".encode(), hashlib.sha256).hexdigest()
    return {"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": sig}


class TestUrlVerification:
    def test_challenge_echoed_without_signature(self):
        body = json.dumps({"type": "url_verification", "challenge": "abc123"})
        resp = h.handler({"body": body, "headers": {}}, None)
        assert resp["statusCode"] == 200
        assert resp["body"] == "abc123"


class TestSignatureVerification:
    def test_rejects_missing_signature(self):
        body = json.dumps({"type": "event_callback", "event": {"type": "app_mention"}})
        resp = h.handler({"body": body, "headers": {}}, None)
        assert resp["statusCode"] == 401

    def test_rejects_bad_signature(self, monkeypatch):
        body = json.dumps({"type": "event_callback", "event": {"type": "app_mention"}})
        headers = {"X-Slack-Request-Timestamp": str(int(time.time())), "X-Slack-Signature": "v0=deadbeef"}
        with patch.object(h.boto3, "client") as mock_client:
            mock_client.return_value.get_parameter.return_value = {"Parameter": {"Value": SIGNING_SECRET}}
            resp = h.handler({"body": body, "headers": headers}, None)
        assert resp["statusCode"] == 401

    def test_rejects_stale_timestamp(self):
        body = json.dumps({"type": "event_callback"})
        old_ts = str(int(time.time()) - 99999)
        headers = _signed_headers(body, ts=old_ts)
        resp = h.handler({"body": body, "headers": headers}, None)
        assert resp["statusCode"] == 401


class TestAppMention:
    def test_valid_mention_invokes_runtime(self, monkeypatch):
        body = json.dumps(
            {
                "type": "event_callback",
                "event_id": "Ev123",
                "event": {"type": "app_mention", "text": "<@U1> hi", "channel": "C1", "ts": "1.0"},
            }
        )
        headers = _signed_headers(body)
        ctx = MagicMock()
        ctx.function_name = "omnisummary-dev-slack-events"
        clients = {}

        def fake_client(name, *a, **k):
            clients[name] = clients.get(name) or MagicMock()
            return clients[name]

        with patch.object(h.boto3, "client", side_effect=fake_client):
            with patch.object(h, "_verify_slack_signature", return_value=True):
                with patch.object(h, "_is_duplicate_event", return_value=False):
                    resp = h.handler({"body": body, "headers": headers}, ctx)
        assert resp["statusCode"] == 200
        # the lambda self-invoke was issued
        assert clients["lambda"].invoke.called
        payload = json.loads(clients["lambda"].invoke.call_args.kwargs["Payload"])
        assert payload["action"] == "invoke_agentcore"
        assert payload["channel"] == "C1"

    def test_duplicate_event_short_circuits(self):
        body = json.dumps({"type": "event_callback", "event_id": "Ev1", "event": {"type": "app_mention"}})
        headers = _signed_headers(body)
        with patch.object(h, "_verify_slack_signature", return_value=True):
            with patch.object(h, "_is_duplicate_event", return_value=True):
                with patch.object(h.boto3, "client") as mock_client:
                    resp = h.handler({"body": body, "headers": headers}, MagicMock())
        assert resp["statusCode"] == 200
        mock_client.return_value.invoke.assert_not_called()


class TestAsyncInvocation:
    def test_invoke_agentcore_calls_runtime(self, monkeypatch):
        monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:::runtime/x")
        event = {
            "action": "invoke_agentcore",
            "text": "<@U1> explain item 1",
            "channel": "C1",
            "thread_ts": "1.0",
            "event_id": "Ev1",
        }
        with patch.object(h, "_is_duplicate_event", return_value=False):
            with patch.object(h, "_post_ack") as ack:
                with patch.object(h.boto3, "client") as mock_client:
                    resp = h.handler(event, MagicMock())
        assert resp["statusCode"] == 200
        mock_client.return_value.invoke_agent_runtime.assert_called_once()
        # The user gets an immediate acknowledgement before the multi-minute runtime call.
        ack.assert_called_once_with("C1", "1.0")

    def test_ack_posts_to_thread(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        client = MagicMock()
        with patch("slack_sdk.web.WebClient", return_value=client):
            h._post_ack("C9", "ts-1")
        client.chat_postMessage.assert_called_once()
        kwargs = client.chat_postMessage.call_args.kwargs
        assert kwargs["channel"] == "C9"
        assert kwargs["thread_ts"] == "ts-1"
        assert ":hourglass_flowing_sand:" in kwargs["blocks"][1]["elements"][0]["text"]

    def test_ack_noop_without_channel(self):
        with patch("slack_sdk.web.WebClient") as wc:
            h._post_ack("", "")
        wc.assert_not_called()
