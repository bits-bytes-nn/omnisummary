import hashlib
import hmac
import json
import time
from unittest.mock import MagicMock, patch

from lambda_handlers import slack_event_handler as h

SIGNING_SECRET = "test-signing-secret"


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
            with patch.object(h.boto3, "client") as mock_client:
                resp = h.handler(event, MagicMock())
        assert resp["statusCode"] == 200
        mock_client.return_value.invoke_agent_runtime.assert_called_once()
