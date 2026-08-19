import ast
import hashlib
import hmac
import importlib.util
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

from lambda_handlers import slack_event_handler as h

SIGNING_SECRET = "test-signing-secret"


def test_handler_imports_nothing_outside_the_zip():
    # This handler ships as a standalone zip containing ONLY lambda_handlers/ — no sibling packages
    # (shared, agent, ...) AND no third-party deps (slack_sdk, httpx, ...). Importing either crashes
    # at cold start: 'No module named shared' (sibling) / 'No module named slack_sdk' (third-party),
    # which 502s the Slack ingress. boto3 + botocore (boto3's own dependency, always co-installed in
    # the Lambda runtime) + the stdlib are the only things present. The test env CAN import these, so
    # scan the source instead of importing at runtime.
    allowed = {"boto3", "botocore"}  # present in the AWS Lambda Python runtime (botocore ships with boto3)
    src = Path(h.__file__).read_text()
    tree = ast.parse(src)
    bad: list[str] = []
    for node in ast.walk(tree):
        names: list[str] = []
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names = [node.module]
        elif isinstance(node, ast.Import):
            names = [n.name for n in node.names]
        for name in names:
            top = name.split(".")[0]
            if top in allowed:
                continue
            # Anything resolved from site-packages isn't in the zip (only stdlib + boto3 are).
            spec = importlib.util.find_spec(top)
            if spec and "site-packages" in (spec.origin or ""):
                bad.append(name)
    assert not bad, f"slack_event_handler must import only stdlib + boto3 (zip has nothing else): {bad}"


def _signed_headers(body: str, secret: str = SIGNING_SECRET, ts: str | None = None) -> dict[str, str]:
    ts = ts or str(int(time.time()))
    sig = "v0=" + hmac.new(secret.encode(), f"v0:{ts}:{body}".encode(), hashlib.sha256).hexdigest()
    return {"X-Slack-Request-Timestamp": ts, "X-Slack-Signature": sig}


class TestUrlVerification:
    def test_challenge_echoed_for_a_signed_request(self):
        body = json.dumps({"type": "url_verification", "challenge": "abc123"})
        with patch.object(h.boto3, "client") as mock_client:
            mock_client.return_value.get_parameter.return_value = {"Parameter": {"Value": SIGNING_SECRET}}
            resp = h.handler({"body": body, "headers": _signed_headers(body)}, None)
        assert resp["statusCode"] == 200
        assert resp["body"] == "abc123"

    def test_unsigned_challenge_is_rejected(self):
        # The handshake used to be answered BEFORE verification, so anyone who could reach the
        # endpoint got an unauthenticated echo of attacker-chosen bytes.
        body = json.dumps({"type": "url_verification", "challenge": "abc123"})
        resp = h.handler({"body": body, "headers": {}}, None)
        assert resp["statusCode"] == 401
        assert resp["body"] == "Unauthorized"

    def test_badly_signed_challenge_is_rejected(self):
        body = json.dumps({"type": "url_verification", "challenge": "abc123"})
        headers = {"X-Slack-Request-Timestamp": str(int(time.time())), "X-Slack-Signature": "v0=deadbeef"}
        with patch.object(h.boto3, "client") as mock_client:
            mock_client.return_value.get_parameter.return_value = {"Parameter": {"Value": SIGNING_SECRET}}
            resp = h.handler({"body": body, "headers": headers}, None)
        assert resp["statusCode"] == 401


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

    def test_non_numeric_timestamp_rejected_cleanly(self):
        # A malformed timestamp must fail verification (401), not raise a ValueError → 502 that
        # Slack would then retry.
        body = json.dumps({"type": "event_callback", "event": {"type": "app_mention"}})
        headers = {"X-Slack-Request-Timestamp": "not-a-number", "X-Slack-Signature": "v0=abc"}
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
        # Minted at the ingress and carried onward, so one research run is traceable across hops.
        assert payload["correlation_id"]

    def test_duplicate_event_short_circuits(self):
        body = json.dumps({"type": "event_callback", "event_id": "Ev1", "event": {"type": "app_mention"}})
        headers = _signed_headers(body)
        with patch.object(h, "_verify_slack_signature", return_value=True):
            with patch.object(h, "_is_duplicate_event", return_value=True):
                with patch.object(h.boto3, "client") as mock_client:
                    resp = h.handler({"body": body, "headers": headers}, MagicMock())
        assert resp["statusCode"] == 200
        mock_client.return_value.invoke.assert_not_called()

    def test_dispatch_failure_releases_marker_and_500s(self):
        # If the self-invoke fails AFTER the dedup marker was written, the marker must be released
        # and a 500 returned so Slack's retry hits a clean state instead of being dropped as a dup.
        body = json.dumps(
            {"type": "event_callback", "event_id": "EvX", "event": {"type": "app_mention", "channel": "C1"}}
        )
        headers = _signed_headers(body)
        ctx = MagicMock()
        ctx.function_name = "fn"
        lambda_client = MagicMock()
        lambda_client.invoke.side_effect = RuntimeError("throttled")
        with patch.object(h, "_verify_slack_signature", return_value=True):
            with patch.object(h, "_is_duplicate_event", return_value=False):
                with patch.object(h, "_release_event_marker") as release:
                    with patch.object(h.boto3, "client", return_value=lambda_client):
                        resp = h.handler({"body": body, "headers": headers}, ctx)
        assert resp["statusCode"] == 500
        release.assert_called_once_with("EvX")


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
        ctx = MagicMock()
        ctx.aws_request_id = "abcdef12-3456-7890-abcd-ef1234567890"
        with patch.object(h, "_is_duplicate_event", return_value=False):
            with patch.object(h, "_post_ack") as ack:
                with patch.object(h.boto3, "client") as mock_client:
                    resp = h.handler(event, ctx)
        assert resp["statusCode"] == 200
        mock_client.return_value.invoke_agent_runtime.assert_called_once()
        # The user gets an immediate acknowledgement before the multi-minute runtime call.
        ack.assert_called_once_with("C1", "1.0")
        kwargs = mock_client.return_value.invoke_agent_runtime.call_args.kwargs
        # The runtime already reads correlation_id off the payload; it just never had a producer.
        assert json.loads(kwargs["payload"])["correlation_id"] == "abcdef123456"
        # Same trace in AgentCore's own per-session logs; the API requires >= 33 characters.
        assert "abcdef123456" in kwargs["runtimeSessionId"] and len(kwargs["runtimeSessionId"]) >= 33

    def test_the_ingress_correlation_id_is_preserved_across_the_self_invoke(self, monkeypatch):
        monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:::runtime/x")
        event = {
            "action": "invoke_agentcore",
            "text": "hi",
            "channel": "C1",
            "thread_ts": "1.0",
            "event_id": "Ev1",
            "correlation_id": "fromingress1",
        }
        ctx = MagicMock()
        ctx.aws_request_id = "a-different-request-id"
        with patch.object(h, "_is_duplicate_event", return_value=False):
            with patch.object(h, "_post_ack"):
                with patch.object(h.boto3, "client") as mock_client:
                    h.handler(event, ctx)
        kwargs = mock_client.return_value.invoke_agent_runtime.call_args.kwargs
        assert json.loads(kwargs["payload"])["correlation_id"] == "fromingress1"

    def test_read_timeout_is_treated_as_successful_dispatch(self, monkeypatch):
        # invoke_agent_runtime blocks for minutes; we fire it with a short read timeout and do
        # NOT await the streamed result (the runtime delivers to Slack itself). A ReadTimeoutError
        # therefore means "dispatched OK" → 200 and NO error fallback, so the async self-invoke
        # never retries and double-runs the research.
        from botocore.exceptions import ReadTimeoutError

        monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:::runtime/x")
        event = {"action": "invoke_agentcore", "text": "<@U1> research", "channel": "C1", "event_id": "Ev2"}
        client = MagicMock()
        client.invoke_agent_runtime.side_effect = ReadTimeoutError(endpoint_url="https://x")
        with patch.object(h, "_is_duplicate_event", return_value=False):
            with patch.object(h, "_post_ack"):
                with patch.object(h, "_post_fallback") as fallback:
                    with patch.object(h.boto3, "client", return_value=client):
                        resp = h.handler(event, MagicMock())
        assert resp["statusCode"] == 200
        fallback.assert_not_called()  # a read timeout is success, not a failure

    def test_real_dispatch_error_posts_fallback(self, monkeypatch):
        # A genuine dispatch failure (bad ARN, throttle) must still surface a fallback to the user.
        monkeypatch.setenv("AGENTCORE_RUNTIME_ARN", "arn:aws:bedrock-agentcore:::runtime/x")
        event = {"action": "invoke_agentcore", "text": "<@U1> research", "channel": "C1", "event_id": "Ev3"}
        client = MagicMock()
        client.invoke_agent_runtime.side_effect = RuntimeError("AccessDenied")
        with patch.object(h, "_is_duplicate_event", return_value=False):
            with patch.object(h, "_post_ack"):
                with patch.object(h, "_post_fallback") as fallback:
                    with patch.object(h.boto3, "client", return_value=client):
                        resp = h.handler(event, MagicMock())
        assert resp["statusCode"] == 500
        fallback.assert_called_once()

    def test_ack_posts_to_thread_via_stdlib(self, monkeypatch):
        # ack must post with stdlib urllib (no slack_sdk — not in the zip). Capture the request.
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        captured = {}

        class _Resp:
            def read(self):
                return b'{"ok": true}'

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=0):
            captured["url"] = req.full_url
            captured["auth"] = req.headers.get("Authorization")
            captured["body"] = json.loads(req.data.decode())
            return _Resp()

        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            h._post_ack("C9", "ts-1")
        assert captured["url"] == "https://slack.com/api/chat.postMessage"
        assert captured["auth"] == "Bearer xoxb-test"
        assert captured["body"]["channel"] == "C9"
        assert captured["body"]["thread_ts"] == "ts-1"
        assert ":hourglass_flowing_sand:" in captured["body"]["blocks"][1]["elements"][0]["text"]

    def test_ack_noop_without_channel(self, monkeypatch):
        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-test")
        with patch("urllib.request.urlopen") as urlopen:
            h._post_ack("", "")
        urlopen.assert_not_called()


def _ddb(table: MagicMock) -> MagicMock:
    """A boto3 dynamodb resource stand-in whose Table() returns `table` and whose client exposes
    the real-shaped ConditionalCheckFailedException class."""
    resource = MagicMock()
    resource.Table.return_value = table
    resource.meta.client.exceptions.ConditionalCheckFailedException = _ConditionalCheckFailed
    return resource


class _ConditionalCheckFailed(Exception):
    pass


class TestEventDedup:
    """Exercises the real _is_duplicate_event / _release_event_marker (no patching of the function
    under test): the conditional PutItem IS the dedup, and Slack retries every mention up to 3x."""

    def test_no_table_configured_never_dedups(self, monkeypatch):
        monkeypatch.delenv("DDB_TABLE_NAME", raising=False)
        with patch.object(h.boto3, "resource") as resource:
            assert h._is_duplicate_event("Ev1") is False
        resource.assert_not_called()  # no table -> no AWS call at all

    def test_first_event_writes_marker_with_ttl(self, monkeypatch):
        monkeypatch.setenv("DDB_TABLE_NAME", "dedup")
        table = MagicMock()
        with patch.object(h.boto3, "resource", return_value=_ddb(table)):
            assert h._is_duplicate_event("Ev1") is False
        kwargs = table.put_item.call_args.kwargs
        assert kwargs["Item"]["event_id"] == "Ev1"
        assert kwargs["Item"]["ttl"] > int(time.time())  # marker expires, never accumulates
        # The conditional write is what makes this atomic against Slack's concurrent retries.
        assert kwargs["ConditionExpression"] == "attribute_not_exists(event_id)"

    def test_second_event_is_a_duplicate(self, monkeypatch):
        monkeypatch.setenv("DDB_TABLE_NAME", "dedup")
        table = MagicMock()
        table.put_item.side_effect = _ConditionalCheckFailed()
        with patch.object(h.boto3, "resource", return_value=_ddb(table)):
            assert h._is_duplicate_event("Ev1") is True

    def test_dedup_store_failure_fails_open(self, monkeypatch):
        # A throttled/missing table must not swallow a user's mention: fail open and let it run.
        monkeypatch.setenv("DDB_TABLE_NAME", "dedup")
        table = MagicMock()
        table.put_item.side_effect = RuntimeError("throttled")
        with patch.object(h.boto3, "resource", return_value=_ddb(table)):
            with patch.object(h, "logger") as log:
                assert h._is_duplicate_event("Ev1") is False
        assert log.warning.called

    def test_release_marker_deletes_the_key(self, monkeypatch):
        monkeypatch.setenv("DDB_TABLE_NAME", "dedup")
        table = MagicMock()
        with patch.object(h.boto3, "resource", return_value=_ddb(table)):
            h._release_event_marker("EvX")
        assert table.delete_item.call_args.kwargs["Key"] == {"event_id": "EvX"}

    def test_release_marker_noop_without_table(self, monkeypatch):
        monkeypatch.delenv("DDB_TABLE_NAME", raising=False)
        with patch.object(h.boto3, "resource") as resource:
            h._release_event_marker("EvX")
        resource.assert_not_called()

    def test_release_marker_swallows_errors(self, monkeypatch):
        monkeypatch.setenv("DDB_TABLE_NAME", "dedup")
        table = MagicMock()
        table.delete_item.side_effect = RuntimeError("denied")
        with patch.object(h.boto3, "resource", return_value=_ddb(table)):
            with patch.object(h, "logger") as log:
                h._release_event_marker("EvX")  # must not raise into the handler
        assert log.warning.called

    def test_slack_retry_of_the_same_mention_is_dispatched_once(self, monkeypatch):
        # End-to-end through handler(): Slack redelivers the same event_id, and only the first
        # delivery may reach the self-invoke.
        monkeypatch.setenv("DDB_TABLE_NAME", "dedup")
        body = json.dumps(
            {
                "type": "event_callback",
                "event_id": "EvDup",
                "event": {"type": "app_mention", "text": "<@U1> hi", "channel": "C1", "ts": "1.0"},
            }
        )
        headers = _signed_headers(body)
        table = MagicMock()
        seen: set[str] = set()

        def put_item(**kwargs):
            event_id = kwargs["Item"]["event_id"]
            if event_id in seen:
                raise _ConditionalCheckFailed()
            seen.add(event_id)

        table.put_item.side_effect = put_item
        lambda_client = MagicMock()
        ctx = MagicMock()
        ctx.function_name = "fn"

        def fake_client(name, *a, **k):
            return lambda_client

        with patch.object(h, "_verify_slack_signature", return_value=True):
            with patch.object(h.boto3, "resource", return_value=_ddb(table)):
                with patch.object(h.boto3, "client", side_effect=fake_client):
                    first = h.handler({"body": body, "headers": headers}, ctx)
                    second = h.handler({"body": body, "headers": headers}, ctx)
        assert first["statusCode"] == 200 and second["statusCode"] == 200
        assert lambda_client.invoke.call_count == 1  # the retry was suppressed
