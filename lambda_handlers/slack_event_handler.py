from __future__ import annotations

import hashlib
import hmac
import json
import logging
import os
import re
import time
from typing import Any

import boto3
from botocore.config import Config as BotoConfig
from botocore.exceptions import ReadTimeoutError

# This handler is packaged as a standalone zip containing ONLY lambda_handlers/, so it MUST NOT
# import from `shared` (or any sibling package) — those aren't in the zip and the import fails at
# cold start with ImportModuleError. A self-contained stdlib logger keeps the ingress dependency-free.
logger = logging.getLogger("omnisummary.slack_events")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO)

SIGNATURE_EXPIRATION_SEC = int(os.environ.get("SLACK_SIGNATURE_EXPIRATION_SEC", "300"))
EVENT_DEDUP_TTL_SEC = int(os.environ.get("EVENT_DEDUPLICATION_TTL_SEC", "300"))


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    action = event.get("action")

    if action == "invoke_agentcore":
        return _handle_async_invocation(event, context)

    return _handle_slack_event(event, context)


def _handle_slack_event(event: dict[str, Any], context: Any) -> dict[str, Any]:
    headers = event.get("headers", {})
    body = event.get("body", "")

    try:
        data = json.loads(body) if isinstance(body, str) and body else (body if isinstance(body, dict) else {})
    except (json.JSONDecodeError, TypeError):
        return {"statusCode": 400, "body": "Bad Request"}

    if data.get("type") == "url_verification":
        return {"statusCode": 200, "body": data.get("challenge", "")}

    if not _verify_slack_signature(headers, body):
        logger.warning("Slack signature verification failed")
        return {"statusCode": 401, "body": "Unauthorized"}

    if data.get("type") == "event_callback":
        evt = data.get("event", {})
        if evt.get("type") == "app_mention":
            event_id = data.get("event_id", "")
            if _is_duplicate_event(event_id):
                logger.info("Duplicate event '%s', skipping", event_id)
                return {"statusCode": 200, "body": "OK"}

            lambda_client = boto3.client("lambda")
            lambda_client.invoke(
                FunctionName=context.function_name,
                InvocationType="Event",
                Payload=json.dumps(
                    {
                        "action": "invoke_agentcore",
                        "text": evt.get("text", ""),
                        "channel": evt.get("channel", ""),
                        "thread_ts": evt.get("thread_ts") or evt.get("ts", ""),
                        "event_id": event_id,
                    }
                ),
            )

    return {"statusCode": 200, "body": "OK"}


def _handle_async_invocation(event: dict[str, Any], context: Any) -> dict[str, Any]:
    text = event.get("text", "")
    channel = event.get("channel", "")
    thread_ts = event.get("thread_ts", "")
    event_id = event.get("event_id", "")

    invocation_id = hashlib.sha256(f"{event_id}:{text}".encode()).hexdigest()[:16]
    if _is_duplicate_event(invocation_id):
        logger.info("Duplicate invocation '%s', skipping", invocation_id)
        return {"statusCode": 200, "body": "OK"}

    try:
        agentcore_arn = os.environ["AGENTCORE_RUNTIME_ARN"]
        # invoke_agent_runtime is a SYNCHRONOUS, streaming call that blocks until the agent
        # finishes — but deep research takes minutes, far beyond this Lambda's 60s timeout. The
        # runtime delivers its own result to Slack/Threads (via the deliver_report tool), so we
        # only need to START it, not await the response body. Use a short read timeout and treat
        # the resulting ReadTimeoutError as "successfully fired": the runtime keeps running on its
        # own after we disconnect. Without this the Lambda times out, which (a) trips the Errors +
        # Timeout alarms and (b) makes the async self-invoke RETRY, double-running the research.
        agentcore_client = boto3.client(
            "bedrock-agentcore",
            config=BotoConfig(read_timeout=5, connect_timeout=5, retries={"max_attempts": 0}),
        )

        clean_text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()

        # Immediate acknowledgement so the user knows the request was received — deep research
        # takes minutes, so without this the thread stays silent. Mirrors scholar-lens' ack
        # convention (intent line + hourglass "I'll post the result here when it's ready").
        _post_ack(channel, thread_ts)

        try:
            agentcore_client.invoke_agent_runtime(
                agentRuntimeArn=agentcore_arn,
                qualifier="DEFAULT",
                payload=json.dumps(
                    {
                        "prompt": clean_text,
                        "channel_id": channel,
                        "thread_ts": thread_ts,
                    }
                ),
            )
        except ReadTimeoutError:
            # Expected: the request reached the runtime and it's now working; we intentionally
            # don't wait for the (minutes-long) streamed response. NOT a failure.
            logger.info("AgentCore invocation dispatched for event '%s' (not awaiting result)", event_id)
            return {"statusCode": 200, "body": "OK"}

        logger.info("AgentCore invocation returned synchronously for event '%s'", event_id)
        return {"statusCode": 200, "body": "OK"}

    except Exception as e:
        logger.error("AgentCore invocation failed: %s", e, exc_info=True)
        # A real dispatch failure (bad ARN, throttle, access denied) — the outer Slack request
        # already got 200, so post a visible fallback to the originating thread.
        _post_fallback(channel, thread_ts)
        return {"statusCode": 500, "body": f"Error: {e}"}


def _slack_post_message(channel: str, text: str, blocks: list[dict[str, Any]], thread_ts: str) -> None:
    """POST to chat.postMessage with stdlib urllib only. This handler ships as a standalone zip
    with NO third-party deps (no slack_sdk), so we must not import it — doing so crashed _post_ack
    at runtime ('No module named slack_sdk'). Best-effort: any failure is logged, never raised."""
    import urllib.request

    if not channel:
        return
    token = _resolve_slack_bot_token()
    if not token:
        return
    payload: dict[str, Any] = {"channel": channel, "text": text, "blocks": blocks}
    if thread_ts:
        payload["thread_ts"] = thread_ts
    req = urllib.request.Request(
        "https://slack.com/api/chat.postMessage",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json; charset=utf-8"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    if not result.get("ok"):
        logger.error("Slack chat.postMessage failed: %s", result.get("error"))


def _post_ack(channel: str, thread_ts: str) -> None:
    """Post an immediate 'research started' acknowledgement to the originating thread, so the
    user gets feedback during the multi-minute run. Mirrors scholar-lens' ack format (an intent
    line + a muted hourglass hint). Best-effort: never blocks the runtime invocation."""
    blocks: list[dict[str, Any]] = [
        {"type": "section", "text": {"type": "mrkdwn", "text": ":satellite: *Deep research* started."}},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":hourglass_flowing_sand: Gathering and synthesizing sources — "
                    "I'll post the result in this thread when it's ready.",
                }
            ],
        },
    ]
    try:
        _slack_post_message(channel, "Deep research started.", blocks, thread_ts)
    except Exception as e:
        logger.error("Failed to post ack message: %s", e)


def _post_fallback(channel: str, thread_ts: str) -> None:
    # Mirror the family's Slack error convention (see scholar-lens notifier/bot): a header line
    # + a muted retry hint, rather than one bare warning sentence.
    blocks: list[dict[str, Any]] = [
        {
            "type": "header",
            "text": {"type": "plain_text", "text": ":x: I couldn't process that request", "emoji": True},
        },
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": ":arrows_counterclockwise: Please mention me "
                    "again in a moment, or check the logs if it keeps happening.",
                }
            ],
        },
    ]
    try:
        _slack_post_message(channel, "Sorry, I couldn't process that request.", blocks, thread_ts)
    except Exception as e:
        logger.error("Failed to post fallback message: %s", e)


def _resolve_slack_bot_token() -> str:
    token = os.environ.get("SLACK_BOT_TOKEN", "")
    if token:
        return token
    project = os.environ.get("PROJECT_NAME", "omnisummary")
    stage = os.environ.get("STAGE", "dev")
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"))
    try:
        return boto3.client("ssm", region_name=region).get_parameter(
            Name=f"/{project}/{stage}/slack-bot-token",
            WithDecryption=True,
        )["Parameter"]["Value"]
    except Exception as e:
        logger.error("Failed to resolve Slack bot token for fallback: %s", e)
        return ""


def _verify_slack_signature(headers: dict[str, str], body: str) -> bool:
    lower_headers = {k.lower(): v for k, v in headers.items()}
    timestamp = lower_headers.get("x-slack-request-timestamp", "")
    signature = lower_headers.get("x-slack-signature", "")

    if not timestamp or not signature:
        return False

    if abs(time.time() - float(timestamp)) > SIGNATURE_EXPIRATION_SEC:
        return False

    project_name = os.environ.get("PROJECT_NAME", "omnisummary")
    stage = os.environ.get("STAGE", "dev")
    ssm = boto3.client("ssm")

    try:
        secret = ssm.get_parameter(
            Name=f"/{project_name}/{stage}/slack-signing-secret",
            WithDecryption=True,
        )[
            "Parameter"
        ]["Value"]
    except Exception as e:
        logger.error("Failed to fetch Slack signing secret: %s", e)
        return False

    sig_basestring = f"v0:{timestamp}:{body}"
    my_signature = "v0=" + hmac.new(secret.encode(), sig_basestring.encode(), hashlib.sha256).hexdigest()
    return hmac.compare_digest(my_signature, signature)


def _is_duplicate_event(event_id: str) -> bool:
    table_name = os.environ.get("DDB_TABLE_NAME", "")
    if not table_name:
        return False

    ddb = boto3.resource("dynamodb")
    table = ddb.Table(table_name)

    try:
        table.put_item(
            Item={"event_id": event_id, "ttl": int(time.time()) + EVENT_DEDUP_TTL_SEC},
            ConditionExpression="attribute_not_exists(event_id)",
        )
        return False
    except ddb.meta.client.exceptions.ConditionalCheckFailedException:
        return True
