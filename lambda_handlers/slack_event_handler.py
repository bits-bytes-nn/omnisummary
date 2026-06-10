from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import time
from typing import Any

import boto3

# Use the shared logger so records carry correlation_id (via _CorrelationFilter) for
# continuity from API Gateway through the async AgentCore invocation. Imported directly
# from shared.logger to avoid pulling the heavy shared package __init__ at cold start.
from shared.logger import logger

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
        agentcore_client = boto3.client("bedrock-agentcore")

        clean_text = re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()

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

        logger.info("AgentCore invocation completed for event '%s'", event_id)
        return {"statusCode": 200, "body": "OK"}

    except Exception as e:
        logger.error("AgentCore invocation failed: %s", e, exc_info=True)
        # The outer Slack request already got 200, so without this the user sees
        # nothing when the runtime invocation itself throws (throttle, cold-start
        # timeout). Post a visible fallback to the originating thread.
        _post_fallback(channel, thread_ts)
        return {"statusCode": 500, "body": f"Error: {e}"}


def _post_fallback(channel: str, thread_ts: str) -> None:
    if not channel:
        return
    try:
        from slack_sdk.web import WebClient

        token = _resolve_slack_bot_token()
        if not token:
            return
        # Mirror the family's Slack error convention (see scholar-lens
        # notifier/bot): a header line + a muted retry hint, rather than one
        # bare warning sentence.
        kwargs: dict[str, Any] = {
            "channel": channel,
            "text": "Sorry, I couldn't process that request.",
            "blocks": [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": ":x: I couldn't process that request",
                        "emoji": True,
                    },
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
            ],
        }
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        WebClient(token=token).chat_postMessage(**kwargs)
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
