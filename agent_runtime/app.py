from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from slack_sdk.web import WebClient

from agent import create_research_agent
from agent.research_tools import DeliveryContext, request_context
from output.renderers import render_agent_blocks
from shared import logger, sanitize_slack_mrkdwn, set_correlation_id

app = BedrockAgentCoreApp()


def _emit_agent_error_metric() -> None:
    """Emit a CloudWatch EMF error metric so a systemic agent break is alarmable — the runtime
    catches its own exceptions and replies with text, so nothing else would record a failure."""
    emf = {
        "_aws": {
            "Timestamp": int(datetime.now().timestamp() * 1000),
            "CloudWatchMetrics": [
                {"Namespace": "OmniSummary", "Dimensions": [[]], "Metrics": [{"Name": "AgentErrors"}]}
            ],
        },
        "AgentErrors": 1,
    }
    print(json.dumps(emf))


def _resolve_bot_token() -> str:
    bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
    if bot_token:
        return bot_token
    project = os.environ.get("PROJECT_NAME", "omnisummary")
    stage = os.environ.get("STAGE", "dev")
    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"))
    try:
        return boto3.client("ssm", region_name=region).get_parameter(
            Name=f"/{project}/{stage}/slack-bot-token",
            WithDecryption=True,
        )["Parameter"]["Value"]
    except Exception as e:
        logger.error("Failed to get Slack token: %s", e)
        return ""


def _send_slack_message(channel: str, text: str, thread_ts: str = "") -> None:
    """Fallback delivery: post the agent's final text to Slack when the agent finished without
    calling deliver_report. The happy path delivers through the deliver_report tool instead."""
    bot_token = _resolve_bot_token()
    if not bot_token:
        return
    client = WebClient(token=bot_token)
    for blocks in render_agent_blocks(text):
        kwargs: dict[str, Any] = {"channel": channel, "blocks": blocks, "text": text[:200]}
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        client.chat_postMessage(**kwargs)


@app.entrypoint
def invoke(payload: dict[str, Any]) -> str:
    prompt = payload.get("prompt", "")
    channel_id = payload.get("channel_id", "")
    thread_ts = payload.get("thread_ts", "")

    set_correlation_id(payload.get("correlation_id") or None)
    logger.info("AgentCore invoked: prompt='%s', channel='%s'", prompt[:100], channel_id)

    delivery = DeliveryContext(channel_id=channel_id, thread_ts=thread_ts)
    agent = create_research_agent()

    # contextvar-scoped per-invocation delivery: a warm container handling concurrent
    # invocations can't leak one request's channel into another.
    with request_context(delivery):
        try:
            response = sanitize_slack_mrkdwn(str(agent(prompt)))
        except Exception as e:
            logger.error("Agent execution failed: %s", e, exc_info=True)
            _emit_agent_error_metric()
            response = f"Error processing request: {e}"

        # Fallback ONLY when the agent delivered to NO channel at all (it never called
        # deliver_report, or every delivery failed) — so the user always gets something. Do NOT
        # fall back to Slack just because Slack wasn't a target: a Threads-only request that
        # succeeded on Threads must not also dump the (Threads-formatted) report into Slack.
        # Prefer the actual report the agent produced over its terminal one-line confirmation.
        if channel_id and not delivery.delivered_channels:
            fallback_text = delivery.last_report or response
            _send_slack_message(channel_id, sanitize_slack_mrkdwn(fallback_text), thread_ts)

    return response


if __name__ == "__main__":
    app.run()
