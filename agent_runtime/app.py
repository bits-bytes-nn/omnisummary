from __future__ import annotations

import os
from typing import Any

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from slack_sdk.web import WebClient

from agent import create_digest_agent
from agent.agent_tools import DeliveryContext, request_context
from agent.tool_state import DigestStateManager
from output.slack_handler import _split_message
from shared import create_memory_store, logger, sanitize_slack_mrkdwn, set_correlation_id

app = BedrockAgentCoreApp()


def _load_latest_state() -> DigestStateManager:
    state = DigestStateManager()
    memory = create_memory_store()
    data = memory.get_latest_digest()
    if not data:
        logger.warning("No digest state available in AgentCore Memory")
        return state

    state.load_from(DigestStateManager.load_from_dict(data))
    logger.info("Loaded %d items from AgentCore Memory", state.get_item_count())
    return state


def _send_slack_message(channel: str, text: str, thread_ts: str = "") -> None:

    bot_token = os.environ.get("SLACK_BOT_TOKEN", "")
    if not bot_token:
        project = os.environ.get("PROJECT_NAME", "omnisummary")
        stage = os.environ.get("STAGE", "dev")
        region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"))
        ssm = boto3.client("ssm", region_name=region)
        try:
            bot_token = ssm.get_parameter(
                Name=f"/{project}/{stage}/slack-bot-token",
                WithDecryption=True,
            )[
                "Parameter"
            ]["Value"]
        except Exception as e:
            logger.error("Failed to get Slack token: %s", e)
            return

    client = WebClient(token=bot_token)
    chunks = _split_message(text)
    for chunk in chunks:
        kwargs: dict[str, Any] = {"channel": channel, "text": chunk}
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

    state = _load_latest_state()
    delivery = DeliveryContext(channel_id=channel_id, thread_ts=thread_ts)

    agent = create_digest_agent()

    # contextvar-scoped per-invocation state: a warm container handling concurrent
    # invocations can't leak one request's channel/state into another.
    with request_context(state, delivery):
        try:
            response = str(agent(prompt))
            response = sanitize_slack_mrkdwn(response)
        except Exception as e:
            logger.error("Agent execution failed: %s", e, exc_info=True)
            response = f"Error processing request: {e}"

    if channel_id:
        _send_slack_message(channel_id, response, thread_ts)

    return response


if __name__ == "__main__":
    app.run()
