from __future__ import annotations

import json
import os
from typing import Any

import boto3
from bedrock_agentcore.runtime import BedrockAgentCoreApp
from slack_sdk.web import WebClient

from agent import create_digest_agent
from agent.agent_tools import state_manager
from agent.tool_state import DigestStateManager
from output.slack_handler import _split_message
from shared import S3StateStore, logger, sanitize_slack_mrkdwn, set_correlation_id

app = BedrockAgentCoreApp()


def _load_latest_state() -> None:
    bucket = os.environ.get("STATE_BUCKET", "")
    if not bucket:
        logger.warning("STATE_BUCKET not set, agent has no digest state")
        return

    region = os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "ap-northeast-2"))
    session = boto3.Session(region_name=region)
    store = S3StateStore(session, bucket, prefix="digest_state")

    s3_client = session.client("s3")
    prefix = "digest_state/"
    response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix)

    files = sorted(
        [obj["Key"] for obj in response.get("Contents", []) if obj["Key"].endswith(".json")],
        reverse=True,
    )

    if not files:
        logger.warning("No digest state files found in S3")
        return

    latest_key = files[0]
    content = store.read(latest_key.removeprefix(prefix))
    if not content:
        return

    data = json.loads(content)
    loaded = DigestStateManager.load_from_dict(data)
    state_manager.load_from(loaded)
    logger.info("Loaded %d items from S3 ('%s')", state_manager.get_item_count(), latest_key)


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

    _load_latest_state()

    agent = create_digest_agent()

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
