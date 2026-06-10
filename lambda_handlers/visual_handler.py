from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import boto3

from agent.tool_state import DigestStateManager
from pipeline.daily_visual import DailyVisualMaker
from shared import BedrockLanguageModelFactory, Config, create_memory_store, logger, set_correlation_id


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Daily-visual Lambda, invoked asynchronously by the digest Lambda so visual
    generation (LLM editor + Tavily + gpt-image, ~1-2 min) stays off the digest's
    critical path. Loads today's ranked items from AgentCore Memory and posts one
    fun visual to Slack. Best-effort: any failure is logged, never retried hard."""
    set_correlation_id(getattr(context, "aws_request_id", "") or None)
    logger.info("Visual Lambda invoked")
    try:
        asyncio.run(_run())
        return {"statusCode": 200, "body": "Visual completed"}
    except Exception as e:
        logger.error("Visual Lambda failed: %s", e, exc_info=True)
        return {"statusCode": 500, "body": f"Visual error: {e}"}


async def _run() -> None:
    config = Config.load()
    if not config.pipeline.enable_daily_visual:
        logger.info("Daily visual disabled, skipping")
        return

    data = create_memory_store().get_latest_digest()
    if not data:
        logger.warning("No digest state in AgentCore Memory, skipping visual")
        return
    state = DigestStateManager.load_from_dict(data)
    ranked_items = state.get_ranked_items()
    content = state.get_content()

    session = boto3.Session(region_name=config.aws.bedrock_region)
    factory = BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)

    today = datetime.now(ZoneInfo(config.aws.timezone)).date()
    posted = await DailyVisualMaker(config, factory).run(ranked_items, content, today=today)
    logger.info("Daily visual %s", "posted" if posted else "skipped")
