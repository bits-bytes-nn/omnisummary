#!/usr/bin/env python3
"""One-off: regenerate today's daily visual as a 4-panel cartoon and post to Slack."""

import asyncio
import os

os.environ.setdefault("AWS_REGION", "ap-northeast-2")
os.environ.setdefault("MEMORY_ID", "omnisummary_dev_digest_state-9GEp6f8MtL")

import boto3

from agent.tool_state import DigestStateManager
from agent.visuals import VisualGenerator
from output.slack_handler import send_image_to_slack
from pipeline.daily_visual import DailyVisualMaker
from shared import BedrockLanguageModelFactory, Config, create_memory_store, logger


async def main() -> None:
    config = Config.load()
    session = boto3.Session(region_name=config.aws.bedrock_region, profile_name=config.aws.profile or None)
    factory = BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)

    data = create_memory_store().get_latest_digest()
    if not data:
        logger.error("No digest state in AgentCore Memory")
        return
    mgr = DigestStateManager.load_from_dict(data)
    ranked = mgr.get_ranked_items()
    logger.info("Loaded %d ranked items", len(ranked))

    maker = DailyVisualMaker(config, factory)
    plan = await maker._pick_story(ranked)
    n = plan.get("item_number", 0)
    if not (1 <= n <= len(ranked)):
        logger.error("Editor returned no usable story: %s", plan)
        return
    chosen = ranked[n - 1]
    logger.info("Editor picked item #%d: %s", n, chosen.item.title)

    source = f"{chosen.item.title}\n\n{chosen.item.text}"
    context = await maker._gather_context(plan.get("search_query", ""))
    instruction = (
        "A 4-panel cartoon (2x2 grid) telling ONE connected story about this item. "
        "Keep the same characters and art style across all four panels; each panel follows "
        "causally from the previous one (setup -> escalation -> twist -> punchline). Make it "
        "genuinely funny with internet-meme energy and parody. Bake in recognizable real-world "
        "cues (the actual people's likenesses, company logos, brand colors) so it reads without "
        "a caption. All on-image text/speech bubbles must be SHORT ENGLISH (the image model "
        "garbles Korean glyphs). Original angle: " + plan.get("instruction", "")
    )

    generator = VisualGenerator(factory, config.pipeline.digest_model)
    image_bytes, brief = await generator.generate(instruction, source, context)
    logger.info("Brief title: %s", brief.get("title"))

    ok = await send_image_to_slack(
        image_bytes,
        channel_id=config.slack.channel_id,
        title=brief.get("title", "4컷 카툰"),
        comment=f"*{brief.get('title', '4컷 카툰')}* (4컷 재생성)\n{brief.get('caption', '')}",
        bot_token=config.slack.bot_token,
    )
    logger.info("Posted to Slack: %s", ok)


if __name__ == "__main__":
    asyncio.run(main())
