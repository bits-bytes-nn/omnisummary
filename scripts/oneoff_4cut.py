#!/usr/bin/env python3
"""One-off: regenerate today's daily visual as a 4-panel cartoon and post to Slack."""

import asyncio
import os

os.environ.setdefault("AWS_REGION", "ap-northeast-2")
os.environ.setdefault("MEMORY_ID", "omnisummary_dev_digest_state-9GEp6f8MtL")

import boto3

from agent.tool_state import DigestStateManager
from agent.visuals import VisualGenerator
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
    # Let the editor's own format/instruction drive it; just nudge toward a 4-panel comic.
    instruction = "A 4-panel comic. " + plan.get("instruction", "")

    generator = VisualGenerator(
        factory,
        config.pipeline.digest_model,
        image_model=config.pipeline.image_model,
        image_size=config.pipeline.image_size,
    )
    image_bytes, brief = await generator.generate(instruction, source, context)
    logger.info("Brief title: %s", brief.title)
    logger.info("Image prompt: %s", brief.prompt[:600])

    with open("/tmp/oneoff_4cut.png", "wb") as f:
        f.write(image_bytes)
    logger.info("Saved /tmp/oneoff_4cut.png (%d bytes)", len(image_bytes))


if __name__ == "__main__":
    asyncio.run(main())
