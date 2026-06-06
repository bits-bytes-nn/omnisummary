#!/usr/bin/env python3
"""Produce the daily-visual editor pick + synopsis brief (TEXT only, no image render) so
the article-quality workflow can evaluate the visual concept's message/expression/aesthetics
without paying for gpt-image. Reuses the same cached candidate set as sample_digest.py.

Usage: uv run python scripts/sample_visual_brief.py <out_json> [cache_path]
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3

from agent.visuals import VisualGenerator
from pipeline import ContentAggregator, ContentRanker
from pipeline.daily_visual import DailyVisualMaker
from shared import BedrockLanguageModelFactory, CollectedItem, Config, logger


async def main() -> None:
    out_path = Path(sys.argv[1])
    cache_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    config = Config.load()
    session = boto3.Session(region_name=config.aws.bedrock_region, profile_name=config.aws.profile or None)
    factory = BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)

    if cache_path and cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        collected = [CollectedItem.model_validate(it) for it in raw]
    else:
        from main import run_collectors_with_health

        collected, _ = await run_collectors_with_health(config, factory)

    items = ContentAggregator().aggregate(collected)
    ranked = await ContentRanker(config.pipeline, factory).rank(items)

    maker = DailyVisualMaker(config, factory)
    plan = await maker._pick_story(ranked)
    out: dict = {"editor_plan": plan}

    if plan and not plan.get("skip"):
        n = plan.get("item_number", 0)
        if 1 <= n <= len(ranked):
            chosen = ranked[n - 1]
            source = f"{chosen.item.title}\n\n{chosen.item.text}"
            instruction = plan.get("instruction", "")
            generator = VisualGenerator(
                factory,
                config.pipeline.digest_model,
                source_max_tokens=config.pipeline.visual_synopsis_source_max_tokens,
                context_max_tokens=config.pipeline.visual_synopsis_context_max_tokens,
            )
            brief = await generator.brief(instruction, source, "")
            out["chosen_item_title"] = chosen.item.title
            out["brief"] = brief.model_dump()

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote visual brief to %s (skip=%s)", out_path, bool(not out.get("brief")))


if __name__ == "__main__":
    asyncio.run(main())
