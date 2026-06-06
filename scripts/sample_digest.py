#!/usr/bin/env python3
"""Generate one digest (no Slack send) and write the digest text to a file, so the
article-quality workflow can evaluate real output of the current DigestPrompt against a
fixed, cached candidate set. Reuses a cached collected-items snapshot when present so
each round only varies the prompt, not the upstream data.

Usage: uv run python scripts/sample_digest.py <out_path> [cache_path]
"""
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3

from pipeline import ContentAggregator, ContentRanker, DigestGenerator, TrendTracker
from shared import BedrockLanguageModelFactory, CollectedItem, Config, create_state_store, logger


async def _collect(config, factory) -> list[CollectedItem]:
    from main import run_collectors_with_health

    items, _ = await run_collectors_with_health(config, factory)
    return items


async def main() -> None:
    out_path = Path(sys.argv[1])
    cache_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    config = Config.load()
    session = boto3.Session(region_name=config.aws.bedrock_region, profile_name=config.aws.profile or None)
    factory = BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)

    if cache_path and cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        collected = [CollectedItem.model_validate(it) for it in raw]
        logger.info("Loaded %d cached collected items", len(collected))
    else:
        collected = await _collect(config, factory)
        if cache_path:
            cache_path.write_text(
                json.dumps([it.model_dump(mode="json") for it in collected], ensure_ascii=False), encoding="utf-8"
            )
            logger.info("Cached %d collected items to %s", len(collected), cache_path)

    items = ContentAggregator().aggregate(collected)
    ranked = await ContentRanker(config.pipeline, factory).rank(items)
    trends = TrendTracker(config.pipeline, factory, create_state_store(config)).get_trends_context()
    digest = await DigestGenerator(config.pipeline, factory).generate(ranked, items, trends_context=trends)

    out_path.write_text(digest.digest_text, encoding="utf-8")
    logger.info("Wrote digest (%d chars, %d items) to %s", len(digest.digest_text), len(ranked), out_path)


if __name__ == "__main__":
    asyncio.run(main())
