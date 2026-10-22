#!/usr/bin/env python3
"""Generate one digest (no publishing) and write the digest text to a file, so the
article-quality workflow can evaluate real output of the current DigestPrompt against a
fixed, cached candidate set. Reuses a cached collected-items snapshot when present so
each round only varies the prompt, not the upstream data.

The digest comes from run_pipeline(dry_run=True), the production path: the editor sees the
same cross-day exclusions, recent titles and recent leads a real run for today would.

Usage: uv run python scripts/sample_digest.py <out_path> [cache_path]
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3

from pipeline.runner import resolve_digest_window, run_pipeline
from shared import BedrockLanguageModelFactory, CollectedItem, Config, logger


async def _collect(config, factory) -> list[CollectedItem]:
    from pipeline import run_collectors_with_health

    items, _ = await run_collectors_with_health(config, factory)
    return items


async def main() -> None:
    parser = argparse.ArgumentParser(description="Generate one digest (nothing is published) and write it to a file")
    parser.add_argument("out_path", type=Path, help="Where to write the output")
    parser.add_argument("cache_path", type=Path, nargs="?", help="Collected-items cache to reuse (written if absent)")
    args = parser.parse_args()
    out_path, cache_path = args.out_path, args.cache_path

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

    digest_date, _ = resolve_digest_window(config)
    _, _, digest = await run_pipeline(config, factory, collected, digest_date=digest_date, dry_run=True)
    if digest is None:
        raise SystemExit("No digest: nothing survived aggregation and ranking")

    out_path.write_text(digest.digest_text, encoding="utf-8")
    stories = len(digest.content.items) if digest.content else 0
    logger.info("Wrote digest (%d chars, %d stories) to %s", len(digest.digest_text), stories, out_path)


if __name__ == "__main__":
    asyncio.run(main())
