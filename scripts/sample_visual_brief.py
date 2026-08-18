#!/usr/bin/env python3
"""Produce the daily-visual editor pick + synopsis brief (TEXT only, no image render) so
the article-quality workflow can evaluate the visual concept's message/expression/aesthetics
without paying for gpt-image. Reuses the same cached candidate set as sample_digest.py.

It grades what PRODUCTION sends: the digest is generated first (so the real DigestContent — and
therefore the real editorial angle — exists), and the art-director instruction is built by the
maker's own DailyVisualMaker._build_instruction. Briefing the bare `plan["instruction"]` instead
scored a prompt that never ships: no editorial angle, no guardrails, no format nudge, no character.

Usage: uv run python scripts/sample_visual_brief.py <out_json> [cache_path]
"""

import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3

from pipeline import ContentAggregator, ContentRanker, DigestGenerator
from pipeline.daily_visual import DailyVisualMaker
from shared import BedrockLanguageModelFactory, CollectedItem, Config, logger


async def main() -> None:
    out_path = Path(sys.argv[1])
    cache_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    config = Config.load()
    post_date = datetime.now(ZoneInfo(config.aws.timezone)).date()
    session = boto3.Session(region_name=config.aws.bedrock_region, profile_name=config.aws.profile or None)
    factory = BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)

    if cache_path and cache_path.exists():
        raw = json.loads(cache_path.read_text(encoding="utf-8"))
        collected = [CollectedItem.model_validate(it) for it in raw]
    else:
        from main import run_collectors_with_health

        collected, _ = await run_collectors_with_health(config, factory)

    items = ContentAggregator().aggregate(collected)
    ranked = await ContentRanker(config.pipeline, factory).rank(
        items,
        select_count=config.pipeline.top_n + config.pipeline.digest_candidate_buffer,
        core_count=config.pipeline.top_n,
    )
    # The digest comes FIRST because the visual's editorial angle is derived from its lead: the
    # sampler must pass in the real DigestContent rather than invent a take it does not have.
    digest = await DigestGenerator(config.pipeline, factory).generate(ranked, items, today=post_date)
    content = digest.content

    maker = DailyVisualMaker(config, factory)
    headline_index = maker._headline_ranked_index(content, ranked)
    marker_index, headline_title, source = maker._headline_brief(content, ranked, headline_index)
    recent_formats = maker.format_log.entries() if maker.format_log else []
    preferred_orientation = maker._least_recent_orientation(recent_formats)
    plan = await maker._pick_story(ranked, marker_index, recent_formats, preferred_orientation)
    out: dict = {"editor_plan": plan}

    if plan and not plan.get("skip"):
        instruction, use_character = maker._build_instruction(
            plan, content, post_date, headline_title, recent_formats, preferred_orientation
        )
        # Reuse the maker's generator so the sampled brief uses the exact production config
        # (VisualGenerator carries no defaults of its own).
        brief = await maker.generator.brief(instruction, source, "")
        out["chosen_item_title"] = headline_title
        out["use_character"] = use_character
        out["instruction"] = instruction
        out["brief"] = brief.model_dump()

    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote visual brief to %s (skip=%s)", out_path, bool(not out.get("brief")))


if __name__ == "__main__":
    asyncio.run(main())
