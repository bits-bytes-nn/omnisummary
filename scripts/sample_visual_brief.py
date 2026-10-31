#!/usr/bin/env python3
"""Write the daily-visual editor plan, the art-director instruction and the image brief production
would send for a stored digest, and optionally render it. Nothing is published or recorded.

It grades what PRODUCTION sends: the digest snapshot for the date is read from AgentCore Memory (the
same one the visual Lambda loads), and the text comes from DailyVisualMaker._compose. The recent
comics are the deployed log's (only the last few days), unless --recent-comics names a JSON list of
{"date", "orientation", "comic"} records (to replay an older day against the comics before it).

Usage: uv run python scripts/sample_visual_brief.py 2026-10-17 out/ [--render] [--recent-comics FILE]
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from domain.digest_snapshot import DigestSnapshot
from entrypoints import bootstrap
from shared import Config, logger


async def main() -> None:
    parser = argparse.ArgumentParser(description="Write (and optionally render) a stored digest's daily visual")
    parser.add_argument("digest_date", type=date.fromisoformat, help="The stored digest's date (YYYY-MM-DD)")
    parser.add_argument("out_dir", type=Path, help="Where to write <date>.json (and <date>.png)")
    parser.add_argument("--render", action="store_true", help="Also render the image (one gpt-image call)")
    parser.add_argument("--recent-comics", type=Path, help="JSON list of recent comics to use instead of the log")
    args = parser.parse_args()

    config = Config.load()
    data = bootstrap.memory_store().get_digest(args.digest_date.isoformat())
    if not data:
        raise SystemExit(f"No digest snapshot for {args.digest_date}")
    snapshot = DigestSnapshot.from_dict(data)

    maker = bootstrap.daily_visual_maker(config)
    if args.recent_comics:
        logged = json.loads(args.recent_comics.read_text(encoding="utf-8"))
        recent = [e for e in logged if e["date"] < args.digest_date.isoformat()][-config.pipeline.recent_leads_window :]
    else:
        recent = maker._recent_comics(args.digest_date)
    # The log keeps only the last few comics, so a date older than that window replays with less
    # history than production had; pass --recent-comics for an older day.
    logger.info("Replaying %s against %d recent comic(s)", args.digest_date, len(recent))

    plan, instruction, source = await maker._compose(
        snapshot.ranked_items, snapshot.content, args.digest_date, recent_comics=recent
    )
    out: dict = {"editor_plan": plan, "instruction": instruction}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if instruction:
        if args.render:
            image, brief = await maker.generator.generate(instruction, source, references=maker._references(plan))
            (args.out_dir / f"{args.digest_date}.png").write_bytes(image)
        else:
            brief = await maker.generator.brief(instruction, source)
        out["brief"] = brief.model_dump()

    out_path = args.out_dir / f"{args.digest_date}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s (skip=%s, rendered=%s)", out_path, not instruction, bool(instruction and args.render))


if __name__ == "__main__":
    asyncio.run(main())
