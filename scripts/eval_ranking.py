#!/usr/bin/env python3
"""Replay a FROZEN candidate pool through ContentRanker.rank() N times and report how stable the
ranking is: top-k overlap between runs, per-item score spread, and above-threshold counts.

CLAUDE.md says "Re-measure before revisiting" about the ranking model choice (Opus vs Sonnet) and
records a measured 0.02-0.06 score jitter, but there was no way to run that measurement — every
comparison was a one-off done by hand against whatever pool the day happened to produce. Freezing the
pool is what makes two runs comparable at all: the same items, only the model or the prompt varying.

Usage:
  # Once: collect today's pool and freeze it (then commit the file so the numbers are reproducible).
  uv run python scripts/eval_ranking.py --capture [--pool eval/ranking_pool.json]

  # Then, any time: replay it.
  uv run python scripts/eval_ranking.py [--runs 3] [--model anthropic.claude-sonnet-5]

Every run is a real Bedrock call, so `--runs` is the whole cost knob: runs x pool-size tokens.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3

from pipeline import ContentAggregator, ContentRanker
from shared import BedrockLanguageModelFactory, CollectedItem, Config, LanguageModelId, RankedItem, logger

DEFAULT_POOL_PATH = Path("eval/ranking_pool.json")


def _factory(config: Config) -> BedrockLanguageModelFactory:
    session = boto3.Session(region_name=config.aws.bedrock_region, profile_name=config.aws.profile or None)
    return BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)


async def _capture(config: Config, pool_path: Path) -> None:
    """Collect today's pool once and freeze it. Deliberately separate from the replay: a pool that
    changed between two runs makes every difference below unattributable."""
    from pipeline import run_collectors_with_health

    collected, health = await run_collectors_with_health(config, _factory(config))
    logger.info("Collected %d items\n%s", len(collected), health.summary())
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    pool_path.write_text(
        json.dumps([item.model_dump(mode="json") for item in collected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("Froze %d items to '%s' — commit it so the replay is reproducible", len(collected), pool_path)


def _load_pool(pool_path: Path) -> list[CollectedItem]:
    if not pool_path.exists():
        raise SystemExit(f"No frozen pool at '{pool_path}'. Capture one first: --capture --pool {pool_path}")
    raw = json.loads(pool_path.read_text(encoding="utf-8"))
    return [CollectedItem.model_validate(item) for item in raw]


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _report(runs: list[list[RankedItem]], top_k: int, min_score: float) -> str:
    """The three numbers a ranking change has to be judged on, from the same replay."""
    lines: list[str] = []

    top_sets = [{r.item.item_id for r in run[:top_k]} for run in runs]
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            shared = len(top_sets[i] & top_sets[j])
            lines.append(
                f"top-{top_k} overlap run{i + 1} vs run{j + 1}: {shared}/{top_k} "
                f"(jaccard {_jaccard(top_sets[i], top_sets[j]):.2f})"
            )

    scores: dict[str, list[float]] = {}
    for run in runs:
        for ranked in run:
            scores.setdefault(ranked.item.item_id, []).append(ranked.score)
    spreads = {item_id: max(values) - min(values) for item_id, values in scores.items() if len(values) == len(runs)}
    if spreads:
        worst = sorted(spreads.items(), key=lambda kv: -kv[1])[:5]
        mean_spread = sum(spreads.values()) / len(spreads)
        lines.append(f"score spread across runs: mean {mean_spread:.3f} over {len(spreads)} item(s) in every run")
        lines.extend(f"  {item_id}: {spread:.3f}" for item_id, spread in worst)

    for i, run in enumerate(runs, start=1):
        above = sum(1 for r in run if r.score >= min_score)
        lines.append(f"run{i}: {len(run)} selected, {above} at/above min_score {min_score:.2f}")
    return "\n".join(lines)


async def _replay(config: Config, pool_path: Path, runs: int) -> None:
    pool = _load_pool(pool_path)
    items = ContentAggregator().aggregate(pool)
    logger.info(
        "Replaying %d aggregated items (from %d frozen) x %d run(s) through '%s'",
        len(items),
        len(pool),
        runs,
        config.pipeline.ranking_model.value,
    )
    factory = _factory(config)
    select_count = config.pipeline.top_n + config.pipeline.digest_candidate_buffer
    results: list[list[RankedItem]] = []
    for _ in range(runs):
        ranker = ContentRanker(config.pipeline, factory)
        results.append(await ranker.rank(items, select_count=select_count, core_count=config.pipeline.top_n))
    print(_report(results, config.pipeline.top_n, config.pipeline.min_score))


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", action="store_true", help="collect today's pool and freeze it, then exit")
    parser.add_argument("--pool", type=Path, default=DEFAULT_POOL_PATH, help="frozen pool path")
    parser.add_argument("--runs", type=int, default=3, help="how many times to re-rank the frozen pool")
    parser.add_argument("--model", default="", help="override pipeline.ranking_model for this replay")
    args = parser.parse_args()

    config = Config.load()
    if args.model:
        config.pipeline.ranking_model = LanguageModelId(args.model)
    if args.capture:
        await _capture(config, args.pool)
        return
    await _replay(config, args.pool, max(1, args.runs))


if __name__ == "__main__":
    asyncio.run(main())
