import argparse
import asyncio
import json
import os
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3

from collectors import (
    RedditCollector,
    RSSCollector,
    RSSHubCollector,
    WebSearchCollector,
    YouTubeCollector,
)
from collectors.base import gather_collector_results
from output import send_digest_to_slack
from agent.tool_state import DigestStateManager
from pipeline import ContentAggregator, ContentRanker, DigestGenerator, TrendTracker
from shared import (
    BedrockLanguageModelFactory,
    CollectedItem,
    Config,
    DigestResult,
    LocalPaths,
    LocalStateStore,
    RankedItem,
    S3StateStore,
    is_running_in_aws,
    logger,
)
from shared.state_store import StateStore


async def run_collectors(
    config: Config,
    llm_factory: BedrockLanguageModelFactory,
    sources: list[str] | None = None,
) -> list[CollectedItem]:
    collectors_map = {
        "reddit": (RedditCollector, config.collectors.reddit, {}),
        "rsshub": (RSSHubCollector, config.collectors.rsshub, {}),
        "rss": (RSSCollector, config.collectors.rss, {}),
        "web_search": (WebSearchCollector, config.collectors.web_search, {"llm_factory": llm_factory}),
        "youtube": (YouTubeCollector, config.collectors.youtube, {}),
    }

    active_sources = (
        [name for name, (_, cfg, _kw) in collectors_map.items() if cfg.enabled] if sources is None else sources
    )
    tasks = []
    labels = []

    for source_name in active_sources:
        if source_name not in collectors_map:
            logger.warning("Unknown source: '%s'", source_name)
            continue
        collector_cls, collector_cfg, kwargs = collectors_map[source_name]
        if not collector_cfg.enabled:
            logger.info("Skipping disabled source: '%s'", source_name)
            continue
        collector = collector_cls(collector_cfg, **kwargs)
        logger.info("Starting collector: '%s'", source_name)
        tasks.append(collector.collect())
        labels.append(source_name)

    if not tasks:
        logger.warning("No active collectors")
        return []

    return await gather_collector_results(tasks, labels=labels)


async def run_pipeline(
    config: Config,
    llm_factory: BedrockLanguageModelFactory,
    collected_items: list[CollectedItem],
    digest_date: date,
    dry_run: bool = False,
) -> tuple[list[CollectedItem], list[RankedItem], DigestResult] | tuple[None, None, None]:
    aggregator = ContentAggregator()
    items = aggregator.aggregate(collected_items)
    logger.info("Aggregated %d unique items", len(items))

    if not items:
        logger.warning("No items to process after aggregation")
        return None, None, None

    ranker = ContentRanker(config.pipeline, llm_factory)
    ranked_items = await ranker.rank(items)
    logger.info("Ranked %d items (from %d)", len(ranked_items), len(items))

    if not ranked_items:
        logger.warning("No items passed ranking threshold")
        return None, None, None

    state_store = _create_state_store()
    trend_tracker = TrendTracker(config.pipeline, llm_factory, state_store)
    trends_context = trend_tracker.get_trends_context()

    generator = DigestGenerator(config.pipeline, llm_factory)
    digest = await generator.generate(ranked_items, items, trends_context=trends_context)
    logger.info("Generated digest with %d ranked items", len(digest.ranked_items))

    if dry_run:
        logger.info("DRY RUN - Digest output:\n%s", digest.digest_text)
        return items, ranked_items, digest

    await trend_tracker.update_trends(digest.digest_text, digest_date.isoformat())

    success = await send_digest_to_slack(digest, config.slack)
    if success:
        logger.info("Digest sent to Slack successfully")
    else:
        logger.error("Failed to send digest to Slack")

    if not is_running_in_aws():
        _save_state(items, ranked_items, digest, digest_date)
    return items, ranked_items, digest


def _create_state_store(config: Config | None = None) -> StateStore:
    if is_running_in_aws():
        bucket = os.environ.get("STATE_BUCKET", "")
        if bucket:
            prefix = os.environ.get("S3_PREFIX", "digest_state")
            return S3StateStore(boto3.Session(), bucket, prefix=prefix)
    if config and config.aws.state_bucket_name:
        prefix = f"{config.aws.s3_prefix}/digest_state" if config.aws.s3_prefix else "digest_state"
        return S3StateStore(
            boto3.Session(profile_name=config.aws.profile or None, region_name=config.aws.region),
            config.aws.state_bucket_name,
            prefix=prefix,
        )
    return LocalStateStore(Path(LocalPaths.DIGEST_STATE_DIR.value))


def _save_state(
    items: list[CollectedItem],
    ranked_items: list[RankedItem],
    digest: DigestResult,
    digest_date: date,
) -> None:
    state_dir = Path(LocalPaths.DIGEST_STATE_DIR.value)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_file = state_dir / f"digest_{digest_date.isoformat()}.json"

    mgr = DigestStateManager()
    mgr.store_digest(items, ranked_items, digest)
    mgr.save_to_file(state_file)


async def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSummary - Daily AI Digest")
    parser.add_argument("--sources", nargs="+", help="Specific sources to collect from")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending to Slack")
    parser.add_argument("--top-n", type=int, help="Override top_n from config")
    parser.add_argument("--date", type=str, help="Digest date (YYYY-MM-DD). Defaults to today")
    parser.add_argument("--interactive", action="store_true", help="Enter agent chat mode after digest")
    args = parser.parse_args()

    config = Config.load()

    if args.top_n:
        config.pipeline.top_n = args.top_n

    KST = ZoneInfo("Asia/Seoul")
    digest_date = date.fromisoformat(args.date) if args.date else datetime.now(KST).date()

    next_day = digest_date + timedelta(days=1)
    reference_time = datetime(next_day.year, next_day.month, next_day.day, tzinfo=KST)
    config.collectors.set_reference_time(reference_time)

    logger.info(
        "Starting OmniSummary digest pipeline (date: '%s', reference_time: '%s')",
        digest_date,
        reference_time.isoformat(),
    )

    boto_session = boto3.Session(
        profile_name=config.aws.profile or None,
        region_name=config.aws.region,
    )
    llm_factory = BedrockLanguageModelFactory(
        boto_session=boto_session,
        region_name=config.aws.bedrock_region,
    )

    collected_items = await run_collectors(config, llm_factory, args.sources)
    logger.info("Collected %d total items", len(collected_items))

    if not collected_items:
        logger.warning("No items collected. Exiting.")
        return

    result = await run_pipeline(config, llm_factory, collected_items, digest_date=digest_date, dry_run=args.dry_run)
    logger.info("OmniSummary pipeline completed")

    if args.interactive and result and result[0] is not None:
        _run_interactive(*result)


def _run_interactive(items: list[CollectedItem], ranked_items: list[RankedItem], digest: DigestResult) -> None:
    from agent import DigestStateManager, create_digest_agent
    from agent.agent_tools import state_manager

    state_manager.store_digest(items, ranked_items, digest)

    logger.info("Entering interactive agent mode (%d items). Type 'quit' to exit.", state_manager.get_item_count())
    print("\n=== OmniSummary Agent ===")
    print(f"Loaded {state_manager.get_item_count()} digest items. Type 'quit' to exit.")
    print("Examples: '1번 자세히', '1번 관련 논문', '1번 커뮤니티 반응'\n")

    agent = create_digest_agent()

    while True:
        try:
            query = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not query or query in ("quit", "exit", "q"):
            break
        try:
            response = agent(query)
            print(f"\n{response}\n")
        except Exception as e:
            logger.error("Agent error: %s", e)
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    asyncio.run(main())
