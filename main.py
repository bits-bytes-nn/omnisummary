import argparse
import asyncio

import boto3

from pipeline import resolve_digest_window, run_collectors_with_health, run_pipeline
from shared import BedrockLanguageModelFactory, Config, logger


def _positive_int(value: str) -> int:
    """argparse type for a >=1 count: `--top-n 0`/negative fails loudly instead of being silently
    ignored (truthiness check) or corrupting select_count = top_n + buffer downstream."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {ivalue}")
    return ivalue


async def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSummary - Daily AI Digest")
    parser.add_argument("--sources", nargs="+", help="Specific sources to collect from")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending to Slack")
    parser.add_argument("--top-n", type=_positive_int, help="Override top_n from config (>= 1)")
    parser.add_argument("--date", type=str, help="Digest date (YYYY-MM-DD). Defaults to today")
    parser.add_argument(
        "--force-republish",
        action="store_true",
        help="Re-post to Threads even if today's digest was already posted (bypass idempotency guard)",
    )
    parser.add_argument(
        "--pin-url",
        nargs="+",
        default=None,
        help="One or more URLs to force into the digest's top stories regardless of ranking score",
    )
    args = parser.parse_args()

    config = Config.load()

    if args.top_n is not None:
        config.pipeline.top_n = args.top_n

    digest_date, reference_time = resolve_digest_window(config, args.date)
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

    collected_items, health = await run_collectors_with_health(config, llm_factory, args.sources)
    logger.info("Collected %d total items", len(collected_items))
    logger.info("Source health report:\n%s", health.summary())

    if args.pin_url:
        from collectors.web_search import fetch_pinned_items

        pinned = await fetch_pinned_items(args.pin_url)
        logger.info("Fetched %d pinned item(s) from %d URL(s)", len(pinned), len(args.pin_url))
        collected_items = pinned + collected_items

    if not collected_items:
        logger.warning("No items collected. Exiting.")
        return

    await run_pipeline(
        config,
        llm_factory,
        collected_items,
        digest_date=digest_date,
        dry_run=args.dry_run,
        force_republish=args.force_republish,
    )
    logger.info("OmniSummary pipeline completed")


if __name__ == "__main__":
    asyncio.run(main())
