import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from application.digest_pipeline import (
    PublishIncomplete,
    already_published,
    persist_digest,
    publish_inline,
    resolve_digest_window,
    run_collectors_with_health,
    run_pipeline,
)
from entrypoints import bootstrap
from shared import Config, LocalPaths, logger
from shared.constants import COLLECTOR_NAMES


def _positive_int(value: str) -> int:
    """argparse type for a >=1 count: `--top-n 0`/negative fails loudly instead of being silently
    ignored (truthiness check) or corrupting select_count = top_n + buffer downstream."""
    ivalue = int(value)
    if ivalue < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {ivalue}")
    return ivalue


async def main() -> int:
    """Exit code: 0 when the run did what it was asked, 1 when nothing was collected, no digest was
    produced or the Threads post did not fully land (locally there is no alert to say so)."""
    parser = argparse.ArgumentParser(description="OmniSummary - Daily AI Digest")
    parser.add_argument("--sources", nargs="+", choices=COLLECTOR_NAMES, help="Specific sources to collect from")
    parser.add_argument("--dry-run", action="store_true", help="Run without publishing or persisting")
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
    if digest_date > datetime.now(ZoneInfo(config.aws.timezone)).date():
        # A typo'd future date stored that day's snapshot and post marker, and the real run then skipped.
        parser.error(f"--date {digest_date} is in the future")
    config.collectors.set_reference_time(reference_time)

    logger.info(
        "Starting OmniSummary digest pipeline (date: '%s', reference_time: '%s')",
        digest_date,
        reference_time.isoformat(),
    )

    state_store = bootstrap.state_store(config)
    if not args.dry_run and not args.force_republish and already_published(config, digest_date, state_store):
        return 0

    models = bootstrap.language_models(config)
    collected_items, health = await run_collectors_with_health(
        config, bootstrap.collectors(config, models, args.sources)
    )
    logger.info("Collected %d total items", len(collected_items))
    logger.info("Source health report:\n%s", health.summary())

    if args.pin_url:
        from collectors.web_search import fetch_pinned_items

        pinned = await fetch_pinned_items(args.pin_url)
        logger.info("Fetched %d pinned item(s) from %d URL(s)", len(pinned), len(args.pin_url))
        collected_items = pinned + collected_items

    if not collected_items:
        logger.error("No items collected. Exiting.")
        return 1

    _, ranked_items, digest = await run_pipeline(
        config,
        models,
        collected_items,
        digest_date,
        state_store=state_store,
        memory_store=bootstrap.memory_store,
        dry_run=args.dry_run,
    )
    if digest is None or ranked_items is None:
        logger.error("No digest was produced")
        return 1
    if not args.dry_run:
        # Persisted BEFORE the post: a store that fails after the post went up would leave the day's
        # posted digest stored nowhere, and a re-run would record a different one.
        persist_digest(
            ranked_items, digest, digest_date, bootstrap.memory_store(Path(LocalPaths.DIGEST_STATE_DIR.value))
        )
        try:
            await publish_inline(
                bootstrap.daily_visual_maker(config, models),
                ranked_items,
                digest,
                digest_date,
                force_republish=args.force_republish,
            )
        except PublishIncomplete as e:
            logger.error("%s", e)
            return 1
    logger.info("OmniSummary pipeline completed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
