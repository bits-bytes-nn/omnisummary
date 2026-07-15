import argparse
import asyncio
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import boto3

from agent.tool_state import DigestStateManager
from collectors import (
    RedditCollector,
    RSSCollector,
    RSSHubCollector,
    WebSearchCollector,
    YouTubeCollector,
)
from output import send_digest_to_slack
from pipeline import ContentAggregator, ContentRanker, DigestGenerator, TrendTracker
from pipeline.aggregator import normalize_url
from shared import (
    BedrockLanguageModelFactory,
    CollectedItem,
    Config,
    DigestResult,
    HealthReport,
    LocalPaths,
    PublishedUrlLedger,
    RankedItem,
    RollingLog,
    SourceHealth,
    SourceStatus,
    agi_countdown_intro,
    create_memory_store,
    create_state_store,
    is_running_in_aws,
    logger,
    published_urls_from_snapshots,
)
from shared.history_store import RECENT_LEADS_KEY


def _build_collector_tasks(
    config: Config,
    llm_factory: BedrockLanguageModelFactory,
    sources: list[str] | None = None,
) -> tuple[list, list[str]]:
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

    return tasks, labels


async def run_collectors_with_health(
    config: Config,
    llm_factory: BedrockLanguageModelFactory,
    sources: list[str] | None = None,
) -> tuple[list[CollectedItem], HealthReport]:
    tasks, labels = _build_collector_tasks(config, llm_factory, sources)
    if not tasks:
        logger.warning("No active collectors")
        return [], HealthReport()

    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[CollectedItem] = []
    health: list[SourceHealth] = []
    for label, result in zip(labels, results, strict=True):
        if isinstance(result, BaseException):
            health.append(SourceHealth(name=label, item_count=0, status=SourceStatus.FAILED, detail=str(result)[:200]))
            logger.warning("Collector '%s' failed: %s", label, result)
        else:
            items.extend(result)
            status = SourceStatus.OK if result else SourceStatus.EMPTY
            health.append(SourceHealth(name=label, item_count=len(result), status=status))
    return items, HealthReport(sources=health)


async def run_pipeline(
    config: Config,
    llm_factory: BedrockLanguageModelFactory,
    collected_items: list[CollectedItem],
    digest_date: date,
    dry_run: bool = False,
    force_republish: bool = False,
) -> tuple[list[CollectedItem], list[RankedItem], DigestResult] | tuple[None, None, None]:
    state_store = create_state_store(config)
    ledger = PublishedUrlLedger(state_store, config.pipeline.published_url_ttl_days)
    leads_log = RollingLog(state_store, RECENT_LEADS_KEY, config.pipeline.recent_leads_window)

    # Cross-day dedup draws from BOTH the ledger AND recent AgentCore Memory snapshots, so it
    # self-heals from existing history (and survives a lost/empty ledger) rather than only
    # suppressing dupes for runs after the ledger is first populated. URLs are normalized to the
    # ledger's canonical form so http/https + trailing-slash variants of a past story still match.
    exclude_urls = ledger.recent_urls(digest_date)
    try:
        # Seed the same TTL window the ledger uses: skip today's own snapshot (exclude_date) so a
        # same-day re-run keeps its stories, and floor at digest_date - ttl (after_date) so a story
        # that legitimately recurs past the window isn't suppressed by a stale snapshot.
        ttl = config.pipeline.published_url_ttl_days
        snapshots = create_memory_store().get_recent_digests(
            ttl,
            exclude_date=digest_date.isoformat(),
            after_date=(digest_date - timedelta(days=ttl)).isoformat(),
        )
        exclude_urls |= {normalize_url(u) for u in published_urls_from_snapshots(snapshots)}
    except Exception:
        logger.warning("Could not seed cross-day dedup from memory snapshots (non-fatal)", exc_info=True)

    aggregator = ContentAggregator()
    items = aggregator.aggregate(collected_items, exclude_urls=exclude_urls)
    logger.info("Aggregated %d unique items (excluding %d recently-published URLs)", len(items), len(exclude_urls))

    if not items:
        logger.warning("No items to process after aggregation")
        return None, None, None

    # Over-select (top_n + buffer) so the digest editor can backfill after merging same-event
    # items and still land on exactly top_n distinct stories.
    select_count = config.pipeline.top_n + config.pipeline.digest_candidate_buffer
    ranker = ContentRanker(config.pipeline, llm_factory)
    ranked_items = await ranker.rank(items, select_count=select_count)
    logger.info("Ranked %d items (from %d)", len(ranked_items), len(items))

    if not ranked_items:
        logger.warning("No items passed ranking threshold")
        return None, None, None

    trend_tracker = TrendTracker(config.pipeline, llm_factory, state_store)
    trends_context = trend_tracker.get_trends_context(today=digest_date)

    recent_leads = [e.get("lead", "") for e in leads_log.entries()]
    generator = DigestGenerator(config.pipeline, llm_factory)
    digest = await generator.generate(
        ranked_items, items, trends_context=trends_context, today=digest_date, recent_leads=recent_leads
    )
    logger.info("Generated digest with %d ranked items", len(digest.ranked_items))

    if dry_run:
        logger.info("DRY RUN - Digest output:\n%s", digest.digest_text)
        return items, ranked_items, digest

    await trend_tracker.update_trends(digest.digest_text, digest_date.isoformat())

    # Record the stories that became TODAY'S digest of record so future runs don't re-publish
    # them, and remember the lead so the next digest avoids repeating the same opening angle.
    # Recorded at generation (post trend-update) — the same content.items the snapshot carries
    # and the visual Lambda delivers — not gated on downstream delivery, which is async/
    # best-effort and alarmed separately. The lead is stored WITHOUT the AGI-countdown prefix
    # (a fixed daily template) so the novelty signal is the editorial angle, not the boilerplate.
    try:
        if digest.content and digest.content.items:
            ledger.record([normalize_url(it.url) for it in digest.content.items], digest_date)
            leads_log.append(
                {"date": digest_date.isoformat(), "lead": _editorial_lead(config, digest, digest_date)},
                dedup_key="date",
            )
    except Exception:
        logger.warning("Failed to update published-URL / leads history (non-fatal)", exc_info=True)

    if config.pipeline.enable_slack_post:
        success = await send_digest_to_slack(digest, config.slack)
        if success:
            logger.info("Digest sent to Slack successfully")
        else:
            logger.error("Failed to send digest to Slack")

    # In AWS the digest Lambda fires a separate visual Lambda (off the critical path);
    # locally we run it inline so `uv run python main.py` still produces the visual.
    # The visual step also fans out to Threads (when enabled), so it carries the digest text.
    if config.pipeline.enable_daily_visual and not is_running_in_aws():
        try:
            from pipeline.daily_visual import DailyVisualMaker

            posted = await DailyVisualMaker(config, llm_factory).run(
                ranked_items, digest.content, today=digest_date, force_republish=force_republish
            )
            logger.info("Daily visual %s", "posted" if posted else "skipped")
        except Exception:
            logger.warning("Daily visual step failed (non-fatal)", exc_info=True)

    if not is_running_in_aws():
        persist_digest(items, ranked_items, digest, digest_date, base_dir=Path(LocalPaths.DIGEST_STATE_DIR.value))
    return items, ranked_items, digest


def _editorial_lead(config: Config, digest: DigestResult, digest_date: date) -> str:
    """The digest lead with the AGI-countdown prefix removed, so recent-leads novelty compares
    the editorial angle (not the fixed daily countdown template every lead starts with)."""
    lead = digest.content.lead if digest.content else ""
    intro = agi_countdown_intro(
        config.pipeline.agi_countdown_date,
        config.pipeline.agi_countdown_template,
        digest_date,
        config.pipeline.agi_countdown_after,
    )
    return lead[len(intro) :] if intro and lead.startswith(intro) else lead


def persist_digest(
    items: list[CollectedItem],
    ranked_items: list[RankedItem],
    digest: DigestResult,
    digest_date: date,
    *,
    base_dir: Path | None = None,
) -> None:
    """Persist the digest snapshot to the memory store (single path used by both the
    local CLI and the Lambda handler). base_dir selects the local fallback; pass None
    in AWS so create_memory_store picks the AgentCore-backed store."""
    mgr = DigestStateManager()
    mgr.store_digest(items, ranked_items, digest)
    memory = create_memory_store(base_dir)
    memory.put_digest(digest_date.isoformat(), mgr.export_state())


async def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSummary - Daily AI Digest")
    parser.add_argument("--sources", nargs="+", help="Specific sources to collect from")
    parser.add_argument("--dry-run", action="store_true", help="Run without sending to Slack")
    parser.add_argument("--top-n", type=int, help="Override top_n from config")
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

    if args.top_n:
        config.pipeline.top_n = args.top_n

    tz = ZoneInfo(config.aws.timezone)
    digest_date = date.fromisoformat(args.date) if args.date else datetime.now(tz).date()

    next_day = digest_date + timedelta(days=1)
    reference_time = datetime(next_day.year, next_day.month, next_day.day, tzinfo=tz)
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
