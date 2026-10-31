"""Digest orchestration: the single path from collected items to a stored digest, taken by both the
CLI (main.py) and the digest Lambda. The entrypoints hand in the sources, the models and the stores
(entrypoints/bootstrap.py); this module builds none of them."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from application.digest_generator import DigestGenerator
from application.history import (
    RECENT_LEADS_KEY,
    PublishedUrlLedger,
    RollingLog,
    ThreadsPostLedger,
    published_urls_from_snapshots,
)
from application.ports import ContentSource, LanguageModels, MemoryStore, StateStore
from application.ranker import ContentRanker
from application.trend_tracker import TrendTracker
from domain.aggregator import ContentAggregator, normalize_url
from domain.digest_snapshot import DigestSnapshot
from domain.formatting import agi_countdown_intro, editorial_lead
from domain.models import CollectedItem, DigestResult, HealthReport, RankedItem, SourceHealth, SourceStatus
from shared import Config, logger

if TYPE_CHECKING:
    from application.daily_visual import DailyVisualMaker


def resolve_digest_window(config: Config, date_arg: str | None = None) -> tuple[date, datetime]:
    """The digest's date and the collection cutoff derived from it.

    The reference time is MIDNIGHT AT THE END of the digest date in config.aws.timezone, and every
    collector's lookback_hours counts back from it — so a run at 19:00 KST still sees the whole day,
    and `--date` re-runs a past day with exactly the window that day had rather than the window
    ending now. Pure, so the boundary is testable without running the pipeline."""
    tz = ZoneInfo(config.aws.timezone)
    digest_date = date.fromisoformat(date_arg) if date_arg else datetime.now(tz).date()
    next_day = digest_date + timedelta(days=1)
    return digest_date, datetime(next_day.year, next_day.month, next_day.day, tzinfo=tz)


async def run_collectors_with_health(
    config: Config, sources: list[tuple[str, ContentSource]]
) -> tuple[list[CollectedItem], HealthReport]:
    """Run every (name, source) at once and report each one's health: a failure reads FAILED, a stalled
    or unreadable park file STALE, a fraction of inputs DEGRADED."""
    if not sources:
        logger.warning("No active collectors")
        return [], HealthReport()
    labels = [label for label, _ in sources]
    collectors = [collector for _, collector in sources]
    for label in labels:
        logger.info("Starting collector: '%s'", label)
    tasks = [collector.collect() for collector in collectors]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[CollectedItem] = []
    health: list[SourceHealth] = []
    for label, collector, result in zip(labels, collectors, results, strict=True):
        if isinstance(result, BaseException):
            health.append(SourceHealth(name=label, item_count=0, status=SourceStatus.FAILED, detail=str(result)[:200]))
            logger.warning("Collector '%s' failed: %s", label, result)
            continue
        items.extend(result)
        # Items arrived, but the source says they are stale or came from a fraction of its inputs.
        # Reported, never filtered: every item still reaches the aggregator.
        if (issue := collector.health_issue) is not None:
            status, detail = issue
            health.append(SourceHealth(name=label, item_count=len(result), status=status, detail=detail[:200]))
            logger.warning("Collector '%s' is %s: %s", label, status.value.upper(), detail)
            continue
        status = SourceStatus.OK if result else SourceStatus.EMPTY
        health.append(SourceHealth(name=label, item_count=len(result), status=status))
    report = HealthReport(sources=health)
    # A source that ran clean and returned nothing is DARK: no exception, no stale park file, no
    # failure ratio. The config names which sources that is an incident for (the digest Lambda
    # alerts on the same list); log it loudly here so a local run sees it too.
    dark = [name for name in report.empty_sources if name in set(config.collectors.alert_on_empty)]
    if dark:
        logger.error("Source(s) returned NO items and are configured as must-not-be-empty: %s", ", ".join(dark))
    return items, report


class PublishIncomplete(RuntimeError):
    """A local run's Threads post failed or did not fully land. The digest is already persisted."""


def already_published(config: Config, digest_date: date, state_store: StateStore) -> bool:
    """True when the date's digest is already on Threads. A second run for it (a duplicate invoke, a
    local run) still persisted its own snapshot, leads and ledger entries before the post step skipped,
    so the record of the day became a digest nobody saw. With Threads posting off nothing can post, so a
    re-run is how a day's record is rebuilt."""
    if config.pipeline.enable_threads_post and ThreadsPostLedger(state_store).already_posted(digest_date):
        logger.warning("The %s digest is already on Threads; this run persists and posts nothing", digest_date)
        return True
    return False


async def run_pipeline(
    config: Config,
    models: LanguageModels,
    collected_items: list[CollectedItem],
    digest_date: date,
    *,
    state_store: StateStore,
    memory_store: Callable[[], MemoryStore],
    dry_run: bool = False,
) -> tuple[list[CollectedItem], list[RankedItem], DigestResult] | tuple[None, None, None]:
    """Aggregate, rank and write the day's digest, then record its stories and lead. `memory_store`
    is called only to read the recent snapshots, and a failure there costs only the cross-day dedup
    seed. The digest is neither stored nor published here: see persist_digest and publish_inline."""
    ledger = PublishedUrlLedger(state_store, config.pipeline.published_url_ttl_days)
    leads_log = RollingLog(state_store, RECENT_LEADS_KEY, config.pipeline.recent_leads_window)

    # Cross-day dedup draws from BOTH the ledger AND recent AgentCore Memory snapshots, so it
    # self-heals from existing history (and survives a lost/empty ledger) rather than only
    # suppressing dupes for runs after the ledger is first populated. URLs are normalized to the
    # ledger's canonical form so http/https + trailing-slash variants of a past story still match.
    # Re-normalized on read: a key written under an older normalize_url must still match.
    exclude_urls = {normalize_url(u) for u in ledger.recent_urls(digest_date)}
    snapshots: list[dict] = []
    try:
        # Seed the same window the ledger excludes: ages 1..ttl-1 (today's own snapshot is excluded
        # so a same-day re-run keeps its stories, and a story back on the TTL day is allowed again).
        snapshots = memory_store().get_recent_digests(digest_date, config.pipeline.published_url_ttl_days - 1)
        exclude_urls |= {normalize_url(u) for u in published_urls_from_snapshots(snapshots)}
    except Exception:
        logger.warning("Could not seed cross-day dedup from memory snapshots (non-fatal)", exc_info=True)

    aggregator = ContentAggregator()
    items = aggregator.aggregate(collected_items, exclude_urls=exclude_urls)
    logger.info("Aggregated %d unique items (excluding %d recently-published URLs)", len(items), len(exclude_urls))

    if not items:
        logger.warning("No items to process after aggregation")
        return None, None, None

    # Over-select (top_n + buffer): the editor picks the strongest top_n distinct stories from it.
    select_count = config.pipeline.top_n + config.pipeline.digest_candidate_buffer
    ranker = ContentRanker(config.pipeline, models)
    recent_titles = _recent_story_titles(snapshots)
    ranked_items = await ranker.rank(items, select_count=select_count, recent_titles=recent_titles)
    logger.info("Ranked %d items (from %d)", len(ranked_items), len(items))

    if not ranked_items:
        logger.warning("No items passed ranking threshold")
        return None, None, None

    trend_tracker = TrendTracker(config.pipeline, models, state_store)
    trends_context = trend_tracker.get_trends_context(today=digest_date)

    recent_leads = [e.get("lead", "") for e in leads_log.entries()]
    generator = DigestGenerator(config.pipeline, models)
    digest = await generator.generate(
        ranked_items,
        trends_context=trends_context,
        today=digest_date,
        recent_leads=recent_leads,
        recent_titles=recent_titles,
    )
    logger.info("Generated digest with %d ranked items", len(digest.ranked_items))
    # Carry the ranking verdict on the result so the delivery/alerting layer can say the digest was
    # built on an incomplete candidate pool — the digest itself looks normal either way.
    digest.ranking_health = ranker.health
    stories = len(digest.content.items) if digest.content else 0
    if stories < config.pipeline.top_n:
        # WARNING, not ERROR: keeping one of two same-event items legitimately shortens the digest, and the
        # generator is explicitly allowed to emit fewer than the target. It still must not pass
        # unremarked — a day that quietly halves is the same shape as a broken one.
        logger.warning(
            "Digest carries %d stories, fewer than the target %d (dropping a same-event duplicate can do this)",
            stories,
            config.pipeline.top_n,
        )
    if ranker.health.degraded:
        logger.error("Digest was built on an incomplete candidate pool: %s", ranker.health.summary())

    if dry_run:
        logger.info("DRY RUN - Digest output:\n%s", digest.digest_text)
        return items, ranked_items, digest

    try:
        # Trends only feed tomorrow's lead: a failed update must not cost today's post.
        await trend_tracker.update_trends(digest.digest_text, digest_date.isoformat())
    except Exception:
        logger.error("Trend update failed (non-fatal; today's digest still ships)", exc_info=True)

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

    return items, ranked_items, digest


def _recent_story_titles(snapshots: list[dict]) -> list[str]:
    """Story titles from every snapshot the cross-day dedup seed already fetched (the ledger's
    window, ages 1..published_url_ttl_days-1, newest first), so the editor sees what recent digests ran. One
    day was not enough: a weekly roundup brings a story back under a NEW URL the ledger cannot
    match, and on 2026-10-11 three of five stories had run 1-4 days earlier. The ranker reads it for
    Novelty and the editor for selection; it is information, not a filter."""
    titles: list[str] = []
    for snap in snapshots or []:
        content = ((snap or {}).get("digest_result") or {}).get("content") or {}
        titles += [t for it in (content.get("items") or []) if isinstance(it, dict) and (t := it.get("title", ""))]
    return list(dict.fromkeys(titles))


def _editorial_lead(config: Config, digest: DigestResult, digest_date: date) -> str:
    """The digest lead with the AGI-countdown prefix removed, so recent-leads novelty compares
    the editorial angle (not the fixed daily countdown template every lead starts with)."""
    intro = agi_countdown_intro(
        config.pipeline.agi_countdown_date,
        config.pipeline.agi_countdown_template,
        digest_date,
        config.pipeline.agi_countdown_after,
    )
    return editorial_lead(digest.content.lead if digest.content else "", intro)


def persist_digest(
    ranked_items: list[RankedItem], digest: DigestResult, digest_date: date, memory: MemoryStore
) -> None:
    """Persist the digest snapshot (the single path used by both the local CLI and the Lambda handler):
    the visual Lambda publishes off it, and later runs dedup against it."""
    memory.put_digest(digest_date.isoformat(), DigestSnapshot(ranked_items, digest).to_dict())


async def publish_inline(
    maker: DailyVisualMaker,
    ranked_items: list[RankedItem],
    digest: DigestResult,
    digest_date: date,
    *,
    force_republish: bool = False,
) -> None:
    """The local run's visual and Threads post, inline (in AWS the digest Lambda fires the visual
    Lambda instead). Locally there is no alert: PublishIncomplete is the only place a failed or
    partial post can show, as main.py's exit code. Persist the digest BEFORE calling this: a store
    that fails after the post went up would leave the day's posted digest stored nowhere."""
    try:
        posted = await maker.run(ranked_items, digest.content, today=digest_date, force_republish=force_republish)
    except Exception as e:
        logger.error("Daily visual step failed", exc_info=True)
        raise PublishIncomplete(f"The {digest_date} visual/Threads step failed: {e}") from e
    logger.info("Daily visual %s", "posted" if posted else "skipped")
    if maker.threads_unprotected:
        logger.error("Do NOT re-run %s: it is on Threads but not recorded as posted", digest_date)
    outcome = maker.threads_outcome
    if outcome is not None and outcome.posted < outcome.expected:
        raise PublishIncomplete(
            f"The {digest_date} Threads post did not fully land ({outcome.summary()}); root {outcome.root_id or '-'}"
            + (f", standalone {', '.join(outcome.detached)}" if outcome.detached else "")
        )
