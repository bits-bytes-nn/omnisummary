"""Digest orchestration: the single path from collected items to a stored digest, taken by both the
CLI (main.py) and the digest Lambda. The entrypoints hand in the sources, the models and the stores
(entrypoints/bootstrap.py); this module builds none of them."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import date, datetime, timedelta
from typing import TYPE_CHECKING, NamedTuple
from zoneinfo import ZoneInfo

from application.digest_generator import DigestGenerator
from application.history import (
    PublishedUrlLedger,
    ThreadsPostLedger,
    published_urls_from_snapshots,
)
from application.ports import ContentSource, LanguageModels, MemoryStore, NoteText, PaidPageCheck, StateStore
from application.ranker import ContentRanker
from application.trend_tracker import TrendTracker
from domain.aggregator import ContentAggregator, normalize_url
from domain.digest_snapshot import DigestSnapshot
from domain.formatting import resolve_origin_key
from domain.models import CollectedItem, DigestResult, HealthReport, RankedItem, SourceHealth, SourceStatus
from shared import Config, SourceType, logger
from shared.config import PipelineConfig

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
    # The collectors cut their own fan-out at collection_deadline_sec and keep what finished; this
    # backstop cancels one that is still running past it.
    backstop = config.collectors.collection_backstop_sec
    tasks = [_collect_within(collector, backstop) for collector in collectors]

    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[CollectedItem] = []
    health: list[SourceHealth] = []
    for label, collector, result in zip(labels, collectors, results, strict=True):
        if isinstance(result, BaseException):
            health.append(SourceHealth(name=label, item_count=0, status=SourceStatus.FAILED, detail=str(result)))
            logger.warning("Collector '%s' failed: %s", label, result)
            continue
        items.extend(result)
        # Items arrived, but the source says they are stale or came from a fraction of its inputs.
        # Reported, never filtered: every item still reaches the aggregator.
        if (issue := collector.health_issue) is not None:
            status, detail = issue
            health.append(SourceHealth(name=label, item_count=len(result), status=status, detail=detail))
            logger.warning("Collector '%s' is %s: %s", label, status.value.upper(), detail)
            continue
        status = SourceStatus.OK if result else SourceStatus.EMPTY
        health.append(SourceHealth(name=label, item_count=len(result), status=status))
    report = HealthReport(sources=health)
    # A source that ran clean and returned nothing is DARK: no exception, no stale park file, no
    # failure ratio. The config names which sources that is an incident for (the digest Lambda
    # alerts on the same list); log it loudly here so a local run sees it too.
    dark = report.watched_empty(config.collectors.alert_on_empty)
    if dark:
        logger.error("Source(s) returned NO items and are configured as must-not-be-empty: %s", ", ".join(dark))
    return items, report


class CollectionDeadlineExceeded(RuntimeError):
    """A collector ran past collectors.collection_backstop_sec and was cancelled."""


async def _collect_within(collector: ContentSource, deadline: int) -> list[CollectedItem]:
    """One collector's collect(), cancelled at the backstop so a stuck source cannot spend the Lambda's
    time budget the ranking and writing still need."""
    scope = asyncio.timeout(deadline)
    try:
        async with scope:
            return await collector.collect()
    except TimeoutError as e:
        if not scope.expired():
            raise
        raise CollectionDeadlineExceeded(f"exceeded collection deadline ({deadline}s)") from e


class PipelineRun(NamedTuple):
    """What run_pipeline produced: the aggregated items, the candidate pool and the digest."""

    items: list[CollectedItem]
    ranked_items: list[RankedItem]
    digest: DigestResult


class RecentHistory(NamedTuple):
    """The recent snapshots, read once: `dedup` is the URL ledger's window, `titles` the longer title window."""

    dedup: list[DigestSnapshot]
    titles: list[DigestSnapshot]


def read_recent_history(config: Config, memory_store: Callable[[], MemoryStore], digest_date: date) -> RecentHistory:
    """Ages 1..ttl-1 seed cross-day dedup (today's own snapshot is excluded so a same-day re-run keeps its
    stories, and a story back on the TTL day is allowed again); ages 1..recent_title_days feed the titles.
    Best-effort: a failed read costs only the dedup seed and the titles, never the run."""
    dedup_days = config.pipeline.published_url_ttl_days - 1
    title_days = config.pipeline.recent_title_days
    try:
        dated = memory_store().get_recent_digests_dated(digest_date, max(dedup_days, title_days))
    except Exception:
        logger.warning("Could not read the recent digest snapshots (non-fatal)", exc_info=True)
        return RecentHistory([], [])
    return RecentHistory(
        dedup=[snap for day, snap in dated if (digest_date - day).days <= dedup_days],
        titles=[snap for day, snap in dated if (digest_date - day).days <= title_days],
    )


class PublishIncomplete(RuntimeError):
    """A local run's Threads post failed or did not fully land. The digest is already persisted."""


def already_published(config: Config, digest_date: date, state_store: StateStore) -> bool:
    """True when the date's digest is already on Threads. A second run for it (a duplicate invoke, a
    local run) still persisted its own snapshot, leads and ledger entries before the post step skipped,
    so the record of the day became a digest nobody saw. With Threads posting off nothing can post, so a
    re-run is how a day's record is rebuilt."""
    if config.pipeline.enable_threads_post and ThreadsPostLedger(state_store).already_posted(digest_date):
        logger.warning("The %s digest is already on Threads; this run stops before collecting", digest_date)
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
    sibling_memory: Callable[[], list[MemoryStore]],
    paid_check: PaidPageCheck,
    note_text: NoteText,
    dry_run: bool = False,
) -> PipelineRun | None:
    """Aggregate, rank and write the day's digest, then record its stories and lead. `memory_store`
    is called once, to read the recent snapshots (read_recent_history). The digest is neither stored
    nor published here: see persist_digest and publish_inline."""
    ledger = PublishedUrlLedger(state_store, config.pipeline.published_url_ttl_days)

    # Cross-day dedup draws from BOTH the ledger AND recent AgentCore Memory snapshots, so it
    # self-heals from existing history (and survives a lost/empty ledger) rather than only
    # suppressing dupes for runs after the ledger is first populated. URLs are normalized to the
    # ledger's canonical form so http/https + trailing-slash variants of a past story still match.
    # Re-normalized on read: a key written under an older normalize_url must still match.
    exclude_urls = {normalize_url(u) for u in ledger.recent_urls(digest_date)}
    history = await asyncio.to_thread(read_recent_history, config, memory_store, digest_date)
    exclude_urls |= {normalize_url(u) for u in published_urls_from_snapshots(history.dedup)}

    aggregator = ContentAggregator()
    items = aggregator.aggregate(collected_items, exclude_urls=exclude_urls)
    logger.info("Aggregated %d unique items (excluding %d recently-published URLs)", len(items), len(exclude_urls))

    if not items:
        logger.warning("No items to process after aggregation")
        return None

    # Over-select (top_n + buffer): the editor picks the strongest top_n distinct stories from it.
    select_count = config.pipeline.top_n + config.pipeline.digest_candidate_buffer
    ranker = ContentRanker(config.pipeline, models, paid_check)
    recent = editor_context(history, sibling_snapshots(sibling_memory, digest_date), config.pipeline)
    ranked_items = await ranker.rank(items, select_count=select_count, recent_titles=recent.titles)
    logger.info("Ranked %d items (from %d)", len(ranked_items), len(items))

    if not ranked_items:
        logger.warning("No items passed ranking threshold")
        return None

    await require_primary(ranked_items, config.pipeline.primary_quota, config.pipeline.primary_source_types, note_text)

    trend_tracker = TrendTracker(config.pipeline, models, state_store)
    trends_context = trend_tracker.get_trends_context(today=digest_date)

    digest = await write_digest(
        config.pipeline, models, ranked_items, recent, trends_context=trends_context, today=digest_date
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
        return PipelineRun(items, ranked_items, digest)

    try:
        # Trends only feed tomorrow's lead: a failed update must not cost today's post. The classifier
        # reads the stories, not the editor's lead and implications: fed those, the editor's own theses
        # came back as trends ("the digest cautioned ...") and then as tomorrow's arc.
        story_text = "\n\n".join(
            f"{item.title}\n{item.body}" for item in (digest.content.items if digest.content else [])
        )
        await trend_tracker.update_trends(story_text or digest.digest_text, digest_date.isoformat())
    except Exception:
        logger.error("Trend update failed (non-fatal; today's digest still ships)", exc_info=True)

    try:
        # Every scored item's score and origin, not just the pool: enough to replay a change to the slots,
        # the caps or the threshold on a past day without re-ranking. A changed prompt, criteria or model
        # still needs the ranking re-run (the items' text is not kept).
        state_store.write_json(
            f"{SCORED_ITEMS_PREFIX}{digest_date.isoformat()}.json",
            [
                {
                    "url": r.item.url,
                    "title": r.item.title,
                    "source": r.item.source_type.value,
                    "origin": resolve_origin_key(r.item),
                    "score": r.score,
                    "reasoning": r.reasoning,
                }
                for r in ranker.scored
            ],
        )
    except Exception:
        logger.warning("Failed to keep the day's scored items (non-fatal)", exc_info=True)

    try:
        # The pool exactly as the editor read it (full text, quota flags, a note's PDF text): the stored
        # snapshot cuts each item to its share of the AgentCore event limit, so a replay from it judged
        # a shorter source than production wrote from.
        state_store.write_json(
            f"{POOLS_PREFIX}{digest_date.isoformat()}.json", [r.model_dump(mode="json") for r in ranked_items]
        )
    except Exception:
        logger.warning("Failed to keep the day's candidate pool (non-fatal)", exc_info=True)

    # Record the stories that became TODAY'S digest of record so future runs don't re-publish them.
    # Recorded at generation (post trend-update) — the same content.items the snapshot carries and
    # the visual Lambda delivers — not gated on downstream delivery, which is alarmed separately.
    try:
        if digest.content and digest.content.items:
            ledger.record([normalize_url(it.url) for it in digest.content.items], digest_date)
    except Exception:
        logger.warning("Failed to update the published-URL ledger (non-fatal)", exc_info=True)

    return PipelineRun(items, ranked_items, digest)


async def require_primary(
    pool: list[RankedItem], quota: int, source_types: list[SourceType], note_text: NoteText
) -> None:
    """Mark the `quota` best-scored pool items of `source_types` as stories the editor must carry, and
    give each research note among them its full text: the editor picked the press report of a view over
    the view's own note or post, whose summary or post was too thin to write from."""
    if not quota:
        return
    chosen = sorted((r for r in pool if r.item.source_type in source_types), key=lambda r: -r.score)[:quota]
    for r in chosen:
        r.item.primary = True
        if r.item.source_type == SourceType.RESEARCH:
            text = await note_text(r.item)
            if text:
                r.item.text = f"{r.item.text}\n\n{text}"
    logger.info(
        "Primary stories the editor must carry: %s", ", ".join(f"{r.item.title[:50]!r} ({r.score:.2f})" for r in chosen)
    )


def sibling_snapshots(stores: Callable[[], list[MemoryStore]], digest_date: date) -> list[DigestSnapshot]:
    """The other editions' snapshots for the digest date and the day before, newest first: what this
    Threads account posted in the last day or so. The editions run at different hours (07:30 and 19:00),
    so the same date can already be live. Best-effort: it only steers the ranker and the editor."""
    out: list[DigestSnapshot] = []
    try:
        for store in stores():
            today = store.get_digest(digest_date.isoformat())
            out += ([today] if today else []) + store.get_recent_digests(digest_date, 1)
    except Exception:
        logger.warning("Could not read the other editions' recent digests (non-fatal)", exc_info=True)
    return out


class EditorContext(NamedTuple):
    """What the recent digests tell the ranker and the editor: story titles and angles, newest first."""

    titles: list[str]
    leads: list[str]


def editor_context(history: RecentHistory, sibling: list[DigestSnapshot], config: PipelineConfig) -> EditorContext:
    """This edition's recent snapshots joined by the other editions' on the same account."""
    return EditorContext(
        titles=list(dict.fromkeys(_recent_story_titles(history.titles) + _recent_story_titles(sibling))),
        leads=recent_angles(history.dedup, config) + recent_angles(sibling, config),
    )


async def write_digest(
    config: PipelineConfig,
    models: LanguageModels,
    pool: list[RankedItem],
    recent: EditorContext,
    *,
    trends_context: str,
    today: date,
) -> DigestResult:
    """The editor's call on a ranked pool: the one path run_pipeline and the eval harness's replay share,
    so an input added to the editor reaches both."""
    return await DigestGenerator(config, models).generate(
        pool, trends_context=trends_context, today=today, recent_leads=recent.leads, recent_titles=recent.titles
    )


def _recent_story_titles(snapshots: list[DigestSnapshot]) -> list[str]:
    """Story titles from the recent snapshots (newest first), so the ranker can judge Novelty and the
    editor selection against what recent digests ran: information, not a filter."""
    titles = [item.title for snap in snapshots if snap.content for item in snap.content.items if item.title]
    return list(dict.fromkeys(titles))


def recent_angles(snapshots: list[DigestSnapshot], config: PipelineConfig) -> list[str]:
    """The angles the next digest must not repeat (newest first): the last recent_leads_window leads (the
    editor is shown each one's thesis sentence, so the countdown gag never reaches the comparison) and
    the implications of the last recent_implication_days snapshots."""
    leads = [snap.content.lead for snap in snapshots if snap.content and snap.content.lead.strip()]
    recent = [snap.content for snap in snapshots[: config.recent_implication_days] if snap.content]
    implications = [item.implication for content in recent for item in content.items if item.implication.strip()]
    return leads[: config.recent_leads_window] + implications


SCORED_ITEMS_PREFIX = "scored_items/"
POOLS_PREFIX = "pools/"


def persist_digest(
    ranked_items: list[RankedItem], digest: DigestResult, digest_date: date, memory: MemoryStore
) -> None:
    """Persist the digest snapshot (the single path used by both the local CLI and the Lambda handler):
    the visual Lambda publishes off it, and later runs dedup against it."""
    memory.put_digest(digest_date.isoformat(), DigestSnapshot(ranked_items, digest))


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
