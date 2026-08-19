from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime, timedelta
from enum import Enum
from http.client import HTTPException
from typing import Any

import boto3
import feedparser
import httpx
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field, ValidationError

from shared import CollectedItem, SourceType, generate_item_id, logger, parse_feed_published_date, retry_async
from shared.constants import BROWSER_USER_AGENT
from shared.proxy import fetch_with_proxy_fallback

# HTTP statuses worth another attempt: rate limiting and server-side faults. Everything else —
# notably 403 (quota exhausted / revoked key) and 404 (unknown resource) — is a verdict retrying
# cannot change. Lives here so every collector classifies a status the same way.
RETRIABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})

# Exception types feedparser reports for a TRANSPORT failure rather than a malformed document.
# OSError covers urllib's URLError, socket.timeout/TimeoutError, ConnectionResetError and ssl.SSLError;
# HTTPException covers a truncated/aborted response (IncompleteRead, RemoteDisconnected).
_TRANSPORT_EXCEPTIONS = (OSError, HTTPException)


class TransientStatusError(RuntimeError):
    """A response that should be retried (429 / 5xx, or a transport-level fetch failure). A
    RuntimeError like the permanent rejections, so an exhausted retry chain still reads as an
    input FAILURE upstream."""


def feed_status_failure(description: str, status: int) -> Exception:
    """The exception for a feed that answered with an error status: transient for 429/5xx (the
    caller retries), permanent for everything else."""
    message = f"{description} returned HTTP {status}"
    return TransientStatusError(message) if status in RETRIABLE_STATUS_CODES else RuntimeError(message)


FEED_FETCH_HEADERS = {"User-Agent": BROWSER_USER_AGENT}


def feed_parse_failure(description: str, bozo_exception: object) -> Exception:
    """The exception for a feed that yielded no entries.

    feedparser does NOT raise on a connection error: it returns a feed with no `status` and the
    transport exception in `bozo_exception`. Classifying that as a permanent parse error meant a DNS
    hiccup lost the feed for the whole day while an HTTP 503 on the same feed got three attempts."""
    message = f"Failed to parse {description}: {bozo_exception}"
    if isinstance(bozo_exception, _TRANSPORT_EXCEPTIONS):
        return TransientStatusError(message)
    return RuntimeError(message)


# ClientError codes that mean "the park file simply isn't there" — an expected state (first run,
# local dev, a source that isn't synced) that must stay a quiet fall-through to live collection.
_ABSENT_ERROR_CODES = frozenset({"NoSuchKey", "NoSuchBucket", "404"})

# Keys of the park-file `meta` block a sync script writes and the collector reads back. Named here
# (the way park_file_key pins the layout) so writer and reader cannot drift. The names are historic
# — RSSHub accounts were the first inputs counted this way — but they mean "the source's INPUTS":
# RSSHub account feeds, YouTube channels, RSS feeds, search queries.
PARK_META_ACCOUNTS_TOTAL = "accounts_total"
PARK_META_ACCOUNTS_FAILED = "accounts_failed"
PARK_META_ACCOUNTS_EMPTY = "accounts_empty"


class ParkOutcome(str, Enum):
    ABSENT = "absent"  # no bucket configured, or no object -> collect live
    FRESH = "fresh"  # park file inside the age window -> use its items
    STALE = "stale"  # park file older than the age window -> use its items, report STALE
    ERROR = "error"  # park file unreadable/misconfigured -> collect live, report STALE


class ParkedItems(BaseModel):
    """Outcome of reading a collector's S3 park file, with the park's age carried explicitly so a
    stalled sync can be reported by the health check. Returned (never a bare list + a hidden
    module-level flag) so staleness travels with the data it describes."""

    outcome: ParkOutcome
    items: list[CollectedItem] = Field(default_factory=list)
    age_hours: float | None = None
    detail: str = ""
    # Whatever the writing sync recorded about HOW the collection went (e.g. how many of its feeds
    # failed). A fresh, on-time park file says nothing about a sync that collected from 2 of 40
    # accounts, so without this a half-dead source reads as perfectly healthy. Always optional:
    # legacy files (bare list, or an envelope with no `meta`) load as an empty dict.
    meta: dict[str, Any] = Field(default_factory=dict)

    @property
    def usable(self) -> bool:
        """True when the park file supplied the items (fresh or stale); False means the caller
        must fall back to live collection."""
        return self.outcome in (ParkOutcome.FRESH, ParkOutcome.STALE)

    @property
    def degraded(self) -> bool:
        """True when the source's items are stale, or the park file could not be read at all —
        either way the health report must say STALE rather than OK."""
        return self.outcome in (ParkOutcome.STALE, ParkOutcome.ERROR)


def degradation_reason(
    *,
    total: int,
    failed: int,
    empty: int,
    what: str,
    threshold: float,
    empty_threshold: float,
    max_failed: int,
) -> str:
    """Why a source that produced items is nonetheless DEGRADED, or "" when it is healthy. Shared by
    the live path and the park-file path so the two verdicts cannot drift.

    Three independent tripwires, any of which is enough:
    - `threshold`: percent of inputs that FAILED (a rate, for sources with many inputs);
    - `max_failed`: an ABSOLUTE failed count, because a rate cannot see a small input list — with
      2 subreddits, 1 of 2 is exactly 50% (clean at the default) and 2 of 2 already raises FAILED,
      so DEGRADED was unreachable;
    - `empty_threshold`: percent of inputs that answered with ZERO items. All-200-and-empty (expired
      RSSHub cookies, a paywalled 200, a playlist that resolves to nothing) is the same
      disappearance shape as a failure, but it trips no failure rate at all — and as long as ONE
      input still produced an item, the source reported a clean OK."""
    fail_rate = failed / total * 100
    empty_rate = empty / total * 100
    if failed > 0 and fail_rate > threshold:
        return f"{failed}/{total} {what} failed (>{threshold:.0f}%)"
    if max_failed > 0 and failed >= max_failed:
        return f"{failed}/{total} {what} failed (>={max_failed})"
    if empty > 0 and empty_rate > empty_threshold:
        return f"{empty}/{total} {what} returned nothing (>{empty_threshold:.0f}%)"
    return ""


class BaseCollector(ABC):
    # Set by collectors that read an S3 park file (YouTube, RSSHub), so run_collectors_with_health
    # can classify a stalled/unreadable park as STALE instead of a healthy OK.
    park_status: ParkedItems | None = None
    # Set by a collector that DID return items but collected them from only a fraction of its
    # inputs (e.g. most RSSHub account feeds failed). Reporting/alerting only — it must never
    # change which items reach the aggregator; without it a source could shrink from 40 feeds to 2
    # and still be logged as OK.
    degraded_detail: str = ""
    # How the LIVE fan-out went (park-meta keys), for a sync script to park alongside the items so
    # the Lambda-side reader can see that a fresh park file came from a half-dead sync. Always
    # REPLACED wholesale (record_run_health), never mutated in place — the class-level default is a
    # shared empty dict, exactly like park_status/degraded_detail above.
    run_meta: dict[str, int] = {}

    @abstractmethod
    async def collect(self) -> list[CollectedItem]: ...

    def record_run_health(
        self,
        *,
        total: int,
        failed: int,
        empty: int = 0,
        threshold: float,
        what: str,
        hint: str = "",
        empty_threshold: float = 100.0,
        max_failed: int = 0,
    ) -> None:
        """Record how many of the source's inputs answered, and report the source DEGRADED when too
        many of them failed OR came back empty.

        One implementation for every collector: a fresh, on-time result says nothing about a run
        that collected from 3 of 40 inputs, which is the shape of a source quietly vanishing from
        the digest. Reporting only — the items themselves are never filtered."""
        self.run_meta = {
            PARK_META_ACCOUNTS_TOTAL: total,
            PARK_META_ACCOUNTS_FAILED: failed,
            PARK_META_ACCOUNTS_EMPTY: empty,
        }
        if total <= 0:
            return
        reason = degradation_reason(
            total=total,
            failed=failed,
            empty=empty,
            what=what,
            threshold=threshold,
            empty_threshold=empty_threshold,
            max_failed=max_failed,
        )
        if not reason:
            return
        self.degraded_detail = reason + (f"; {hint}" if hint else "")
        logger.warning("Collector is DEGRADED: %s", self.degraded_detail)

    def flag_degraded_park(
        self,
        parked: ParkedItems,
        *,
        threshold: float,
        what: str,
        hint: str = "",
        empty_threshold: float = 100.0,
        max_failed: int = 0,
    ) -> None:
        """Report a park file that a HALF-DEAD sync wrote as DEGRADED. The file itself is fresh and
        carries items, so nothing else in the health check can tell that the local sync collected
        from 3 of 40 inputs — or that every input it reached answered with nothing. Judged with the
        SAME thresholds the live path uses; silent for legacy files that carry no meta block.
        Reporting only: every item still reaches the aggregator."""
        total = parked.meta.get(PARK_META_ACCOUNTS_TOTAL) or 0
        failed = parked.meta.get(PARK_META_ACCOUNTS_FAILED) or 0
        empty = parked.meta.get(PARK_META_ACCOUNTS_EMPTY) or 0
        if not all(isinstance(value, int) for value in (total, failed, empty)) or total <= 0:
            return
        reason = degradation_reason(
            total=total,
            failed=failed,
            empty=empty,
            what=what,
            threshold=threshold,
            empty_threshold=empty_threshold,
            max_failed=max_failed,
        )
        if not reason:
            return
        self.degraded_detail = f"parked sync: {reason}" + (f"; {hint}" if hint else "")
        logger.warning("Park file is DEGRADED: %s", self.degraded_detail)


def cutoff_datetime(lookback_hours: int, reference_time: datetime | None = None) -> datetime:
    return (reference_time or datetime.now(UTC)) - timedelta(hours=lookback_hours)


# Default age budget for sync-parked items: older ones are still used (better stale than empty)
# but reported STALE, so a silently-stopped local cron (laptop asleep, cron disabled) surfaces
# instead of looking like a healthy run that keeps re-ingesting the same days-old YouTube/X items.
# Per-collector override: collectors.<source>.park_max_age_hours.
S3_ITEMS_MAX_AGE_HOURS = 36


def dump_items_envelope(
    items: list[CollectedItem], generated_at: datetime | None = None, meta: dict[str, Any] | None = None
) -> str:
    """Serialize sync-collected items with a `generated_at` stamp so the loader can detect a
    stale (long-unrun) sync. Written by the local sync scripts; read by load_items_from_s3.

    `meta` optionally records HOW the sync went (how many of its inputs answered), so the reader
    can report a half-collected source as DEGRADED instead of trusting a fresh timestamp. Omitted
    entirely when empty, keeping the file byte-compatible with readers that never look for it."""
    stamp = (generated_at or datetime.now(UTC)).isoformat()
    payload: dict[str, Any] = {"generated_at": stamp, "items": [item.model_dump(mode="json") for item in items]}
    if meta:
        payload["meta"] = meta
    return json.dumps(payload, ensure_ascii=False, indent=2)


def park_file_key(filename: str, root_prefix: str) -> str:
    """S3 key of a collector's park file under `root_prefix` (the bucket ROOT prefix, i.e. the
    config's aws.s3_prefix). Single source of truth for the layout: the sync scripts write with it
    and load_items_from_s3 reads with it, so the writer and reader can't drift — they used to
    disagree whenever s3_prefix was empty (writer '<file>' vs reader 'digest_state/<file>')."""
    root = root_prefix.strip("/")
    return f"{root}/{filename}" if root else filename


def park_root_prefix(state_prefix: str) -> str:
    """Bucket ROOT prefix derived from the digest-state prefix (S3_PREFIX, set by the CDK Lambda
    env as '<root>/digest_state'). Parked items live one level up from the digest state, so the
    root is the state prefix's parent — and '' when there is no parent (bare 'digest_state')."""
    prefix = state_prefix.strip("/")
    return prefix.rsplit("/", 1)[0] if "/" in prefix else ""


def load_items_from_s3(filename: str, max_age_hours: int = S3_ITEMS_MAX_AGE_HOURS) -> ParkedItems:
    """Load a collector's pre-fetched items from S3 (uploaded by a local sync script).

    Sources that YouTube/X block from datacenter (Lambda) IPs are collected locally on a
    residential IP and parked in S3; in AWS the collector reads that file instead of fetching
    live. The S3 key mirrors trends.json: the prefix's parent + filename (S3_PREFIX is
    '<root>/digest_state', the items live one level up at '<root>/'), computed by the same
    park_file_key() the sync scripts write with.

    Returns a ParkedItems describing the outcome — `usable` says whether the items came from the
    park file, `degraded` says whether the health report must read STALE. ABSENT (no STATE_BUCKET,
    or no object) and ERROR (unreadable object, denied read, unexpected S3 error) both fall back to
    live collection; the difference is only how loudly they are reported, and neither ever raises.

    Accepts both the newer envelope ({"generated_at", "items"}) and the legacy bare list; when a
    stamp is present and older than max_age_hours the items are still returned (stale beats empty)
    but the outcome is STALE.

    A zero-item file is treated as ABSENT only when it is ALSO older than max_age_hours: that
    combination means the local sync stopped producing, and falling through to live collection lets
    a real outage report FAILED instead of a silent EMPTY. A FRESH zero-item envelope is a
    legitimately quiet sync day and is returned as-is, so it can't trigger a false alert."""
    bucket = os.environ.get("STATE_BUCKET", "")
    if not bucket:
        return ParkedItems(outcome=ParkOutcome.ABSENT)

    s3_key = park_file_key(filename, park_root_prefix(os.environ.get("S3_PREFIX", "")))

    try:
        resp = boto3.client("s3").get_object(Bucket=bucket, Key=s3_key)
        data = json.loads(resp["Body"].read().decode("utf-8"))
        raw_items, generated_at, meta = _unwrap_items_envelope(data)
        items = [CollectedItem.model_validate(item) for item in raw_items]
        age_hours = _age_hours(generated_at)
        stale = age_hours is not None and age_hours > max_age_hours
        if not items and stale:
            detail = f"park file 's3://{bucket}/{s3_key}' is empty and {age_hours:.1f}h old (>{max_age_hours}h)"
            logger.warning("%s — treating as absent and falling back to live collection", detail)
            return ParkedItems(outcome=ParkOutcome.ABSENT, age_hours=age_hours, detail=detail)
        if stale:
            detail = (
                f"park file 's3://{bucket}/{s3_key}' is {age_hours:.1f}h old (>{max_age_hours}h); "
                "the local sync may have stalled — using stale data"
            )
            logger.warning("%s", detail)
            return ParkedItems(outcome=ParkOutcome.STALE, items=items, age_hours=age_hours, detail=detail, meta=meta)
        logger.info("Loaded %d items from 's3://%s/%s'", len(items), bucket, s3_key)
        return ParkedItems(outcome=ParkOutcome.FRESH, items=items, age_hours=age_hours, meta=meta)
    except ClientError as e:
        # Classification changes the log level and the reported status ONLY — every ClientError
        # still falls through to live collection, so an unrecognised S3 failure can never abort a
        # run. An absent object is routine; a denied/throttled/misconfigured read is not, and used
        # to be logged at info as if the file simply didn't exist.
        code = str(e.response.get("Error", {}).get("Code", ""))
        if code in _ABSENT_ERROR_CODES:
            logger.info("No items found at 's3://%s/%s', falling back to live collection", bucket, s3_key)
            return ParkedItems(outcome=ParkOutcome.ABSENT)
        detail = f"could not read park file 's3://{bucket}/{s3_key}': {e}"
        logger.warning("%s — falling back to live collection", detail)
        return ParkedItems(outcome=ParkOutcome.ERROR, detail=detail)
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as e:
        detail = f"park file 's3://{bucket}/{s3_key}' is unreadable: {e}"
        logger.warning("%s — falling back to live collection", detail)
        return ParkedItems(outcome=ParkOutcome.ERROR, detail=detail)


def _unwrap_items_envelope(data: object) -> tuple[list, datetime | None, dict[str, Any]]:
    """Return (items, generated_at, meta) from either the envelope dict or a legacy bare list.

    Every part after `items` is optional: a legacy bare list, an envelope without `generated_at`,
    and an envelope without `meta` all load — only what is present is reported."""
    if isinstance(data, dict):
        items = data.get("items", [])
        stamp = data.get("generated_at")
        generated_at: datetime | None = None
        if isinstance(stamp, str):
            try:
                generated_at = datetime.fromisoformat(stamp)
            except ValueError:
                generated_at = None
        meta = data.get("meta")
        return items if isinstance(items, list) else [], generated_at, meta if isinstance(meta, dict) else {}
    return data if isinstance(data, list) else [], None, {}


def _age_hours(generated_at: datetime | None) -> float | None:
    """Age of a sync envelope in hours; None when it carries no (parsable) `generated_at`."""
    if generated_at is None:
        return None
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    return (datetime.now(UTC) - generated_at).total_seconds() / 3600


class CollectorRunResult(BaseModel):
    """A fan-out's items PLUS how many of its inputs answered.

    Returned instead of a bare list so the caller can report a source that produced items from only
    a fraction of its inputs: 2 of 40 feeds answering used to look exactly like a healthy run."""

    items: list[CollectedItem] = Field(default_factory=list)
    total: int = 0
    failed: int = 0
    empty: int = 0


async def gather_collector_results(
    tasks: Sequence[Awaitable[list[CollectedItem]]],
    labels: list[str] | None = None,
    raise_if_all_failed: bool = False,
) -> CollectorRunResult:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[CollectedItem] = []
    failures: list[BaseException] = []
    empty = 0
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            label = labels[i] if labels else f"task-{i}"
            logger.warning("Collector task '%s' failed: %s", label, result)
            failures.append(result)
        elif result:
            items.extend(result)
        else:
            empty += 1

    # When every task errored (and produced nothing), surface it so the health check
    # marks the source FAILED instead of reporting a silent empty result on an outage.
    if raise_if_all_failed and results and len(failures) == len(results):
        raise RuntimeError(f"All {len(failures)} collector tasks failed: {failures[0]}")

    return CollectorRunResult(items=items, total=len(results), failed=len(failures), empty=empty)


async def fetch_feed(url: str, *, description: str, timeout: float) -> Any:
    """Fetch ONE feed and parse it, applying the shared transient/permanent classification.

    The body is downloaded with httpx and only then handed to feedparser. feedparser.parse(url)
    fetches through urllib with NO socket timeout, so a hung host held its worker forever — and
    asyncio.wait_for cannot cancel an asyncio.to_thread worker, so every timed-out attempt leaked a
    thread for the rest of the process's life (up to max_retries per feed). An httpx timeout inside
    a coroutine is a real, cancellable timeout, and it costs nothing once given up on.

    Raises TransientStatusError for a timeout / transport failure / 429 / 5xx (the caller retries),
    and a plain RuntimeError for a permanent status or an unparseable document."""
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=FEED_FETCH_HEADERS) as client:
            response = await client.get(url)
    except httpx.TimeoutException as e:
        raise TransientStatusError(f"{description} timed out after {timeout}s") from e
    except httpx.HTTPError as e:
        # A transport failure (DNS, reset, TLS) is transient, exactly as feedparser's transport-level
        # bozo_exception is: a DNS hiccup used to lose the feed for the whole day.
        raise TransientStatusError(f"{description} fetch failed: {e}") from e

    if response.status_code >= 400:
        raise feed_status_failure(description, response.status_code)
    feed = feedparser.parse(response.content)
    if feed.bozo and not feed.entries:
        raise feed_parse_failure(description, feed.get("bozo_exception"))
    return feed


async def fetch_feed_with_retry(
    url: str,
    *,
    description: str,
    timeout: float,
    max_retries: int,
    backoff_sec: float,
    proxy_fallback: bool = False,
) -> Any:
    """fetch_feed with the retry policy every feed collector shares: a timeout / transport failure /
    429 / 5xx is retried with jittered linear backoff, a permanent verdict (403/404, malformed body)
    is not. The jitter seed is the URL, so the dozens of feeds retrying at once don't resynchronise
    into the burst the upstream rate-limited.

    With proxy_fallback each attempt tries the direct URL first and then the Cloudflare proxy,
    keeping the best usable response of the two (see fetch_with_proxy_fallback)."""

    async def _attempt() -> Any:
        if proxy_fallback:
            return await fetch_with_proxy_fallback(
                url,
                lambda candidate: fetch_feed(candidate, description=description, timeout=timeout),
                has_entries=lambda feed: bool(feed.entries),
            )
        return await fetch_feed(url, description=description, timeout=timeout)

    return await retry_async(
        _attempt,
        max_retries=max_retries,
        backoff_sec=backoff_sec,
        retry_on=(TransientStatusError,),
        description=description,
        jitter_seed=url,
    )


def parse_feed_entries(
    feed: Any,
    *,
    source_type: SourceType,
    cutoff: datetime,
    description: str,
    metadata: dict[str, Any],
    author: str | None = None,
    item_id_of: Callable[[Any, str], str] | None = None,
    limit: int | None = None,
) -> list[CollectedItem]:
    """Turn a parsed feed's entries into CollectedItems: drop anything published before `cutoff`,
    take the title/link, prefer full content over the summary, and skip (never fail on) a
    structurally broken entry.

    One implementation for every feed-based source — RSS, RSSHub and Reddit carried byte-for-byte
    identical loops, so a fix to one silently left the other two behind. `author` overrides the
    entry's own (RSSHub attributes every item to the account) and `item_id_of` overrides the
    entry-id-or-hash default (Reddit derives the post id from the permalink). `limit` caps how many
    entries are READ (YouTube over-fetches a fixed depth and then keeps the latest N by date,
    because its feed is not reliably newest-first)."""
    items: list[CollectedItem] = []
    for entry in feed.entries if limit is None else feed.entries[:limit]:
        try:
            published_at = parse_feed_published_date(entry)
            if published_at and published_at < cutoff:
                continue

            title = entry.get("title", "")
            link = entry.get("link", "")

            text = ""
            if hasattr(entry, "content") and entry.content:
                text = entry.content[0].get("value", "")
            elif hasattr(entry, "summary"):
                text = entry.summary or ""

            item_id = item_id_of(entry, link) if item_id_of else (entry.get("id", "") or generate_item_id(link))

            items.append(
                CollectedItem(
                    item_id=item_id,
                    source_type=source_type,
                    title=title,
                    url=link,
                    text=text,
                    author=entry.get("author") if author is None else author,
                    published_at=published_at,
                    metadata=metadata,
                )
            )
            logger.info("Collected item from %s: '%s'", description, title)
        except (AttributeError, KeyError, TypeError, ValueError):
            logger.warning("Failed to process an entry from %s", description, exc_info=True)

    return items
