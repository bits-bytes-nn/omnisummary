from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Sequence
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import boto3
from botocore.exceptions import ClientError
from pydantic import BaseModel, Field, ValidationError

from shared import CollectedItem, logger

# ClientError codes that mean "the park file simply isn't there" — an expected state (first run,
# local dev, a source that isn't synced) that must stay a quiet fall-through to live collection.
_ABSENT_ERROR_CODES = frozenset({"NoSuchKey", "NoSuchBucket", "404"})


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


class BaseCollector(ABC):
    # Set by collectors that read an S3 park file (YouTube, RSSHub), so run_collectors_with_health
    # can classify a stalled/unreadable park as STALE instead of a healthy OK.
    park_status: ParkedItems | None = None
    # Set by a collector that DID return items but collected them from only a fraction of its
    # inputs (e.g. most RSSHub account feeds failed). Reporting/alerting only — it must never
    # change which items reach the aggregator; without it a source could shrink from 40 feeds to 2
    # and still be logged as OK.
    degraded_detail: str = ""

    @abstractmethod
    async def collect(self) -> list[CollectedItem]: ...


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


async def gather_collector_results(
    tasks: Sequence[Awaitable[list[CollectedItem]]],
    labels: list[str] | None = None,
    raise_if_all_failed: bool = False,
) -> list[CollectedItem]:
    results = await asyncio.gather(*tasks, return_exceptions=True)
    items: list[CollectedItem] = []
    failures: list[BaseException] = []
    for i, result in enumerate(results):
        if isinstance(result, BaseException):
            label = labels[i] if labels else f"task-{i}"
            logger.warning("Collector task '%s' failed: %s", label, result)
            failures.append(result)
        else:
            items.extend(result)

    # When every task errored (and produced nothing), surface it so the health check
    # marks the source FAILED instead of reporting a silent empty result on an outage.
    if raise_if_all_failed and results and len(failures) == len(results):
        raise RuntimeError(f"All {len(failures)} collector tasks failed: {failures[0]}")

    return items
