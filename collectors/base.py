from __future__ import annotations

import asyncio
import json
import os
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Sequence
from datetime import UTC, datetime, timedelta

import boto3
from botocore.exceptions import ClientError
from pydantic import ValidationError

from shared import CollectedItem, logger


class BaseCollector(ABC):
    @abstractmethod
    async def collect(self) -> list[CollectedItem]: ...


def cutoff_datetime(lookback_hours: int, reference_time: datetime | None = None) -> datetime:
    return (reference_time or datetime.now(UTC)) - timedelta(hours=lookback_hours)


# Sync-parked items older than this are still used (better stale than empty) but logged LOUDLY,
# so a silently-stopped local cron (laptop asleep, cron disabled) surfaces as a warning instead of
# looking like a healthy run that keeps re-ingesting the same days-old YouTube/X items.
S3_ITEMS_MAX_AGE_HOURS = 36


def dump_items_envelope(items: list[CollectedItem], generated_at: datetime | None = None) -> str:
    """Serialize sync-collected items with a `generated_at` stamp so the loader can detect a
    stale (long-unrun) sync. Written by the local sync scripts; read by load_items_from_s3."""
    stamp = (generated_at or datetime.now(UTC)).isoformat()
    payload = {"generated_at": stamp, "items": [item.model_dump(mode="json") for item in items]}
    return json.dumps(payload, ensure_ascii=False, indent=2)


def load_items_from_s3(filename: str) -> list[CollectedItem] | None:
    """Load a collector's pre-fetched items from S3 (uploaded by a local sync script).

    Sources that YouTube/X block from datacenter (Lambda) IPs are collected locally on a
    residential IP and parked in S3; in AWS the collector reads that file instead of fetching
    live. Returns None when no STATE_BUCKET is set (local dev) or the file is absent, so the
    caller falls back to live collection. The S3 key mirrors trends.json: the prefix's parent
    + filename (S3_PREFIX is '<root>/digest_state', the items live one level up at '<root>/').

    Accepts both the newer envelope ({"generated_at", "items"}) and the legacy bare list; when a
    stamp is present and older than S3_ITEMS_MAX_AGE_HOURS, the items are still returned but a
    warning is logged so a stalled sync doesn't masquerade as a fresh run."""
    bucket = os.environ.get("STATE_BUCKET", "")
    if not bucket:
        return None

    prefix = os.environ.get("S3_PREFIX", "").rstrip("/")
    base_prefix = prefix.rsplit("/", 1)[0] if "/" in prefix else prefix
    s3_key = f"{base_prefix}/{filename}" if base_prefix else filename

    try:
        resp = boto3.client("s3").get_object(Bucket=bucket, Key=s3_key)
        data = json.loads(resp["Body"].read().decode("utf-8"))
        raw_items, generated_at = _unwrap_items_envelope(data)
        items = [CollectedItem.model_validate(item) for item in raw_items]
        _warn_if_stale(s3_key, generated_at)
        logger.info("Loaded %d items from 's3://%s/%s'", len(items), bucket, s3_key)
        return items
    except ClientError:
        logger.info("No items found at 's3://%s/%s', falling back to live collection", bucket, s3_key)
        return None
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as e:
        logger.warning("Failed to load items from 's3://%s/%s': %s", bucket, s3_key, e)
        return None


def _unwrap_items_envelope(data: object) -> tuple[list, datetime | None]:
    """Return (items, generated_at) from either the envelope dict or a legacy bare list."""
    if isinstance(data, dict):
        items = data.get("items", [])
        stamp = data.get("generated_at")
        generated_at: datetime | None = None
        if isinstance(stamp, str):
            try:
                generated_at = datetime.fromisoformat(stamp)
            except ValueError:
                generated_at = None
        return items if isinstance(items, list) else [], generated_at
    return data if isinstance(data, list) else [], None


def _warn_if_stale(s3_key: str, generated_at: datetime | None) -> None:
    if generated_at is None:
        return
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=UTC)
    age_hours = (datetime.now(UTC) - generated_at).total_seconds() / 3600
    if age_hours > S3_ITEMS_MAX_AGE_HOURS:
        logger.warning(
            "S3 items at '%s' are %.1fh old (>%dh) — the local sync may have stalled; using stale data",
            s3_key,
            age_hours,
            S3_ITEMS_MAX_AGE_HOURS,
        )


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
