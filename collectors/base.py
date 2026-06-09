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


def load_items_from_s3(filename: str) -> list[CollectedItem] | None:
    """Load a collector's pre-fetched items from S3 (uploaded by a local sync script).

    Sources that YouTube/X block from datacenter (Lambda) IPs are collected locally on a
    residential IP and parked in S3; in AWS the collector reads that file instead of fetching
    live. Returns None when no STATE_BUCKET is set (local dev) or the file is absent, so the
    caller falls back to live collection. The S3 key mirrors trends.json: the prefix's parent
    + filename (S3_PREFIX is '<root>/digest_state', the items live one level up at '<root>/')."""
    bucket = os.environ.get("STATE_BUCKET", "")
    if not bucket:
        return None

    prefix = os.environ.get("S3_PREFIX", "").rstrip("/")
    base_prefix = prefix.rsplit("/", 1)[0] if "/" in prefix else prefix
    s3_key = f"{base_prefix}/{filename}" if base_prefix else filename

    try:
        resp = boto3.client("s3").get_object(Bucket=bucket, Key=s3_key)
        data = json.loads(resp["Body"].read().decode("utf-8"))
        items = [CollectedItem.model_validate(item) for item in data]
        logger.info("Loaded %d items from 's3://%s/%s'", len(items), bucket, s3_key)
        return items
    except ClientError:
        logger.info("No items found at 's3://%s/%s', falling back to live collection", bucket, s3_key)
        return None
    except (json.JSONDecodeError, UnicodeDecodeError, ValidationError) as e:
        logger.warning("Failed to load items from 's3://%s/%s': %s", bucket, s3_key, e)
        return None


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
