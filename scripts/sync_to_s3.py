#!/usr/bin/env python3
"""Collect a residential-IP-only source locally and park its items in S3.

X (via the local RSSHub container) and YouTube transcripts are blocked from datacenter (Lambda)
IPs, so they are collected here and the digest Lambda reads the parked file instead.

Run locally via cron (sync_all_to_s3.sh runs every source):
    uv run python scripts/sync_to_s3.py rsshub
    uv run python scripts/sync_to_s3.py youtube
"""

import argparse
import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from dotenv import load_dotenv

load_dotenv()

# Collect live (residential IP) — never read back the parked S3 file during the sync itself
# (with STATE_BUCKET set in .env the collector would return yesterday's park file and this script
# would just re-upload it).
os.environ.pop("STATE_BUCKET", None)

from collectors.base import PARK_META_PREVIOUS_SYNC_AT, dump_items_envelope, park_file_key
from collectors.rsshub import RSSHubCollector
from collectors.youtube import YouTubeCollector
from shared import Config, LocalPaths, logger
from shared.constants import DIGEST_PERIOD_HOURS

_SOURCES = {
    "rsshub": (RSSHubCollector, "rsshub_items.json"),
    "youtube": (YouTubeCollector, "youtube_items.json"),
}


_SYNC_HISTORY = 14


def _sync_stamp_path(source: str) -> Path:
    return Path(__file__).resolve().parent.parent / LocalPaths.DIGEST_STATE_DIR.value / f"last_park_sync_{source}.txt"


def _healthy_syncs(source: str) -> list[tuple[datetime, datetime]]:
    """(start, upload) of this machine's recent healthy syncs of `source`, oldest first. Kept locally: the
    sync's IAM user may only PUT the park file, so reading the file it replaces came back AccessDenied. A
    line holding one time (the older format) is read as both."""
    try:
        lines = _sync_stamp_path(source).read_text().splitlines()
    except (OSError, ValueError):
        return []
    syncs = []
    for line in lines:
        try:
            times = [datetime.fromisoformat(part).astimezone(UTC) for part in line.split()]
        except ValueError:
            continue
        if times:
            syncs.append((times[0], times[-1]))
    return sorted(syncs)


def _last_digest_run(config: Config, now: datetime) -> datetime | None:
    """The latest scheduled digest run (aws.digest_cron_hour/minute, UTC) at or before `now`."""
    try:
        today = now.replace(hour=int(config.aws.digest_cron_hour), minute=int(config.aws.digest_cron_minute))
    except (TypeError, ValueError):
        return None
    today = today.replace(second=0, microsecond=0)
    return today if today <= now else today - timedelta(days=1)


def _previous_sync_at(source: str, config: Config, now: datetime) -> str | None:
    """Where the next digest must read back to: the START of the healthy sync whose file the LAST digest
    read, i.e. the latest one UPLOADED before that digest ran. Not simply the latest sync: one that ran
    late, or ran across the digest time, parked a file no digest read before it was replaced."""
    last_run = _last_digest_run(config, now)
    consumed = [start for start, uploaded in _healthy_syncs(source) if last_run is None or uploaded < last_run]
    return consumed[-1].isoformat() if consumed else None


def _record_healthy_sync(source: str, started_at: datetime, uploaded_at: datetime) -> None:
    syncs = [*_healthy_syncs(source), (started_at, uploaded_at)][-_SYNC_HISTORY:]
    stamp = _sync_stamp_path(source)
    stamp.parent.mkdir(parents=True, exist_ok=True)
    stamp.write_text("".join(f"{start.isoformat()} {uploaded.isoformat()}\n" for start, uploaded in syncs))


async def main(source: str) -> None:
    config = Config.load()
    source_config = getattr(config.collectors, source)
    if not source_config.enabled:
        logger.info("Collector '%s' is disabled, skipping", source)
        return

    bucket = config.aws.state_bucket_name
    if not bucket:
        # A local file nothing reads would let the sync report success while the digest gets no park.
        raise SystemExit("aws.state_bucket_name is not configured; the park file has nowhere to go")

    # Park one digest period more than the Lambda reads: after a skipped or failed sync the next park
    # must still reach back to the sync the last digest read. Repeats are excluded by the cross-day
    # ledger, not by the window.
    sync_config = source_config.model_copy(
        update={"lookback_hours": source_config.lookback_hours + DIGEST_PERIOD_HOURS}
    )
    collector_cls, park_name = _SOURCES[source]
    collector = collector_cls(sync_config)
    # The START is what the next park reaches back to: a post published while this sync ran may be in
    # neither file.
    started_at = datetime.now(UTC)
    items = await collector.collect()
    with_body = sum(1 for item in items if item.text and item.text.strip())
    logger.info("Collected %d '%s' items (%d with body text)", len(items), source, with_body)

    if not items:
        # Park an EMPTY envelope rather than returning early: its `generated_at` stamp is the only
        # proof the sync RAN. A collector exception propagates instead and leaves the previous
        # (good) park file untouched.
        logger.info("No '%s' items collected; parking an empty envelope to stamp the sync time", source)

    s3_key = park_file_key(park_name, config.aws.s3_prefix)
    s3_access_key = os.getenv("S3_SYNC_ACCESS_KEY_ID", "")
    s3_secret_key = os.getenv("S3_SYNC_SECRET_ACCESS_KEY", "")
    if s3_access_key and s3_secret_key:
        session = boto3.Session(
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            region_name=config.aws.region,
        )
    else:
        session = boto3.Session(profile_name=config.aws.profile or None, region_name=config.aws.region)
    s3 = session.client("s3")
    # Park HOW the sync went next to the items: a fresh file full of items from 3 of 40 inputs is
    # otherwise indistinguishable from a healthy one.
    meta = dict(collector.run_meta)
    previous = _previous_sync_at(source, config, datetime.now(UTC))
    if previous:
        meta[PARK_META_PREVIOUS_SYNC_AT] = previous
    payload = dump_items_envelope(items, meta=meta)
    s3.put_object(Bucket=bucket, Key=s3_key, Body=payload.encode("utf-8"))
    logger.info("Uploaded %d '%s' items to 's3://%s/%s'", len(items), source, bucket, s3_key)
    if collector.degraded_detail:
        # A degraded sync (expired X cookies answer with empty feeds) is not a healthy one: the next park
        # then reaches back past it. A single failing input does not count, or one suspended account
        # would switch the read-back off for good.
        logger.warning("'%s' sync was degraded; it does not count as a healthy sync", source)
        return
    try:
        _record_healthy_sync(source, started_at, datetime.now(UTC))
    except OSError:
        logger.warning("Could not record the '%s' sync time; the next park carries no previous_sync_at", source)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", choices=sorted(_SOURCES))
    asyncio.run(main(parser.parse_args().source))
