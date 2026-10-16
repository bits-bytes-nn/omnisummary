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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from dotenv import load_dotenv

load_dotenv()

# Collect live (residential IP) — never read back the parked S3 file during the sync itself
# (with STATE_BUCKET set in .env the collector would return yesterday's park file and this script
# would just re-upload it).
os.environ.pop("STATE_BUCKET", None)

from collectors.base import dump_items_envelope, park_file_key
from collectors.rsshub import RSSHubCollector
from collectors.youtube import YouTubeCollector
from shared import Config, logger

_SOURCES = {
    "rsshub": (RSSHubCollector, "rsshub_items.json"),
    "youtube": (YouTubeCollector, "youtube_items.json"),
}


async def main(source: str) -> None:
    config = Config.load()
    source_config = getattr(config.collectors, source)
    if not source_config.enabled:
        logger.info("Collector '%s' is disabled, skipping", source)
        return

    collector_cls, park_name = _SOURCES[source]
    collector = collector_cls(source_config)
    items = await collector.collect()
    with_body = sum(1 for item in items if item.text and item.text.strip())
    logger.info("Collected %d '%s' items (%d with body text)", len(items), source, with_body)

    if not items:
        # Park an EMPTY envelope rather than returning early: its `generated_at` stamp is the only
        # proof the sync RAN. A collector exception propagates instead and leaves the previous
        # (good) park file untouched.
        logger.info("No '%s' items collected; parking an empty envelope to stamp the sync time", source)

    # Park HOW the sync went next to the items: a fresh file full of items from 3 of 40 inputs is
    # otherwise indistinguishable from a healthy one.
    payload = dump_items_envelope(items, meta=collector.run_meta)

    bucket = config.aws.state_bucket_name
    if not bucket:
        # A local file nothing reads would let the sync report success while the digest gets no park.
        raise SystemExit("aws.state_bucket_name is not configured; the park file has nowhere to go")

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
    session.client("s3").put_object(Bucket=bucket, Key=s3_key, Body=payload.encode("utf-8"))
    logger.info("Uploaded %d '%s' items to 's3://%s/%s'", len(items), source, bucket, s3_key)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", choices=sorted(_SOURCES))
    asyncio.run(main(parser.parse_args().source))
