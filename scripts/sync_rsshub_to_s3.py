#!/usr/bin/env python3
"""Collect X/Twitter posts via local RSSHub and upload to S3.

Run locally via cron (e.g., daily before the AWS digest pipeline):
    python scripts/sync_rsshub_to_s3.py
"""

import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import boto3
from dotenv import load_dotenv

load_dotenv()

os.environ.pop("CLOUDFLARE_PROXY_URL", None)
os.environ.pop("CLOUDFLARE_PROXY_TOKEN", None)

from collectors.rsshub import RSSHubCollector
from shared import Config, logger


async def main() -> None:
    config = Config.load()

    if not config.collectors.rsshub.enabled:
        logger.info("RSSHub collector is disabled, skipping")
        return

    collector = RSSHubCollector(config.collectors.rsshub)
    items = await collector.collect()

    if not items:
        logger.info("No RSSHub items collected")
        return

    data = [item.model_dump(mode="json") for item in items]

    bucket = config.aws.state_bucket_name
    prefix = config.aws.s3_prefix
    if not bucket:
        logger.warning("No state_bucket_name configured, saving locally only")
        from pathlib import Path

        out = Path("digest_state/rsshub_items.json")
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Saved %d items to '%s'", len(items), out)
        return

    s3_key = f"{prefix}/rsshub_items.json" if prefix else "rsshub_items.json"
    s3_access_key = os.getenv("S3_SYNC_ACCESS_KEY_ID", "")
    s3_secret_key = os.getenv("S3_SYNC_SECRET_ACCESS_KEY", "")

    if s3_access_key and s3_secret_key:
        session = boto3.Session(
            aws_access_key_id=s3_access_key,
            aws_secret_access_key=s3_secret_key,
            region_name=config.aws.region,
        )
    else:
        session = boto3.Session(
            profile_name=config.aws.profile or None,
            region_name=config.aws.region,
        )
    s3 = session.client("s3")
    s3.put_object(
        Bucket=bucket,
        Key=s3_key,
        Body=json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8"),
    )
    logger.info("Uploaded %d RSSHub items to 's3://%s/%s'", len(items), bucket, s3_key)


if __name__ == "__main__":
    asyncio.run(main())
