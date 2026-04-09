from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import boto3

from agent.tool_state import DigestStateManager
from main import run_collectors, run_pipeline
from shared import BedrockLanguageModelFactory, Config, S3StateStore, logger


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    logger.info("Digest pipeline Lambda invoked")

    try:
        asyncio.run(_run())
        return {"statusCode": 200, "body": "Digest pipeline completed"}
    except Exception as e:
        logger.error("Digest pipeline failed: %s", e, exc_info=True)
        return {"statusCode": 500, "body": f"Pipeline error: {e}"}


async def _run() -> None:
    config = Config.load()

    rsshub_url = os.environ.get("RSSHUB_BASE_URL")
    if rsshub_url:
        config.collectors.rsshub.base_url = rsshub_url

    KST = ZoneInfo("Asia/Seoul")
    digest_date = datetime.now(KST).date()
    next_day = digest_date + timedelta(days=1)
    reference_time = datetime(next_day.year, next_day.month, next_day.day, tzinfo=KST)
    config.collectors.set_reference_time(reference_time)

    boto_session = boto3.Session(region_name=config.aws.bedrock_region)
    llm_factory = BedrockLanguageModelFactory(
        boto_session=boto_session,
        region_name=config.aws.bedrock_region,
    )

    collected_items = await run_collectors(config, llm_factory)
    logger.info("Collected %d total items", len(collected_items))

    if not collected_items:
        logger.warning("No items collected. Exiting.")
        return

    result = await run_pipeline(config, llm_factory, collected_items, digest_date=digest_date)

    bucket = os.environ.get("STATE_BUCKET", "")
    if bucket and result:
        items, ranked_items, digest = result
        if items and ranked_items and digest:
            state_store = S3StateStore(boto_session, bucket, prefix="digest_state")
            mgr = DigestStateManager()
            mgr.store_digest(items, ranked_items, digest)
            state_store.write(
                f"digest_{digest_date.isoformat()}.json",
                json.dumps(mgr.export_state(), ensure_ascii=False, indent=2),
            )

    logger.info("Digest pipeline completed for %s", digest_date)
