from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import boto3

from main import persist_digest, run_collectors_with_health, run_pipeline
from shared import (
    BedrockLanguageModelFactory,
    Config,
    HealthReport,
    SourceStatus,
    logger,
    set_correlation_id,
)


def _maybe_alert(health: HealthReport) -> None:
    topic_arn = os.environ.get("ALERT_SNS_TOPIC_ARN", "")
    if not topic_arn or not health.has_failures:
        return
    failed = [s.name for s in health.sources if s.status == SourceStatus.FAILED]
    try:
        sns = boto3.client("sns")
        sns.publish(
            TopicArn=topic_arn,
            Subject=f"[omnisummary] {len(failed)} source(s) failed",
            Message="Source health report:\n\n" + health.summary(),
        )
        logger.warning("Published SNS alert for failed sources: %s", failed)
    except Exception as e:
        logger.error("Failed to publish SNS alert: %s", e)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    request_id = getattr(context, "aws_request_id", "") if context else ""
    set_correlation_id(request_id or None)
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

    collected_items, health = await run_collectors_with_health(config, llm_factory)
    logger.info("Collected %d total items", len(collected_items))
    logger.info("Source health report:\n%s", health.summary())
    _maybe_alert(health)

    if not collected_items:
        logger.warning("No items collected. Exiting.")
        return

    result = await run_pipeline(config, llm_factory, collected_items, digest_date=digest_date)

    if result:
        items, ranked_items, digest = result
        if items and ranked_items and digest:
            # base_dir=None → AgentCore-backed memory store in AWS.
            persist_digest(items, ranked_items, digest, digest_date, base_dir=None)

    logger.info("Digest pipeline completed for %s", digest_date)
