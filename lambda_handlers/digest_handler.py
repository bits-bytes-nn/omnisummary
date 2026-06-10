from __future__ import annotations

import asyncio
import json
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
    format_alarm,
    logger,
    set_correlation_id,
)

METRIC_NAMESPACE = "OmniSummary"
DIGEST_ITEMS_METRIC = "DigestItemsPublished"


def _emit_digest_items_metric(count: int) -> None:
    """Emit the published-item count as a CloudWatch EMF metric on stdout. A CDK alarm fires
    when this is 0 or missing — catching the 'ran clean but produced an empty digest' (or didn't
    run at all) failure that no error/timeout alarm would surface."""
    emf = {
        "_aws": {
            "Timestamp": int(datetime.now().timestamp() * 1000),
            "CloudWatchMetrics": [
                {"Namespace": METRIC_NAMESPACE, "Dimensions": [[]], "Metrics": [{"Name": DIGEST_ITEMS_METRIC}]}
            ],
        },
        DIGEST_ITEMS_METRIC: count,
    }
    print(json.dumps(emf))


def _maybe_alert(health: HealthReport) -> None:
    topic_arn = os.environ.get("ALERT_SNS_TOPIC_ARN", "")
    if not topic_arn or not health.has_failures:
        return
    failed = [s.name for s in health.sources if s.status == SourceStatus.FAILED]
    try:
        sns = boto3.client("sns")
        subject, message = format_alarm(
            event="Source Health",
            status="ALERT",
            fields={
                "Failed sources": ", ".join(failed),
                "Report": health.summary(),
            },
        )
        sns.publish(TopicArn=topic_arn, Subject=subject, Message=message)
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

    tz = ZoneInfo(config.aws.timezone)
    digest_date = datetime.now(tz).date()
    next_day = digest_date + timedelta(days=1)
    reference_time = datetime(next_day.year, next_day.month, next_day.day, tzinfo=tz)
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
        _emit_digest_items_metric(0)
        return

    result = await run_pipeline(config, llm_factory, collected_items, digest_date=digest_date)

    published = len(result[1]) if result and result[1] else 0
    _emit_digest_items_metric(published)

    if result:
        items, ranked_items, digest = result
        if items and ranked_items and digest:
            # Persist for the follow-up agent. A persistence failure must NOT abort the
            # run or block the daily visual — the Slack digest is already sent by now.
            try:
                # base_dir=None → AgentCore-backed memory store in AWS.
                persist_digest(items, ranked_items, digest, digest_date, base_dir=None)
            except Exception:
                logger.error("Failed to persist digest state (non-fatal)", exc_info=True)
            _trigger_visual()

    logger.info("Digest pipeline completed for %s", digest_date)


def _trigger_visual() -> None:
    """Fire the daily-visual Lambda asynchronously so its ~1-2 min of work doesn't
    count against the digest Lambda's 15-min timeout. Best-effort."""
    fn = os.environ.get("VISUAL_FUNCTION_NAME", "")
    if not fn:
        return
    try:
        boto3.client("lambda").invoke(FunctionName=fn, InvocationType="Event", Payload=b"{}")
        logger.info("Triggered visual Lambda '%s'", fn)
    except Exception as e:
        logger.warning("Failed to trigger visual Lambda: %s", e)
