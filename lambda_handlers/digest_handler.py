from __future__ import annotations

import asyncio
import json
import os
from datetime import date, datetime, timedelta
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
    is_running_in_aws,
    logger,
    set_correlation_id,
)

METRIC_NAMESPACE = "OmniSummary"
DIGEST_ITEMS_METRIC = "DigestItemsPublished"


def _emit_digest_items_metric(count: int) -> None:
    """Emit the number of STORIES the digest carries as a CloudWatch EMF metric on stdout. A CDK
    alarm fires when this is 0 or missing — catching the 'ran clean but produced an empty digest'
    (or didn't run at all) failure that no error/timeout alarm would surface. It must count the
    curated digest items, never the ranker's candidate list, or an empty digest reads as full."""
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
    failed = [s.name for s in health.sources if s.status == SourceStatus.FAILED]
    # A STALE source also alerts: it produced items, but off an S3 park file whose local sync has
    # stopped (or that couldn't be read), which otherwise stays invisible for days. So does a
    # DEGRADED one: it produced items on time, but from only a fraction of its feeds.
    stale = health.stale_sources
    degraded = health.degraded_sources
    if not topic_arn or not (failed or stale or degraded):
        return
    try:
        sns = boto3.client("sns")
        fields = {}
        if failed:
            fields["Failed sources"] = ", ".join(failed)
        if stale:
            fields["Stale sources"] = ", ".join(stale)
        if degraded:
            fields["Degraded sources"] = ", ".join(degraded)
        fields["Report"] = health.summary()
        subject, message = format_alarm(event="Source Health", status="ALERT", fields=fields)
        sns.publish(TopicArn=topic_arn, Subject=subject, Message=message)
        logger.warning("Published SNS alert (failed: %s, stale: %s, degraded: %s)", failed, stale, degraded)
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
        # Log (with the correlation id) and then RE-RAISE: returning a 500 body made the
        # invocation look successful to Lambda, so neither the Errors alarm nor the async DLQ
        # ever fired on a broken digest. retry_attempts=0 means re-raising can't re-post.
        logger.error("Digest pipeline failed: %s", e, exc_info=True)
        raise


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

    items, ranked_items, digest = result
    # Count the STORIES the digest actually carries (the curated content items), not the ranker's
    # candidate list: on 2026-08-13/08-17 the editor's output failed to parse and the digest shipped
    # zero stories, yet this metric reported the full candidate count and the alarm stayed green.
    content_items = digest.content.items if digest and digest.content else []
    _emit_digest_items_metric(len(content_items))

    if items and ranked_items and digest:
        try:
            # base_dir=None → AgentCore-backed memory store in AWS.
            persist_digest(items, ranked_items, digest, digest_date, base_dir=None)
        except Exception as e:
            # The visual Lambda publishes off this snapshot and is the only Threads delivery path,
            # so a failed persist means the day produces NOTHING. Don't trigger the visual (it
            # would load an older date and re-publish stale content) and fail LOUD: re-raising is
            # what fires the Errors alarm and lands the invoke in the DLQ for replay.
            # retry_attempts=0, so this cannot re-run the pipeline.
            logger.error("Failed to persist digest state; visual/Threads delivery skipped", exc_info=True)
            raise RuntimeError(f"Digest snapshot persist failed for {digest_date}: {e}") from e
        _trigger_visual(digest_date)

    logger.info("Digest pipeline completed for %s", digest_date)


def _trigger_visual(digest_date: date) -> None:
    """Fire the daily-visual Lambda asynchronously so its ~1-2 min of work doesn't count against
    the digest Lambda's 15-min timeout. The digest date is passed EXPLICITLY so the visual publishes
    the snapshot this run produced, instead of re-deriving a clock that may have rolled over (or
    reading whatever snapshot happens to be newest).

    NOT best-effort in AWS: that Lambda is the only Threads delivery path, so a missing function
    name or a failed invoke means the day is never published. Both raise — the snapshot is already
    persisted at this point, so failing here loses nothing and is what fires the Errors alarm and
    puts the invoke in the DLQ for replay. Locally the visual runs inline from main.py instead, so
    an unset VISUAL_FUNCTION_NAME is the normal case and stays a quiet no-op."""
    fn = os.environ.get("VISUAL_FUNCTION_NAME", "")
    if not fn:
        if is_running_in_aws():
            raise RuntimeError(
                f"VISUAL_FUNCTION_NAME is not set; the {digest_date} digest was persisted but never delivered"
            )
        return
    payload = json.dumps({"digest_date": digest_date.isoformat()}).encode()
    try:
        boto3.client("lambda").invoke(FunctionName=fn, InvocationType="Event", Payload=payload)
        logger.info("Triggered visual Lambda '%s' for %s", fn, digest_date)
    except Exception as e:
        logger.error("Failed to trigger visual Lambda '%s': %s", fn, e, exc_info=True)
        raise RuntimeError(f"Could not trigger visual delivery for {digest_date}: {e}") from e
