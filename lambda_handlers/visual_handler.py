from __future__ import annotations

import asyncio
import os
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import boto3

from agent.tool_state import DigestStateManager
from output.threads_handler import ThreadsDelivery
from pipeline.daily_visual import DailyVisualMaker
from shared import BedrockLanguageModelFactory, Config, create_memory_store, format_alarm, logger, set_correlation_id


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Daily-visual Lambda, invoked asynchronously by the digest Lambda so visual
    generation (LLM editor + Tavily + gpt-image, ~1-2 min) stays off the digest's
    critical path. Loads the digest snapshot for the requested date from AgentCore Memory
    and publishes the digest (image + text) to Slack/Threads.

    Failures are logged and RE-RAISED so Lambda records an invocation error: that is what makes
    the Errors alarm fire and puts the async invoke in the DLQ. Re-raising is safe because the
    function is configured with retry_attempts=0 and the ThreadsPostLedger marker blocks a
    duplicate post anyway."""
    set_correlation_id(getattr(context, "aws_request_id", "") or None)
    logger.info("Visual Lambda invoked")
    try:
        asyncio.run(_run(event or {}))
        return {"statusCode": 200, "body": "Visual completed"}
    except Exception as e:
        logger.error("Visual Lambda failed: %s", e, exc_info=True)
        raise


def _requested_date(event: dict[str, Any], tz: ZoneInfo) -> date:
    """The digest date this invocation must publish. The digest Lambda passes it explicitly so the
    visual publishes the SAME day's content it was fired for, rather than re-deriving a clock that
    can have rolled over. A DLQ replay hands back the failed invoke's envelope, whose original
    payload sits under `requestPayload` — honour that too so a replay isn't silently re-dated."""
    payload = event
    if "digest_date" not in payload and isinstance(payload.get("requestPayload"), dict):
        payload = payload["requestPayload"]
    raw = str(payload.get("digest_date", "") or "")
    if raw:
        try:
            return date.fromisoformat(raw)
        except ValueError:
            logger.warning("Ignoring malformed digest_date '%s'; falling back to today", raw)
    return datetime.now(tz).date()


def _maybe_alert_threads_outcome(outcome: ThreadsDelivery | None, digest_date: date) -> None:
    """SNS notice when the Threads post did not fully land: a partial reply chain (the reader sees
    a digest whose stories stop mid-way) or a total delivery failure. No-op without
    ALERT_SNS_TOPIC_ARN, so local runs and un-wired stages stay silent."""
    topic_arn = os.environ.get("ALERT_SNS_TOPIC_ARN", "")
    if not topic_arn or outcome is None or outcome.posted >= outcome.expected:
        return
    status = "ALERT" if outcome.published else "FAILED"
    try:
        subject, message = format_alarm(
            event="Threads Delivery",
            status=status,
            fields={
                "Digest date": digest_date.isoformat(),
                "Delivered": outcome.summary(),
                "Detail": (
                    "reply chain incomplete — some stories are missing from the thread"
                    if outcome.published
                    else "the digest was NOT published to Threads"
                ),
            },
        )
        boto3.client("sns").publish(TopicArn=topic_arn, Subject=subject, Message=message)
        logger.warning("Published SNS alert for Threads delivery (%s)", outcome.summary())
    except Exception as e:
        logger.error("Failed to publish Threads delivery alert: %s", e)


async def _run(event: dict[str, Any] | None = None) -> None:
    config = Config.load()
    if not config.pipeline.enable_daily_visual:
        logger.info("Daily visual disabled, skipping")
        return

    digest_date = _requested_date(event or {}, ZoneInfo(config.aws.timezone))
    # Load the snapshot BY DATE. 'Load the latest' published yesterday's stories whenever today's
    # snapshot was missing, and comparing digest_result.generated_at instead is not an option: it
    # is a UTC timestamp, so it disagrees with the KST digest date on every pre-09:00 KST run.
    data = create_memory_store().get_digest(digest_date.isoformat())
    if not data:
        logger.error("No digest state for %s in AgentCore Memory; nothing to publish", digest_date)
        return
    state = DigestStateManager.load_from_dict(data)
    ranked_items = state.get_ranked_items()
    content = state.get_content()

    session = boto3.Session(region_name=config.aws.bedrock_region)
    factory = BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)

    maker = DailyVisualMaker(config, factory)
    posted = await maker.run(ranked_items, content, today=digest_date)
    logger.info("Daily visual %s for %s", "posted" if posted else "skipped", digest_date)
    _maybe_alert_threads_outcome(maker.threads_outcome, digest_date)
