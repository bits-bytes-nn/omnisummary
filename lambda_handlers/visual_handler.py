from __future__ import annotations

import asyncio
import time
from datetime import date, datetime
from typing import Any
from zoneinfo import ZoneInfo

import boto3

from agent.tool_state import DigestStateManager
from output.threads_handler import ThreadsDelivery
from pipeline.daily_visual import DailyVisualMaker
from shared import (
    BedrockLanguageModelFactory,
    Config,
    create_memory_store,
    emit_emf,
    logger,
    publish_alert,
    set_correlation_id,
)

THREADS_POSTS_METRIC = "ThreadsPostsPublished"
THREADS_IMAGE_METRIC = "ThreadsImagePublished"


def _emit_threads_metrics(outcome: ThreadsDelivery | None) -> None:
    """Emit what the day's Threads delivery actually produced, as ONE CloudWatch EMF record: how
    many posts (root + replies) landed, and whether the root carried the visual.

    Emitted UNCONDITIONALLY — including for a run with no outcome at all (0 posts). This Lambda is
    the only delivery path, and skipping the datapoint when nothing was delivered is exactly the
    case worth measuring: a missing datapoint reads as "no data", not as a zero."""
    posted = outcome.posted if outcome else 0
    with_image = 1 if (outcome and outcome.with_image) else 0
    emit_emf({THREADS_POSTS_METRIC: posted, THREADS_IMAGE_METRIC: with_image})


def _remaining_deadline(context: Any) -> float | None:
    """A plain monotonic deadline for the publish path, derived from the Lambda's own remaining
    time. Returns None when there is no context (local runs, tests) — the deadline is optional
    everywhere downstream, so None keeps behaviour byte-identical to having no bound at all.
    The CONTEXT OBJECT never leaves this function; only a float is threaded through."""
    remaining_ms = getattr(context, "get_remaining_time_in_millis", None)
    if remaining_ms is None:
        return None
    try:
        return time.monotonic() + float(remaining_ms()) / 1000.0
    except Exception:
        return None


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Daily-visual Lambda, invoked asynchronously by the digest Lambda so visual
    generation (LLM editor + Tavily + gpt-image, ~1-2 min) stays off the digest's
    critical path. Loads the digest snapshot for the requested date from AgentCore Memory
    and publishes the digest (image + text) to Slack/Threads.

    Failures are logged and RE-RAISED so Lambda records an invocation error: that is what makes
    the Errors alarm fire and puts the async invoke in the DLQ. Re-raising is safe because the
    function is configured with retry_attempts=0 and the ThreadsPostLedger marker blocks a
    duplicate post anyway."""
    # Prefer the id the digest run passed in, so both halves of one digest share a correlation id;
    # fall back to this invocation's request id for a manual/DLQ-replayed invoke that carries none.
    set_correlation_id(_requested_correlation_id(event or {}) or getattr(context, "aws_request_id", "") or None)
    logger.info("Visual Lambda invoked")
    try:
        asyncio.run(_run(event or {}, deadline=_remaining_deadline(context)))
        return {"statusCode": 200, "body": "Visual completed"}
    except Exception as e:
        logger.error("Visual Lambda failed: %s", e, exc_info=True)
        raise


def _invoke_payload(event: dict[str, Any], key: str) -> dict[str, Any]:
    """The payload carrying `key`. A DLQ replay hands back the failed invoke's envelope, whose
    original payload sits under `requestPayload` — honour that too so a replay isn't silently
    re-dated or stripped of its correlation id."""
    if key not in event and isinstance(event.get("requestPayload"), dict):
        return event["requestPayload"]
    return event


def _requested_correlation_id(event: dict[str, Any]) -> str:
    """The correlation id the digest run passed in, so its pipeline half and this delivery half of
    the same digest share one traceable id. Empty when the invoke carries none."""
    return str(_invoke_payload(event, "correlation_id").get("correlation_id", "") or "")


def _requested_date(event: dict[str, Any], tz: ZoneInfo) -> tuple[date, bool]:
    """(digest date this invocation must publish, whether the invoke NAMED that date).

    The digest Lambda passes the date explicitly so the visual publishes the SAME day's content it
    was fired for, rather than re-deriving a clock that can have rolled over.

    The flag matters because it says whether a MISSING snapshot is a real failure: an explicit date
    comes from a run that just persisted one, while a today-fallback invoke (local/manual) may
    legitimately find nothing yet."""
    payload = _invoke_payload(event, "digest_date")
    raw = str(payload.get("digest_date", "") or "")
    if raw:
        try:
            return date.fromisoformat(raw), True
        except ValueError:
            logger.warning("Ignoring malformed digest_date '%s'; falling back to today", raw)
    return datetime.now(tz).date(), False


def _maybe_alert_threads_outcome(outcome: ThreadsDelivery | None, digest_date: date) -> None:
    """SNS notice when the Threads post did not fully land: a partial reply chain (the reader sees
    a digest whose stories stop mid-way), a total delivery failure, or a day that published without
    the visual. Silent on a complete delivery; publish_alert itself is a no-op without
    ALERT_SNS_TOPIC_ARN, so local runs and un-wired stages stay quiet."""
    if outcome is None:
        return
    if outcome.posted < outcome.expected:
        publish_alert(
            "Threads Delivery",
            "ALERT" if outcome.published else "FAILED",
            {
                "Digest date": digest_date.isoformat(),
                "Delivered": outcome.summary(),
                "Detail": (
                    "reply chain incomplete — some stories are missing from the thread"
                    if outcome.published
                    else "the digest was NOT published to Threads"
                ),
            },
        )
        return
    if outcome.published and not outcome.with_image:
        # `expected` counts posts only, so a text-only day was complete success by that measure and
        # said nothing at all: the image is silently dropped on a render failure, a missing OpenAI
        # key or an unreadable secret, and the only trace was one log line inside the maker.
        publish_alert(
            "Threads Delivery",
            "ALERT",
            {
                "Digest date": digest_date.isoformat(),
                "Delivered": outcome.summary(),
                "Detail": "published TEXT-ONLY — the day's visual never reached the root post",
            },
        )


async def _run(event: dict[str, Any] | None = None, *, deadline: float | None = None) -> None:
    config = Config.load()
    if not config.pipeline.enable_daily_visual:
        logger.info("Daily visual disabled, skipping")
        return

    digest_date, dated_invoke = _requested_date(event or {}, ZoneInfo(config.aws.timezone))
    # Load the snapshot BY DATE. 'Load the latest' published yesterday's stories whenever today's
    # snapshot was missing, and comparing digest_result.generated_at instead is not an option: it
    # is a UTC timestamp, so it disagrees with the KST digest date on every pre-09:00 KST run.
    # An unreadable store raises MemoryReadError from here — never mistaken for an empty day.
    data = create_memory_store().get_digest(digest_date.isoformat())
    if not data:
        # An invoke that NAMED its date was fired by a pipeline run that had just persisted the
        # snapshot, so a miss means the day's only delivery path has nothing to publish: RAISE, so
        # the Errors alarm fires and the invoke lands in the DLQ (retry_attempts=0, so re-raising
        # can't duplicate a post). A today-fallback invoke stays quiet — nothing has run yet.
        if dated_invoke:
            raise RuntimeError(f"No digest state for {digest_date} in AgentCore Memory; nothing to publish")
        logger.info("No digest state for %s yet; nothing to publish", digest_date)
        return
    state = DigestStateManager.load_from_dict(data)
    ranked_items = state.get_ranked_items()
    content = state.get_content()

    session = boto3.Session(region_name=config.aws.bedrock_region)
    factory = BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)

    maker = DailyVisualMaker(config, factory)
    posted = await maker.run(ranked_items, content, today=digest_date, deadline=deadline)
    logger.info("Daily visual %s for %s", "posted" if posted else "skipped", digest_date)
    outcome = maker.threads_outcome
    _emit_threads_metrics(outcome)
    _maybe_alert_threads_outcome(outcome, digest_date)
