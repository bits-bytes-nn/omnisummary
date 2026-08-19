from __future__ import annotations

import asyncio
import json
import os
from datetime import date
from typing import Any

import boto3

from pipeline import persist_digest, resolve_digest_window, run_collectors_with_health, run_pipeline
from shared import (
    BedrockLanguageModelFactory,
    Config,
    DigestResult,
    HealthReport,
    SourceStatus,
    emit_emf,
    get_correlation_id,
    is_running_in_aws,
    logger,
    publish_alert,
    set_correlation_id,
)

DIGEST_ITEMS_METRIC = "DigestItemsPublished"


def _emit_digest_items_metric(count: int) -> None:
    """Emit the number of STORIES the digest carries as a CloudWatch EMF metric. A CDK alarm fires
    when this is 0 or missing — catching the 'ran clean but produced an empty digest' (or didn't run
    at all) failure that no error/timeout alarm would surface. It must count the curated digest
    items, never the ranker's candidate list, or an empty digest reads as full."""
    emit_emf({DIGEST_ITEMS_METRIC: count})


def _maybe_alert(health: HealthReport, alert_on_empty: list[str] | None = None) -> None:
    failed = [s.name for s in health.sources if s.status == SourceStatus.FAILED]
    # A STALE source also alerts: it produced items, but off an S3 park file whose local sync has
    # stopped (or that couldn't be read), which otherwise stays invisible for days. So does a
    # DEGRADED one: it produced items on time, but from only a fraction of its feeds.
    stale = health.stale_sources
    degraded = health.degraded_sources
    # A source that ran clean and returned NOTHING leaves no exception, no stale park file and no
    # degraded ratio — it is simply dark. Only the sources config NAMES alert (collectors.
    # alert_on_empty), so reddit/x quiet days can't page every morning.
    watched = set(alert_on_empty or [])
    empty = [name for name in health.empty_sources if name in watched]
    if not (failed or stale or degraded or empty):
        return
    fields = {}
    if failed:
        fields["Failed sources"] = ", ".join(failed)
    if stale:
        fields["Stale sources"] = ", ".join(stale)
    if degraded:
        fields["Degraded sources"] = ", ".join(degraded)
    if empty:
        fields["Empty sources"] = ", ".join(empty)
    fields["Report"] = health.summary()
    publish_alert("Source Health", "ALERT", fields)


def _maybe_alert_ranking(digest: DigestResult | None, digest_date: date) -> None:
    """Notice when the digest was built on an INCOMPLETE candidate pool (a ranking batch that failed
    every retry deletes ~40 candidates from the day), or when the digest AS SHIPPED broke a diversity
    cap the ranker guarantees only on the ranked core. Either way the digest itself reads perfectly
    normally, which is why it needs saying. Silent when nothing was lost and nothing was breached."""
    health = digest.ranking_health if digest else None
    breaches = digest.diversity_breaches if digest else []
    if not breaches and (health is None or not health.degraded):
        return
    fields = {"Digest date": digest_date.isoformat()}
    if health is not None and health.degraded:
        fields["Detail"] = health.summary()
    if breaches:
        fields["Diversity"] = "; ".join(breaches)
    publish_alert("Ranking Health", "ALERT", fields)


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

    digest_date, reference_time = resolve_digest_window(config)
    config.collectors.set_reference_time(reference_time)

    boto_session = boto3.Session(region_name=config.aws.bedrock_region)
    llm_factory = BedrockLanguageModelFactory(
        boto_session=boto_session,
        region_name=config.aws.bedrock_region,
    )

    collected_items, health = await run_collectors_with_health(config, llm_factory)
    logger.info("Collected %d total items", len(collected_items))
    logger.info("Source health report:\n%s", health.summary())
    _maybe_alert(health, config.collectors.alert_on_empty)

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
    # Published SEPARATELY from the collector-health alert above (which runs before the pipeline):
    # keeping them apart means a pipeline exception can never swallow the collector notice, and a
    # digest built on a pool that lost a whole ranking batch is reported even though the digest
    # itself looks entirely normal.
    _maybe_alert_ranking(digest, digest_date)

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
    # Pass this run's correlation id along: the visual Lambda otherwise mints a fresh unrelated one,
    # so the two halves of a single digest (pipeline + the only delivery path) could not be traced
    # as one run — which is the whole point of the structured logger's correlation filter.
    payload = json.dumps({"digest_date": digest_date.isoformat(), "correlation_id": get_correlation_id()}).encode()
    try:
        boto3.client("lambda").invoke(FunctionName=fn, InvocationType="Event", Payload=payload)
        logger.info("Triggered visual Lambda '%s' for %s", fn, digest_date)
    except Exception as e:
        logger.error("Failed to trigger visual Lambda '%s': %s", fn, e, exc_info=True)
        raise RuntimeError(f"Could not trigger visual delivery for {digest_date}: {e}") from e
