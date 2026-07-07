from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from slack_sdk.web.async_client import AsyncWebClient

from shared import Config, ImageAsset, logger, resolve_secret, sanitize_slack_mrkdwn
from shared.media import extension_for

from .renderers import _strip_slack_mrkdwn, render_research_blocks, render_threads_research

# Header shown above a deep-research report in Slack — mirrors the daily digest's
# ":satellite: OmniSummary — <date> · N stories" so the two read as one product.
_SLACK_RESEARCH_HEADER = ":satellite: OmniSummary Deep Research"


@dataclass
class DeliveryContext:
    """Per-invocation delivery target + staging for the deep-research agent. `staged_images`
    holds OG images the agent attached via attach_image; `delivered_channels` records which
    channels were successfully posted (so the runtime can fall back only for a channel that
    failed). `dry_run` short-circuits delivery to stdout for the local CLI."""

    channel_id: str = ""
    thread_ts: str = ""
    dry_run: bool = False
    staged_images: list[ImageAsset] = field(default_factory=list)
    delivered_channels: set[str] = field(default_factory=set)
    # Content hashes of images already uploaded to Slack this invocation, so a retried
    # deliver_report (after a mid-delivery failure that left the channel un-recorded) doesn't
    # re-upload the same image bytes.
    _slack_uploaded_image_hashes: set[str] = field(default_factory=set)
    # The last report text handed to deliver_report — so the runtime fallback can re-post the
    # actual report to Slack, not the agent's terminal one-line confirmation.
    last_report: str = ""

    @property
    def delivered(self) -> bool:
        return bool(self.delivered_channels)


_request_delivery: ContextVar[DeliveryContext | None] = ContextVar("research_request_delivery", default=None)


def current_delivery_context() -> DeliveryContext:
    # Return a fresh context when nothing is bound rather than a shared module-level singleton —
    # a process-wide mutable default would accumulate staged_images/channel across invocations in
    # a warm container. Real entrypoints always bind via request_context, so this is a safe stub.
    return _request_delivery.get() or DeliveryContext()


@contextmanager
def request_context(delivery: DeliveryContext):
    """Bind per-invocation delivery state so concurrent invocations don't share globals."""
    token = _request_delivery.set(delivery)
    try:
        yield
    finally:
        _request_delivery.reset(token)


async def _deliver_slack(report: str, delivery: DeliveryContext, *, bot_token: str = "") -> bool:
    """Post a research report to Slack: the staged OG images first (each as a bare file upload,
    no caption), then the mrkdwn report as Block Kit messages. Best-effort."""
    from output.slack_handler import send_image_to_slack

    token = bot_token or resolve_secret("SLACK_BOT_TOKEN", "slack-bot-token")
    channel_id = delivery.channel_id or resolve_secret("SLACK_CHANNEL_ID", "slack-channel-id")
    if not token or not channel_id:
        logger.warning("Slack bot_token or channel_id not configured. Skipping Slack delivery.")
        return False

    for image in delivery.staged_images:
        img_hash = hashlib.sha256(image.data).hexdigest()
        if img_hash in delivery._slack_uploaded_image_hashes:
            continue  # already uploaded on a prior (partially-failed) attempt — don't duplicate
        await send_image_to_slack(
            image.data,
            channel_id=channel_id,
            title=image.alt[:100] or "research image",
            thread_ts=delivery.thread_ts,
            bot_token=token,
            file_ext=extension_for(image.content_type),
        )
        delivery._slack_uploaded_image_hashes.add(img_hash)

    # Enforce the prompt's Slack-mrkdwn contract in code: repair any markup the model slipped
    # (## headings, **bold**, [text](url), emoji) so this primary path matches the fallback path.
    report = sanitize_slack_mrkdwn(report)
    # Notification/preview text: plain prose, not raw <url|...>/*bold* markup.
    notify = _strip_slack_mrkdwn(report)[:200]

    client = AsyncWebClient(token=token)
    try:
        for blocks in render_research_blocks(report, header=_SLACK_RESEARCH_HEADER):
            kwargs: dict = {"channel": channel_id, "blocks": blocks, "text": notify}
            if delivery.thread_ts:
                kwargs["thread_ts"] = delivery.thread_ts
            await client.chat_postMessage(**kwargs)
        logger.info("Delivered research report to Slack channel '%s'", channel_id)
        return True
    except Exception as e:
        logger.warning("Failed to deliver research report to Slack: %s", e)
        return False


async def _deliver_threads(report: str, delivery: DeliveryContext) -> bool:
    """Post a research report to Threads: a root + flat reply chain (each <=500 chars). At most
    one staged image rides the root (Threads media indexing is slow; extra images stay Slack-only)."""
    from output.threads_handler import post_to_threads

    max_posts = Config.load().agent.research_max_threads_posts
    root_text, replies = render_threads_research(report, max_posts=max_posts)
    if not root_text.strip():
        # Empty report → nothing to post. An empty root would 400 the Threads API; skip cleanly.
        logger.warning("Research report rendered to empty Threads root; skipping Threads delivery.")
        return False

    image_bytes = None
    image_bucket = ""
    image_key = ""
    image_content_type = "image/png"
    if delivery.staged_images:
        config = Config.load()
        bucket = config.aws.state_bucket_name or os.environ.get("STATE_BUCKET", "")
        if not bucket:
            # No host for the image bytes → make 'text-only' explicit at the call site rather
            # than relying on post_to_threads' downstream three-part guard.
            logger.warning("No state bucket configured; Threads image will be skipped (text-only post)")
        else:
            image = delivery.staged_images[0]
            image_bytes = image.data
            image_content_type = image.content_type
            image_bucket = bucket
            prefix = config.aws.s3_prefix.rstrip("/") + "/" if config.aws.s3_prefix else ""
            ext = extension_for(image_content_type)
            image_key = f"{prefix}threads/research_{hashlib.sha256(image_bytes).hexdigest()[:16]}.{ext}"

    return await post_to_threads(
        root_text=root_text,
        replies=replies,
        image_bytes=image_bytes,
        image_bucket=image_bucket,
        image_key=image_key,
        image_content_type=image_content_type,
    )


def _dry_run_print(report: str, channel: str, delivery: DeliveryContext) -> bool:
    print(f"\n===== DRY RUN: deliver to {channel} =====")
    if delivery.staged_images:
        print("Staged images:")
        for i, img in enumerate(delivery.staged_images):
            # Mirror actual behavior: Threads attaches only the first image; Slack attaches all.
            note = "(attached to root)" if i == 0 else "(Slack-only, not posted to Threads)"
            tag = note if channel == "threads" else f"({img.content_type})"
            print(f"  - {img.image_url} (from {img.source_url}, {len(img.data)} bytes) {tag}")
    if channel == "threads":
        max_posts = Config.load().agent.research_max_threads_posts
        root, replies = render_threads_research(report, max_posts=max_posts)
        print(f"\n[ROOT]\n{root}\n")
        for i, r in enumerate(replies, 1):
            print(f"[REPLY {i}]\n{r}\n")
    else:
        # Show what _deliver_slack actually posts: sanitized, header + sectioned blocks.
        chunks = render_research_blocks(sanitize_slack_mrkdwn(report), header=_SLACK_RESEARCH_HEADER)
        for mi, blocks in enumerate(chunks, 1):
            print(f"\n[SLACK MESSAGE {mi} — {len(blocks)} block(s)]")
            for b in blocks:
                if b.get("type") == "divider":
                    print("  ---")
                elif b.get("type") == "header":
                    print(f"  [HEADER] {b['text']['text']}")
                else:
                    text = b.get("text", {}).get("text", "")
                    print(f"  ({len(text)} chars) {text}")
    print("===== END DRY RUN =====\n")
    return True


async def deliver_research_report(report: str, *, channel: str, delivery: DeliveryContext) -> bool:
    """Render and post a finished research report to the chosen channel, attaching any staged
    OG images. Records the channel in delivery.delivered_channels on success so a partial
    multi-channel failure (e.g. Threads ok but Slack failed) still triggers the runtime's
    per-channel fallback. In dry-run mode the rendered output is printed instead of posted."""
    delivery.last_report = report
    if channel in delivery.delivered_channels:
        # Idempotency: a retried/duplicated tool call must not double-post the report.
        logger.info("Report already delivered to '%s'; skipping repeat post", channel)
        return True
    if delivery.dry_run:
        ok = _dry_run_print(report, channel, delivery)
    elif channel == "threads":
        ok = await _deliver_threads(report, delivery)
    else:
        ok = await _deliver_slack(report, delivery)
    if ok:
        delivery.delivered_channels.add(channel)
    return ok
