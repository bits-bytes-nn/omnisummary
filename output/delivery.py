from __future__ import annotations

import hashlib
import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import Enum, auto

from shared import THREADS_MAX_POST_CHARS, ImageAsset, get_config, logger
from shared.media import extension_for

from .renderers import render_threads_research

# The two places a research report can go. MARKDOWN is the report handed back to the caller (the CLI
# prints it and saves it); THREADS publishes a plain-text thread to the public account and is allowed
# only when the caller asked for it.
MARKDOWN = "markdown"
THREADS = "threads"


@dataclass
class DeliveryStats:
    """What the last delivery actually put in front of the reader. `rendered` is the number of posts
    the report became, `delivered` how many landed, and dropped/trimmed how much content the renderer
    had to discard to fit the channel's limits. Carried back to the agent so its tool result states
    the truth: a thread whose second half was dropped is not "delivered"."""

    rendered: int = 0
    delivered: int = 0
    dropped: int = 0
    trimmed: int = 0

    @property
    def complete(self) -> bool:
        return self.delivered >= self.rendered and not self.dropped and not self.trimmed

    def summary(self) -> str:
        parts = [f"{self.delivered}/{self.rendered} posts delivered"]
        if self.dropped:
            parts.append(f"{self.dropped} post(s) DROPPED (report too long for the channel)")
        if self.trimmed:
            parts.append(f"{self.trimmed} post(s) trimmed to fit the {THREADS_MAX_POST_CHARS}-char cap")
        return "; ".join(parts)


class DeliveryOutcome(Enum):
    """What ONE deliver_research_report call did. Deliberately not a bool: "nothing was posted
    because this channel already has the report" and "the post failed" are different answers, and
    reporting the first as success let a REVISED report be announced as delivered while the reader
    still had the old one."""

    POSTED = auto()
    NOT_POSTED = auto()
    FAILED = auto()


@dataclass
class DeliveryContext:
    """Per-invocation state for one research run: which channels it may use, the OG images the agent
    staged via attach_image, the URLs its tools surfaced, and what each delivery did."""

    # Publishing to the PUBLIC Threads account is decided by the caller (the CLI's --publish), never
    # inferred from the request's wording.
    publish_threads: bool = False
    staged_images: list[ImageAsset] = field(default_factory=list)
    delivered_channels: set[str] = field(default_factory=set)
    # Channels where SOMETHING landed but the delivery was not usable as a whole (a Threads root with
    # no replies). A repeat deliver_report must not re-post over what did land.
    partial_channels: set[str] = field(default_factory=set)
    # Every URL a tool actually surfaced this invocation, in normalize_citation_url form.
    # deliver_report refuses a report citing a URL that appeared in no tool result — the
    # deterministic guard against a fabricated citation reaching the requester or a public post.
    seen_urls: set[str] = field(default_factory=set)
    # The Markdown report the agent submitted (the latest one: a revision replaces it).
    report: str = ""
    # What the Threads delivery put in front of the reader (MARKDOWN has nothing to count).
    threads_stats: DeliveryStats = field(default_factory=DeliveryStats)

    @property
    def allowed_channels(self) -> set[str]:
        return {MARKDOWN, THREADS} if self.publish_threads else {MARKDOWN}


_request_delivery: ContextVar[DeliveryContext | None] = ContextVar("research_request_delivery", default=None)


def current_delivery_context() -> DeliveryContext:
    # A fresh context when nothing is bound rather than a shared module-level singleton, so state
    # can never accumulate across invocations. Real entrypoints always bind via request_context.
    return _request_delivery.get() or DeliveryContext()


@contextmanager
def request_context(delivery: DeliveryContext):
    """Bind per-invocation delivery state so concurrent invocations don't share globals."""
    token = _request_delivery.set(delivery)
    try:
        yield
    finally:
        _request_delivery.reset(token)


async def _deliver_threads(report: str, delivery: DeliveryContext) -> bool:
    """Post a research report to Threads: a root + flat reply chain (each <=500 chars). At most one
    staged image rides the root (Threads media indexing is slow)."""
    from output.threads_handler import post_to_threads

    max_posts = get_config().agent.research_max_threads_posts
    root_text, replies, dropped, trimmed = render_threads_research(report, max_posts=max_posts)
    delivery.threads_stats = DeliveryStats(dropped=dropped, trimmed=trimmed)
    if not root_text.strip():
        # Empty report → nothing to post. An empty root would 400 the Threads API; skip cleanly.
        logger.warning("Research report rendered to empty Threads root; skipping Threads delivery.")
        return False

    image_bytes = None
    image_bucket = ""
    image_key = ""
    image_content_type = "image/png"
    if delivery.staged_images:
        config = get_config()
        bucket = config.aws.state_bucket_name or os.environ.get("STATE_BUCKET", "")
        if not bucket:
            logger.warning("No state bucket configured; Threads image will be skipped (text-only post)")
        else:
            image = delivery.staged_images[0]
            image_bytes = image.data
            image_content_type = image.content_type
            image_bucket = bucket
            prefix = config.aws.s3_prefix.rstrip("/") + "/" if config.aws.s3_prefix else ""
            ext = extension_for(image_content_type)
            image_key = f"{prefix}threads/research_{hashlib.sha256(image_bytes).hexdigest()[:16]}.{ext}"

    outcome = await post_to_threads(
        root_text=root_text,
        replies=replies,
        image_bytes=image_bytes,
        image_bucket=image_bucket,
        image_key=image_key,
        image_content_type=image_content_type,
        topic_tag=get_config().pipeline.threads_topic_tag,
    )
    # Explicitly read the counts: the outcome is a NamedTuple and therefore always truthy, so
    # `return outcome` would report a 0-of-8 post as a success.
    delivery.threads_stats.rendered = outcome.expected
    delivery.threads_stats.delivered = outcome.posted
    return outcome.published


async def deliver_research_report(report: str, *, channel: str, delivery: DeliveryContext) -> DeliveryOutcome:
    """Hand a finished report to `channel`.

    MARKDOWN records the report for the caller; a later call replaces it (nothing external happened,
    so a revision is just a better answer). THREADS publishes, and is idempotent: a second call for a
    channel that already carries the report returns NOT_POSTED rather than success, because the text
    was not published — and the likely reason for the second call is a REVISED report, which must not
    be announced as delivered. A partially-delivered thread is NOT re-sent (that would duplicate what
    did land); the outcome says so instead."""
    if channel == MARKDOWN:
        delivery.report = report
        delivery.delivered_channels.add(MARKDOWN)
        return DeliveryOutcome.POSTED

    if channel in delivery.delivered_channels or channel in delivery.partial_channels:
        logger.warning("Report already posted to '%s'; the text of this call was NOT published", channel)
        return DeliveryOutcome.NOT_POSTED
    # Reset before dispatch so the verdict below reads THIS attempt's counts, never a previous one's.
    delivery.threads_stats = DeliveryStats()
    ok = await _deliver_threads(report, delivery)
    if ok:
        delivery.delivered_channels.add(channel)
        if not delivery.threads_stats.complete:
            logger.warning("Research report delivered INCOMPLETE to %s: %s", channel, delivery.threads_stats.summary())
        return DeliveryOutcome.POSTED
    if delivery.threads_stats.delivered > 0:
        # Not usable as a delivery (a Threads root whose replies all failed), but posts DID land on a
        # public account. Re-sending would duplicate them, so the channel is spent.
        delivery.partial_channels.add(channel)
        logger.warning("Research report delivered INCOMPLETE to %s: %s", channel, delivery.threads_stats.summary())
        return DeliveryOutcome.POSTED
    return DeliveryOutcome.FAILED
