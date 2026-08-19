from __future__ import annotations

import hashlib
import os
from datetime import date, datetime
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from shared import (
    DigestContent,
    VisualBrief,
    agi_countdown_intro,
    create_state_store,
    get_correlation_id,
    logger,
)
from shared.config import Config
from shared.history_store import ThreadsPostLedger
from shared.state_store import StateStore

if TYPE_CHECKING:
    from output.threads_handler import ThreadsDelivery


class DigestPublisher:
    """The digest's publish path: the Slack image upload, the Threads root + reply chain, that
    chain's idempotency ledger and the S3 hosting of its image.

    It lives here, beside the channel handlers, because it used to live inside DailyVisualMaker —
    a class named for image generation that also owned the digest's ONLY publish path. Two of that
    method's three early returns existed purely to stop a visual failure from skipping delivery, and
    its return value had to become `slack_ok or threads_ok` to stop reporting "skipped" for every
    Threads-only run. The image is an OPTIONAL argument here, so a failed render is just a text-only
    publish and cannot short-circuit the day's digest by construction — and the publish path is
    exercisable without an OpenAI key.

    Best-effort in both directions: no failure escapes into the pipeline, and every outcome is left
    on `threads_outcome` so the caller can alert on a partial or missing delivery."""

    def __init__(self, config: Config, store: StateStore | None = None) -> None:
        self.config = config
        # Last Threads publish outcome (posted/expected posts), for the caller's metrics/alerts.
        self.threads_outcome: ThreadsDelivery | None = None
        if store is None:
            store = _open_state_store(config)
        self.threads_ledger: ThreadsPostLedger | None = ThreadsPostLedger(store) if store is not None else None

    def has_a_destination(self, content: DigestContent | None, post_date: date, force_republish: bool) -> bool:
        """Whether anything can still be published for this date — checked BEFORE the visual is
        rendered, so a day with no destination doesn't pay for an LLM editor pass plus a gpt-image
        render it can never publish. Owns the logging and the delivery verdict for both refusals."""
        if self._nothing_left_to_publish(post_date, force_republish):
            logger.info(
                "Threads digest for %s already posted and no other channel is enabled, skipping "
                "(use force to re-publish)",
                post_date,
            )
            return False
        if self._no_destination_for_the_image(content):
            # A story-less digest is deliberately NOT posted to Threads (that shape published the
            # broken 2026-08-13/08-17 roots), so with Slack off there is no destination left for an
            # image. Record the same verdict _post_threads would have, so the caller's delivery
            # alert still fires for the unpublished day.
            logger.error(
                "Digest for %s carries no stories and Slack is disabled; skipping the visual render "
                "(the day stays retryable)",
                post_date,
            )
            if self.config.pipeline.enable_threads_post:
                from output.threads_handler import ThreadsDelivery

                self.threads_outcome = ThreadsDelivery(0, 1)
            return False
        return True

    async def publish(
        self,
        content: DigestContent | None,
        *,
        image_bytes: bytes | None = None,
        brief: VisualBrief | None = None,
        today: date | None = None,
        force_republish: bool = False,
        deadline: float | None = None,
    ) -> bool:
        """Publish the digest to every enabled channel, with the image when there is one.

        Success = at least one enabled channel published. Returning only the Slack verdict reported
        "skipped" for every Threads-only run (the current config), hiding real outcomes."""
        post_date = today or datetime.now(ZoneInfo(self.config.aws.timezone)).date()
        slack_ok = await self._post_slack(image_bytes, brief)
        threads_ok = await self._post_threads(
            image_bytes, content, today=post_date, force_republish=force_republish, deadline=deadline
        )
        return slack_ok or threads_ok

    def _nothing_left_to_publish(self, post_date: date, force_republish: bool) -> bool:
        """True when the day is provably done: Threads already carries this date's digest AND no
        other channel could take the visual.

        Deliberately narrow — with enable_slack_post on, the Slack image upload is a separate
        delivery the Threads marker says nothing about, so the run must proceed."""
        if force_republish or self.config.pipeline.enable_slack_post:
            return False
        if not self.config.pipeline.enable_threads_post:
            return False
        return bool(self.threads_ledger and self.threads_ledger.already_posted(post_date))

    def _no_destination_for_the_image(self, content: DigestContent | None) -> bool:
        """True when a rendered image has no possible destination: the digest carries no stories
        (Threads refuses that shape) AND Slack — the other real destination — is off."""
        if self.config.pipeline.enable_slack_post:
            return False
        return not (content and content.items)

    async def _post_slack(self, image_bytes: bytes | None, brief: VisualBrief | None) -> bool:
        if not self.config.pipeline.enable_slack_post:
            return False
        if not image_bytes or not brief:
            return False
        from output.slack_handler import send_image_to_slack

        emoji = self.config.pipeline.visual_caption_emoji
        return await send_image_to_slack(
            image_bytes,
            channel_id=self.config.slack.channel_id,
            title=brief.title,
            comment=f"{emoji} *{brief.title}*\n{brief.caption}",
            bot_token=self.config.slack.bot_token,
        )

    async def _post_threads(
        self,
        image_bytes: bytes | None,
        content: DigestContent | None,
        *,
        today: date | None = None,
        force_republish: bool = False,
        deadline: float | None = None,
    ) -> bool:
        if not self.config.pipeline.enable_threads_post:
            return False
        from output.renderers import render_threads_posts
        from output.threads_handler import ThreadsDelivery as Delivery
        from output.threads_handler import post_to_threads

        # Idempotency: a same-day re-run (manual `main.py`) or an automatic async retry of the
        # visual Lambda after a timeout would otherwise post the whole root+replies set again.
        # Skip if today's digest already went to Threads, unless explicitly forced.
        post_date = today or datetime.now(ZoneInfo(self.config.aws.timezone)).date()
        if self.threads_ledger and not force_republish and self.threads_ledger.already_posted(post_date):
            logger.info("Threads digest for %s already posted, skipping (use force to re-publish)", post_date)
            return False

        # Root = the digest lead (which already carries the AGI-countdown intro, prepended at digest
        # generation) plus the visual when there is one; replies = one per story.
        #
        # A digest with no stories is NOT posted. There used to be a fallback that published the
        # visual's own title/caption as a lone root with no replies: on 2026-08-13 and 2026-08-17 a
        # digest whose content failed to parse took that branch and published a story-less post
        # (one of them carrying leaked `</caption>` markup), consuming the day's ledger slot and
        # logging success. Skipping instead keeps the day retryable and never ships a broken digest.
        if not (content and content.items):
            logger.error("No digest stories to post to Threads for %s; skipping (day stays retryable)", post_date)
            # Leave a VERDICT behind, don't just return: this is the 2026-08-13/08-17 story-loss
            # shape (the channel was enabled, the day was unpublished, and nothing went out), and
            # with threads_outcome left at None the caller's alert was a no-op. expected=1 keeps
            # posted >= expected false, so the "nothing to report" early-return can't swallow it.
            self.threads_outcome = Delivery(0, 1)
            return False
        # Hand the countdown gag over so an over-long lead drops the fixed daily template rather
        # than the sentence carrying the day's argument.
        root_text, replies = render_threads_posts(content, self._countdown_intro(post_date))

        bucket = self.config.aws.state_bucket_name or os.environ.get("STATE_BUCKET", "")
        prefix = self.config.aws.s3_prefix.rstrip("/") + "/" if self.config.aws.s3_prefix else ""
        image_key = f"{prefix}threads/{hashlib.sha256(image_bytes).hexdigest()[:16]}.png" if image_bytes else ""

        # Claim the date BEFORE the multi-minute post so concurrent invocations (e.g. a client
        # that retried a timed-out invoke) see it already taken and skip, instead of all passing
        # the already_posted() check above and each posting. Roll back if the post fails so a
        # genuine failure stays retryable — but only if WE added the mark, so a force-republish
        # failure doesn't wipe out a prior day's successful-post record.
        # run_id scopes marker ownership: a rollback only releases the marker THIS run wrote, so a
        # concurrent invocation's failure can't erase the marker of one that succeeded.
        run_id = get_correlation_id() or ""
        was_marked = bool(self.threads_ledger and self.threads_ledger.already_posted(post_date))
        if self.threads_ledger and not was_marked:
            try:
                self.threads_ledger.mark(post_date, run_id)
            except Exception:
                logger.warning("Failed to record Threads post marker (non-fatal)", exc_info=True)

        try:
            outcome = await post_to_threads(
                root_text=root_text,
                replies=replies,
                image_bytes=image_bytes,
                image_bucket=bucket,
                image_key=image_key,
                deadline=deadline,
            )
        except Exception:
            # Best-effort like the rest of this path: roll the claim back so the post stays
            # retryable, log, and don't let a Threads failure escape into the caller. The verdict
            # is still recorded (nothing landed of the posts we intended) so the caller can alert.
            logger.warning("Threads post failed", exc_info=True)
            self.threads_outcome = Delivery(0, 1 + len([r for r in replies if r.strip()]))
            if not was_marked:
                self._release_threads_marker(post_date, run_id)
            return False
        # Expose the (posted, expected) counts so the caller can report/alert on a partial chain.
        # Branch on .published explicitly — the outcome tuple itself is ALWAYS truthy, so `if
        # outcome:` would treat a 0-of-6 post as a success and skip the ledger rollback.
        self.threads_outcome = outcome
        if not outcome.published and not was_marked:
            self._release_threads_marker(post_date, run_id)
        return outcome.published

    def _release_threads_marker(self, post_date: date, run_id: str = "") -> None:
        if not self.threads_ledger:
            return
        try:
            self.threads_ledger.unmark(post_date, run_id)
        except Exception:
            logger.warning("Failed to roll back Threads post marker (non-fatal)", exc_info=True)

    def _countdown_intro(self, post_date: date) -> str:
        """The AGI-countdown gag for this date, exactly as the digest generator computed it — the one
        part of the lead CODE owns, so the renderer can identify (and drop) it if the lead
        overflows."""
        return agi_countdown_intro(
            self.config.pipeline.agi_countdown_date,
            self.config.pipeline.agi_countdown_template,
            post_date,
            self.config.pipeline.agi_countdown_after,
        )


def _open_state_store(config: Config) -> StateStore | None:
    """The store behind the Threads idempotency ledger, or None when it cannot be opened.

    ERROR, not warning: without the store this run cannot tell an already-published day from a fresh
    one. Deliberately NOT raised — see StateReadError's contract: a lost digest is strictly worse
    than a run without idempotency, and retry_attempts=0 means nothing would auto-retry the
    publish."""
    try:
        return create_state_store(config)
    except Exception:
        logger.error("The Threads post ledger is unavailable (state store init failed)", exc_info=True)
        return None
