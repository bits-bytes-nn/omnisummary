from __future__ import annotations

import asyncio
import hashlib
import os
import time
from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import datetime
from typing import Any

import httpx

from adapters.aws import resolve_secret
from adapters.og_image import extension_for
from application.ports import PublishError
from domain.models import ThreadsComment, ThreadsDelivery
from shared import THREADS_MAX_POST_CHARS, logger
from shared.config import Config

# Meta's cap on an image post's alt_text.
THREADS_ALT_TEXT_MAX_CHARS = 1000

THREADS_API_BASE = "https://graph.threads.net/v1.0"
# Meta processes the media container asynchronously; publishing too early fails.
THREADS_MEDIA_PROCESS_WAIT_SEC = 30
# After an image root is published it isn't immediately addressable as a reply target; replies
# to it 400 with "media not found" (code 24) until Meta finishes indexing, which takes minutes for
# image roots. Each reply retries against one shared deadline, so a never-indexing root can't blow
# the 15-min Lambda timeout mid-chain. ~4.5 min: render (~4 min) + this leaves margin.
THREADS_INDEXING_BUDGET_SEC = 270
# Retry cap on a reply's transient container-processing 400/429/5xx (a single such 400 dropped one
# story from a 2026-07-17 digest). A genuine content rejection burns these attempts too, but is
# logged with its body.
THREADS_REPLY_RETRY_ATTEMPTS = 3
THREADS_REPLY_RETRY_BACKOFF_SEC = 10
# Seconds the publish path still needs AFTER the indexing wait: the media-processing sleep the root
# already paid for plus one reply's full retry ladder. Derived from the constants above — never a
# second hand-written number that could drift from them — and used only to shorten the indexing
# budget when the caller supplied a deadline that cannot cover the full 270s.
THREADS_PUBLISH_RESERVE_SEC = THREADS_MEDIA_PROCESS_WAIT_SEC + THREADS_REPLY_RETRY_ATTEMPTS * (
    THREADS_REPLY_RETRY_BACKOFF_SEC
)
# How long the hosted-image presigned URL stays valid — must outlast the
# create-container + media-processing window with margin.
THREADS_IMAGE_URL_TTL_SEC = 900


def _upload_image_for_hosting(image_bytes: bytes, bucket: str, key: str, content_type: str = "image/png") -> str:
    """Threads can only fetch images from a public URL (no byte upload), so host the image on
    S3 and hand back a short-lived presigned URL Meta can cURL once. The ContentType must match
    the real image type or Meta may reject/mis-render it. The client is on the state store's session
    (configured profile + region): a local shell's inherited AWS_PROFILE / AWS_REGION may be another
    account, or sign the URL for a region Meta cannot fetch from."""
    from adapters.state_store import state_boto_session
    from shared.config import get_config

    s3 = state_boto_session(get_config()).client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=image_bytes, ContentType=content_type)
    return s3.generate_presigned_url(
        "get_object", Params={"Bucket": bucket, "Key": key}, ExpiresIn=THREADS_IMAGE_URL_TTL_SEC
    )


async def _create_container(client: httpx.AsyncClient, user_id: str, token: str, **params: Any) -> str:
    params["access_token"] = token
    resp = await client.post(f"{THREADS_API_BASE}/{user_id}/threads", data=params)
    resp.raise_for_status()
    return resp.json()["id"]


async def _publish_container(client: httpx.AsyncClient, user_id: str, token: str, creation_id: str) -> str:
    resp = await client.post(
        f"{THREADS_API_BASE}/{user_id}/threads_publish",
        data={"creation_id": creation_id, "access_token": token},
    )
    resp.raise_for_status()
    return resp.json()["id"]


async def _publish_post(
    client: httpx.AsyncClient,
    user_id: str,
    token: str,
    *,
    text: str = "",
    image_url: str = "",
    image_alt: str = "",
    topic_tag: str = "",
    reply_to_id: str = "",
) -> str:
    params: dict[str, Any] = {"media_type": "IMAGE" if image_url else "TEXT"}
    if text:
        params["text"] = text[:THREADS_MAX_POST_CHARS]
    if image_url:
        params["image_url"] = image_url
        if image_alt:
            params["alt_text"] = image_alt[:THREADS_ALT_TEXT_MAX_CHARS]
    if topic_tag:
        params["topic_tag"] = topic_tag
    if reply_to_id:
        params["reply_to_id"] = reply_to_id
    creation_id = await _create_container(client, user_id, token, **params)
    if image_url:
        await asyncio.sleep(THREADS_MEDIA_PROCESS_WAIT_SEC)
    return await _publish_container(client, user_id, token, creation_id)


def publish_may_be_live(exc: Exception) -> bool:
    """True when a failure came back from the PUBLISH step in a way that can answer a post that went
    live: a 5xx, or a transport failure after the request may have been sent (a timeout, a dropped
    connection). Retrying it creates a new container, so the same text could post twice."""
    try:
        request = exc.request if isinstance(exc, httpx.HTTPStatusError | httpx.RequestError) else None
    except RuntimeError:
        return False
    if request is None or not request.url.path.endswith("/threads_publish"):
        return False
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code >= 500
    return not isinstance(exc, httpx.ConnectError)


def is_media_not_found(exc: httpx.HTTPStatusError) -> bool:
    # A just-published post isn't instantly addressable as a reply target — Meta returns
    # code 24 / subcode 4279009 ("media not found") until indexing completes.
    try:
        err = exc.response.json().get("error", {})
        return err.get("code") == 24 or err.get("error_subcode") == 4279009
    except Exception:
        return False


def is_transient_reply_error(exc: httpx.HTTPStatusError) -> bool:
    """True if a reply failure is worth another attempt. Meta's container-publish endpoint returns
    transient 400s while it finishes processing a freshly-created container (distinct from the
    'media not found' indexing case), plus the usual 429/5xx. A single such 400 silently dropped
    one of five stories on 2026-07-17. Genuine content rejections also surface as 400 and will burn
    the retries, but they are now logged with the response body so the cause is identifiable."""
    if is_media_not_found(exc):
        return True
    status = exc.response.status_code
    return status == 400 or status == 429 or status >= 500


def error_detail(exc: httpx.HTTPStatusError) -> str:
    """The Threads API response body, truncated — the only place Meta states WHY a post was
    rejected (subcode / message). Logged on reply failure so a text rejection is distinguishable
    from a transient processing error without re-running the pipeline."""
    try:
        return exc.response.text[:300]
    except Exception:
        return "<no response body>"


def _indexing_budget_sec(deadline: float | None) -> float:
    """How long this run may wait for the image root to index.

    THREADS_INDEXING_BUDGET_SEC unless the caller handed over a hard deadline that cannot cover it:
    then wait for whatever is left minus the publish reserve, so a root that never indexes can't eat
    the time the reply chain still needs. The budget is NEVER shortened while the remaining time
    allows the full 270s — too little indexing patience is what dropped stories in the first place.
    A None deadline (local runs, research reports) keeps the fixed budget, exactly as before."""
    if deadline is None:
        return float(THREADS_INDEXING_BUDGET_SEC)
    left = deadline - time.monotonic() - THREADS_PUBLISH_RESERVE_SEC
    return max(0.0, min(float(THREADS_INDEXING_BUDGET_SEC), left))


async def _publish_reply_with_retry(
    client: httpx.AsyncClient, user_id: str, token: str, text: str, reply_to_id: str, *, indexing_deadline: float = 0.0
) -> str:
    """Publish one reply. The code-24 'media not found' case is the root still indexing, so it keeps
    retrying until the SHARED indexing deadline (the first reply to land proves the root is ready);
    other transient 400/429/5xx errors cap at THREADS_REPLY_RETRY_ATTEMPTS, and a non-transient error
    (e.g. auth) raises immediately. The response body is logged on each retry."""
    attempt = 0
    while True:
        attempt += 1
        try:
            return await _publish_post(client, user_id, token, text=text, reply_to_id=reply_to_id)
        except httpx.HTTPStatusError as e:
            # code-24 is pure root-indexing lag: ride the shared deadline (once one reply lands the
            # root is ready, so this only bites the first reply). Other transient errors keep the cap.
            budget_left = indexing_deadline - time.monotonic()
            keep_for_indexing = is_media_not_found(e) and budget_left > THREADS_REPLY_RETRY_BACKOFF_SEC
            # A 5xx from the publish step may come back for a reply that went live, and a retry
            # creates a new container: the same story would post twice. A 4xx there is retried as it
            # long was (all 80 threads checked on 2026-10-27 carry five replies, none twice).
            maybe_live = publish_may_be_live(e)
            # Past a digest's indexing deadline a code 24 is the root that never indexed: every further
            # retry only pushes the chain past the Lambda's time.
            indexing_over = indexing_deadline > 0 and is_media_not_found(e) and not keep_for_indexing
            capped = (
                maybe_live
                or indexing_over
                or not is_transient_reply_error(e)
                or attempt >= THREADS_REPLY_RETRY_ATTEMPTS
            )
            if maybe_live or (capped and not keep_for_indexing):
                raise
            logger.info(
                "Reply attempt %d failed (%s), retrying: %s",
                attempt,
                e.response.status_code,
                error_detail(e),
            )
            await asyncio.sleep(THREADS_REPLY_RETRY_BACKOFF_SEC)


THREADS_CONVERSATION_FIELDS = "id,text,username,timestamp,replied_to,is_reply_owned_by_me,hide_status"


def threads_client(token: str, request_timeout: int) -> httpx.AsyncClient:
    """A client for the Graph API READS. The token rides in the Authorization header, never in the
    query string, because httpx puts the request URL into its error text and the logs carry that."""
    return httpx.AsyncClient(timeout=request_timeout, headers={"Authorization": f"Bearer {token}"})


async def _get_all(client: httpx.AsyncClient, url: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    next_url: str | None = url
    while next_url:
        resp = await client.get(next_url, params=params)
        resp.raise_for_status()
        body = resp.json()
        data.extend(body.get("data") or [])
        # The `next` link already carries the query; under header auth it holds no token.
        next_url = (body.get("paging") or {}).get("next")
        params = {}
    return data


async def fetch_own_posts(client: httpx.AsyncClient, since: datetime) -> list[dict[str, Any]]:
    """The account's own top-level posts published at or after `since`."""
    posts = await _get_all(
        client,
        f"{THREADS_API_BASE}/me/threads",
        {"fields": "id,text,timestamp,is_reply", "since": int(since.timestamp()), "limit": 100},
    )
    return [p for p in posts if not p.get("is_reply")]


async def fetch_conversation(client: httpx.AsyncClient, media_id: str) -> list[dict[str, Any]]:
    """Every reply under a top-level post, at any depth, the account's own included."""
    return await _get_all(
        client,
        f"{THREADS_API_BASE}/{media_id}/conversation",
        {"fields": THREADS_CONVERSATION_FIELDS, "reverse": "false", "limit": 100},
    )


async def _landed_detached(client: httpx.AsyncClient, token: str, media_id: str) -> bool:
    """True only when Threads says the post is NOT a reply. Any failure to ask answers False: the check
    reports a detached reply, it must never be able to fail a reply that did land. The token rides in a
    header, never the URL (httpx puts the URL into its error text)."""
    try:
        resp = await client.get(
            f"{THREADS_API_BASE}/{media_id}",
            params={"fields": "is_reply"},
            headers={"Authorization": f"Bearer {token}"},
        )
        return resp.status_code == 200 and resp.json().get("is_reply") is False
    except Exception:
        logger.debug("Could not read back reply '%s'", media_id, exc_info=True)
        return False


async def _publish_root(
    client: httpx.AsyncClient,
    user_id: str,
    token: str,
    *,
    text: str,
    topic_tag: str,
    image_url: str,
    image_alt: str,
    deadline: float | None = None,
) -> tuple[str, str]:
    """(root id, the image url it carries). Only a root known to have posted nothing is redone: a 5xx
    from the publish step may come back for a post that went live, and a second root is a duplicate.
    A rate limit, code 24 (the container still processing) or a 5xx at container creation is retried
    up to THREADS_REPLY_RETRY_ATTEMPTS times; without the retry one such answer cost the day's comic,
    or the whole day when the root was text. Any other rejection of the IMAGE root (unfetchable URL,
    unsupported media) posts the text without it, since the text stands on its own. A retry is only
    started while `deadline` still leaves the reply chain its THREADS_PUBLISH_RESERVE_SEC."""
    attempt = 0
    while True:
        attempt += 1
        try:
            root_id = await _publish_post(
                client, user_id, token, text=text, topic_tag=topic_tag, image_url=image_url, image_alt=image_alt
            )
            return root_id, image_url
        except httpx.HTTPStatusError as e:
            if _root_may_be_live(e):
                raise
            status = e.response.status_code
            retryable = is_media_not_found(e) or status == 429 or status >= 500
            # Room for one more attempt too, or the retry itself eats the reply chain's reserve.
            attempt_cost = THREADS_REPLY_RETRY_BACKOFF_SEC + (THREADS_MEDIA_PROCESS_WAIT_SEC if image_url else 0)
            time_left = deadline is None or deadline - time.monotonic() > THREADS_PUBLISH_RESERVE_SEC + attempt_cost
            if retryable and attempt < THREADS_REPLY_RETRY_ATTEMPTS and time_left:
                logger.info("Threads root attempt %d failed (%s), retrying: %s", attempt, status, error_detail(e))
                await asyncio.sleep(THREADS_REPLY_RETRY_BACKOFF_SEC)
                continue
            if not image_url:
                raise
            logger.warning("Threads image root rejected (%s: %s); posting the root text-only", status, error_detail(e))
            image_url, image_alt, attempt = "", "", 0


async def post_to_threads(
    *,
    root_text: str,
    replies: list[str] | None = None,
    image_bytes: bytes | None = None,
    image_bucket: str = "",
    image_key: str = "",
    image_content_type: str = "image/png",
    image_alt: str = "",
    topic_tag: str = "",
    request_timeout: int = 60,
    deadline: float | None = None,
) -> ThreadsDelivery:
    """Post a digest to Threads as a root post (image + lead) followed by a reply chain — one
    pre-rendered reply per story; `_publish_post` hard-caps every text at the 500-char limit.
    Best-effort: missing credentials or any API failure is logged and skipped, never raising to the
    caller.

    `deadline` is an optional monotonic timestamp bounding this call (the visual Lambda's remaining
    time); it only ever SHORTENS the root-indexing wait and stops a root retry that would eat the reply
    chain's reserve; None behaves exactly as before.

    Returns a ThreadsDelivery (posted, expected) count — NOT a bool, so a partial reply chain is
    distinguishable from a complete post. Branch on `.published`, never on the value itself."""
    expected = 1 + len([r for r in (replies or []) if r.strip()])
    token = resolve_secret("THREADS_ACCESS_TOKEN", "threads-access-token")
    user_id = resolve_secret("THREADS_USER_ID", "threads-user-id")
    if not token or not user_id:
        # ERROR, not INFO: with enable_threads_post on, empty credentials mean the day's digest is
        # NOT delivered anywhere. That read as a routine "skipping" line in the logs.
        logger.error("Threads access token / user id not configured — digest NOT delivered to Threads")
        return ThreadsDelivery(0, expected)

    image_url = ""
    if image_bytes and image_bucket and image_key:
        try:
            image_url = await asyncio.to_thread(
                _upload_image_for_hosting, image_bytes, image_bucket, image_key, image_content_type
            )
        except Exception as e:
            logger.warning("Failed to host Threads image on S3, posting text-only: %s", e)

    # Renderer already fits each item into one <=500-char post at a sentence boundary; keep the
    # one-item-one-reply mapping and only hard-cap as a last-resort safety net (no re-splitting).
    posts: list[str] = [r[:THREADS_MAX_POST_CHARS] for r in (replies or []) if r.strip()]

    try:
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            root_id, image_url = await _publish_root(
                client,
                user_id,
                token,
                text=root_text[:THREADS_MAX_POST_CHARS],
                topic_tag=topic_tag,
                image_url=image_url,
                image_alt=image_alt,
                deadline=deadline,
            )
            logger.info("Posted Threads root '%s'", root_id)
            # An image root needs time to become addressable as a reply target, so every reply
            # carries this deadline and retries its own create-container against it (code 24 names the
            # reply's own container, so polling the root cannot gate this delay).
            indexing_deadline = time.monotonic() + _indexing_budget_sec(deadline)
            # All replies hang off the ROOT (a flat thread), not off each other — otherwise
            # they nest as reply-of-reply and only the first shows under the root. Each reply is
            # best-effort: a single failure must not abandon the rest, so the digest never posts a
            # half-finished comment chain ("댓글이 달리다 말았다").
            posted = 0
            detached: list[str] = []
            for i, post in enumerate(posts, start=1):
                try:
                    reply_id = await _publish_reply_with_retry(
                        client, user_id, token, post, root_id, indexing_deadline=indexing_deadline
                    )
                    if await _landed_detached(client, token, reply_id):
                        # 2026-10-27: a reply retried after code 24 published as a standalone post although
                        # Meta answered success, and the day logged "5/5 replies" with no alert.
                        logger.error(
                            "Threads reply %d/%d published as a STANDALONE post '%s', not under the root",
                            i,
                            len(posts),
                            reply_id,
                        )
                        detached.append(reply_id)
                        continue
                    posted += 1
                    logger.debug("Posted Threads reply %d/%d", i, len(posts))
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "Threads reply %d/%d failed (%s), continuing: %s",
                        i,
                        len(posts),
                        e.response.status_code,
                        error_detail(e),
                    )
                except Exception as e:
                    logger.warning("Threads reply %d/%d failed, continuing: %s", i, len(posts), e)
        outcome = ThreadsDelivery(
            1 + posted, expected, with_image=bool(image_url), root_id=root_id, detached=tuple(detached)
        )
        # Say what actually happened. "Successfully posted digest ... (0/0 reply posts)" was logged
        # on the two days the digest shipped with no stories at all, so the logs read green while
        # the post was broken. A reply-less root is legitimate for a single research report, but it
        # is never a digest, so don't call it one.
        if not posts:
            logger.info("Posted Threads root '%s' with no replies", root_id)
        elif not outcome.published:
            # Every reply failed (e.g. the image root never indexed within the budget): a lone
            # image with no stories. The root is live, so the caller keeps the date marked.
            logger.error("Threads root '%s' posted but ALL %d reply posts failed", root_id, len(posts))
        elif outcome.partial:
            logger.error("Threads digest posted PARTIALLY: %s (reply chain is incomplete)", outcome.summary())
        else:
            logger.info("Posted Threads root '%s' with %d/%d replies", root_id, posted, len(posts))
        return outcome
    except httpx.HTTPStatusError as e:
        logger.warning("Threads API error: %s — %s", e.response.status_code, e.response.text[:300])
        return ThreadsDelivery(0, expected, uncertain=_root_may_be_live(e))
    except Exception as e:
        logger.warning("Unexpected error posting to Threads: %s", e)
        return ThreadsDelivery(0, expected, uncertain=_root_may_be_live(e))


def _root_may_be_live(exc: Exception) -> bool:
    """False when the root is known to have posted nothing: it failed at CONTAINER creation, or the
    publish step was rate-limited (429) or answered code 24, which names the container being published
    as not found. Replies have retried code 24 77+ times, and all 80 digest threads checked on
    2026-10-27 carry exactly five story replies with no duplicate. The root is held to that evidence and
    no further, deliberately stricter than the replies' long-standing 4xx retry: a wrong guess here is a
    second public root. Any other publish-step failure, or one whose request is unknown, may be live."""
    try:
        request = exc.request if isinstance(exc, httpx.HTTPStatusError | httpx.RequestError) else None
    except RuntimeError:
        request = None
    if (
        isinstance(exc, httpx.HTTPStatusError)
        and exc.response.status_code < 500
        and (is_media_not_found(exc) or exc.response.status_code == 429)
    ):
        return False
    return request is None or request.url.path.endswith("/threads_publish")


# The comment-reply reads and replies; a digest post set takes longer per call (an image root).
COMMENT_REQUEST_TIMEOUT_SEC = 30


# Meta's moderation states a reader still sees; a HIDDEN, COVERED or BLOCKED comment is left alone.
VISIBLE_HIDE_STATUSES = frozenset({"", "NOT_HUSHED", "UNHUSHED"})


def to_comment(raw: dict[str, Any], *, own: bool | None = None) -> ThreadsComment:
    """One Graph API post or reply as the comment model. `own` overrides is_reply_owned_by_me (the
    account's top-level posts do not carry it)."""
    try:
        created_at = datetime.strptime(str(raw.get("timestamp", "")), "%Y-%m-%dT%H:%M:%S%z")
    except ValueError:
        created_at = None
    return ThreadsComment(
        id=str(raw["id"]),
        text=raw.get("text") or "",
        username=raw.get("username") or "",
        created_at=created_at,
        parent_id=str((raw.get("replied_to") or {}).get("id", "")),
        own=bool(raw.get("is_reply_owned_by_me")) if own is None else own,
        visible=raw.get("hide_status", "") in VISIBLE_HIDE_STATUSES,
    )


class _CommentSession:
    def __init__(self, client: httpx.AsyncClient, user_id: str, token: str) -> None:
        self._client = client
        self._user_id = user_id
        self._token = token

    async def fetch_own_posts(self, since: datetime) -> list[ThreadsComment]:
        return [to_comment(post, own=True) for post in await fetch_own_posts(self._client, since)]

    async def fetch_conversation(self, media_id: str) -> list[ThreadsComment]:
        return [to_comment(reply) for reply in await fetch_conversation(self._client, media_id)]

    async def publish_reply(self, text: str, reply_to_id: str) -> str:
        """Publish one text reply to `reply_to_id`, with the reply chain's retry on a transient error."""
        try:
            return await _publish_reply_with_retry(self._client, self._user_id, self._token, text, reply_to_id)
        except httpx.HTTPStatusError as e:
            raise PublishError(
                e.response.status_code,
                error_detail(e),
                may_be_live=publish_may_be_live(e),
                transient=is_transient_reply_error(e),
            ) from e
        except httpx.RequestError as e:
            if not publish_may_be_live(e):
                raise
            raise PublishError(0, f"publish failed ({type(e).__name__})", may_be_live=True, transient=False) from e


class ThreadsGateway:
    """The account on Threads: posts a thread, hosting its image on the state bucket, and opens comment
    sessions. Credentials are read per call, so a token the refresh Lambda rotated is picked up."""

    publish_budget_sec = float(THREADS_INDEXING_BUDGET_SEC + THREADS_PUBLISH_RESERVE_SEC)

    def __init__(self, config: Config) -> None:
        self.config = config

    async def post_thread(
        self,
        root_text: str,
        replies: list[str],
        *,
        image_bytes: bytes | None = None,
        image_content_type: str = "image/png",
        image_alt: str = "",
        deadline: float | None = None,
    ) -> ThreadsDelivery:
        bucket = key = ""
        if image_bytes:
            bucket = self.config.aws.state_bucket_name or os.environ.get("STATE_BUCKET", "")
            if bucket:
                prefix = self.config.aws.s3_prefix.rstrip("/") + "/" if self.config.aws.s3_prefix else ""
                digest = hashlib.sha256(image_bytes).hexdigest()[:16]
                key = f"{prefix}threads/{digest}.{extension_for(image_content_type)}"
            else:
                logger.warning("No state bucket configured; the Threads image is skipped (text-only post)")
        return await post_to_threads(
            root_text=root_text,
            replies=replies,
            image_bytes=image_bytes,
            image_bucket=bucket,
            image_key=key,
            image_content_type=image_content_type,
            image_alt=image_alt,
            topic_tag=self.config.pipeline.threads_topic_tag,
            deadline=deadline,
        )

    @asynccontextmanager
    async def comments(self) -> AsyncIterator[_CommentSession]:
        # Strict: an SSM read failure must fail the run, not read as "nothing to answer".
        token = resolve_secret("THREADS_ACCESS_TOKEN", "threads-access-token", strict=True)
        user_id = resolve_secret("THREADS_USER_ID", "threads-user-id", strict=True)
        if not token or not user_id:
            raise RuntimeError("Threads access token / user id not configured; cannot answer comments")
        async with threads_client(token, COMMENT_REQUEST_TIMEOUT_SEC) as client:
            yield _CommentSession(client, user_id, token)


class ThreadsPreview:
    """A ThreadsPublisher that posts NOTHING: it keeps the thread it was handed, so a research run can
    be read (and compared) exactly as it would have gone out."""

    publish_budget_sec = 0.0

    def __init__(self) -> None:
        self.posts: list[str] = []
        self.with_image = False

    async def post_thread(
        self,
        root_text: str,
        replies: list[str],
        *,
        image_bytes: bytes | None = None,
        image_content_type: str = "image/png",
        image_alt: str = "",
        deadline: float | None = None,
    ) -> ThreadsDelivery:
        self.posts = [root_text, *(r for r in replies if r.strip())]
        self.with_image = bool(image_bytes)
        return ThreadsDelivery(len(self.posts), len(self.posts), with_image=self.with_image)

    def comments(self) -> AbstractAsyncContextManager[_CommentSession]:
        raise NotImplementedError("a preview has no comments")
