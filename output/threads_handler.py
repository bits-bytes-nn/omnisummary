from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, NamedTuple

import boto3
import httpx

from shared import THREADS_MAX_POST_CHARS, logger, resolve_secret

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


class ThreadsDelivery(NamedTuple):
    """How much of the intended post set actually landed. `posted`/`expected` count the ROOT plus
    one per reply, so a 5-story digest is 6 expected posts.

    Deliberately NOT a bool: "the root went up" and "the digest went up" are different outcomes,
    and a partial reply chain (4 of 6 posts on 2026-07-17) used to be reported as plain success.
    CALLERS: a NamedTuple is ALWAYS truthy, so never branch on the value itself — use `published`.
    """

    posted: int
    expected: int
    # Whether the root that went up actually carried the day's image. Defaults to False so every
    # existing construction (a failure verdict, a research report) is unchanged; the visual Lambda
    # emits it as a metric, because "posted 6/6" says nothing about a text-only fallback that
    # silently dropped the visual.
    with_image: bool = False

    @property
    def published(self) -> bool:
        """True when what landed is usable: the root, plus at least one reply when replies were
        intended. An all-replies-failed root is a lone image with no stories, never a digest."""
        return self.posted > 1 if self.expected > 1 else self.posted > 0

    @property
    def partial(self) -> bool:
        """Published, but some intended posts are missing — the reader sees a truncated chain."""
        return self.published and self.posted < self.expected

    def summary(self) -> str:
        return f"{self.posted}/{self.expected} posts"


def _upload_image_for_hosting(image_bytes: bytes, bucket: str, key: str, content_type: str = "image/png") -> str:
    """Threads can only fetch images from a public URL (no byte upload), so host the image on
    S3 and hand back a short-lived presigned URL Meta can cURL once. The ContentType must match
    the real image type or Meta may reject/mis-render it."""
    s3 = boto3.client("s3")
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
            capped = not is_transient_reply_error(e) or attempt >= THREADS_REPLY_RETRY_ATTEMPTS
            if capped and not keep_for_indexing:
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


async def publish_reply(client: httpx.AsyncClient, user_id: str, token: str, text: str, reply_to_id: str) -> str:
    """Publish one text reply to `reply_to_id`, with the reply chain's retry on a transient error."""
    return await _publish_reply_with_retry(client, user_id, token, text[:THREADS_MAX_POST_CHARS], reply_to_id)


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
    pre-rendered reply per story. Each reply is re-split here as a safety net so nothing exceeds
    the 500-char cap. Best-effort: missing credentials or any API failure is logged and skipped,
    never raising to the caller.

    `deadline` is an optional monotonic timestamp bounding this call (the visual Lambda's remaining
    time); it only ever SHORTENS the root-indexing wait, and None behaves exactly as before.

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
            root_id = await _publish_post(
                client,
                user_id,
                token,
                text=root_text[:THREADS_MAX_POST_CHARS],
                image_url=image_url,
                image_alt=image_alt,
                topic_tag=topic_tag,
            )
            logger.info("Posted Threads root '%s'", root_id)
            # An image root needs time to become addressable as a reply target, so every reply
            # carries this deadline and retries its own create-container against it. A root-readiness
            # GET poll used to run here first; over 30 production runs it never once reported the
            # root unready (GET 200 on the first probe every time) while the reply retries fired 77
            # times, because decoding error_user_msg on all 82 code-24 failures showed the media id
            # Meta names is the REPLY's own container, not the root — so a root probe structurally
            # cannot gate the delay it was written for.
            indexing_deadline = time.monotonic() + _indexing_budget_sec(deadline)
            # All replies hang off the ROOT (a flat thread), not off each other — otherwise
            # they nest as reply-of-reply and only the first shows under the root. Each reply is
            # best-effort: a single failure must not abandon the rest, so the digest never posts a
            # half-finished comment chain ("댓글이 달리다 말았다").
            posted = 0
            for i, post in enumerate(posts, start=1):
                try:
                    await _publish_reply_with_retry(
                        client, user_id, token, post, root_id, indexing_deadline=indexing_deadline
                    )
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
        outcome = ThreadsDelivery(1 + posted, expected, with_image=bool(image_url))
        # Say what actually happened. "Successfully posted digest ... (0/0 reply posts)" was logged
        # on the two days the digest shipped with no stories at all, so the logs read green while
        # the post was broken. A reply-less root is legitimate for a single research report, but it
        # is never a digest, so don't call it one.
        if not posts:
            logger.info("Posted Threads root '%s' with no replies", root_id)
        elif not outcome.published:
            # Every reply failed (e.g. the image root never indexed within the budget): a lone
            # image with no stories. The caller's ledger rollback keeps the day retryable.
            logger.error("Threads root '%s' posted but ALL %d reply posts failed", root_id, len(posts))
        elif outcome.partial:
            logger.error("Threads digest posted PARTIALLY: %s (reply chain is incomplete)", outcome.summary())
        else:
            logger.info("Posted Threads root '%s' with %d/%d replies", root_id, posted, len(posts))
        return outcome
    except httpx.HTTPStatusError as e:
        logger.warning("Threads API error: %s — %s", e.response.status_code, e.response.text[:300])
        return ThreadsDelivery(0, expected)
    except Exception as e:
        logger.warning("Unexpected error posting to Threads: %s", e)
        return ThreadsDelivery(0, expected)
