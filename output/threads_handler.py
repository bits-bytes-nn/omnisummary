from __future__ import annotations

import asyncio
import time
from typing import Any

import boto3
import httpx

from shared import logger, resolve_secret

THREADS_API_BASE = "https://graph.threads.net/v1.0"
THREADS_MAX_TEXT_LENGTH = 500
# Meta processes the media container asynchronously; publishing too early fails.
THREADS_MEDIA_PROCESS_WAIT_SEC = 30
# After an image root is published it isn't immediately addressable as a reply target;
# replies to it can 400 with "media not found" until Meta finishes indexing (observed to take
# well over 3 minutes for image roots). Wait once before the first reply, then retry the link
# with backoff over a generous window before giving up.
#
# "media not found" means the ROOT isn't indexed yet — a condition ALL replies share, not a
# per-reply problem. So the indexing wait is bounded by a single SHARED deadline across the whole
# chain (THREADS_INDEXING_BUDGET_SEC), not by a fresh per-reply budget. Otherwise, if the root
# never indexes, each of N replies would burn attempts×backoff independently (e.g. 5×270s ≈ 22min)
# and blow the 15-min Lambda timeout mid-chain — the exact half-finished chain this guards against.
THREADS_REPLY_INITIAL_WAIT_SEC = 30
THREADS_REPLY_RETRY_ATTEMPTS = 18
THREADS_REPLY_RETRY_BACKOFF_SEC = 15
# Shared ceiling for total time spent waiting on root indexing across ALL replies (~4.5 min).
# render (~4 min) + this leaves margin under the 15-min Lambda timeout.
THREADS_INDEXING_BUDGET_SEC = 270
# How long the hosted-image presigned URL stays valid — must outlast the
# create-container + media-processing window with margin.
THREADS_IMAGE_URL_TTL_SEC = 900


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
    reply_to_id: str = "",
) -> str:
    params: dict[str, Any] = {"media_type": "IMAGE" if image_url else "TEXT"}
    if text:
        params["text"] = text[:THREADS_MAX_TEXT_LENGTH]
    if image_url:
        params["image_url"] = image_url
    if reply_to_id:
        params["reply_to_id"] = reply_to_id
    creation_id = await _create_container(client, user_id, token, **params)
    if image_url:
        await asyncio.sleep(THREADS_MEDIA_PROCESS_WAIT_SEC)
    return await _publish_container(client, user_id, token, creation_id)


def _is_media_not_found(exc: httpx.HTTPStatusError) -> bool:
    # A just-published post isn't instantly addressable as a reply target — Meta returns
    # code 24 / subcode 4279009 ("media not found") until indexing completes.
    try:
        err = exc.response.json().get("error", {})
        return err.get("code") == 24 or err.get("error_subcode") == 4279009
    except Exception:
        return False


async def _publish_reply_with_retry(
    client: httpx.AsyncClient, user_id: str, token: str, text: str, reply_to_id: str, indexing_deadline: float
) -> str:
    """Publish one reply, retrying only the "root not indexed yet" 400. Indexing retries stop at
    the SHARED indexing_deadline (monotonic seconds) so the whole chain can't exceed the budget —
    a per-attempt cap still applies. Non-indexing errors raise immediately."""
    last: httpx.HTTPStatusError | None = None
    for attempt in range(1, THREADS_REPLY_RETRY_ATTEMPTS + 1):
        try:
            return await _publish_post(client, user_id, token, text=text, reply_to_id=reply_to_id)
        except httpx.HTTPStatusError as e:
            budget_left = indexing_deadline - time.monotonic()
            if not _is_media_not_found(e) or attempt == THREADS_REPLY_RETRY_ATTEMPTS or budget_left <= 0:
                raise
            last = e
            logger.info(
                "Reply target not indexed yet (attempt %d, ~%ds indexing budget left), retrying",
                attempt,
                int(budget_left),
            )
            await asyncio.sleep(min(THREADS_REPLY_RETRY_BACKOFF_SEC, budget_left))
    assert last is not None
    raise last


async def post_to_threads(
    *,
    root_text: str,
    replies: list[str] | None = None,
    image_bytes: bytes | None = None,
    image_bucket: str = "",
    image_key: str = "",
    image_content_type: str = "image/png",
    request_timeout: int = 60,
) -> bool:
    """Post a digest to Threads as a root post (image + lead) followed by a reply chain — one
    pre-rendered reply per story. Each reply is re-split here as a safety net so nothing exceeds
    the 500-char cap. Best-effort: missing credentials or any API failure is logged and skipped,
    never raising to the caller."""
    token = resolve_secret("THREADS_ACCESS_TOKEN", "threads-access-token")
    user_id = resolve_secret("THREADS_USER_ID", "threads-user-id")
    if not token or not user_id:
        logger.info("Threads access token / user id not configured. Skipping Threads delivery.")
        return False

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
    posts: list[str] = [r[:THREADS_MAX_TEXT_LENGTH] for r in (replies or []) if r.strip()]

    try:
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            root_id = await _publish_post(
                client, user_id, token, text=root_text[:THREADS_MAX_TEXT_LENGTH], image_url=image_url
            )
            logger.info("Posted Threads root '%s'", root_id)
            # An image root needs time to become addressable as a reply target; wait once
            # up front so the first reply usually lands without burning retry attempts.
            if image_url and posts:
                await asyncio.sleep(THREADS_REPLY_INITIAL_WAIT_SEC)
            # Shared indexing deadline across ALL replies — the wait is for the root to index,
            # a condition every reply shares, so it must not reset per reply.
            indexing_deadline = time.monotonic() + THREADS_INDEXING_BUDGET_SEC
            # All replies hang off the ROOT (a flat thread), not off each other — otherwise
            # they nest as reply-of-reply and only the first shows under the root. Each reply is
            # best-effort: a single failure (or exhausted indexing retries) must not abandon the
            # rest, so the digest never posts a half-finished comment chain ("댓글이 달리다 말았다").
            posted = 0
            for i, post in enumerate(posts, start=1):
                try:
                    await _publish_reply_with_retry(client, user_id, token, post, root_id, indexing_deadline)
                    posted += 1
                    logger.debug("Posted Threads reply %d/%d", i, len(posts))
                except Exception as e:
                    logger.warning("Threads reply %d/%d failed, continuing: %s", i, len(posts), e)
        logger.info("Successfully posted digest to Threads (%d/%d reply posts)", posted, len(posts))
        return True
    except httpx.HTTPStatusError as e:
        logger.warning("Threads API error: %s — %s", e.response.status_code, e.response.text[:300])
        return False
    except Exception as e:
        logger.warning("Unexpected error posting to Threads: %s", e)
        return False
