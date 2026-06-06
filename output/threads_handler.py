from __future__ import annotations

import asyncio
from typing import Any

import boto3
import httpx

from shared import logger, resolve_secret

THREADS_API_BASE = "https://graph.threads.net/v1.0"
THREADS_MAX_TEXT_LENGTH = 500
# Meta processes the media container asynchronously; publishing too early fails.
THREADS_MEDIA_PROCESS_WAIT_SEC = 30
# How long the hosted-image presigned URL stays valid — must outlast the
# create-container + media-processing window with margin.
THREADS_IMAGE_URL_TTL_SEC = 900


def _split_text(text: str, max_len: int = THREADS_MAX_TEXT_LENGTH) -> list[str]:
    if len(text) <= max_len:
        return [text] if text else []
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        if len(paragraph) > max_len:
            if current:
                chunks.append(current.strip())
                current = ""
            for line in paragraph.split("\n"):
                if len(line) > max_len:
                    for i in range(0, len(line), max_len):
                        chunks.append(line[i : i + max_len])
                elif len(current) + len(line) + 1 > max_len:
                    if current:
                        chunks.append(current.strip())
                    current = line
                else:
                    current = f"{current}\n{line}" if current else line
        elif len(current) + len(paragraph) + 2 > max_len:
            if current:
                chunks.append(current.strip())
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current.strip())
    return [c for c in chunks if c]


def _upload_image_for_hosting(image_bytes: bytes, bucket: str, key: str) -> str:
    """Threads can only fetch images from a public URL (no byte upload), so host the
    PNG on S3 and hand back a short-lived presigned URL Meta can cURL once."""
    s3 = boto3.client("s3")
    s3.put_object(Bucket=bucket, Key=key, Body=image_bytes, ContentType="image/png")
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


async def post_to_threads(
    *,
    root_text: str,
    replies: list[str] | None = None,
    image_bytes: bytes | None = None,
    image_bucket: str = "",
    image_key: str = "",
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
            image_url = await asyncio.to_thread(_upload_image_for_hosting, image_bytes, image_bucket, image_key)
        except Exception as e:
            logger.warning("Failed to host Threads image on S3, posting text-only: %s", e)

    posts: list[str] = []
    for reply in replies or []:
        posts.extend(_split_text(reply))

    try:
        async with httpx.AsyncClient(timeout=request_timeout) as client:
            reply_to = await _publish_post(
                client, user_id, token, text=root_text[:THREADS_MAX_TEXT_LENGTH], image_url=image_url
            )
            logger.info("Posted Threads root '%s'", reply_to)
            for i, post in enumerate(posts, start=1):
                reply_to = await _publish_post(client, user_id, token, text=post, reply_to_id=reply_to)
                logger.debug("Posted Threads reply %d/%d", i, len(posts))
        logger.info("Successfully posted digest to Threads (%d reply posts)", len(posts))
        return True
    except httpx.HTTPStatusError as e:
        logger.warning("Threads API error: %s — %s", e.response.status_code, e.response.text[:300])
        return False
    except Exception as e:
        logger.warning("Unexpected error posting to Threads: %s", e)
        return False
