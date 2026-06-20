from __future__ import annotations

from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup

from shared import Config, ImageAsset, logger

_BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
# og:image / twitter:image, in preference order.
_META_IMAGE_KEYS = (
    ("property", "og:image"),
    ("property", "og:image:url"),
    ("name", "twitter:image"),
    ("name", "twitter:image:src"),
)
# Raster types Slack and Threads render reliably. SVG and other vector/exotic types are excluded —
# Slack file previews and Meta's Threads fetcher don't handle them dependably.
_RENDERABLE_IMAGE_TYPES = frozenset({"image/jpeg", "image/png", "image/webp", "image/gif"})
# Map content-type → file extension for the Slack filename and the Threads S3 key.
_CONTENT_TYPE_EXT = {
    "image/jpeg": "jpg",
    "image/png": "png",
    "image/webp": "webp",
    "image/gif": "gif",
}


def extension_for(content_type: str) -> str:
    """File extension for an image MIME type, defaulting to 'png'."""
    return _CONTENT_TYPE_EXT.get(content_type.split(";")[0].strip().lower(), "png")


def _meta_content(tag) -> str:
    """A meta tag's `content`. BeautifulSoup may return a list for multi-valued attributes,
    so coerce to a single string."""
    if not tag:
        return ""
    content = tag.get("content") or ""
    if isinstance(content, list):
        content = content[0] if content else ""
    return content.strip()


def _extract_image_url(html: str, page_url: str) -> tuple[str, str]:
    """Return (image_url, alt) from a page's OpenGraph/Twitter card meta tags. Relative
    image URLs are resolved against the page URL. Returns ("", "") when none is present."""
    soup = BeautifulSoup(html, "html.parser")
    for attr, value in _META_IMAGE_KEYS:
        content = _meta_content(soup.find("meta", attrs={attr: value}))
        if content:
            alt_tag = soup.find("meta", attrs={"property": "og:title"}) or soup.find(
                "meta", attrs={"name": "twitter:title"}
            )
            return urljoin(page_url, content), _meta_content(alt_tag)
    return "", ""


async def fetch_og_image(url: str, *, timeout: int | None = None, max_bytes: int | None = None) -> ImageAsset | None:
    """Fetch a page's representative image (og:image / twitter:image) and download its bytes.
    Best-effort: any network error, missing tag, non-image content-type, or oversize image
    returns None. Never raises to the caller."""
    cfg = Config.load().agent
    timeout = timeout if timeout is not None else cfg.og_image_timeout_sec
    max_bytes = max_bytes if max_bytes is not None else cfg.og_image_max_bytes

    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=True, headers=_BROWSER_HEADERS) as client:
            page = await client.get(url)
            page.raise_for_status()
            image_url, alt = _extract_image_url(page.text, str(page.url))
            if not image_url:
                logger.info("No og:image found for '%s'", url)
                return None

            # Stream the image so an oversize body is aborted mid-download instead of being
            # fully buffered into memory (a non-streaming get() reads the whole body first).
            async with client.stream("GET", image_url) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
                if content_type not in _RENDERABLE_IMAGE_TYPES:
                    logger.info("og:image for '%s' is not a renderable image (%s)", url, content_type or "unknown")
                    return None
                declared = resp.headers.get("content-length")
                if declared and declared.isdigit() and int(declared) > max_bytes:
                    logger.info("og:image for '%s' too large (Content-Length %s)", url, declared)
                    return None
                chunks: list[bytes] = []
                total = 0
                async for chunk in resp.aiter_bytes():
                    total += len(chunk)
                    if total > max_bytes:
                        logger.info("og:image for '%s' exceeded %d bytes mid-stream, aborting", url, max_bytes)
                        return None
                    chunks.append(chunk)

            data = b"".join(chunks)
            if not data:
                logger.info("og:image for '%s' was empty", url)
                return None

            return ImageAsset(
                data=data,
                source_url=url,
                image_url=image_url,
                content_type=content_type,
                alt=alt,
            )
    except Exception as e:
        logger.info("Failed to fetch og:image for '%s': %s", url, e)
        return None
