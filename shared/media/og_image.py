from __future__ import annotations

import asyncio
import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx
from bs4 import BeautifulSoup

from shared import ImageAsset, get_config, logger

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


# These URLs come from a page the AGENT chose (and from that page's own og:image tag), so they are
# attacker-influenced input to a server-side fetch. Only plain http(s) to a public address is
# followed, and redirects are walked manually so EVERY hop is checked — a permitted first hop that
# 302s to http://169.254.169.254/ would otherwise reach the instance metadata service.
_ALLOWED_SCHEMES = frozenset({"http", "https"})
_MAX_REDIRECT_HOPS = 5


def _resolve_addresses(host: str) -> list[str]:
    """Every address `host` resolves to. Split out so tests can stub resolution instead of
    depending on real DNS."""
    return [str(info[4][0]) for info in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)]


def _is_public_address(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return not (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local  # incl. 169.254.169.254, the instance metadata service
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


async def _is_fetchable(url: str) -> bool:
    """True when `url` is a plain http(s) URL whose host resolves ONLY to public addresses.
    Rejection is logged and returns False — the caller then yields None, never raising.

    Resolution happens off the event loop. This is a pre-flight check, so a host that re-resolves
    between the check and the connect (DNS rebinding) is out of scope; the point is to stop the
    ordinary cases — localhost, RFC1918, and link-local metadata endpoints."""
    parts = urlsplit(url)
    if parts.scheme.lower() not in _ALLOWED_SCHEMES:
        logger.info("Refusing to fetch '%s': unsupported scheme", url)
        return False
    host = parts.hostname
    if not host:
        logger.info("Refusing to fetch '%s': no host", url)
        return False
    try:
        addresses = await asyncio.to_thread(_resolve_addresses, host)
    except OSError as e:
        logger.info("Refusing to fetch '%s': host does not resolve (%s)", url, e)
        return False
    if not addresses or not all(_is_public_address(a) for a in addresses):
        logger.info("Refusing to fetch '%s': host resolves to a non-public address", url)
        return False
    return True


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
    cfg = get_config().agent
    timeout = timeout if timeout is not None else cfg.og_image_timeout_sec
    max_bytes = max_bytes if max_bytes is not None else cfg.og_image_max_bytes

    try:
        # Redirects are followed MANUALLY (bounded by _MAX_REDIRECT_HOPS) so every hop passes
        # _is_fetchable — httpx's own follow_redirects would chase a public URL into a private one.
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, headers=_BROWSER_HEADERS) as client:
            page = await _get_guarded(client, url)
            if page is None:
                return None
            page.raise_for_status()
            image_url, alt = _extract_image_url(page.text, str(page.url))
            if not image_url:
                logger.info("No og:image found for '%s'", url)
                return None

            downloaded = await _download_image(client, image_url, source_url=url, max_bytes=max_bytes)
            if downloaded is None:
                return None
            data, content_type = downloaded

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


async def _get_guarded(client: httpx.AsyncClient, url: str) -> httpx.Response | None:
    """GET `url`, following redirects manually and re-checking each hop. None when a hop is not
    fetchable or the hop budget is exhausted."""
    current = url
    for _ in range(_MAX_REDIRECT_HOPS + 1):
        if not await _is_fetchable(current):
            return None
        resp = await client.get(current)
        location = resp.headers.get("location", "") if resp.is_redirect else ""
        if not location:
            return resp
        current = urljoin(current, location)
    logger.info("Refusing to fetch '%s': more than %d redirects", url, _MAX_REDIRECT_HOPS)
    return None


async def _download_image(
    client: httpx.AsyncClient, image_url: str, *, source_url: str, max_bytes: int
) -> tuple[bytes, str] | None:
    """Stream an image, following redirects manually (each hop re-checked). Streaming means an
    oversize body is aborted mid-download instead of being fully buffered into memory (a
    non-streaming get() reads the whole body first). None when any check rejects it."""
    current = image_url
    for _ in range(_MAX_REDIRECT_HOPS + 1):
        if not await _is_fetchable(current):
            return None
        async with client.stream("GET", current) as resp:
            location = resp.headers.get("location", "") if resp.is_redirect else ""
            if location:
                current = urljoin(current, location)
                continue
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "").split(";")[0].strip().lower()
            if content_type not in _RENDERABLE_IMAGE_TYPES:
                logger.info("og:image for '%s' is not a renderable image (%s)", source_url, content_type or "unknown")
                return None
            declared = resp.headers.get("content-length")
            if declared and declared.isdigit() and int(declared) > max_bytes:
                logger.info("og:image for '%s' too large (Content-Length %s)", source_url, declared)
                return None
            chunks: list[bytes] = []
            total = 0
            async for chunk in resp.aiter_bytes():
                total += len(chunk)
                if total > max_bytes:
                    logger.info("og:image for '%s' exceeded %d bytes mid-stream, aborting", source_url, max_bytes)
                    return None
                chunks.append(chunk)
        data = b"".join(chunks)
        if not data:
            logger.info("og:image for '%s' was empty", source_url)
            return None
        return data, content_type
    logger.info("Refusing to fetch '%s': more than %d redirects", image_url, _MAX_REDIRECT_HOPS)
    return None
