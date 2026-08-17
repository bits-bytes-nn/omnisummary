from __future__ import annotations

import os
from urllib.parse import quote

import feedparser

from .logger import logger


def get_proxied_url(target_url: str) -> str:
    """Wrap a URL so it is fetched through the Cloudflare Worker (cloudflare-proxy/), which has a
    residential-ish egress IP for sources that block AWS datacenter IPs.

    The token rides in the QUERY STRING because the callers hand the result straight to
    feedparser.parse, which cannot attach headers. The worker only forwards hosts on its
    ALLOWED_HOSTS var, so proxying an unlisted host returns 403 rather than silently working."""
    proxy_base = os.getenv("CLOUDFLARE_PROXY_URL", "")
    proxy_token = os.getenv("CLOUDFLARE_PROXY_TOKEN", "")

    if not proxy_base or not proxy_token:
        return target_url

    # Percent-encode BOTH values: an unencoded token containing '&' or '#' would truncate the
    # query string and turn every proxied fetch into a 401.
    proxied = f"{proxy_base.rstrip('/')}/?url={quote(target_url, safe='')}&token={quote(proxy_token, safe='')}"
    logger.debug("Proxying '%s' via Cloudflare Worker", target_url[:80])
    return proxied


def is_proxy_configured() -> bool:
    return bool(os.getenv("CLOUDFLARE_PROXY_URL")) and bool(os.getenv("CLOUDFLARE_PROXY_TOKEN"))


def parse_feed_with_fallback(feed_url: str):
    """feedparser.parse a URL, trying direct first then the Cloudflare proxy.

    Some hosts (e.g. Reddit) block the proxy's datacenter IP but allow residential
    IPs, while others block datacenter IPs but the proxy works. Trying direct first
    then proxy covers both: locally direct usually succeeds; from AWS the direct call
    fails and the proxy fallback is used. Returns the first response that looks usable
    (HTTP < 400 with entries), else the last attempt so callers can inspect bozo/status.
    """
    candidates = [feed_url]
    proxied = get_proxied_url(feed_url)
    if proxied != feed_url:
        candidates.append(proxied)

    last = None
    for url in candidates:
        feed = feedparser.parse(url)
        last = feed
        status = feed.get("status")
        if (status is None or status < 400) and feed.entries:
            return feed
    return last
