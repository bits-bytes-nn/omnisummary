from __future__ import annotations

import os
from collections.abc import Awaitable, Callable
from typing import Any
from urllib.parse import quote

from .logger import logger


def get_proxied_url(target_url: str) -> str:
    """Wrap a URL so it is fetched through the Cloudflare Worker (cloudflare-proxy/), which has a
    residential-ish egress IP for sources that block AWS datacenter IPs.

    The token rides in the QUERY STRING because the worker is reached with a plain GET whose headers
    the caller does not control. The worker only forwards hosts on its ALLOWED_HOSTS var, so proxying
    an unlisted host returns 403 rather than silently working."""
    proxy_base = os.getenv("CLOUDFLARE_PROXY_URL", "")
    proxy_token = os.getenv("CLOUDFLARE_PROXY_TOKEN", "")

    if not proxy_base or not proxy_token:
        return target_url

    # Percent-encode BOTH values: an unencoded token containing '&' or '#' would truncate the
    # query string and turn every proxied fetch into a 401.
    proxied = f"{proxy_base.rstrip('/')}/?url={quote(target_url, safe='')}&token={quote(proxy_token, safe='')}"
    logger.debug("Proxying '%s' via Cloudflare Worker", target_url[:80])
    return proxied


async def fetch_with_proxy_fallback(
    target_url: str, fetch: Callable[[str], Awaitable[Any]], *, has_entries: Callable[[Any], bool]
) -> Any:
    """Fetch a feed directly first, then through the Cloudflare proxy, and return the BEST usable
    response rather than the last attempt.

    Some hosts (e.g. Reddit) block the proxy's datacenter IP but allow residential IPs, while others
    block datacenter IPs and only the proxy works — so both are tried. Selection order:
    1. the first candidate that returned entries;
    2. otherwise the first candidate that answered WITHOUT an error — a genuinely quiet feed;
    3. only when EVERY candidate errored does an error propagate (the last one, which is the
       proxy's when one is configured and therefore the more informative for a blocked host).

    Returning the last attempt meant a direct 200 with zero entries (a quiet subreddit) was
    overwritten by a proxy 429: the caller saw a transient failure, burned every retry, and with two
    configured subreddits reported the whole source FAILED on a clean empty day."""
    candidates = [target_url]
    proxied = get_proxied_url(target_url)
    if proxied != target_url:
        candidates.append(proxied)

    usable: Any | None = None
    last_error: BaseException | None = None
    for url in candidates:
        try:
            feed = await fetch(url)
        except Exception as e:
            last_error = e
            continue
        if has_entries(feed):
            return feed
        if usable is None:
            usable = feed
    if usable is not None:
        return usable
    assert last_error is not None
    raise last_error
