from __future__ import annotations

import os
from urllib.parse import quote

from .logger import logger


def get_proxied_url(target_url: str) -> str:
    proxy_base = os.getenv("CLOUDFLARE_PROXY_URL", "")
    proxy_token = os.getenv("CLOUDFLARE_PROXY_TOKEN", "")

    if not proxy_base or not proxy_token:
        return target_url

    proxied = f"{proxy_base.rstrip('/')}/?url={quote(target_url, safe='')}&token={proxy_token}"
    logger.debug("Proxying '%s' via Cloudflare Worker", target_url[:80])
    return proxied


def is_proxy_configured() -> bool:
    return bool(os.getenv("CLOUDFLARE_PROXY_URL")) and bool(os.getenv("CLOUDFLARE_PROXY_TOKEN"))
