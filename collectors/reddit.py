from __future__ import annotations

import asyncio
from datetime import UTC, datetime

import httpx

from shared import CollectedItem, SourceType, logger
from shared.config import RedditCollectorConfig
from shared.proxy import get_proxied_url

from .base import BaseCollector, cutoff_datetime, gather_collector_results

USER_AGENT = "omnisummary:v1.0 (by /u/omnisummary)"
MAX_RETRIES = 3
RETRY_BACKOFF = 5


class RedditCollector(BaseCollector):
    def __init__(self, config: RedditCollectorConfig):
        self.config = config

    async def collect(self) -> list[CollectedItem]:
        if not self.config.subreddits:
            logger.info("No subreddits configured, skipping")
            return []

        tasks = [self._collect_subreddit(sub) for sub in self.config.subreddits]
        items = await gather_collector_results(tasks, labels=self.config.subreddits)
        logger.info("Reddit collector gathered %d items total", len(items))
        return items

    async def _collect_subreddit(self, subreddit_name: str) -> list[CollectedItem]:
        logger.info("Collecting posts from 'r/%s'", subreddit_name)
        base_url = f"https://www.reddit.com/r/{subreddit_name}/{self.config.sort}.json"
        params: dict[str, str | int] = {"limit": self.config.limit}
        if self.config.sort == "top":
            params["t"] = "day"
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = get_proxied_url(f"{base_url}?{query}")

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with httpx.AsyncClient(headers={"User-Agent": USER_AGENT}, timeout=30) as client:
                    response = await client.get(url)
                    response.raise_for_status()
                    data = response.json()
                return self._parse_listing(data, subreddit_name)
            except Exception:
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "Failed to fetch 'r/%s' (attempt %d/%d), retrying in %ds...",
                        subreddit_name, attempt, MAX_RETRIES, RETRY_BACKOFF * attempt,
                    )
                    await asyncio.sleep(RETRY_BACKOFF * attempt)
                else:
                    logger.warning("Failed to fetch 'r/%s' after %d attempts", subreddit_name, MAX_RETRIES, exc_info=True)
                    return []
        return []

    def _parse_listing(self, data: dict, subreddit_name: str) -> list[CollectedItem]:
        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)
        items: list[CollectedItem] = []

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            try:
                created_utc = post.get("created_utc")
                if not created_utc:
                    continue
                created_at = datetime.fromtimestamp(created_utc, tz=UTC)
                if created_at < cutoff:
                    continue

                permalink = post.get("permalink", "")
                is_self = post.get("is_self", True)

                items.append(
                    CollectedItem(
                        item_id=post.get("id", ""),
                        source_type=SourceType.REDDIT,
                        title=post.get("title", ""),
                        url=f"https://www.reddit.com{permalink}",
                        text=post.get("selftext", ""),
                        author=post.get("author"),
                        published_at=created_at,
                        metadata={
                            "subreddit": subreddit_name,
                            "score": post.get("score", 0),
                            "num_comments": post.get("num_comments", 0),
                            "link_url": post.get("url") if not is_self else None,
                        },
                    )
                )
                logger.info("Collected Reddit post: '%s'", post.get("title", ""))
            except Exception:
                logger.warning("Failed to process Reddit post in 'r/%s'", subreddit_name, exc_info=True)

        return items
