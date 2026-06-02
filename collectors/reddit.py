from __future__ import annotations

import asyncio
import re

import feedparser

from shared import CollectedItem, SourceType, generate_item_id, logger, parse_feed_published_date
from shared.config import RedditCollectorConfig
from shared.proxy import get_proxied_url

from .base import BaseCollector, cutoff_datetime, gather_collector_results

RSS_BASE = "https://www.reddit.com"


class RedditCollector(BaseCollector):
    """Collects subreddit posts via Reddit's public .rss feed.

    Reddit froze self-serve OAuth app creation (Responsible Builder Policy, 2025-11)
    and blocks the .json API from datacenter IPs, but the .rss feed remains open.
    Routed through the Cloudflare proxy so it also works from AWS Lambda IPs.
    Trade-off vs the old OAuth API: RSS carries no score/num_comments engagement.
    """

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
        feed_url = f"{RSS_BASE}/r/{subreddit_name}/{self.config.sort}/.rss?limit={self.config.limit}"
        return await asyncio.to_thread(self._parse_feed, feed_url, subreddit_name)

    def _parse_feed(self, feed_url: str, subreddit_name: str) -> list[CollectedItem]:
        feed = feedparser.parse(get_proxied_url(feed_url))
        if feed.bozo and not feed.entries:
            logger.warning("Failed to parse Reddit feed 'r/%s': %s", subreddit_name, feed.bozo_exception)
            return []

        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)
        items: list[CollectedItem] = []

        for entry in feed.entries:
            try:
                published_at = parse_feed_published_date(entry)
                if published_at and published_at < cutoff:
                    continue

                link = entry.get("link", "")
                text = ""
                if hasattr(entry, "content") and entry.content:
                    text = entry.content[0].get("value", "")
                elif hasattr(entry, "summary"):
                    text = entry.summary or ""

                item_id = self._extract_post_id(entry.get("id", ""), link)

                items.append(
                    CollectedItem(
                        item_id=item_id,
                        source_type=SourceType.REDDIT,
                        title=entry.get("title", ""),
                        url=link,
                        text=text,
                        author=entry.get("author"),
                        published_at=published_at,
                        metadata={"subreddit": subreddit_name},
                    )
                )
                logger.info("Collected Reddit post: '%s'", entry.get("title", ""))
            except Exception:
                logger.warning("Failed to process Reddit entry in 'r/%s'", subreddit_name, exc_info=True)

        return items

    @staticmethod
    def _extract_post_id(entry_id: str, link: str) -> str:
        match = re.search(r"/comments/([a-z0-9]+)/", link)
        if match:
            return match.group(1)
        if entry_id:
            return entry_id.rsplit("_", 1)[-1] if "_" in entry_id else entry_id
        return generate_item_id(link)
