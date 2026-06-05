from __future__ import annotations

import asyncio

import feedparser

from shared import CollectedItem, SourceType, generate_item_id, logger, parse_feed_published_date
from shared.config import RSSCollectorConfig

from .base import BaseCollector, cutoff_datetime, gather_collector_results


class RSSCollector(BaseCollector):
    def __init__(self, config: RSSCollectorConfig):
        self.config = config

    async def collect(self) -> list[CollectedItem]:
        if not self.config.feeds:
            logger.info("No RSS feeds configured, skipping")
            return []

        tasks = [self._collect_feed(feed_url) for feed_url in self.config.feeds]
        items = await gather_collector_results(tasks, labels=self.config.feeds, raise_if_all_failed=True)
        logger.info("RSS collector gathered %d items total", len(items))
        return items

    async def _collect_feed(self, feed_url: str) -> list[CollectedItem]:
        logger.info("Collecting posts from feed '%s'", feed_url)
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._parse_feed, feed_url),
                timeout=self.config.request_timeout,
            )
        except TimeoutError:
            logger.warning("RSS feed '%s' timed out after %ds, skipping", feed_url, self.config.request_timeout)
            return []

    def _parse_feed(self, feed_url: str) -> list[CollectedItem]:
        feed = feedparser.parse(feed_url)
        status = feed.get("status")
        if (status is not None and status >= 400) or (feed.bozo and not feed.entries):
            reason = f"HTTP {status}" if status and status >= 400 else feed.get("bozo_exception")
            logger.warning("Failed RSS feed '%s': %s", feed_url, reason)
            return []
        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)

        items: list[CollectedItem] = []
        for entry in feed.entries:
            try:
                published_at = parse_feed_published_date(entry)
                if published_at and published_at < cutoff:
                    continue

                title = entry.get("title", "")
                link = entry.get("link", "")

                text = ""
                if hasattr(entry, "content") and entry.content:
                    text = entry.content[0].get("value", "")
                elif hasattr(entry, "summary"):
                    text = entry.summary or ""

                item_id = entry.get("id", "") or generate_item_id(link)

                items.append(
                    CollectedItem(
                        item_id=item_id,
                        source_type=SourceType.RSS,
                        title=title,
                        url=link,
                        text=text,
                        author=entry.get("author"),
                        published_at=published_at,
                        metadata={"feed_url": feed_url, "feed_title": feed.feed.get("title", "")},
                    )
                )
                logger.info("Collected RSS post: '%s'", title)
            except (AttributeError, KeyError, TypeError, ValueError):
                logger.warning("Failed to process feed entry from '%s'", feed_url, exc_info=True)

        return items
