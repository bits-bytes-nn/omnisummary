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

        # Bound the fan-out. Every feed's feedparser.parse occupies a worker thread, and with
        # dozens of feeds the default executor is oversubscribed: a feed's wait_for could expire
        # while its parse had not even started, so a healthy feed counted as a timeout FAILURE.
        # The semaphore is created HERE (on the running loop, never at import/__init__) and is
        # acquired BEFORE the per-feed timeout, so the timeout measures the fetch, not the queue
        # wait. Mirrors rsshub.max_concurrency.
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        tasks = [self._collect_feed(feed_url, semaphore) for feed_url in self.config.feeds]
        items = await gather_collector_results(tasks, labels=self.config.feeds, raise_if_all_failed=True)
        logger.info("RSS collector gathered %d items total", len(items))
        return items

    async def _collect_feed(self, feed_url: str, semaphore: asyncio.Semaphore) -> list[CollectedItem]:
        async with semaphore:
            logger.info("Collecting posts from feed '%s'", feed_url)
            try:
                return await asyncio.wait_for(
                    asyncio.to_thread(self._parse_feed, feed_url),
                    timeout=self.config.request_timeout,
                )
            except TimeoutError as e:
                # Raise (not return []) so gather_collector_results counts this as a task FAILURE:
                # a total outage (every feed hung) then surfaces as FAILED instead of a silent empty
                # result. One hung feed among many is still just logged there and skipped.
                logger.warning("RSS feed '%s' timed out after %ds, skipping", feed_url, self.config.request_timeout)
                raise RuntimeError(f"RSS feed '{feed_url}' timed out after {self.config.request_timeout}s") from e

    def _parse_feed(self, feed_url: str) -> list[CollectedItem]:
        feed = feedparser.parse(feed_url)
        status = feed.get("status")
        if (status is not None and status >= 400) or (feed.bozo and not feed.entries):
            reason = f"HTTP {status}" if status and status >= 400 else feed.get("bozo_exception")
            # A dead feed is a FAILURE, not an empty one: raising lets the all-failed check mark
            # the whole source FAILED when every feed is dead, while a single dead feed among
            # many is still tolerated (logged and skipped by gather_collector_results).
            raise RuntimeError(f"Failed RSS feed '{feed_url}': {reason}")
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
