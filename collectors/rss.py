from __future__ import annotations

import asyncio

from shared import CollectedItem, SourceType, logger
from shared.config import RSSCollectorConfig

from .base import (
    BaseCollector,
    cutoff_datetime,
    fetch_feed_with_retry,
    gather_collector_results,
    parse_feed_entries,
)


class RSSCollector(BaseCollector):
    def __init__(self, config: RSSCollectorConfig):
        self.config = config

    async def collect(self) -> list[CollectedItem]:
        if not self.config.feeds:
            logger.info("No RSS feeds configured, skipping")
            return []

        # Bound the fan-out: max_concurrency feeds are in flight at a time, so the collector's
        # worst case is ceil(feeds / max_concurrency) * (max_retries * request_timeout + backoff),
        # which has to stay inside the digest Lambda's 15-minute budget. The semaphore is created
        # HERE (on the running loop, never at import/__init__). Mirrors rsshub.max_concurrency.
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        tasks = [self._collect_feed(feed_url, semaphore) for feed_url in self.config.feeds]
        result = await gather_collector_results(tasks, labels=self.config.feeds, raise_if_all_failed=True)
        logger.info(
            "RSS collector gathered %d items total from %d/%d feeds (%d failed, %d empty)",
            len(result.items),
            result.total - result.failed - result.empty,
            result.total,
            result.failed,
            result.empty,
        )
        # A partial outage that still returns items is neither OK nor FAILED: without this, RSS
        # could shrink to a couple of live feeds and the health report would call it healthy.
        self.record_run_health(
            total=result.total,
            failed=result.failed,
            empty=result.empty,
            threshold=self.config.error_rate_threshold,
            empty_threshold=self.config.empty_rate_threshold,
            max_failed=self.config.max_failed_inputs,
            what="feeds",
        )
        return result.items

    async def _collect_feed(self, feed_url: str, semaphore: asyncio.Semaphore) -> list[CollectedItem]:
        async with semaphore:
            logger.info("Collecting posts from feed '%s'", feed_url)
            # A hung fetch and a 429/5xx are transient: a single blip used to drop the whole feed's
            # items for the day. A dead feed (403/404, unparseable body) is a verdict retrying cannot
            # change, so it raises straight out — and raising (rather than returning []) is what lets
            # gather_collector_results report a whole-source outage as FAILED.
            feed = await fetch_feed_with_retry(
                feed_url,
                description=f"RSS feed '{feed_url}'",
                timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
                backoff_sec=self.config.retry_backoff_sec,
            )
            return parse_feed_entries(
                feed,
                source_type=SourceType.RSS,
                cutoff=cutoff_datetime(self.config.lookback_hours, self.config.reference_time),
                description=f"RSS feed '{feed_url}'",
                metadata={"feed_url": feed_url, "feed_title": feed.feed.get("title", "")},
            )
