from __future__ import annotations

import asyncio

from adapters.research_backends import extract_pages
from domain.models import CollectedItem
from shared import SourceType, logger
from shared.config import RSSCollectorConfig

from .base import (
    BaseCollector,
    fetch_feed_with_retry,
    gather_collector_results,
    parse_feed_entries,
)

# A summary-only entry shorter than this is a teaser, not the post. Measured on the live feeds: the
# thin primary-source summaries ran to a median of 19 (research.google) and 149 (openai.com) chars,
# while Simon Willison's feed ships whole posts as the summary (median 1712), which the page text
# would only replace with navigation and footer.
THIN_SUMMARY_MAX_CHARS = 500


class RSSCollector(BaseCollector):
    config: RSSCollectorConfig

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
            what="feeds",
        )
        return await self._fill_summary_only(result.items)

    @staticmethod
    async def _fill_summary_only(items: list[CollectedItem]) -> list[CollectedItem]:
        """Swap in the page text for entries whose feed ships only a summary. Several primary-source
        feeds (DeepMind, Google Research, Hugging Face) carry an empty or one-line description, and
        the ranker then scored a title alone: an empty "Gemini 4 Argon" post scored 0.84 and shipped
        as a one-sentence story. The feed text stays when the page reads shorter or not at all."""
        thin = [it for it in items if it.metadata.get("summary_only") and len(it.text) < THIN_SUMMARY_MAX_CHARS]
        if not thin:
            return items
        pages = await extract_pages(list(dict.fromkeys(it.url for it in thin)))
        filled = added = 0
        for item in thin:
            page = pages.get(item.url, "")
            if len(page) > len(item.text):
                added += len(page) - len(item.text)
                item.text = page
                filled += 1
        logger.info(
            "RSS: replaced the feed summary with the page text for %d of %d summary-only items (+%d chars)",
            filled,
            len(thin),
            added,
        )
        return items

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
            items = parse_feed_entries(
                feed,
                source_type=SourceType.RSS,
                lookback_hours=self.config.lookback_hours,
                reference_time=self.config.reference_time,
                description=f"RSS feed '{feed_url}'",
                metadata={"feed_url": feed_url, "feed_title": feed.feed.get("title", "")},
            )
            summary_only = {entry.get("link", "") for entry in feed.entries if not entry.get("content")}
            for item in items:
                if item.url in summary_only:
                    item.metadata = {**item.metadata, "summary_only": True}
            return items
