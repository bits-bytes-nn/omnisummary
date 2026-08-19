from __future__ import annotations

import asyncio
import re

from shared import CollectedItem, SourceType, generate_item_id, logger
from shared.config import RedditCollectorConfig

from .base import (
    BaseCollector,
    cutoff_datetime,
    fetch_feed_with_retry,
    parse_feed_entries,
)

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

        # Fetch subreddits SEQUENTIALLY, not via asyncio.gather: firing all feeds in one ~50ms
        # burst from a single IP is exactly the pattern Reddit rate-limits (observed HTTP 429,
        # one subreddit dropped per run). Serial + per-request spacing keeps us under the limit;
        # two or three subreddits don't need parallelism.
        items: list[CollectedItem] = []
        failures: list[BaseException] = []
        empty = 0
        for idx, sub in enumerate(self.config.subreddits):
            if idx:
                await asyncio.sleep(self.config.retry_backoff_sec)
            try:
                collected = await self._collect_subreddit(sub)
            except Exception as e:
                logger.warning("Reddit subreddit 'r/%s' failed: %s", sub, e)
                failures.append(e)
                continue
            if collected:
                items.extend(collected)
            else:
                empty += 1

        # All subreddits failed (proxy/network/upstream outage) -> surface as a failure
        # so the health check marks Reddit FAILED and alerts, rather than a silent empty day.
        if failures and len(failures) == len(self.config.subreddits):
            raise RuntimeError(f"All {len(failures)} Reddit subreddits failed: {failures[0]}")

        total = len(self.config.subreddits)
        logger.info(
            "Reddit collector gathered %d items total from %d/%d subreddits (%d failed, %d empty)",
            len(items),
            total - len(failures) - empty,
            total,
            len(failures),
            empty,
        )
        # A partial outage that still returns items is neither OK nor FAILED: without this, Reddit
        # could shrink from 6 subreddits to 2 (proxy 429s) and still be reported healthy.
        self.record_run_health(
            total=total,
            failed=len(failures),
            empty=empty,
            threshold=self.config.error_rate_threshold,
            empty_threshold=self.config.empty_rate_threshold,
            max_failed=self.config.max_failed_inputs,
            what="subreddits",
        )
        return items

    async def _collect_subreddit(self, subreddit_name: str) -> list[CollectedItem]:
        logger.info("Collecting posts from 'r/%s'", subreddit_name)
        feed_url = f"{RSS_BASE}/r/{subreddit_name}/{self.config.sort}/.rss?limit={self.config.limit}"
        if self.config.sort == "top":
            feed_url += "&t=day"
        # A rate-limited/transient fetch is retried with jittered backoff instead of dropping the
        # subreddit on the first 429; each attempt tries the direct URL first and then the Cloudflare
        # proxy (Reddit blocks datacenter IPs, the proxy is blocked by other hosts).
        feed = await fetch_feed_with_retry(
            feed_url,
            description=f"Reddit feed 'r/{subreddit_name}'",
            timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            backoff_sec=self.config.retry_backoff_sec,
            proxy_fallback=True,
        )
        return parse_feed_entries(
            feed,
            source_type=SourceType.REDDIT,
            cutoff=cutoff_datetime(self.config.lookback_hours, self.config.reference_time),
            description=f"Reddit feed 'r/{subreddit_name}'",
            metadata={"subreddit": subreddit_name},
            item_id_of=lambda entry, link: self._extract_post_id(entry.get("id", ""), link),
        )

    @staticmethod
    def _extract_post_id(entry_id: str, link: str) -> str:
        match = re.search(r"/comments/([a-z0-9]+)/", link)
        if match:
            return match.group(1)
        if entry_id:
            return entry_id.rsplit("_", 1)[-1] if "_" in entry_id else entry_id
        return generate_item_id(link)
