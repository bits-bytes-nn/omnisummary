from __future__ import annotations

import re
from typing import Any

from shared import CollectedItem, SourceType, generate_item_id, logger
from shared.config import RedditCollectorConfig

from .base import BaseCollector, fetch_feed_with_retry, parse_feed_entries

RSS_BASE = "https://www.reddit.com"
# Reddit's .rss caps a listing at 100 entries.
_MAX_FEED_LIMIT = 100
_SUBREDDIT_IN_LINK = re.compile(r"/r/([^/]+)/comments/")


class RedditCollector(BaseCollector):
    """Collects subreddit posts via Reddit's public .rss feed.

    Reddit froze self-serve OAuth app creation (Responsible Builder Policy, 2025-11)
    and blocks the .json API from datacenter IPs, but the .rss feed remains open, Lambda IPs included.
    Trade-off vs the old OAuth API: RSS carries no score/num_comments engagement.

    All subreddits are fetched in ONE multi-subreddit request (`r/A+B/...`): a second request a few
    seconds after the first is what Reddit rate-limits (429 on the second subreddit in 30 of 30 runs).
    """

    config: RedditCollectorConfig

    def __init__(self, config: RedditCollectorConfig):
        self.config = config

    async def collect(self) -> list[CollectedItem]:
        subreddits = self.config.subreddits
        if not subreddits:
            logger.info("No subreddits configured, skipping")
            return []

        description = f"Reddit feed 'r/{'+'.join(subreddits)}'"
        logger.info("Collecting posts from %s", description)
        feed = await fetch_feed_with_retry(
            self._feed_url(),
            description=description,
            timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            backoff_sec=self.config.retry_backoff_sec,
        )
        items = parse_feed_entries(
            feed,
            source_type=SourceType.REDDIT,
            lookback_hours=self.config.lookback_hours,
            reference_time=self.config.reference_time,
            description=description,
            metadata={},
            item_id_of=lambda entry, link: self._extract_post_id(entry.get("id", ""), link),
        )
        items = self._attribute_and_cap(items, feed.entries)

        per_sub = {sub: sum(1 for item in items if item.metadata["subreddit"] == sub) for sub in subreddits}
        empty = sum(1 for count in per_sub.values() if not count)
        logger.info(
            "Reddit collector gathered %d items total from %d/%d subreddits (%d empty)",
            len(items),
            len(subreddits) - empty,
            len(subreddits),
            empty,
        )
        self.record_run_health(
            total=len(subreddits),
            failed=0,
            empty=empty,
            what="subreddits",
        )
        return items

    def _feed_url(self) -> str:
        url = (
            f"{RSS_BASE}/r/{'+'.join(self.config.subreddits)}/{self.config.sort}/.rss"
            f"?limit={min(_MAX_FEED_LIMIT, self.config.limit * len(self.config.subreddits))}"
        )
        if self.config.sort == "top":
            # Reddit's window is relative to the fetch and runs are a day apart. A wider bucket
            # (t=week) is a different ranking: last week's viral posts crowd out today's.
            url += "&t=day"
        return url

    def _attribute_and_cap(self, items: list[CollectedItem], entries: list[Any]) -> list[CollectedItem]:
        """Tag each item with its subreddit and keep at most `limit` per subreddit, in feed order (the
        feed's own ranking), so one busy subreddit cannot crowd out a quiet one."""
        canonical = {sub.lower(): sub for sub in self.config.subreddits}
        tag_of = {entry.get("link", ""): self._entry_subreddit(entry) for entry in entries}
        kept: list[CollectedItem] = []
        counts: dict[str, int] = {}
        for item in items:
            sub = canonical.get(tag_of.get(item.url, "").lower(), "")
            if not sub or counts.get(sub, 0) >= self.config.limit:
                continue
            counts[sub] = counts.get(sub, 0) + 1
            kept.append(item.model_copy(update={"metadata": {"subreddit": sub}}))
        return kept

    @staticmethod
    def _entry_subreddit(entry: Any) -> str:
        match = _SUBREDDIT_IN_LINK.search(entry.get("link", ""))
        if match:
            return match.group(1)
        tags = entry.get("tags") or []
        return str(tags[0].get("term", "")) if tags else ""

    @staticmethod
    def _extract_post_id(entry_id: str, link: str) -> str:
        match = re.search(r"/comments/([a-z0-9]+)/", link)
        if match:
            return match.group(1)
        if entry_id:
            return entry_id.rsplit("_", 1)[-1] if "_" in entry_id else entry_id
        return generate_item_id(link)
