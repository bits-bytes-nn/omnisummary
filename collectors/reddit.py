from __future__ import annotations

import asyncio
import hashlib
import re

from shared import CollectedItem, SourceType, generate_item_id, logger, parse_feed_published_date
from shared.config import RedditCollectorConfig
from shared.proxy import parse_feed_with_fallback

from .base import (
    BaseCollector,
    TransientStatusError,
    cutoff_datetime,
    feed_parse_failure,
    feed_status_failure,
)

RSS_BASE = "https://www.reddit.com"


def _jittered_backoff(base_sec: float, attempt: int, seed: str) -> float:
    """Linear backoff with deterministic per-subreddit jitter. Plain linear backoff would
    re-synchronize concurrent retries into the same burst Reddit rate-limits; the jitter (0..base,
    derived from the subreddit name + attempt so it needs no RNG) spreads them out."""
    frac = int(hashlib.sha256(f"{seed}:{attempt}".encode()).hexdigest(), 16) % 1000 / 1000.0
    return base_sec * attempt + base_sec * frac


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
            what="subreddits",
        )
        return items

    async def _collect_subreddit(self, subreddit_name: str) -> list[CollectedItem]:
        logger.info("Collecting posts from 'r/%s'", subreddit_name)
        feed_url = f"{RSS_BASE}/r/{subreddit_name}/{self.config.sort}/.rss?limit={self.config.limit}"
        if self.config.sort == "top":
            feed_url += "&t=day"
        # Retry a rate-limited/transient fetch with jittered backoff instead of dropping the
        # subreddit on the first 429. The parse itself runs in a thread (feedparser is sync).
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                # feedparser.parse has no timeout; bound each attempt so a hung fetch doesn't block
                # its worker thread indefinitely. A timeout is transient → retried like a 429/5xx.
                return await asyncio.wait_for(
                    asyncio.to_thread(self._parse_feed, feed_url, subreddit_name),
                    timeout=self.config.request_timeout,
                )
            except TimeoutError:
                last_error = TransientStatusError(
                    f"Reddit 'r/{subreddit_name}' timed out after {self.config.request_timeout}s"
                )
                if attempt < self.config.max_retries:
                    delay = _jittered_backoff(self.config.retry_backoff_sec, attempt, subreddit_name)
                    logger.warning(
                        "Reddit 'r/%s' fetch timed out (attempt %d/%d); retrying in %.1fs",
                        subreddit_name,
                        attempt,
                        self.config.max_retries,
                        delay,
                    )
                    await asyncio.sleep(delay)
            except TransientStatusError as e:
                last_error = e
                if attempt < self.config.max_retries:
                    delay = _jittered_backoff(self.config.retry_backoff_sec, attempt, subreddit_name)
                    logger.warning(
                        "Reddit 'r/%s' fetch failed (attempt %d/%d): %s; retrying in %.1fs",
                        subreddit_name,
                        attempt,
                        self.config.max_retries,
                        e,
                        delay,
                    )
                    await asyncio.sleep(delay)
        assert last_error is not None
        raise last_error

    def _parse_feed(self, feed_url: str, subreddit_name: str) -> list[CollectedItem]:
        feed = parse_feed_with_fallback(feed_url)
        description = f"Reddit feed 'r/{subreddit_name}'"
        status = feed.get("status")
        # 429 (rate limit) and 5xx are transient — signal a retry. 4xx (e.g. 404) is permanent, and
        # so is a malformed body; a TRANSPORT failure (feedparser reports it in bozo_exception with
        # no status at all) is transient, and used to lose the subreddit for the whole day.
        if status is not None and status >= 400:
            raise feed_status_failure(description, status)
        if feed.bozo and not feed.entries:
            raise feed_parse_failure(description, feed.get("bozo_exception"))

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
