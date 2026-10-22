from __future__ import annotations

import asyncio

import httpx

from shared import CollectedItem, SourceType, is_running_in_aws, logger, retry_async
from shared.config import RSSHubCollectorConfig

from .base import (
    PARK_META_ACCOUNTS_EMPTY,
    PARK_META_ACCOUNTS_FAILED,
    PARK_META_ACCOUNTS_TOTAL,
    BaseCollector,
    TransientStatusError,
    fetch_feed_with_retry,
    load_items_from_s3,
    parked_items_in_window,
    parse_feed_entries,
)

__all__ = [
    "PARK_META_ACCOUNTS_EMPTY",
    "PARK_META_ACCOUNTS_FAILED",
    "PARK_META_ACCOUNTS_TOTAL",
    "RSSHubCollector",
]

# What the source's inputs are called in the degraded/park reports.
_INPUT_LABEL = "account feeds"


# Expired Twitter cookies are the usual cause of RSSHub feed failures.
_FAILURE_HINT = "Twitter cookies may have expired — update TWITTER_AUTH_TOKEN and TWITTER_CT0 in the RSSHub container"


class RSSHubCollector(BaseCollector):
    config: RSSHubCollectorConfig

    def __init__(self, config: RSSHubCollectorConfig):
        self.config = config
        self.run_meta: dict[str, int] = {}

    async def collect(self) -> list[CollectedItem]:
        parked = load_items_from_s3("rsshub_items.json", max_age_hours=self.config.park_max_age_hours)
        self.park_status = parked
        if parked.usable:
            self.flag_degraded_park(
                parked,
                what=_INPUT_LABEL,
                hint=_FAILURE_HINT,
            )
            # Through the SAME window the live branch applies below: the park file can be stale (its
            # items older than lookback) and a --date backfill must not ingest today's parked posts.
            return parked_items_in_window(
                parked.items,
                lookback_hours=self.config.lookback_hours,
                reference_time=self.config.reference_time,
                description="RSSHub park file",
            )

        if is_running_in_aws():
            # Nothing in AWS runs RSSHub, so probing it would only name the wrong cause.
            raise RuntimeError(
                "X park file unusable, check the local sync (scripts/sync_all_to_s3.sh): "
                f"{parked.detail or parked.outcome.value}"
            )

        if not self.config.accounts:
            logger.info("No RSSHub accounts configured, skipping")
            return []

        await self._check_reachable()

        # Bounded fan-out against one local RSSHub instance. Created on the running loop, never at
        # import/__init__.
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        coros = [self._collect_account(account.username, semaphore) for account in self.config.accounts]
        labels = [f"x/{a.username}" for a in self.config.accounts]
        results = await asyncio.gather(*coros, return_exceptions=True)

        items: list[CollectedItem] = []
        failed_accounts: list[str] = []
        empty_accounts: list[str] = []
        for label, result in zip(labels, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("RSSHub task '%s' failed: %s", label, result)
                failed_accounts.append(label)
            elif result:
                items.extend(result)
            else:
                empty_accounts.append(label)

        total = len(self.config.accounts)
        active = total - len(failed_accounts) - len(empty_accounts)
        # Records run_meta (for the sync script to park) AND flags the source DEGRADED past the
        # configured failure rate — the same helper every other collector uses. It owns the
        # rate-vs-threshold verdict and its log line, so nothing here re-derives it.
        self.record_run_health(
            total=total,
            failed=len(failed_accounts),
            empty=len(empty_accounts),
            what=_INPUT_LABEL,
            hint=_FAILURE_HINT,
        )
        logger.info(
            "RSSHub collector gathered %d items from %d/%d accounts (%d failed, %d empty)",
            len(items),
            active,
            total,
            len(failed_accounts),
            len(empty_accounts),
        )
        if failed_accounts:
            # WHICH accounts failed — the only part record_run_health cannot report.
            logger.warning(
                "RSSHub failed feeds: %s",
                ", ".join(failed_accounts[:10]) + ("..." if len(failed_accounts) > 10 else ""),
            )
        if empty_accounts:
            logger.debug(
                "RSSHub empty feeds (no recent posts): %d/%d — '%s'",
                len(empty_accounts),
                total,
                ", ".join(empty_accounts[:10]) + ("..." if len(empty_accounts) > 10 else ""),
            )
        # Every account failed (reachable service, but nothing could be parsed): that is an
        # outage, not a quiet day, so surface it as FAILED instead of a silent empty result.
        # Partial failures are tolerated — only an all-failed run raises.
        if len(failed_accounts) == total:
            raise RuntimeError(f"All {total} RSSHub feeds failed: {', '.join(failed_accounts[:10])}")
        return items

    async def _check_reachable(self) -> None:
        """Raise if the RSSHub service is unreachable, so a total outage is reported
        as FAILED (→ alert) instead of looking like an all-accounts-empty quiet day."""
        base = self.config.base_url.rstrip("/")

        async def _probe() -> None:
            async with httpx.AsyncClient(timeout=self.config.request_timeout, follow_redirects=True) as client:
                response = await client.get(base)
            if response.status_code >= 500:
                raise TransientStatusError(f"RSSHub at {base} returned HTTP {response.status_code}")

        try:
            await retry_async(
                _probe,
                max_retries=self.config.max_retries,
                backoff_sec=self.config.retry_backoff_sec,
                retry_on=(httpx.HTTPError, TransientStatusError),
                description=f"RSSHub reachability check for {base}",
            )
        except Exception as e:
            raise RuntimeError(f"RSSHub unreachable at {base}: {e}") from e

    async def _collect_account(self, username: str, semaphore: asyncio.Semaphore) -> list[CollectedItem]:
        feed_path = f"twitter/user/{username}"
        feed_url = f"{self.config.base_url.rstrip('/')}/{feed_path}"
        async with semaphore:
            logger.info("Collecting RSSHub feed: '%s'", feed_url)
            # A hung fetch and a 429/5xx are transient: without a retry a single blip on the largest
            # source (~41 accounts) dropped that author for the whole day and could push RSSHub past
            # error_rate_threshold. An unparseable body or a permanent 4xx raises straight out, and
            # that raise is what lets collect() report an all-accounts-failed run as FAILED.
            feed = await fetch_feed_with_retry(
                feed_url,
                description=f"RSSHub feed '{feed_url}'",
                timeout=self.config.request_timeout,
                max_retries=self.config.max_retries,
                backoff_sec=self.config.retry_backoff_sec,
            )
            return parse_feed_entries(
                feed,
                source_type=SourceType.X,
                lookback_hours=self.config.lookback_hours,
                reference_time=self.config.reference_time,
                description=f"RSSHub feed '{feed_url}'",
                metadata={"rsshub_feed": feed_url, "platform": "x"},
                author=username,
            )
