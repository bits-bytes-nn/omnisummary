from __future__ import annotations

import asyncio
from collections.abc import Iterable

import httpx

from shared import CollectedItem, SourceType, logger, retry_async
from shared.config import RSSHubCollectorConfig
from shared.constants import TWITTER_PLATFORMS

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


def _failure_hint(platforms: Iterable[str]) -> str:
    """The actionable hint for the platforms that actually failed. Expired Twitter cookies are the
    usual cause of RSSHub failures HERE, but the collector also serves mastodon/other routes, and
    asserting the Twitter cause unconditionally sent ops to the wrong container setting."""
    if any(platform.lower() in TWITTER_PLATFORMS for platform in platforms):
        return "Twitter cookies may have expired — update TWITTER_AUTH_TOKEN and TWITTER_CT0 in the RSSHub container"
    return ""


class RSSHubCollector(BaseCollector):
    def __init__(self, config: RSSHubCollectorConfig):
        self.config = config
        self.run_meta: dict[str, int] = {}

    async def collect(self) -> list[CollectedItem]:
        if not self.config.enabled:
            logger.info("RSSHub collector is disabled, skipping")
            return []

        parked = load_items_from_s3("rsshub_items.json", max_age_hours=self.config.park_max_age_hours)
        self.park_status = parked
        if parked.usable:
            # The park file records HOW MANY inputs failed, never which ones, so the hint is derived
            # from the platforms this deployment actually configures.
            self.flag_degraded_park(
                parked,
                threshold=self.config.error_rate_threshold,
                empty_threshold=self.config.empty_rate_threshold,
                what=_INPUT_LABEL,
                hint=_failure_hint(a.platform for a in self.config.accounts),
            )
            # Through the SAME window the live branch applies below: the park file can be stale (its
            # items older than lookback) and a --date backfill must not ingest today's parked posts.
            return parked_items_in_window(
                parked.items,
                lookback_hours=self.config.lookback_hours,
                reference_time=self.config.reference_time,
                description="RSSHub park file",
            )

        if not self.config.accounts:
            logger.info("No RSSHub accounts configured, skipping")
            return []

        await self._check_reachable()

        # Bound the fan-out. Every account's feedparser.parse occupies a worker thread, and with
        # 40+ accounts the default executor is oversubscribed: a feed's wait_for could expire
        # while its parse had not even started, so healthy accounts looked like timeouts. The
        # semaphore is created HERE (on the running loop, never at import/__init__) and is
        # acquired BEFORE the per-account timeout, so the timeout measures the fetch itself
        # rather than the queue wait. Worst case wall time is
        # ceil(accounts / max_concurrency) * request_timeout, which stays inside the Lambda budget.
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        coros = [
            self._collect_account(account.username, account.platform, semaphore) for account in self.config.accounts
        ]
        labels = [f"{a.platform}/{a.username}" for a in self.config.accounts]
        results = await asyncio.gather(*coros, return_exceptions=True)

        items: list[CollectedItem] = []
        failed_accounts: list[str] = []
        failed_platforms: list[str] = []
        empty_accounts: list[str] = []
        for account, label, result in zip(self.config.accounts, labels, results, strict=True):
            if isinstance(result, BaseException):
                logger.warning("RSSHub task '%s' failed: %s", label, result)
                failed_accounts.append(label)
                failed_platforms.append(account.platform)
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
            threshold=self.config.error_rate_threshold,
            empty_threshold=self.config.empty_rate_threshold,
            what=_INPUT_LABEL,
            hint=_failure_hint(failed_platforms if failed_accounts else [a.platform for a in self.config.accounts]),
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

    async def _collect_account(self, username: str, platform: str, semaphore: asyncio.Semaphore) -> list[CollectedItem]:
        feed_path = self._build_feed_path(username, platform)
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
                source_type=self._detect_source_type(platform),
                lookback_hours=self.config.lookback_hours,
                reference_time=self.config.reference_time,
                description=f"RSSHub feed '{feed_url}'",
                metadata={"rsshub_feed": feed_url, "platform": platform},
                author=username,
            )

    @staticmethod
    def _build_feed_path(username: str, platform: str) -> str:
        """Build the RSSHub route path for an account.

        Twitter/X accounts map to `twitter/user/{username}`; any other platform maps
        to `{platform}/user/{username}`.
        """
        platform_lower = platform.lower()
        if platform_lower in TWITTER_PLATFORMS:
            return f"twitter/user/{username}"
        return f"{platform_lower}/user/{username}"

    @staticmethod
    def _detect_source_type(platform: str) -> SourceType:
        platform_lower = platform.lower()
        if platform_lower in TWITTER_PLATFORMS:
            return SourceType.X
        return SourceType.WEB
