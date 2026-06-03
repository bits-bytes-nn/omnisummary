from __future__ import annotations

import asyncio
import json
import os
import time

import boto3
import feedparser
from botocore.exceptions import ClientError

from shared import CollectedItem, SourceType, generate_item_id, logger, parse_feed_published_date
from shared.config import RSSHubCollectorConfig
from shared.constants import TWITTER_PLATFORMS

from .base import BaseCollector, cutoff_datetime


class RSSHubCollector(BaseCollector):
    def __init__(self, config: RSSHubCollectorConfig):
        self.config = config

    async def collect(self) -> list[CollectedItem]:
        if not self.config.enabled:
            logger.info("RSSHub collector is disabled, skipping")
            return []

        s3_items = self._load_from_s3()
        if s3_items is not None:
            return s3_items

        if not self.config.accounts:
            logger.info("No RSSHub accounts configured, skipping")
            return []

        await asyncio.to_thread(self._check_reachable)

        coros = [self._collect_account(account.username, account.platform) for account in self.config.accounts]
        labels = [f"{a.platform}/{a.username}" for a in self.config.accounts]
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
        logger.info(
            "RSSHub collector gathered %d items from %d/%d accounts (%d failed, %d empty)",
            len(items),
            active,
            total,
            len(failed_accounts),
            len(empty_accounts),
        )
        if failed_accounts:
            fail_rate = len(failed_accounts) / total * 100
            logger.warning(
                "RSSHub failed feeds: %d/%d (%.0f%%) — %s",
                len(failed_accounts),
                total,
                fail_rate,
                ", ".join(failed_accounts[:10]) + ("..." if len(failed_accounts) > 10 else ""),
            )
            if fail_rate > self.config.error_rate_threshold:
                logger.warning(
                    "RSSHub failure rate >%.0f%% — Twitter cookies may have expired. "
                    "Update TWITTER_AUTH_TOKEN and TWITTER_CT0 in the RSSHub container.",
                    self.config.error_rate_threshold,
                )
        if empty_accounts:
            logger.debug(
                "RSSHub empty feeds (no recent posts): %d/%d — '%s'",
                len(empty_accounts),
                total,
                ", ".join(empty_accounts[:10]) + ("..." if len(empty_accounts) > 10 else ""),
            )
        return items

    def _check_reachable(self) -> None:
        """Raise if the RSSHub service is unreachable, so a total outage is reported
        as FAILED (→ alert) instead of looking like an all-accounts-empty quiet day."""
        import httpx

        base = self.config.base_url.rstrip("/")
        last_error: Exception | None = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = httpx.get(base, timeout=self.config.request_timeout, follow_redirects=True)
                if resp.status_code >= 500:
                    raise RuntimeError(f"RSSHub at {base} returned HTTP {resp.status_code}")
                return
            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries:
                    logger.warning("RSSHub reachability check failed (attempt %d): %s", attempt, e)
                    time.sleep(self.config.retry_backoff_sec * attempt)
        raise RuntimeError(f"RSSHub unreachable at {base}: {last_error}") from last_error

    async def _collect_account(self, username: str, platform: str) -> list[CollectedItem]:
        feed_path = self._build_feed_path(username, platform)
        feed_url = f"{self.config.base_url.rstrip('/')}/{feed_path}"
        logger.info("Collecting RSSHub feed: '%s'", feed_url)
        return await asyncio.to_thread(self._parse_feed, feed_url, username, platform)

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

    def _parse_feed(self, feed_url: str, username: str, platform: str) -> list[CollectedItem]:
        feed = feedparser.parse(feed_url)
        if feed.bozo and not feed.entries:
            logger.warning("Failed to parse RSSHub feed '%s': %s", feed_url, feed.bozo_exception)
            return []

        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)
        source_type = self._detect_source_type(platform)

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
                        source_type=source_type,
                        title=title,
                        url=link,
                        text=text,
                        author=username,
                        published_at=published_at,
                        metadata={"rsshub_feed": feed_url, "platform": platform},
                    )
                )
                logger.info("Collected RSSHub item: '%s'", title)
            except Exception:
                logger.warning("Failed to process RSSHub entry from '%s'", feed_url, exc_info=True)

        return items

    @staticmethod
    def _detect_source_type(platform: str) -> SourceType:
        platform_lower = platform.lower()
        if platform_lower in TWITTER_PLATFORMS:
            return SourceType.X
        return SourceType.WEB

    @staticmethod
    def _load_from_s3() -> list[CollectedItem] | None:
        bucket = os.environ.get("STATE_BUCKET", "")
        if not bucket:
            return None

        prefix = os.environ.get("S3_PREFIX", "").rstrip("/")
        base_prefix = prefix.rsplit("/", 1)[0] if "/" in prefix else prefix
        s3_key = f"{base_prefix}/rsshub_items.json" if base_prefix else "rsshub_items.json"

        try:
            s3 = boto3.client("s3")
            resp = s3.get_object(Bucket=bucket, Key=s3_key)
            data = json.loads(resp["Body"].read().decode("utf-8"))
            items = [CollectedItem.model_validate(item) for item in data]
            logger.info("Loaded %d RSSHub items from 's3://%s/%s'", len(items), bucket, s3_key)
            return items
        except ClientError:
            logger.info("No RSSHub items found in S3, falling back to live collection")
            return None
        except Exception as e:
            logger.warning("Failed to load RSSHub items from S3: %s", e)
            return None
