from __future__ import annotations

import asyncio
import os
from datetime import UTC, datetime

import boto3
import httpx

from shared import CollectedItem, SourceType, logger
from shared.config import RedditCollectorConfig

from .base import BaseCollector, cutoff_datetime, gather_collector_results

USER_AGENT = "omnisummary:v1.0 (by /u/omnisummary)"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
OAUTH_BASE = "https://oauth.reddit.com"


def _resolve_reddit_credentials() -> tuple[str, str] | None:
    client_id = os.getenv("REDDIT_CLIENT_ID", "")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET", "")
    if client_id and client_secret:
        return client_id, client_secret

    project = os.getenv("PROJECT_NAME", "omnisummary")
    stage = os.getenv("STAGE", "dev")
    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "ap-northeast-2"))
    ssm = boto3.client("ssm", region_name=region)
    try:
        client_id = ssm.get_parameter(Name=f"/{project}/{stage}/reddit-client-id", WithDecryption=True)[
            "Parameter"
        ]["Value"]
        client_secret = ssm.get_parameter(Name=f"/{project}/{stage}/reddit-client-secret", WithDecryption=True)[
            "Parameter"
        ]["Value"]
    except Exception as e:
        logger.warning("Reddit credentials unavailable (env + SSM): %s", e)
        return None
    return client_id, client_secret


class RedditCollector(BaseCollector):
    def __init__(self, config: RedditCollectorConfig):
        self.config = config

    async def collect(self) -> list[CollectedItem]:
        if not self.config.subreddits:
            logger.info("No subreddits configured, skipping")
            return []

        creds = await asyncio.to_thread(_resolve_reddit_credentials)
        if not creds:
            logger.warning("Reddit credentials missing — skipping Reddit collection")
            return []

        client_id, client_secret = creds
        try:
            token = await self._fetch_token(client_id, client_secret)
        except Exception:
            logger.warning("Failed to obtain Reddit OAuth token", exc_info=True)
            return []
        if not token:
            logger.warning("Reddit OAuth token empty — skipping Reddit collection")
            return []

        tasks = [self._collect_subreddit(sub, token) for sub in self.config.subreddits]
        items = await gather_collector_results(tasks, labels=self.config.subreddits)
        logger.info("Reddit collector gathered %d items total", len(items))
        return items

    async def _fetch_token(self, client_id: str, client_secret: str) -> str:
        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            response = await client.post(
                TOKEN_URL,
                data={"grant_type": "client_credentials"},
                auth=httpx.BasicAuth(client_id, client_secret),
                headers={"User-Agent": USER_AGENT},
            )
            response.raise_for_status()
            return response.json().get("access_token", "")

    async def _collect_subreddit(self, subreddit_name: str, token: str) -> list[CollectedItem]:
        logger.info("Collecting posts from 'r/%s'", subreddit_name)
        url = f"{OAUTH_BASE}/r/{subreddit_name}/{self.config.sort}"
        params: dict[str, str | int] = {"limit": self.config.limit}
        if self.config.sort == "top":
            params["t"] = "day"
        headers = {"Authorization": f"Bearer {token}", "User-Agent": USER_AGENT}
        max_retries = self.config.max_retries
        backoff = self.config.retry_backoff_sec

        for attempt in range(1, max_retries + 1):
            try:
                async with httpx.AsyncClient(headers=headers, timeout=self.config.request_timeout) as client:
                    response = await client.get(url, params=params)
                    response.raise_for_status()
                    data = response.json()
                return self._parse_listing(data, subreddit_name)
            except Exception:
                if attempt < max_retries:
                    logger.warning(
                        "Failed to fetch 'r/%s' (attempt %d/%d), retrying in %ds...",
                        subreddit_name, attempt, max_retries, backoff * attempt,
                    )
                    await asyncio.sleep(backoff * attempt)
                else:
                    logger.warning("Failed to fetch 'r/%s' after %d attempts", subreddit_name, max_retries, exc_info=True)
                    return []
        return []

    def _parse_listing(self, data: dict, subreddit_name: str) -> list[CollectedItem]:
        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)
        items: list[CollectedItem] = []

        for child in data.get("data", {}).get("children", []):
            post = child.get("data", {})
            try:
                created_utc = post.get("created_utc")
                if not created_utc:
                    continue
                created_at = datetime.fromtimestamp(created_utc, tz=UTC)
                if created_at < cutoff:
                    continue

                permalink = post.get("permalink", "")
                is_self = post.get("is_self", True)

                items.append(
                    CollectedItem(
                        item_id=post.get("id", ""),
                        source_type=SourceType.REDDIT,
                        title=post.get("title", ""),
                        url=f"https://www.reddit.com{permalink}",
                        text=post.get("selftext", ""),
                        author=post.get("author"),
                        published_at=created_at,
                        metadata={
                            "subreddit": subreddit_name,
                            "score": post.get("score", 0),
                            "num_comments": post.get("num_comments", 0),
                            "link_url": post.get("url") if not is_self else None,
                        },
                    )
                )
                logger.info("Collected Reddit post: '%s'", post.get("title", ""))
            except Exception:
                logger.warning("Failed to process Reddit post in 'r/%s'", subreddit_name, exc_info=True)

        return items
