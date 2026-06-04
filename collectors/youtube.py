from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import datetime

import feedparser
import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from shared import CollectedItem, SourceType, logger, parse_feed_published_date, retry_async
from shared.config import YouTubeCollectorConfig
from shared.proxy import get_proxied_url, is_proxy_configured

from .base import BaseCollector, cutoff_datetime, gather_collector_results

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"


class YouTubeCollector(BaseCollector):
    def __init__(self, config: YouTubeCollectorConfig):
        self.config = config
        self.api_key = os.getenv("YOUTUBE_API_KEY", "")
        # Reuse one pooled sync client across channel-id resolution and transcript fetches
        # so warm Lambda containers keep connections alive instead of opening one per call.
        self._sync_client = httpx.Client(follow_redirects=True)

    def __del__(self) -> None:
        # Release pooled sockets when the collector is garbage-collected so warm Lambda
        # containers don't leak connections across invocations.
        client = getattr(self, "_sync_client", None)
        if client is not None:
            client.close()

    async def collect(self) -> list[CollectedItem]:
        if not self.config.channels:
            logger.info("No YouTube channels configured, skipping")
            return []

        tasks = [self._collect_channel(ch) for ch in self.config.channels]
        items = await gather_collector_results(tasks, labels=self.config.channels, raise_if_all_failed=True)
        logger.info("YouTube collector gathered %d items total", len(items))
        return items

    async def _collect_channel(self, channel_url: str) -> list[CollectedItem]:
        logger.info("Collecting videos from channel '%s'", channel_url)

        if self.api_key:
            return await self._collect_via_api(channel_url)
        return await self._collect_via_rss(channel_url)

    async def _collect_via_api(self, channel_url: str) -> list[CollectedItem]:
        channel_id = await self._resolve_channel_id_async(channel_url)
        if not channel_id:
            logger.warning("Could not resolve channel ID for '%s'", channel_url)
            return []

        uploads_playlist = f"UU{channel_id[2:]}"
        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)
        items: list[CollectedItem] = []

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            response = await retry_async(
                lambda: client.get(
                    f"{YOUTUBE_API_BASE}/playlistItems",
                    params={
                        "part": "snippet",
                        "playlistId": uploads_playlist,
                        "maxResults": self.config.max_videos_per_channel,
                        "key": self.api_key,
                    },
                ),
                max_retries=self.config.max_retries,
                backoff_sec=self.config.retry_backoff_sec,
                retry_on=(httpx.HTTPError,),
                description=f"YouTube playlistItems for '{channel_url}'",
            )
            if response.status_code != 200:
                logger.warning("YouTube API error for '%s': %d", channel_url, response.status_code)
                return []

            try:
                data = response.json()
            except ValueError:
                logger.warning("YouTube playlistItems for '%s' returned malformed JSON", channel_url, exc_info=True)
                return []
            video_ids = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                vid = snippet.get("resourceId", {}).get("videoId", "")
                if vid:
                    video_ids.append(vid)

            if not video_ids:
                return []

            details_resp = await retry_async(
                lambda: client.get(
                    f"{YOUTUBE_API_BASE}/videos",
                    params={
                        "part": "snippet,statistics,contentDetails",
                        "id": ",".join(video_ids),
                        "key": self.api_key,
                    },
                ),
                max_retries=self.config.max_retries,
                backoff_sec=self.config.retry_backoff_sec,
                retry_on=(httpx.HTTPError,),
                description=f"YouTube videos details for '{channel_url}'",
            )
            if details_resp.status_code != 200:
                logger.warning("YouTube API error for '%s': %d", channel_url, details_resp.status_code)
                return []

            try:
                details_data = details_resp.json()
            except ValueError:
                logger.warning("YouTube videos details for '%s' returned malformed JSON", channel_url, exc_info=True)
                return []

            for video in details_data.get("items", []):
                try:
                    snippet = video["snippet"]
                    stats = video.get("statistics", {})
                    video_id = video["id"]

                    published_str = snippet.get("publishedAt", "")
                    published_at = (
                        datetime.fromisoformat(published_str.replace("Z", "+00:00")) if published_str else None
                    )
                    if published_at and published_at < cutoff:
                        continue

                    transcript = await self._fetch_transcript(video_id)
                    text = transcript or snippet.get("description", "")

                    items.append(
                        CollectedItem(
                            item_id=video_id,
                            source_type=SourceType.YOUTUBE,
                            title=snippet.get("title", ""),
                            url=f"https://www.youtube.com/watch?v={video_id}",
                            text=text,
                            author=snippet.get("channelTitle", ""),
                            published_at=published_at,
                            metadata={
                                "view_count": int(stats.get("viewCount", 0)),
                                "channel_url": channel_url,
                            },
                        )
                    )
                    logger.info("Collected YouTube video: '%s'", snippet.get("title", ""))
                except (KeyError, ValueError, TypeError, AttributeError):
                    logger.warning("Failed to process YouTube video '%s'", video.get("id", ""), exc_info=True)

        return items

    async def _collect_via_rss(self, channel_url: str) -> list[CollectedItem]:
        channel_id = await self._resolve_channel_id_async(channel_url)
        if not channel_id:
            logger.warning("Could not resolve channel ID for '%s'", channel_url)
            return []

        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        feed = await asyncio.to_thread(feedparser.parse, get_proxied_url(rss_url))

        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)
        items: list[CollectedItem] = []

        for entry in feed.entries[: self.config.max_videos_per_channel]:
            try:
                video_id = entry.get("yt_videoid", "")
                if not video_id:
                    link = entry.get("link", "")
                    match = re.search(r"v=([a-zA-Z0-9_-]{11})", link)
                    video_id = match.group(1) if match else ""

                if not video_id:
                    continue

                published_at = parse_feed_published_date(entry)
                if published_at and published_at < cutoff:
                    continue

                transcript = await self._fetch_transcript(video_id)
                text = transcript or entry.get("summary", "")

                items.append(
                    CollectedItem(
                        item_id=video_id,
                        source_type=SourceType.YOUTUBE,
                        title=entry.get("title", ""),
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        text=text,
                        author=entry.get("author", ""),
                        published_at=published_at,
                        metadata={"channel_url": channel_url},
                    )
                )
                logger.info("Collected YouTube video: '%s'", entry.get("title", ""))
            except (KeyError, ValueError, TypeError, AttributeError):
                logger.warning("Failed to process YouTube RSS entry", exc_info=True)

        return items

    async def _resolve_channel_id_async(self, channel_url: str) -> str:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._resolve_channel_id, channel_url),
                timeout=self.config.resolve_timeout,
            )
        except TimeoutError:
            logger.warning("Channel ID resolution timed out for '%s', skipping", channel_url)
            return ""

    def _resolve_channel_id(self, channel_url: str) -> str:
        try:
            resp = self._sync_client.get(channel_url, timeout=self.config.resolve_timeout)
            match = re.search(r'"channelId":"(UC[a-zA-Z0-9_-]+)"', resp.text)
            if match:
                return match.group(1)
            match = re.search(r'channel_id=([^"&]+)', resp.text)
            if match:
                return match.group(1)
        except httpx.HTTPError as e:
            logger.warning("Failed to resolve channel ID for '%s': %s", channel_url, e)
        return ""

    async def _fetch_transcript(self, video_id: str) -> str:
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(self._get_transcript, video_id),
                timeout=self.config.transcript_timeout,
            )
        except TimeoutError:
            logger.warning("Transcript fetch timed out for video '%s', skipping", video_id)
            return ""

    def _get_transcript(self, video_id: str) -> str:
        try:
            if is_proxy_configured():
                proxy_url = get_proxied_url(
                    f"https://www.youtube.com/api/timedtext?v={video_id}&lang={self.config.transcript_language}"
                )
                resp = self._sync_client.get(proxy_url, timeout=self.config.transcript_timeout)
                if resp.status_code == 200 and resp.text.strip():

                    try:
                        data = json.loads(resp.text)
                        if isinstance(data, dict) and "events" in data:
                            return " ".join(
                                e.get("segs", [{}])[0].get("utf8", "") for e in data["events"] if e.get("segs")
                            )
                    except (json.JSONDecodeError, KeyError) as e:
                        logger.debug("Failed to parse proxied transcript JSON for video '%s': %s", video_id, e)

            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            return " ".join(snippet.text for snippet in transcript.snippets)
        except (
            YouTubeTranscriptApiException,
            httpx.HTTPError,
            ValueError,
            KeyError,
            TypeError,
            AttributeError,
            RuntimeError,
        ) as e:
            # RuntimeError is retained intentionally: youtube_transcript_api raises a variety
            # of runtime failures (region blocks, parser quirks) and transcript fetch is
            # best-effort — it must degrade to "" rather than fail the whole channel collect.
            logger.warning("Could not fetch transcript for video '%s': %s", video_id, e)
            return ""
