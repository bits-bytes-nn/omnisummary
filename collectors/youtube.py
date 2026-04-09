from __future__ import annotations

import asyncio
import json
import os
import re
from datetime import UTC, datetime
from typing import Any

import feedparser
import httpx
from youtube_transcript_api import YouTubeTranscriptApi

from shared import CollectedItem, SourceType, logger, parse_feed_published_date
from shared.config import YouTubeCollectorConfig
from shared.proxy import get_proxied_url, is_proxy_configured

from .base import BaseCollector, cutoff_datetime, gather_collector_results

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"

class YouTubeCollector(BaseCollector):
    def __init__(self, config: YouTubeCollectorConfig):
        self.config = config
        self.api_key = os.getenv("YOUTUBE_API_KEY", "")

    async def collect(self) -> list[CollectedItem]:
        if not self.config.channels:
            logger.info("No YouTube channels configured, skipping")
            return []

        tasks = [self._collect_channel(ch) for ch in self.config.channels]
        items = await gather_collector_results(tasks, labels=self.config.channels)
        logger.info("YouTube collector gathered %d items total", len(items))
        return items

    async def _collect_channel(self, channel_url: str) -> list[CollectedItem]:
        logger.info("Collecting videos from channel '%s'", channel_url)

        if self.api_key:
            return await self._collect_via_api(channel_url)
        return await self._collect_via_rss(channel_url)

    async def _collect_via_api(self, channel_url: str) -> list[CollectedItem]:
        channel_id = await asyncio.to_thread(self._resolve_channel_id, channel_url)
        if not channel_id:
            logger.warning("Could not resolve channel ID for '%s'", channel_url)
            return []

        uploads_playlist = f"UU{channel_id[2:]}"
        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)
        items: list[CollectedItem] = []

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(
                f"{YOUTUBE_API_BASE}/playlistItems",
                params={
                    "part": "snippet",
                    "playlistId": uploads_playlist,
                    "maxResults": self.config.max_videos_per_channel,
                    "key": self.api_key,
                },
            )
            if response.status_code != 200:
                logger.warning("YouTube API error for '%s': %d", channel_url, response.status_code)
                return []

            data = response.json()
            video_ids = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                vid = snippet.get("resourceId", {}).get("videoId", "")
                if vid:
                    video_ids.append(vid)

            if not video_ids:
                return []

            details_resp = await client.get(
                f"{YOUTUBE_API_BASE}/videos",
                params={
                    "part": "snippet,statistics,contentDetails",
                    "id": ",".join(video_ids),
                    "key": self.api_key,
                },
            )
            if details_resp.status_code != 200:
                return []

            for video in details_resp.json().get("items", []):
                try:
                    snippet = video["snippet"]
                    stats = video.get("statistics", {})
                    video_id = video["id"]

                    published_str = snippet.get("publishedAt", "")
                    published_at = datetime.fromisoformat(published_str.replace("Z", "+00:00")) if published_str else None
                    if published_at and published_at < cutoff:
                        continue

                    transcript = await asyncio.to_thread(self._get_transcript, video_id)
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
                except Exception:
                    logger.warning("Failed to process YouTube video '%s'", video.get("id", ""), exc_info=True)

        return items

    async def _collect_via_rss(self, channel_url: str) -> list[CollectedItem]:
        channel_id = await asyncio.to_thread(self._resolve_channel_id, channel_url)
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

                transcript = await asyncio.to_thread(self._get_transcript, video_id)
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
            except Exception:
                logger.warning("Failed to process YouTube RSS entry", exc_info=True)

        return items

    @staticmethod
    def _resolve_channel_id(channel_url: str) -> str:
        try:
            resp = httpx.get(channel_url, follow_redirects=True, timeout=15)
            match = re.search(r'"channelId":"(UC[a-zA-Z0-9_-]+)"', resp.text)
            if match:
                return match.group(1)
            match = re.search(r'channel_id=([^"&]+)', resp.text)
            if match:
                return match.group(1)
        except Exception:
            logger.warning("Failed to resolve channel ID for '%s'", channel_url)
        return ""

    @staticmethod
    def _get_transcript(video_id: str) -> str:
        try:
            if is_proxy_configured():
                proxy_url = get_proxied_url(
                    f"https://www.youtube.com/api/timedtext?v={video_id}&lang=en"
                )
                resp = httpx.get(proxy_url, timeout=15)
                if resp.status_code == 200 and resp.text.strip():

                    try:
                        data = json.loads(resp.text)
                        if isinstance(data, dict) and "events" in data:
                            return " ".join(e.get("segs", [{}])[0].get("utf8", "") for e in data["events"] if e.get("segs"))
                    except (json.JSONDecodeError, KeyError):
                        pass

            ytt_api = YouTubeTranscriptApi()
            transcript = ytt_api.fetch(video_id)
            return " ".join(snippet.text for snippet in transcript.snippets)
        except Exception:
            logger.warning("Could not fetch transcript for video '%s'", video_id)
            return ""
