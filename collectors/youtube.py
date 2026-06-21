from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime

import feedparser
import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from shared import CollectedItem, SourceType, logger, parse_feed_published_date, resolve_secret, retry_async
from shared.config import YouTubeCollectorConfig
from shared.proxy import get_proxied_url

from .base import BaseCollector, cutoff_datetime, gather_collector_results, load_items_from_s3

YOUTUBE_API_BASE = "https://www.googleapis.com/youtube/v3"
# Canonical YouTube channel ID: "UC" + 22 base64url chars. The uploads playlist is the
# same ID with the "UC" prefix swapped to "UU", so a valid UC id is required.
_CHANNEL_ID_PATTERN = re.compile(r'"channelId":"(UC[a-zA-Z0-9_-]{22})"')
# Extract the @handle from a channel URL (e.g. https://www.youtube.com/@AndrejKarpathy).
_HANDLE_PATTERN = re.compile(r"/@([A-Za-z0-9_.-]+)")
# Fetch this many recent uploads per channel, THEN filter to the lookback window and keep the
# latest max_videos_per_channel. The uploads playlist / RSS feed is NOT reliably newest-first
# (a scheduled/premiered video can sit below older ones), so taking only the top max_per_channel
# rows would drop a fresh video that ranks below a stale one — which is why low-cadence channels
# like Dwarkesh kept getting missed. Over-fetch + sort-by-date fixes that.
_CHANNEL_FETCH_DEPTH = 15


def _latest_within_window(items: list[CollectedItem], limit: int) -> list[CollectedItem]:
    """From the over-fetched per-channel items (already filtered to the lookback window), keep the
    `limit` most recent by published_at. Items with no published_at sort last (a missing date can't
    out-rank a real recent one). Decouples 'how many we look at' from 'how many we keep'."""
    _floor = datetime.min.replace(tzinfo=UTC)
    ordered = sorted(items, key=lambda i: i.published_at or _floor, reverse=True)
    return ordered[:limit]


class YouTubeCollector(BaseCollector):
    def __init__(self, config: YouTubeCollectorConfig):
        self.config = config
        self._api_key: str | None = None
        # Reuse one pooled sync client across channel-id resolution and transcript fetches
        # so warm Lambda containers keep connections alive instead of opening one per call.
        self._sync_client = httpx.Client(follow_redirects=True)

    @property
    def api_key(self) -> str:
        # Resolved lazily (env first, then SSM /{project}/{stage}/youtube-api-key) so
        # construction stays pure — no network I/O until the collector actually runs.
        if self._api_key is None:
            self._api_key = resolve_secret("YOUTUBE_API_KEY", "youtube-api-key")
        return self._api_key

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

        # YouTube blocks transcript fetches from datacenter (Lambda) IPs, so a local sync script
        # collects videos WITH transcripts on a residential IP and parks them in S3 (same pattern
        # as RSSHub/X). In AWS we read that file; live collection from Lambda still works for the
        # metadata but yields transcript-less items, so the S3 file is strongly preferred.
        s3_items = load_items_from_s3("youtube_items.json")
        if s3_items is not None:
            return s3_items

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
        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)
        items: list[CollectedItem] = []

        async with httpx.AsyncClient(timeout=self.config.request_timeout) as client:
            channel_id = await self._resolve_channel_id_via_api(channel_url, client)
            if not channel_id:
                # Fall back to the page scrape only if the API couldn't resolve (e.g. a URL with
                # no @handle). The API path works from datacenter IPs; the scrape does not.
                channel_id = await self._resolve_channel_id_async(channel_url)
            if not channel_id:
                # Raise (not return []) so an unresolvable channel registers as a FAILURE in
                # the health report, not a healthy-but-empty channel.
                raise RuntimeError(f"Could not resolve canonical channel ID for '{channel_url}'")

            uploads_playlist = f"UU{channel_id[2:]}"
            response = await retry_async(
                lambda: client.get(
                    f"{YOUTUBE_API_BASE}/playlistItems",
                    params={
                        "part": "snippet",
                        "playlistId": uploads_playlist,
                        "maxResults": _CHANNEL_FETCH_DEPTH,
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

            # Build window-filtered records WITHOUT transcripts first; the playlist isn't reliably
            # newest-first, so collect every in-window video, then keep the latest N and fetch
            # transcripts only for those (transcript calls are the expensive part).
            in_window: list[CollectedItem] = []
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

                    in_window.append(
                        CollectedItem(
                            item_id=video_id,
                            source_type=SourceType.YOUTUBE,
                            title=snippet.get("title", ""),
                            url=f"https://www.youtube.com/watch?v={video_id}",
                            text=snippet.get("description", ""),
                            author=snippet.get("channelTitle", ""),
                            published_at=published_at,
                            metadata={
                                "view_count": int(stats.get("viewCount", 0)),
                                "channel_url": channel_url,
                            },
                        )
                    )
                except (KeyError, ValueError, TypeError, AttributeError):
                    logger.warning("Failed to process YouTube video '%s'", video.get("id", ""), exc_info=True)

            # Keep the latest N within the window, THEN fetch transcripts only for those.
            kept = _latest_within_window(in_window, self.config.max_videos_per_channel)
            for item in kept:
                video_id = item.url.rsplit("=", 1)[-1]
                transcript = await self._fetch_transcript(video_id)
                if transcript:
                    item.text = transcript
                logger.info("Collected YouTube video: '%s'", item.title)
                items.append(item)

        return items

    async def _collect_via_rss(self, channel_url: str) -> list[CollectedItem]:
        channel_id = await self._resolve_channel_id_async(channel_url)
        if not channel_id:
            logger.warning("Could not resolve channel ID for '%s'", channel_url)
            return []

        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        feed = await asyncio.to_thread(feedparser.parse, get_proxied_url(rss_url))

        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)

        # The RSS feed isn't reliably newest-first either, so scan a fixed depth of entries,
        # collect every in-window one (no transcript yet), then keep the latest N and fetch
        # transcripts only for those — same over-fetch+sort approach as the API path.
        in_window: list[CollectedItem] = []
        for entry in feed.entries[:_CHANNEL_FETCH_DEPTH]:
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

                in_window.append(
                    CollectedItem(
                        item_id=video_id,
                        source_type=SourceType.YOUTUBE,
                        title=entry.get("title", ""),
                        url=f"https://www.youtube.com/watch?v={video_id}",
                        text=entry.get("summary", ""),
                        author=entry.get("author", ""),
                        published_at=published_at,
                        metadata={"channel_url": channel_url},
                    )
                )
            except (KeyError, ValueError, TypeError, AttributeError):
                logger.warning("Failed to process YouTube RSS entry", exc_info=True)

        items: list[CollectedItem] = []
        for item in _latest_within_window(in_window, self.config.max_videos_per_channel):
            video_id = item.url.rsplit("=", 1)[-1]
            transcript = await self._fetch_transcript(video_id)
            if transcript:
                item.text = transcript
            logger.info("Collected YouTube video: '%s'", item.title)
            items.append(item)

        return items

    async def _resolve_channel_id_via_api(self, channel_url: str, client: httpx.AsyncClient) -> str:
        """Resolve the canonical UC channel ID through the YouTube Data API's forHandle
        lookup. Unlike scraping the watch page (blocked / JS-shell on datacenter IPs), this
        works from Lambda. Returns "" if there's no @handle or the lookup fails."""
        match = _HANDLE_PATTERN.search(channel_url)
        if not match:
            return ""
        handle = match.group(1)
        try:
            resp = await retry_async(
                lambda: client.get(
                    f"{YOUTUBE_API_BASE}/channels",
                    params={"part": "id", "forHandle": handle, "key": self.api_key},
                ),
                max_retries=self.config.max_retries,
                backoff_sec=self.config.retry_backoff_sec,
                retry_on=(httpx.HTTPError,),
                description=f"YouTube channels forHandle '{handle}'",
            )
            if resp.status_code != 200:
                logger.warning("YouTube channels.forHandle '%s' returned %d", handle, resp.status_code)
                return ""
            items = resp.json().get("items", [])
            if items:
                return items[0].get("id", "")
            logger.warning("YouTube channels.forHandle '%s' found no channel", handle)
        except (httpx.HTTPError, ValueError, KeyError) as e:
            logger.warning("YouTube channels.forHandle '%s' failed: %s", handle, e)
        return ""

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
        # Only accept the canonical channel ID form (UC + 22 chars). The looser
        # `channel_id=...` fallback could capture a non-UC value, which then produced a
        # malformed `UU...` uploads-playlist ID and a silent empty result, so it's dropped.
        try:
            resp = self._sync_client.get(channel_url, timeout=self.config.resolve_timeout)
            match = _CHANNEL_ID_PATTERN.search(resp.text)
            if match:
                return match.group(1)
            logger.warning("No canonical channel ID found on page for '%s'", channel_url)
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
        # Run by the local sync (residential IP); YouTube blocks transcript fetches from
        # datacenter IPs, so in AWS this fails and the item keeps its description as body.
        return fetch_youtube_transcript(video_id, self.config.transcript_language)


def fetch_youtube_transcript(video_id: str, language: str = "en") -> str:
    """Fetch a video's transcript text (shared by the collector and the --pin-url path). Try the
    given language first, then fall back to ANY transcript the video has (non-English channels,
    auto-generated tracks) so a missing track isn't an empty body. Best-effort: any failure
    (incl. the IpBlocked YouTube throws from datacenter IPs) degrades to "" so the caller keeps
    the video's description as body rather than failing the whole collect."""
    try:
        ytt_api = YouTubeTranscriptApi()
        try:
            fetched = ytt_api.fetch(video_id, languages=(language,))
        except YouTubeTranscriptApiException:
            available = ytt_api.list(video_id)
            codes = [t.language_code for t in available]
            if not codes:
                raise
            fetched = available.find_transcript(codes).fetch()
        return " ".join(snippet.text for snippet in fetched.snippets)
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
