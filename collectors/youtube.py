from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any

import httpx
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import YouTubeTranscriptApiException

from shared import CollectedItem, SourceType, logger, resolve_secret, retry_async
from shared.config import YouTubeCollectorConfig

from .base import (
    RETRIABLE_STATUS_CODES,
    BaseCollector,
    TransientStatusError,
    cutoff_datetime,
    fetch_feed_with_retry,
    gather_collector_results,
    load_items_from_s3,
    parse_feed_entries,
)

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
# A YouTube video id: exactly 11 base64url chars.
_VIDEO_ID_PATTERN = re.compile(r"[a-zA-Z0-9_-]{11}")


def _entry_video_id(entry: Any, link: str) -> str:
    """The video id of a channel-feed entry: the feed's own yt_videoid, else read off the watch
    link. Used as the item id, and everything else about the item (the canonical watch URL, the
    transcript lookup) is derived from it."""
    video_id = entry.get("yt_videoid", "") or ""
    if video_id:
        return video_id
    match = re.search(r"v=([a-zA-Z0-9_-]{11})", link)
    return match.group(1) if match else ""


def _retry_after_delay(response: httpx.Response, cap_sec: float) -> float:
    """Server-requested backoff from a Retry-After header (delta-seconds or HTTP-date), clamped to
    cap_sec so an absurd value can't outlive the channel's timeout budget. 0 when absent/unusable."""
    raw = response.headers.get("Retry-After", "").strip()
    if not raw:
        return 0.0
    try:
        delay = float(raw)
    except ValueError:
        try:
            delay = (parsedate_to_datetime(raw) - datetime.now(UTC)).total_seconds()
        except (TypeError, ValueError):
            return 0.0
    return max(0.0, min(delay, cap_sec))


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
        # Resolved once per collect() (see _resolve_api_key), never lazily per access: the SSM
        # lookup is a blocking boto3 call, and reading it from a property inside the async fan-out
        # stalled the whole event loop for the duration of every channel's first access.
        self.api_key: str = ""
        self._sync_client_instance: httpx.Client | None = None

    @property
    def _sync_client(self) -> httpx.Client:
        # Created lazily so the S3-parked path (the common AWS case, which short-circuits before
        # any HTTP) never opens a client it won't use. Reused across channel-id resolution so a
        # warm Lambda container keeps the connection alive instead of opening one per call.
        if self._sync_client_instance is None:
            self._sync_client_instance = httpx.Client(follow_redirects=True)
        return self._sync_client_instance

    async def _resolve_api_key(self) -> str:
        """Resolve the key ONCE per run (env first, then SSM /{project}/{stage}/youtube-api-key)
        off the event loop. Construction stays pure — no I/O until the collector actually runs —
        and the blocking SSM call never runs on the loop thread."""
        if not self.api_key:
            self.api_key = await asyncio.to_thread(resolve_secret, "YOUTUBE_API_KEY", "youtube-api-key")
        return self.api_key

    def __del__(self) -> None:
        # Release pooled sockets when the collector is garbage-collected so warm Lambda
        # containers don't leak connections — only if a client was actually created.
        client = getattr(self, "_sync_client_instance", None)
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
        parked = load_items_from_s3("youtube_items.json", max_age_hours=self.config.park_max_age_hours)
        self.park_status = parked
        if parked.usable:
            # A FRESH park file says nothing about a sync that collected from 2 of 12 channels;
            # the meta block the sync writes does. Reporting only — every item is still returned.
            self.flag_degraded_park(
                parked,
                threshold=self.config.error_rate_threshold,
                empty_threshold=self.config.empty_rate_threshold,
                max_failed=self.config.max_failed_inputs,
                what="channels",
            )
            return parked.items

        # One resolution for the whole run, before the fan-out, so no channel task blocks the loop.
        await self._resolve_api_key()
        # Bound the fan-out. Each channel occupies worker threads (page scrape, transcript fetches)
        # and burns Data API quota, and with dozens of channels the default executor is
        # oversubscribed — a channel's timeout could expire while its work had not even started.
        # The semaphore is created HERE (on the running loop, never at import/__init__) and is
        # acquired BEFORE the per-channel timeout, so the timeout measures the fetch, not the queue
        # wait. Mirrors rss.max_concurrency / rsshub.max_concurrency.
        semaphore = asyncio.Semaphore(self.config.max_concurrency)
        tasks = [self._collect_channel_bounded(ch, semaphore) for ch in self.config.channels]
        result = await gather_collector_results(tasks, labels=self.config.channels, raise_if_all_failed=True)
        logger.info(
            "YouTube collector gathered %d items total from %d/%d channels (%d failed, %d empty)",
            len(result.items),
            result.total - result.failed - result.empty,
            result.total,
            result.failed,
            result.empty,
        )
        # Records run_meta for the local sync script to park alongside the items (so the Lambda-side
        # reader can see a half-dead sync) and reports the source DEGRADED past the failure rate.
        self.record_run_health(
            total=result.total,
            failed=result.failed,
            empty=result.empty,
            threshold=self.config.error_rate_threshold,
            empty_threshold=self.config.empty_rate_threshold,
            max_failed=self.config.max_failed_inputs,
            what="channels",
        )
        return result.items

    async def _collect_channel_bounded(self, channel_url: str, semaphore: asyncio.Semaphore) -> list[CollectedItem]:
        """Run one channel inside the fan-out bound and a real wall-clock budget. The per-step
        timeouts (resolve, transcript) left the API calls themselves unbounded, so a wedged channel
        could hold the digest until the Lambda itself timed out."""
        async with semaphore:
            budget = self.config.channel_budget_sec
            try:
                return await asyncio.wait_for(self._collect_channel(channel_url), timeout=budget)
            except TimeoutError as e:
                # Raise (not return []) so gather_collector_results counts this as a task FAILURE:
                # an all-channels-hung run then reports FAILED instead of a silent empty result.
                logger.warning("YouTube channel '%s' timed out after %ds, skipping", channel_url, budget)
                raise RuntimeError(f"YouTube channel '{channel_url}' timed out after {budget}s") from e

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
            response = await self._get_api(
                client,
                "playlistItems",
                {
                    "part": "snippet",
                    "playlistId": uploads_playlist,
                    "maxResults": _CHANNEL_FETCH_DEPTH,
                    "key": self.api_key,
                },
                description=f"YouTube playlistItems for '{channel_url}'",
            )
            # Raise (not return []) on an API rejection or a malformed body: these are FAILURES,
            # and gather_collector_results only escalates when EVERY channel failed — so a quota
            # exhaustion / revoked key across all channels reports FAILED instead of a silent EMPTY,
            # while one bad channel among many is still tolerated.
            if response.status_code != 200:
                raise RuntimeError(f"YouTube playlistItems for '{channel_url}' returned {response.status_code}")

            try:
                data = response.json()
            except ValueError as e:
                raise RuntimeError(f"YouTube playlistItems for '{channel_url}' returned malformed JSON") from e
            video_ids = []
            for item in data.get("items", []):
                snippet = item.get("snippet", {})
                vid = snippet.get("resourceId", {}).get("videoId", "")
                if vid:
                    video_ids.append(vid)

            if not video_ids:
                return []

            details_resp = await self._get_api(
                client,
                "videos",
                {
                    "part": "snippet,statistics,contentDetails",
                    "id": ",".join(video_ids),
                    "key": self.api_key,
                },
                description=f"YouTube videos details for '{channel_url}'",
            )
            if details_resp.status_code != 200:
                raise RuntimeError(f"YouTube videos details for '{channel_url}' returned {details_resp.status_code}")

            try:
                details_data = details_resp.json()
            except ValueError as e:
                raise RuntimeError(f"YouTube videos details for '{channel_url}' returned malformed JSON") from e

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
            # Same reasoning as the API path: an unresolvable channel is a failure, so an
            # all-channels-unresolvable run (the RSS-fallback outage) reports FAILED.
            raise RuntimeError(f"Could not resolve channel ID for '{channel_url}'")

        rss_url = f"https://www.youtube.com/feeds/videos.xml?channel_id={channel_id}"
        description = f"YouTube feed for '{channel_url}'"
        # Through the SHARED feed path every other feed source uses, instead of a raw
        # feedparser.parse(url): that fetches through urllib with no socket timeout (so one wedged
        # host held a worker thread that the outer wait_for cannot cancel, and asyncio.run joins the
        # executor at close — minutes of dead wall clock after the digest had already posted), never
        # retried a 5xx/DNS blip, and reported a truncated body as an EMPTY channel instead of a
        # failed one.
        feed = await fetch_feed_with_retry(
            rss_url,
            description=description,
            timeout=self.config.request_timeout,
            max_retries=self.config.max_retries,
            backoff_sec=self.config.retry_backoff_sec,
            proxy_fallback=True,
        )

        # The RSS feed isn't reliably newest-first either, so scan a fixed depth of entries,
        # collect every in-window one (no transcript yet), then keep the latest N and fetch
        # transcripts only for those — same over-fetch+sort approach as the API path.
        parsed = parse_feed_entries(
            feed,
            source_type=SourceType.YOUTUBE,
            cutoff=cutoff_datetime(self.config.lookback_hours, self.config.reference_time),
            description=description,
            metadata={"channel_url": channel_url},
            item_id_of=_entry_video_id,
            limit=_CHANNEL_FETCH_DEPTH,
        )
        # An entry whose video id can't be read is dropped: the transcript fetch and the canonical
        # watch URL are both derived from it, so a hashed fallback id would be useless downstream.
        in_window = [item for item in parsed if _VIDEO_ID_PATTERN.fullmatch(item.item_id)]

        items: list[CollectedItem] = []
        for item in _latest_within_window(in_window, self.config.max_videos_per_channel):
            item.url = f"https://www.youtube.com/watch?v={item.item_id}"
            transcript = await self._fetch_transcript(item.item_id)
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
            resp = await self._get_api(
                client,
                "channels",
                {"part": "id", "forHandle": handle, "key": self.api_key},
                description=f"YouTube channels forHandle '{handle}'",
            )
            if resp.status_code != 200:
                logger.warning("YouTube channels.forHandle '%s' returned %d", handle, resp.status_code)
                return ""
            items = resp.json().get("items", [])
            if items:
                return items[0].get("id", "")
            logger.warning("YouTube channels.forHandle '%s' found no channel", handle)
        except (httpx.HTTPError, TransientStatusError, ValueError, KeyError) as e:
            # A handle lookup that can't be completed (incl. an exhausted 429/5xx retry chain) is
            # not fatal here — the caller falls back to the page scrape.
            logger.warning("YouTube channels.forHandle '%s' failed: %s", handle, e)
        return ""

    async def _get_api(self, client: httpx.AsyncClient, path: str, params: dict, *, description: str) -> httpx.Response:
        """GET a YouTube Data API endpoint, retrying transport errors AND transient statuses
        (429 / 5xx, honouring a Retry-After capped inside the channel's budget). Permanent
        rejections (403 quota/key, 404 unknown resource) are returned as-is on the first response
        so the caller fails the channel immediately instead of retrying a verdict."""

        async def _call() -> httpx.Response:
            resp = await client.get(f"{YOUTUBE_API_BASE}/{path}", params=params)
            if resp.status_code in RETRIABLE_STATUS_CODES:
                delay = _retry_after_delay(resp, self.config.request_timeout)
                if delay:
                    logger.warning(
                        "%s returned %d; honouring Retry-After of %.0fs", description, resp.status_code, delay
                    )
                    await asyncio.sleep(delay)
                raise TransientStatusError(f"{description} returned {resp.status_code}")
            return resp

        return await retry_async(
            _call,
            max_retries=self.config.max_retries,
            backoff_sec=self.config.retry_backoff_sec,
            retry_on=(httpx.HTTPError, TransientStatusError),
            description=description,
        )

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
