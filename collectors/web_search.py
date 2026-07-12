from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal
from urllib.parse import urlparse

import httpx
from langchain_core.output_parsers import StrOutputParser
from tavily import AsyncTavilyClient

from shared import (
    DOMAIN_TO_SOURCE,
    BedrockLanguageModelFactory,
    CollectedItem,
    RefineQueryPrompt,
    SourceType,
    generate_item_id,
    logger,
    parse_json_from_llm_output,
    resolve_secret,
    retry_async,
)
from shared.config import WebSearchCollectorConfig

from .base import BaseCollector, cutoff_datetime, gather_collector_results
from .youtube import YOUTUBE_API_BASE, fetch_youtube_transcript


class WebSearchCollector(BaseCollector):
    def __init__(self, config: WebSearchCollectorConfig, llm_factory: BedrockLanguageModelFactory | None = None):
        self.config = config
        self._api_key = ""
        self._client_instance: AsyncTavilyClient | None = None
        self._llm = llm_factory.get_model(config.refine_model) if llm_factory else None

    @property
    def _client(self) -> AsyncTavilyClient:
        # The key is resolved once in collect() (env -> SSM); reuse it here so a single
        # collect doesn't make repeated SSM round-trips.
        if self._client_instance is None:
            self._client_instance = AsyncTavilyClient(
                api_key=self._api_key or resolve_secret("TAVILY_API_KEY", "tavily-api-key")
            )
        return self._client_instance

    async def collect(self) -> list[CollectedItem]:
        if not self.config.enabled:
            logger.info("Web search collector is disabled, skipping")
            return []

        self._api_key = await asyncio.to_thread(resolve_secret, "TAVILY_API_KEY", "tavily-api-key")
        if not self._api_key:
            logger.warning("TAVILY_API_KEY not set, skipping web search collector")
            return []

        tasks: list[asyncio.Task] = []

        for trend in self.config.trend_searches:
            for query in trend.queries:
                tasks.append(asyncio.ensure_future(self._search_trend(query, trend.domains, trend.name, trend.topic)))

        if not tasks:
            logger.info("No web search queries or accounts configured, skipping")
            return []

        broad_items = self._deduplicate(await gather_collector_results(tasks, raise_if_all_failed=True))
        logger.info("Web search collector gathered %d items (broad phase)", len(broad_items))

        refined_items = await self._refine_search(broad_items)
        all_items = self._deduplicate(broad_items + refined_items)
        logger.info("Web search collector gathered %d items total (after refinement)", len(all_items))

        return all_items

    async def _refine_search(self, broad_items: list[CollectedItem]) -> list[CollectedItem]:
        if not self._llm:
            logger.info("Skipping web search refinement: refine LLM not configured (feature disabled)")
            return []
        if not broad_items:
            logger.info("Skipping web search refinement: no broad-phase items to refine from")
            return []

        queries = await self._generate_refined_queries(broad_items)
        if not queries:
            return []

        logger.info("LLM-generated refined queries: %s", queries)
        tasks = [asyncio.ensure_future(self._search_trend(query, [], "refined")) for query in queries]
        # Refinement is intentionally non-fatal (broad results are the floor), but a total
        # failure should be visible to ops so degraded refinement isn't silent.
        refined_items = await gather_collector_results(tasks)
        if tasks and not refined_items:
            logger.warning("All %d refined web-search queries returned no items; using broad results only", len(tasks))
        return refined_items

    async def _generate_refined_queries(self, items: list[CollectedItem]) -> list[str]:
        if not self._llm:
            return []
        titles = "\n".join(f"- {item.title}" for item in items)
        chain = RefineQueryPrompt.get_prompt() | self._llm | StrOutputParser()

        try:
            raw = await chain.ainvoke(
                {
                    "titles": titles,
                    "max_queries": self.config.max_refine_queries,
                }
            )
            queries = parse_json_from_llm_output(raw)
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str)][: self.config.max_refine_queries]
        except Exception:
            logger.warning("Failed to generate refined queries via LLM, falling back to broad results", exc_info=True)

        return []

    async def _search_trend(
        self, query: str, domains: list[str], trend_name: str, topic: Literal["news", "general"] = "news"
    ) -> list[CollectedItem]:
        logger.info("Searching trend '%s' with query: '%s' (topic='%s')", trend_name, query, topic)
        days = max(1, self.config.lookback_hours // 24)

        async def _search() -> dict:
            return await asyncio.wait_for(
                self._client.search(
                    query=query,
                    max_results=self.config.max_results_per_query,
                    include_domains=domains if domains else None,
                    topic=topic,
                    days=days,
                ),
                timeout=self.config.request_timeout,
            )

        response = await retry_async(
            _search,
            max_retries=self.config.max_retries,
            backoff_sec=self.config.retry_backoff_sec,
            retry_on=(Exception,),
            description=f"Tavily search for trend '{trend_name}' query '{query}'",
        )
        return self._parse_results(response, trend_name=trend_name)

    def _parse_results(
        self,
        response: dict,
        trend_name: str | None = None,
    ) -> list[CollectedItem]:
        cutoff = cutoff_datetime(self.config.lookback_hours, self.config.reference_time)
        items: list[CollectedItem] = []

        for result in response.get("results", []):
            try:
                url = result.get("url", "")
                title = result.get("title", "")
                content = result.get("content", "")
                published_date_str = result.get("published_date")

                if not published_date_str:
                    logger.debug("Skipping item without published_date: '%s'", title[:60])
                    continue

                published_at = _parse_date(published_date_str)
                if not published_at:
                    logger.debug("Unparseable date for: '%s'", title[:60])
                    continue
                if published_at < cutoff:
                    continue

                score = result.get("score")
                if score is not None and score < self.config.min_search_score:
                    logger.debug("Skipping low-relevance result (%.3f): '%s'", score, title[:60])
                    continue

                source_type = self._detect_source_type(url)
                item_id = generate_item_id(url)

                items.append(
                    CollectedItem(
                        item_id=item_id,
                        source_type=source_type,
                        title=title,
                        url=url,
                        text=content,
                        published_at=published_at,
                        metadata={"trend_name": trend_name, "search_score": result.get("score")},
                    )
                )
                logger.info("Collected web result: '%s'", title)
            except (AttributeError, KeyError, TypeError, ValueError):
                logger.warning("Failed to process web search result", exc_info=True)

        return items

    @staticmethod
    def _detect_source_type(url: str) -> SourceType:
        parsed = urlparse(url)
        domain = parsed.netloc.lower().removeprefix("www.")
        return DOMAIN_TO_SOURCE.get(domain, SourceType.WEB)

    @staticmethod
    def _deduplicate(items: list[CollectedItem]) -> list[CollectedItem]:
        seen_urls: set[str] = set()
        unique: list[CollectedItem] = []
        for item in items:
            if item.url not in seen_urls:
                seen_urls.add(item.url)
                unique.append(item)
        return unique


def _title_from_url(url: str) -> str:
    """Fallback title when the extractor returns none. The last path segment works for
    article-style URLs (.../post/some-headline → 'some headline'); for URLs whose tail is an
    opaque id (YouTube /watch?v=ID, X /user/status/ID) it would be meaningless, so fall back
    to the host. The ranker/digest still see the full extracted body either way."""
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    slug = parsed.path.rstrip("/").rsplit("/", 1)[-1]
    opaque = (not slug) or slug.isdigit() or slug in {"watch", "status", "video", "v"}
    if opaque:
        return host or url
    return slug.replace("-", " ").replace("_", " ").strip() or host or url


_YOUTUBE_ID_RE = re.compile(r"(?:v=|/shorts/|youtu\.be/|/embed/)([A-Za-z0-9_-]{11})")
_YOUTUBE_HOSTS = {"youtube.com", "m.youtube.com", "youtu.be"}


def _youtube_video_id(url: str) -> str:
    """Pull the 11-char video id out of any YouTube URL form (watch?v=, youtu.be/, /shorts/,
    /embed/). Returns "" for a non-YouTube or id-less URL so the caller routes it to Tavily.
    (DOMAIN_TO_SOURCE doesn't map youtube.com — it's the collector's own source — so match on
    the host directly rather than _detect_source_type.)"""
    host = urlparse(url).netloc.lower().removeprefix("www.")
    if host not in _YOUTUBE_HOSTS:
        return ""
    match = _YOUTUBE_ID_RE.search(url)
    return match.group(1) if match else ""


async def _fetch_youtube_pinned(url: str, video_id: str) -> CollectedItem | None:
    """Resolve a pinned YouTube URL through the YouTube Data API + transcript API instead of
    Tavily. Tavily's extractor only returns a video page's metadata, never the spoken content,
    and YouTube videos are a common pin target — so go to the source: title + description from
    the Data API, then the transcript (best-effort; YouTube blocks transcript fetches from
    datacenter IPs, so it may be empty, in which case the description carries the body)."""
    api_key = await asyncio.to_thread(resolve_secret, "YOUTUBE_API_KEY", "youtube-api-key")
    if not api_key:
        logger.warning("YOUTUBE_API_KEY not set, cannot fetch pinned YouTube URL '%s'", url)
        return None
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(
                f"{YOUTUBE_API_BASE}/videos",
                params={"part": "snippet", "id": video_id, "key": api_key},
            )
        if resp.status_code != 200:
            logger.warning("YouTube Data API returned %d for pinned '%s'", resp.status_code, url)
            return None
        results = resp.json().get("items", [])
        if not results:
            logger.warning("YouTube Data API found no video for pinned '%s'", url)
            return None
        snippet = results[0].get("snippet", {})
    except (httpx.HTTPError, ValueError, KeyError) as e:
        logger.warning("Failed to fetch pinned YouTube URL '%s': %s", url, e)
        return None

    title = (snippet.get("title") or "").strip() or _title_from_url(url)
    body = snippet.get("description", "")
    transcript = await asyncio.to_thread(fetch_youtube_transcript, video_id)
    if transcript:
        body = transcript
    logger.info("Fetched pinned YouTube URL: '%s' (%s)", url, title)
    return CollectedItem(
        item_id=generate_item_id(url),
        source_type=SourceType.YOUTUBE,
        title=title,
        url=url,
        text=body,
        author=snippet.get("channelTitle", ""),
        metadata={"pinned": True},
    )


async def _fetch_tavily_pinned(urls: list[str]) -> list[CollectedItem]:
    """Resolve non-YouTube pinned URLs via Tavily extract (page text)."""
    api_key = await asyncio.to_thread(resolve_secret, "TAVILY_API_KEY", "tavily-api-key")
    if not api_key:
        logger.warning("TAVILY_API_KEY not set, cannot fetch pinned URLs")
        return []

    client = AsyncTavilyClient(api_key=api_key)
    items: list[CollectedItem] = []
    try:
        response = await client.extract(urls=urls, format="text")
    except Exception:
        logger.warning("Failed to extract pinned URLs", exc_info=True)
        return []

    for result in response.get("results", []):
        url = result.get("url", "")
        content = result.get("raw_content") or result.get("content") or ""
        if not url:
            continue
        # Prefer the extractor's own title (it reads the page's <title>/og:title); fall back to
        # the URL slug only when absent. A pinned item is included regardless of score.
        title = (result.get("title") or "").strip() or _title_from_url(url)
        items.append(
            CollectedItem(
                item_id=generate_item_id(url),
                source_type=WebSearchCollector._detect_source_type(url),
                title=title,
                url=url,
                text=content,
                metadata={"pinned": True},
            )
        )
        logger.info("Fetched pinned URL: '%s' (%s)", url, title)
    return items


async def fetch_pinned_items(urls: list[str]) -> list[CollectedItem]:
    """Fetch user-specified URLs (via `--pin-url`) as CollectedItems so the pipeline can force
    them into the digest regardless of ranking score. YouTube URLs go through the YouTube Data
    API (Tavily only sees a video page's metadata, never its content); everything else goes
    through Tavily extract. Best-effort: a URL that can't be fetched is logged and skipped,
    never aborting the run. Returned items carry metadata['pinned']=True so the ranker
    guarantees them a slot."""
    urls = [u.strip() for u in (urls or []) if u and u.strip()]
    if not urls:
        return []

    youtube_pins = {u: vid for u in urls if (vid := _youtube_video_id(u))}
    other_urls = [u for u in urls if u not in youtube_pins]

    items: list[CollectedItem] = []
    yt_results = await asyncio.gather(*(_fetch_youtube_pinned(u, vid) for u, vid in youtube_pins.items()))
    items.extend(it for it in yt_results if it is not None)
    if other_urls:
        items.extend(await _fetch_tavily_pinned(other_urls))

    # Surface exactly which pinned URLs didn't come back, so a silently-missing pin is visible.
    fetched_urls = {it.url for it in items}
    missing = [u for u in urls if u not in fetched_urls]
    if missing:
        logger.warning("%d of %d pinned URL(s) could not be fetched: %s", len(missing), len(urls), missing)
    return items


def _parse_date(date_str: str) -> datetime | None:
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError) as e:
        logger.debug("ISO date parse failed for '%s' (%s); trying RFC2822", date_str, e)
    try:
        return parsedate_to_datetime(date_str).astimezone(UTC)
    except (ValueError, TypeError) as e:
        logger.debug("RFC2822 date parse failed for '%s' (%s); giving up", date_str, e)
    return None
