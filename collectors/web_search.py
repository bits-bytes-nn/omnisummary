from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal
from urllib.parse import urlparse

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
)
from shared.config import WebSearchCollectorConfig

from .base import BaseCollector, cutoff_datetime, gather_collector_results


class WebSearchCollector(BaseCollector):
    def __init__(self, config: WebSearchCollectorConfig, llm_factory: BedrockLanguageModelFactory | None = None):
        self.config = config
        self._client = AsyncTavilyClient()
        self._llm = llm_factory.get_model(config.refine_model) if llm_factory else None

    async def collect(self) -> list[CollectedItem]:
        if not self.config.enabled:
            logger.info("Web search collector is disabled, skipping")
            return []

        tasks: list[asyncio.Task] = []

        for trend in self.config.trend_searches:
            for query in trend.queries:
                tasks.append(asyncio.ensure_future(self._search_trend(query, trend.domains, trend.name, trend.topic)))

        if not tasks:
            logger.info("No web search queries or accounts configured, skipping")
            return []

        broad_items = self._deduplicate(await gather_collector_results(tasks))
        logger.info("Web search collector gathered %d items (broad phase)", len(broad_items))

        refined_items = await self._refine_search(broad_items)
        all_items = self._deduplicate(broad_items + refined_items)
        logger.info("Web search collector gathered %d items total (after refinement)", len(all_items))

        return all_items

    async def _refine_search(self, broad_items: list[CollectedItem]) -> list[CollectedItem]:
        if not broad_items or not self._llm:
            return []

        queries = await self._generate_refined_queries(broad_items)
        if not queries:
            return []

        logger.info("LLM-generated refined queries: %s", queries)
        tasks = [asyncio.ensure_future(self._search_trend(query, [], "refined")) for query in queries]
        return await gather_collector_results(tasks)

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
            raw = raw.strip()
            if raw.startswith("```"):
                raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            queries = json.loads(raw)
            if isinstance(queries, list):
                return [q for q in queries if isinstance(q, str)][: self.config.max_refine_queries]
        except Exception:
            logger.warning("Failed to generate refined queries via LLM", exc_info=True)

        return []

    async def _search_trend(
        self, query: str, domains: list[str], trend_name: str, topic: Literal["news", "general"] = "news"
    ) -> list[CollectedItem]:
        logger.info("Searching trend '%s' with query: '%s' (topic='%s')", trend_name, query, topic)
        days = max(1, self.config.lookback_hours // 24)
        try:
            response = await self._client.search(
                query=query,
                max_results=self.config.max_results_per_query,
                include_domains=domains if domains else None,
                topic=topic,
                days=days,
            )
            return self._parse_results(response, trend_name=trend_name)
        except Exception:
            logger.warning("Failed to search trend '%s' query '%s'", trend_name, query, exc_info=True)
            return []

    def _parse_results(
        self,
        response: dict,
        trend_name: str | None = None,
        platform: str | None = None,
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

                source_type = self._detect_source_type(url, platform)
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
            except Exception:
                logger.warning("Failed to process web search result", exc_info=True)

        return items

    @staticmethod
    def _detect_source_type(url: str, platform: str | None = None) -> SourceType:
        if platform:
            if platform.lower() in ("x", "twitter"):
                return SourceType.X

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


def _parse_date(date_str: str) -> datetime | None:
    try:
        return datetime.fromisoformat(date_str.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        pass
    try:
        return parsedate_to_datetime(date_str).astimezone(UTC)
    except (ValueError, TypeError):
        pass
    return None
