from __future__ import annotations

import asyncio
import os
from typing import Any

import httpx
from strands import tool

from shared import logger

from .tool_state import DigestStateManager

COMMUNITY_SEARCH_DOMAINS = ["twitter.com", "x.com", "reddit.com", "news.ycombinator.com", "substack.com"]

state_manager = DigestStateManager()


def _get_tavily_client() -> Any | None:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return None
    from tavily import AsyncTavilyClient

    return AsyncTavilyClient(api_key=api_key)


@tool
def get_detail(item_number: int, query: str = "") -> str:
    """Get detailed analysis of a digest item by its number.

    Args:
        item_number: The item number from the digest (1-based, e.g. 1, 2, 3)
        query: Optional specific question about the item
    """
    ranked = state_manager.get_item_by_number(item_number)
    if not ranked:
        total = state_manager.get_item_count()
        return f"Item {item_number} not found. Today's digest has {total} items."

    item = ranked.item
    detail = (
        f"=== Item {item_number} ===\n"
        f"Title: {item.title}\n"
        f"Source: {item.source_type.value}\n"
        f"URL: {item.url}\n"
        f"Author: {item.author or 'Unknown'}\n"
        f"Score: {ranked.score:.2f}\n"
        f"Categories: {', '.join(ranked.categories)}\n"
        f"Reasoning: {ranked.reasoning}\n\n"
        f"Content:\n{item.text[:8000]}"
    )
    if query:
        detail += f"\n\nUser question: {query}"

    logger.info("Retrieved detail for item #%d: '%s'", item_number, item.title[:50])
    return detail


@tool
async def search_papers(query: str) -> str:
    """Search for related academic papers on Semantic Scholar.

    Args:
        query: Search query for finding related papers
    """
    async with httpx.AsyncClient(timeout=30) as client:
        last_error = ""
        for attempt in range(3):
            try:
                response = await client.get(
                    "https://api.semanticscholar.org/graph/v1/paper/search",
                    params={
                        "query": query,
                        "limit": 5,
                        "fields": "title,year,authors,url,abstract",
                    },
                )
            except httpx.HTTPError as e:
                logger.warning("Semantic Scholar API request failed: %s", e)
                return f"Search request failed: {e}"

            if response.status_code == 429:
                last_error = "Rate limited by Semantic Scholar API"
                logger.warning("Semantic Scholar rate limited (attempt %d/3)", attempt + 1)
                await asyncio.sleep(2 * (attempt + 1))
                continue

            if response.status_code != 200:
                logger.warning("Semantic Scholar API returned status %d", response.status_code)
                return f"Search failed (status {response.status_code})"

            break
        else:
            return f"SEARCH_FAILED: {last_error}. Could not retrieve papers."

        try:
            data = response.json()
        except Exception:
            return "SEARCH_FAILED: Invalid response from Semantic Scholar API."
        papers = data.get("data", [])
        if not papers:
            return "No related papers found."

        results: list[str] = []
        for p in papers:
            authors = ", ".join(a["name"] for a in (p.get("authors") or [])[:3])
            abstract = (p.get("abstract") or "")[:200]
            results.append(
                f"- {p.get('title', 'N/A')} ({p.get('year', 'N/A')}) by {authors}\n"
                f"  URL: {p.get('url', '')}\n"
                f"  Abstract: {abstract}"
            )

        logger.info("Found %d papers for query '%s'", len(papers), query)
        return "\n\n".join(results)


@tool
async def search_community(query: str) -> str:
    """Search for community discussions about a topic (Reddit, X, HN, Substack).

    Args:
        query: Search query for community discussions
    """
    client = _get_tavily_client()
    if not client:
        return "TAVILY_API_KEY not configured."

    try:
        response = await client.search(
            query=query,
            max_results=5,
            include_domains=COMMUNITY_SEARCH_DOMAINS,
        )
        results = response.get("results", [])
        if not results:
            return "No community discussions found."

        formatted: list[str] = []
        for r in results:
            formatted.append(
                f"- {r.get('title', 'N/A')}\n" f"  URL: {r.get('url', '')}\n" f"  Content: {r.get('content', '')[:300]}"
            )

        logger.info("Found %d community discussions for query '%s'", len(results), query)
        return "\n\n".join(formatted)
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return f"Search failed: {e}"


@tool
async def search_related_news(query: str) -> str:
    """Search for related news and blog posts (no domain restriction).

    Args:
        query: Search query for related news articles
    """
    client = _get_tavily_client()
    if not client:
        return "TAVILY_API_KEY not configured."

    try:
        response = await client.search(
            query=query,
            max_results=5,
            topic="news",
        )
        results = response.get("results", [])
        if not results:
            return "No related news found."

        formatted: list[str] = []
        for r in results:
            formatted.append(
                f"- {r.get('title', 'N/A')}\n" f"  URL: {r.get('url', '')}\n" f"  Content: {r.get('content', '')[:300]}"
            )

        logger.info("Found %d related news for query '%s'", len(results), query)
        return "\n\n".join(formatted)
    except Exception as e:
        logger.warning("Tavily news search failed: %s", e)
        return f"Search failed: {e}"
