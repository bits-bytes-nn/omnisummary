from __future__ import annotations

import asyncio
import os

import httpx
from strands import tool
from tavily import AsyncTavilyClient

from shared import logger

from .tool_state import DigestStateManager

COMMUNITY_SEARCH_DOMAINS = ["twitter.com", "x.com", "reddit.com", "news.ycombinator.com", "substack.com"]

state_manager = DigestStateManager()


class DeliveryContext:
    """Slack delivery target for tools that produce media (set per-invocation)."""

    channel_id: str = ""
    thread_ts: str = ""


delivery_context = DeliveryContext()


def _get_tavily_client() -> AsyncTavilyClient | None:
    api_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        return None

    return AsyncTavilyClient(api_key=api_key)


def _format_search_results(results: list[dict]) -> str:
    return "\n\n".join(
        f"- {r.get('title', 'N/A')}\n  URL: {r.get('url', '')}\n  Content: {r.get('content', '')[:300]}"
        for r in results
    )


async def _tavily_search(query: str, *, topic: str | None = None, include_domains: list[str] | None = None) -> str:
    client = _get_tavily_client()
    if not client:
        return "TAVILY_API_KEY not configured."

    kwargs: dict = {"query": query, "max_results": 5}
    if topic:
        kwargs["topic"] = topic
    if include_domains:
        kwargs["include_domains"] = include_domains

    try:
        response = await client.search(**kwargs)
        results = response.get("results", [])
        if not results:
            return "No results found."
        logger.info("Tavily search found %d results for query '%s'", len(results), query)
        return _format_search_results(results)
    except Exception as e:
        logger.warning("Tavily search failed: %s", e)
        return f"Search failed: {e}"


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
    return await _tavily_search(query, include_domains=COMMUNITY_SEARCH_DOMAINS)


@tool
async def search_related_news(query: str) -> str:
    """Search for related news and blog posts (no domain restriction).

    Args:
        query: Search query for related news articles
    """
    return await _tavily_search(query, topic="news")


def _build_llm_factory():
    import boto3

    from shared import BedrockLanguageModelFactory, Config, is_running_in_aws

    config = Config.load()
    if is_running_in_aws():
        session = boto3.Session(region_name=config.aws.bedrock_region)
    else:
        session = boto3.Session(
            region_name=config.aws.bedrock_region,
            profile_name=config.aws.profile or None,
        )
    return BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region), config


def _brief_caption(mode: str, brief: dict) -> str:
    if mode == "comic":
        return "\n".join(f"{i + 1}. {p.get('caption', '')}" for i, p in enumerate(brief.get("panels", [])))
    return brief.get("visual", "")[:300]


@tool
async def make_visual(item_number: int, mode: str = "comic", panels: int = 4) -> str:
    """Create a visualization that helps explain a digest item, and post it to Slack.

    Use this when the user asks for a cartoon/comic, an illustration, a diagram, or a
    visual explanation of an item.

    Args:
        item_number: The digest item number (1-based) to visualize
        mode: "comic" for a narrative cartoon, or "diagram" for an explanatory concept diagram
        panels: For comic mode, how many panels to draw (1-6); pick what fits the story
    """
    from agent.visuals import MODES, VisualGenerator
    from output.slack_handler import send_image_to_slack

    if mode not in MODES:
        return f"mode must be one of {sorted(MODES)}."

    ranked = state_manager.get_item_by_number(item_number)
    if not ranked:
        return f"Item {item_number} not found. Today's digest has {state_manager.get_item_count()} items."

    if not os.getenv("OPENAI_API_KEY"):
        return "Visualization is disabled (OPENAI_API_KEY not configured)."

    item = ranked.item
    factory, config = _build_llm_factory()
    generator = VisualGenerator(factory, config.pipeline.digest_model)

    try:
        image_bytes, brief = await generator.generate(item.title, item.text, mode=mode, panels=panels)
    except Exception as e:
        logger.error("Visualization failed: %s", e, exc_info=True)
        return f"Visualization failed: {e}"

    if not delivery_context.channel_id:
        return "Visual generated but no Slack channel is set for delivery."

    visual_title = brief.get("title", item.title)
    uploaded = await send_image_to_slack(
        image_bytes,
        channel_id=delivery_context.channel_id,
        title=visual_title,
        comment=f"*{visual_title}*\n{_brief_caption(mode, brief)}",
        thread_ts=delivery_context.thread_ts,
    )
    if not uploaded:
        return "Visual generated but Slack upload failed."
    descriptor = f"{panels}-panel comic" if mode == "comic" else "diagram"
    return f"Posted a {descriptor} for item {item_number}: '{visual_title}'."
