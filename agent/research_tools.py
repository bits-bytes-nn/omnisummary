from __future__ import annotations

import asyncio

from strands import tool

# DeliveryContext + request scoping live in the delivery layer (output/), which owns the
# delivery contract; re-exported here for the agent entrypoints and tools that bind them.
from output.delivery import DeliveryContext, current_delivery_context, request_context
from shared import Config, logger
from shared.media import fetch_og_image
from shared.research import _search_papers, _tavily_search, extract_url

__all__ = [
    "DeliveryContext",
    "attach_image",
    "community_search",
    "current_delivery_context",
    "deliver_report",
    "read_url",
    "recall_trends",
    "request_context",
    "search_papers",
    "web_search",
]


@tool
async def web_search(query: str, recency: str = "general") -> str:
    """Search the open web for a query. Use recency="news" for recent industry/company/policy
    news, "general" otherwise.

    Args:
        query: The search query.
        recency: "general" (default) or "news" for recent news framing.
    """
    topic = "news" if recency.lower().strip() == "news" else None
    return await _tavily_search(query, topic=topic)


@tool
async def community_search(query: str) -> str:
    """Search community discussion (Reddit, X, Hacker News, Substack) for reactions, sentiment,
    and first-hand reports about a topic.

    Args:
        query: The search query.
    """
    domains = Config.load().agent.community_search_domains
    return await _tavily_search(query, include_domains=domains)


@tool
async def search_papers(query: str) -> str:
    """Search academic papers on Semantic Scholar for a research/technical claim.

    Args:
        query: The search query.
    """
    return await _search_papers(query)


@tool
async def read_url(url: str) -> str:
    """Fetch and read the full text of a specific web page (a primary source you found via
    search) so you can ground your report in its actual content.

    Args:
        url: The page URL to read.
    """
    return await extract_url(url)


@tool
async def recall_trends(query: str) -> str:
    """Recall related AI/ML trends tracked across earlier daily digests (cross-day memory).
    Use this for the "how has this evolved / what came before" angle.

    Args:
        query: What to recall (e.g. "open-weight model releases", "agent frameworks").
    """
    from datetime import date

    from shared import TRENDS_KEY, TrendMemory, create_state_store

    config = Config.load()
    top_k = config.agent.recall_memory_top_k
    half_life = config.pipeline.trend_momentum_half_life_days

    def _load() -> TrendMemory:
        try:
            store = create_state_store(config)
            raw = store.read(TRENDS_KEY) if store.exists(TRENDS_KEY) else None
        except Exception as e:
            logger.warning("Failed to open trend store for recall: %s", e)
            return TrendMemory()
        if not raw:
            return TrendMemory()
        try:
            return TrendMemory.model_validate_json(raw)
        except Exception as e:
            logger.warning("Failed to load trends for recall: %s", e)
            return TrendMemory()

    memory = await asyncio.to_thread(_load)
    matched = memory.search(query, today=date.today(), half_life_days=half_life, top_k=top_k)
    if not matched:
        return "No earlier trends recalled for that query."
    lines = [
        f"- *{t.title}* ({t.status.value}): " + "; ".join(f"[{ev.date}] {ev.summary}" for ev in t.evidence[-3:])
        for t in matched
    ]
    return "Earlier trends:\n\n" + "\n".join(lines)


@tool
async def attach_image(source_url: str) -> str:
    """Download the representative image (og:image) of a source page and stage it to be
    attached to the delivered report. Call this for the best, most on-topic source before
    delivering, so the post carries a real image from the article.

    Args:
        source_url: The article/source URL whose representative image to attach.
    """
    delivery = current_delivery_context()
    limit = Config.load().agent.research_max_staged_images
    if len(delivery.staged_images) >= limit:
        return f"Already staged the maximum of {limit} images; not attaching more."
    asset = await fetch_og_image(source_url)
    if not asset:
        return f"No usable image found for {source_url}."
    delivery.staged_images.append(asset)
    logger.info("Staged OG image from '%s' (%d bytes)", source_url, len(asset.data))
    return f"Attached image from {source_url} ({asset.content_type}, {len(asset.data)} bytes)."


@tool
async def deliver_report(report: str, channel: str = "slack") -> str:
    """Post the finished research report to a channel. Call this exactly once per target
    channel, only after the report is complete. Default channel is "slack"; use "threads"
    only when the user explicitly asked for Threads.

    Args:
        report: The final report text. For Slack use mrkdwn; for Threads use plain text.
        channel: "slack" (default) or "threads".
    """
    from output.delivery import deliver_research_report

    target = channel.lower().strip()
    if target not in ("slack", "threads"):
        # Surface the mistake so the agent can correct itself, rather than silently downgrading
        # an explicit Threads request to Slack.
        return f'Unknown channel "{channel}". Use "slack" or "threads".'

    delivery = current_delivery_context()
    ok = await deliver_research_report(report, channel=target, delivery=delivery)
    if not ok:
        return f"Failed to deliver the report to {target}."
    return f"Delivered the report to {target}."
