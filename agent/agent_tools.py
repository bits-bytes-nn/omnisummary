from __future__ import annotations

import asyncio
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

import httpx
from strands import tool
from tavily import AsyncTavilyClient

from shared import LOGGING_TRUNCATION_CHARS, Config, format_collected_item, logger, resolve_secret, retry_async

from .tool_state import DigestStateManager


@dataclass
class DeliveryContext:
    """Slack delivery target for tools that produce media (set per-invocation)."""

    channel_id: str = ""
    thread_ts: str = ""


# Module-level defaults (used by tests and single-threaded local runs). In a warm
# AgentCore container two concurrent invocations would race on these, so the runtime
# binds per-request copies via contextvars; tools resolve through the accessors below.
state_manager = DigestStateManager()
delivery_context = DeliveryContext()

_request_state: ContextVar[DigestStateManager | None] = ContextVar("request_state", default=None)
_request_delivery: ContextVar[DeliveryContext | None] = ContextVar("request_delivery", default=None)


def current_state_manager() -> DigestStateManager:
    return _request_state.get() or state_manager


def current_delivery_context() -> DeliveryContext:
    return _request_delivery.get() or delivery_context


@contextmanager
def request_context(state: DigestStateManager, delivery: DeliveryContext):
    """Bind per-invocation state so concurrent invocations don't share globals."""
    state_token = _request_state.set(state)
    delivery_token = _request_delivery.set(delivery)
    try:
        yield
    finally:
        _request_state.reset(state_token)
        _request_delivery.reset(delivery_token)


def _get_tavily_client() -> AsyncTavilyClient | None:
    # env first, then SSM SecureString — so search works in the AgentCore runtime and the
    # visual Lambda, which carry the key in SSM rather than the environment.
    api_key = resolve_secret("TAVILY_API_KEY", "tavily-api-key")
    if not api_key:
        return None

    return AsyncTavilyClient(api_key=api_key)


def _format_search_results(results: list[dict], preview_chars: int) -> str:
    return "\n\n".join(
        f"- {r.get('title', 'N/A')}\n  URL: {r.get('url', '')}\n  Content: {r.get('content', '')[:preview_chars]}"
        for r in results
    )


async def _tavily_search(query: str, *, topic: str | None = None, include_domains: list[str] | None = None) -> str:
    client = _get_tavily_client()
    if not client:
        return "TAVILY_API_KEY not configured."

    agent_config = Config.load().agent
    kwargs: dict = {"query": query, "max_results": agent_config.search_result_limit}
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
        return _format_search_results(results, agent_config.search_content_preview_chars)
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
    state = current_state_manager()
    ranked = state.get_item_by_number(item_number)
    if not ranked:
        total = state.get_item_count()
        return f"Item {item_number} not found. Today's digest has {total} items."

    item = ranked.item
    max_tokens = Config.load().agent.detail_max_tokens
    fields = [
        ("Title", item.title),
        ("Source", item.source_type.value),
        ("URL", item.url),
        ("Author", item.author or "Unknown"),
        ("Score", f"{ranked.score:.2f}"),
        ("Categories", ", ".join(ranked.categories)),
        ("Reasoning", ranked.reasoning),
    ]
    detail = format_collected_item(
        item, index=item_number, max_tokens=max_tokens, fields=fields, text_label="Content"
    ).rstrip("\n")
    if query:
        detail += f"\n\nUser question: {query}"

    logger.info(
        "Retrieved detail for item #%d: '%s'", item_number, item.title[: LOGGING_TRUNCATION_CHARS["title_short"]]
    )
    return detail


async def _search_papers(query: str) -> str:
    agent_config = Config.load().agent
    async with httpx.AsyncClient(timeout=agent_config.search_request_timeout) as client:

        async def _fetch() -> httpx.Response:
            resp = await client.get(
                "https://api.semanticscholar.org/graph/v1/paper/search",
                params={
                    "query": query,
                    "limit": agent_config.search_result_limit,
                    "fields": "title,year,authors,url,abstract",
                },
            )
            if resp.status_code == 429:
                raise httpx.HTTPStatusError("Rate limited by Semantic Scholar API", request=resp.request, response=resp)
            return resp

        try:
            response = await retry_async(
                _fetch,
                max_retries=agent_config.search_max_retries,
                backoff_sec=agent_config.search_retry_backoff_sec,
                retry_on=(httpx.HTTPStatusError,),
                description="Semantic Scholar paper search",
            )
        except httpx.HTTPStatusError:
            return "SEARCH_FAILED: Rate limited by Semantic Scholar API. Could not retrieve papers."
        except httpx.HTTPError as e:
            logger.warning("Semantic Scholar API request failed: %s", e)
            return f"Search request failed: {e}"

        if response.status_code != 200:
            logger.warning("Semantic Scholar API returned status %d", response.status_code)
            return f"Search failed (status {response.status_code})"

        try:
            data = response.json()
        except Exception:
            return "SEARCH_FAILED: Invalid response from Semantic Scholar API."
        papers = data.get("data", [])
        if not papers:
            return "No related papers found."

        results: list[str] = []
        for p in papers:
            authors = ", ".join(a["name"] for a in (p.get("authors") or [])[: agent_config.search_paper_max_authors])
            abstract = (p.get("abstract") or "")[: agent_config.search_paper_abstract_max_chars]
            results.append(
                f"- {p.get('title', 'N/A')} ({p.get('year', 'N/A')}) by {authors}\n"
                f"  URL: {p.get('url', '')}\n"
                f"  Abstract: {abstract}"
            )

        logger.info("Found %d papers for query '%s'", len(papers), query)
        return "\n\n".join(results)


@tool
async def search_papers(query: str) -> str:
    """Search for related academic papers on Semantic Scholar.

    Args:
        query: Search query for finding related papers
    """
    return await _search_papers(query)


@tool
async def search_community(query: str) -> str:
    """Search for community discussions about a topic (Reddit, X, HN, Substack).

    Args:
        query: Search query for community discussions
    """
    domains = Config.load().agent.community_search_domains
    return await _tavily_search(query, include_domains=domains)


@tool
async def search_related_news(query: str) -> str:
    """Search for related news and blog posts (no domain restriction).

    Args:
        query: Search query for related news articles
    """
    return await _tavily_search(query, topic="news")


@tool
async def recall_trends(query: str) -> str:
    """Recall related AI/ML trends tracked across earlier digests (cross-day memory).

    Use this when the user asks how a topic has evolved, what was covered before,
    or for historical context beyond today's digest.

    Args:
        query: What to recall (e.g. "open-weight model releases", "agent frameworks")
    """
    from datetime import date

    from pipeline.trend_tracker import TRENDS_KEY
    from shared import TrendMemory, create_state_store

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


def _build_llm_factory():
    import boto3

    from shared import BedrockLanguageModelFactory, is_running_in_aws

    config = Config.load()
    if is_running_in_aws():
        session = boto3.Session(region_name=config.aws.bedrock_region)
    else:
        session = boto3.Session(
            region_name=config.aws.bedrock_region,
            profile_name=config.aws.profile or None,
        )
    return BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region), config


@tool
async def make_visual(instruction: str, item_number: int = 0, context: str = "") -> str:
    """Generate an image from a free-form instruction and post it to Slack.

    The visual format is entirely up to you: a one-page presentation slide, an N-panel
    comic, a concept diagram, an infographic, a poster — describe what you want in
    `instruction`. First gather any extra material yourself (search_papers /
    search_related_news / search_community / get_detail) and pass it via `context`.

    Args:
        instruction: Natural-language description of the image to create (format, content, style).
        item_number: Optional digest item (1-based) to use as the source material.
        context: Optional extra research/notes you gathered to ground the visual.
    """
    from agent.visuals import VisualGenerator
    from output.slack_handler import send_image_to_slack

    if not await asyncio.to_thread(resolve_secret, "OPENAI_API_KEY", "openai-api-key"):
        return "Visualization is disabled (OPENAI_API_KEY not configured)."

    # Fail fast before the paid image generation if there's nowhere to post it.
    delivery = current_delivery_context()
    if not delivery.channel_id:
        return "No Slack channel is set for delivery; skipping image generation."

    state = current_state_manager()
    source = ""
    if item_number:
        ranked = state.get_item_by_number(item_number)
        if not ranked:
            return f"Item {item_number} not found. Today's digest has {state.get_item_count()} items."
        source = f"{ranked.item.title}\n\n{ranked.item.text}"

    factory, config = _build_llm_factory()
    generator = VisualGenerator(
        factory,
        config.pipeline.digest_model,
        image_model=config.pipeline.image_model,
        image_sizes=config.pipeline.image_sizes,
        source_max_tokens=config.pipeline.visual_synopsis_source_max_tokens,
        context_max_tokens=config.pipeline.visual_synopsis_context_max_tokens,
        caption_language=config.pipeline.visual_caption_language,
        on_image_language=config.pipeline.visual_on_image_language,
        moderation_softening_instruction=config.pipeline.visual_moderation_softening_instruction,
        style_guidance=config.pipeline.visual_synopsis_style_guidance,
        humor_guidance=config.pipeline.visual_synopsis_humor_guidance,
        style_aesthetic=config.pipeline.visual_synopsis_style_aesthetic,
    )

    try:
        image_bytes, brief = await generator.generate(instruction, source, context)
    except Exception as e:
        logger.error("Visualization failed: %s", e, exc_info=True)
        return f"Visualization failed: {e}"

    visual_title = brief.title
    caption = brief.caption
    emoji = config.pipeline.visual_caption_emoji
    uploaded = await send_image_to_slack(
        image_bytes,
        channel_id=delivery.channel_id,
        title=visual_title,
        comment=f"{emoji} *{visual_title}*\n{caption}",
        thread_ts=delivery.thread_ts,
    )
    if not uploaded:
        return "Visual generated but Slack upload failed."
    return f"Posted a visual to Slack: '{visual_title}'."
