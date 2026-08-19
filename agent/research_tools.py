from __future__ import annotations

import asyncio

from strands import tool

# DeliveryContext + request scoping live in the delivery layer (output/), which owns the
# delivery contract; re-exported here for the agent entrypoints and tools that bind them.
from output.delivery import DeliveryContext, current_delivery_context, request_context
from shared import extract_urls, get_config, logger, normalize_citation_url
from shared.media import fetch_og_image
from shared.research import extract_url, semantic_scholar_search, tavily_search

__all__ = [
    "DeliveryContext",
    "attach_image",
    "community_search",
    "current_delivery_context",
    "deliver_report",
    "read_url",
    "recall_digest",
    "recall_trends",
    "request_context",
    "search_papers",
    "web_search",
]


def _record_seen_urls(text: str, *extra: str) -> str:
    """Remember every URL a tool actually surfaced, then return the text unchanged.

    deliver_report compares the report's citations against this set, so a URL the model invented can
    be refused before it is published. Recording happens at the TOOL boundary rather than inside the
    research backends: the backends are shared with the digest pipeline, which has no delivery
    context, and the tool result is exactly what the model got to read."""
    delivery = current_delivery_context()
    delivery.seen_urls.update(normalize_citation_url(url) for url in extract_urls(text))
    delivery.seen_urls.update(normalize_citation_url(url) for url in extra if url)
    return text


def _uncited_urls(report: str) -> list[str]:
    """URLs the report cites that no tool result contained, in the order they appear.

    A report is allowed to cite nothing; it is not allowed to cite something that was never
    returned. The comparison is lenient (extract_urls + normalize_citation_url) so an http→https
    rewrite, a URL in parentheses, or the `<url|label>` form the system prompt MANDATES for Slack
    still matches — matching the raw pattern refused every well-formed Slack report."""
    seen = current_delivery_context().seen_urls
    unknown: list[str] = []
    for url in extract_urls(report):
        if normalize_citation_url(url) not in seen and url not in unknown:
            unknown.append(url)
    return unknown


@tool
async def web_search(query: str, recency: str = "general") -> str:
    """Search the open web for a query. Use recency="news" for recent industry/company/policy
    news, "general" otherwise.

    Args:
        query: The search query.
        recency: "general" (default) or "news" for recent news framing.
    """
    topic = "news" if recency.lower().strip() == "news" else None
    return _record_seen_urls(await tavily_search(query, topic=topic))


@tool
async def community_search(query: str) -> str:
    """Search community discussion (Reddit, X, Hacker News, Substack) for reactions, sentiment,
    and first-hand reports about a topic.

    Args:
        query: The search query.
    """
    domains = get_config().agent.community_search_domains
    return _record_seen_urls(await tavily_search(query, include_domains=domains))


@tool
async def search_papers(query: str) -> str:
    """Search academic papers on Semantic Scholar for a research/technical claim.

    Args:
        query: The search query.
    """
    return _record_seen_urls(await semantic_scholar_search(query))


@tool
async def read_url(url: str) -> str:
    """Fetch and read the full text of a specific web page (a primary source you found via
    search) so you can ground your report in its actual content.

    Args:
        url: The page URL to read.
    """
    # The page URL itself counts as surfaced: the agent read it, so citing it is grounded.
    return _record_seen_urls(await extract_url(url), url)


@tool
async def recall_trends(query: str) -> str:
    """Recall related AI/ML trends tracked across earlier daily digests (cross-day memory).
    Use this for the "how has this evolved / what came before" angle.

    Args:
        query: What to recall (e.g. "open-weight model releases", "agent frameworks").
    """
    from datetime import date

    from shared import TRENDS_KEY, TrendMemory, create_state_store

    config = get_config()
    top_k = config.agent.recall_memory_top_k
    half_life = config.pipeline.trend_momentum_half_life_days

    def _load() -> TrendMemory:
        store = create_state_store(config)
        raw = store.read(TRENDS_KEY) if store.exists(TRENDS_KEY) else None
        if not raw:
            return TrendMemory()
        return TrendMemory.model_validate_json(raw)

    try:
        memory = await asyncio.to_thread(_load)
    except Exception as e:
        # Best-effort like every other tool (an exception must not end the run), but "the store
        # wouldn't answer" is NOT "there is no earlier history": returning TrendMemory() made a
        # throttled/denied/misconfigured S3 read look identical to a topic that has never appeared,
        # and the report then asserted that absence. Same distinction recall_digest already draws.
        logger.error("Failed to read the trend store for recall: %s", e, exc_info=True)
        return (
            f"The trend store could not be READ ({e}). This does NOT mean the topic has no earlier "
            "history — do not claim anything about what previous digests tracked."
        )

    matched = memory.search(query, today=date.today(), half_life_days=half_life, top_k=top_k)
    if not matched:
        return "No earlier trends recalled for that query."
    lines = [
        f"- *{t.title}* ({t.status.value}): " + "; ".join(f"[{ev.date}] {ev.summary}" for ev in t.evidence[-3:])
        for t in matched
    ]
    return "Earlier trends:\n\n" + "\n".join(lines)


@tool
async def recall_digest(digest_date: str) -> str:
    """Read back what a specific day's daily digest actually carried (its lead and story titles),
    for a question about what was covered on that date.

    Args:
        digest_date: The digest's date as YYYY-MM-DD.
    """
    from datetime import date as date_cls

    from shared import create_memory_store

    raw = digest_date.strip()
    try:
        day = date_cls.fromisoformat(raw)
    except ValueError:
        return f'"{digest_date}" is not a YYYY-MM-DD date.'

    config = get_config()
    max_stories = config.pipeline.top_n
    max_chars = config.agent.search_content_preview_chars

    def _load() -> dict | None:
        return create_memory_store().get_digest(day.isoformat())

    try:
        data = await asyncio.to_thread(_load)
    except Exception as e:
        # Best-effort like every other tool (an exception must not end the run), but the sentence
        # must NOT read as "that day carried nothing": a throttled/denied/misconfigured store looked
        # identical to an uncovered day, so the report asserted the digest never covered a topic it
        # did. Say the store could not be read and let the agent reason about that instead.
        logger.warning("Failed to recall the digest for '%s': %s", day, e)
        return (
            f"The digest store could not be READ for {day.isoformat()} ({e}). This does NOT mean the "
            "day had no digest — do not claim anything about what that day covered."
        )

    if not data:
        # NEVER fall back to another date: a report citing the wrong day's stories as that day's
        # coverage is worse than no recall at all.
        return f"No digest stored for {day.isoformat()}."

    from agent.tool_state import DigestStateManager

    content = DigestStateManager.load_from_dict(data).get_content()
    if content is None or not content.items:
        return f"The digest for {day.isoformat()} carries no stories."
    lines = [f"- {it.title[:max_chars]}" for it in content.items[:max_stories]]
    return f"Digest for {day.isoformat()}:\n{content.lead[:max_chars]}\n\n" + "\n".join(lines)


@tool
async def attach_image(source_url: str) -> str:
    """Download the representative image (og:image) of a source page and stage it to be
    attached to the delivered report. Call this for the best, most on-topic source before
    delivering, so the post carries a real image from the article.

    Args:
        source_url: The article/source URL whose representative image to attach.
    """
    delivery = current_delivery_context()
    limit = get_config().agent.research_max_staged_images
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

    # A citation no tool ever returned is a fabricated source, and this posts to a PUBLIC Threads
    # account. Refuse rather than publish, and name the offending URLs so the agent can drop or
    # replace them; a re-call with a corrected report is the remedy.
    uncited = _uncited_urls(report)
    if uncited:
        logger.error("Refusing to deliver a report citing %d unverifiable URL(s): %s", len(uncited), uncited)
        return (
            f"NOT delivered. These URLs appear in your report but in NO tool result, so they cannot be "
            f"published as sources: {', '.join(uncited)}. Remove them, or replace them with a URL a "
            "search/read tool actually returned, then call deliver_report again."
        )

    delivery = current_delivery_context()
    ok = await deliver_research_report(report, channel=target, delivery=delivery)
    # Report what the reader actually got. "Delivered the report" was returned even when half the
    # posts were dropped over the channel cap or trimmed mid-sentence, so the agent's final answer
    # asserted a complete delivery that never happened.
    stats = delivery.last_stats
    if not ok:
        return f"Failed to deliver the report to {target} ({stats.summary()})."
    if not stats.complete:
        return (
            f"Delivered the report to {target} INCOMPLETELY: {stats.summary()}. "
            "Do not re-send it (that would duplicate what did land); say so in your final answer."
        )
    return f"Delivered the report to {target} ({stats.summary()})."
