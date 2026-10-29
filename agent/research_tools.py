from __future__ import annotations

import asyncio
import re
from typing import Literal

from strands import tool

# DeliveryContext + request scoping live in the delivery layer (output/), which owns the
# delivery contract; re-exported here for the agent entrypoints and tools that bind them.
from output.delivery import DeliveryContext, current_delivery_context, request_context
from output.renderers import research_post_budget
from shared import extract_urls, get_config, logger, normalize_citation_url
from shared.formatting import citation_key
from shared.media import fetch_og_image
from shared.research import extract_url, paper_search, tavily_search

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


# The line every search and paper result carries for its own URL (shared/research/research_backends.py).
_RESULT_URL = re.compile(r"^\s*URL: (\S+)", re.MULTILINE)


def _record_seen_urls(text: str, *extra: str) -> str:
    """Remember every URL a tool actually surfaced, then return the text unchanged.

    deliver_report compares the report's citations against this set, so a URL the model invented can
    be refused before it is published. Recording happens at the TOOL boundary rather than inside the
    research backends: the backends are shared with the digest pipeline, which has no delivery
    context, and the tool result is exactly what the model got to read."""
    delivery = current_delivery_context()
    # Each result's own "URL:" line only. A link inside a snippet or an abstract is text anyone can
    # write (Reddit and X snippets are full of them): recording it let a planted URL pass the citation
    # and image checks.
    for block in text.split("\n\n"):
        for url in _RESULT_URL.findall(block):
            key = normalize_citation_url(url)
            delivery.seen_urls.add(key)
            delivery.source_texts.setdefault(key, block)
    delivery.seen_urls.update(normalize_citation_url(url) for url in extra if url)
    return text


def _uncited_urls(report: str) -> list[str]:
    """URLs the report cites that no tool result contained, in the order they appear.

    A report is allowed to cite nothing; it is not allowed to cite something that was never
    returned. The comparison is lenient (extract_urls + normalize_citation_url) so an http→https
    rewrite, a URL in parentheses, or the `[label](url)` link the system prompt asks for still
    matches."""
    seen = current_delivery_context().seen_urls
    unknown: list[str] = []
    for url in extract_urls(report):
        if citation_key(url, seen) is None and url not in unknown:
            unknown.append(url)
    return unknown


@tool
async def web_search(
    query: str, news: bool = False, time_range: Literal["day", "week", "month", "year"] | None = None
) -> str:
    """Search the open web for a query. Results carry a Published date when the source has one.

    Args:
        query: The search query.
        news: True for industry/company/policy news coverage (about the last week unless time_range says otherwise).
        time_range: Only results from the last day, week, month or year; omit for any time.
    """
    return _record_seen_urls(await tavily_search(query, topic="news" if news else None, time_range=time_range))


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
    """Search scholarly papers (OpenAlex, incl. arXiv preprints) for a research/technical claim.
    Results carry the publication date and citation count.

    Args:
        query: The search query, in natural language.
    """
    return _record_seen_urls(await paper_search(query))


@tool
async def read_url(url: str) -> str:
    """Fetch and read the full text of a specific web page (a primary source you found via
    search) so you can ground your report in its actual content.

    Args:
        url: The page URL to read.
    """
    text = await extract_url(url)
    if not text:
        # A page that could not be read is not a source: recording it would let an invented URL
        # pass deliver_report's citation check after one failed read.
        return f"Could not read {url}; it is not a source you can cite."
    # Only the page itself counts as surfaced. A link inside it is the page's text, which anyone can
    # write: recording it let a planted URL pass the citation and image checks. Reading it makes it one.
    _record_seen_urls("", url)
    current_delivery_context().source_texts[normalize_citation_url(url)] = text
    return text


@tool
async def recall_trends() -> str:
    """Every AI/ML trend tracked across earlier daily digests (cross-day memory), strongest first,
    with when it was first seen and the dates of its latest evidence (only the last few are kept, so
    they are not a count of every appearance). Judge yourself which ones bear on the topic; the
    trends are written in English. Use this for the "how has this evolved / what came before" angle."""
    from datetime import datetime
    from zoneinfo import ZoneInfo

    from shared import TRENDS_KEY, TrendMemory, create_state_store

    config = get_config()
    half_life = config.pipeline.trend_momentum_half_life_days
    gist_chars = config.agent.search_content_preview_chars

    def _load() -> TrendMemory:
        raw = create_state_store(config).read(TRENDS_KEY)
        return TrendMemory.model_validate_json(raw) if raw else TrendMemory()

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

    # Keyword matching was replaced by the whole list: substring hits ranked "AI Doom Discourse" first
    # for "diffusion LLM", and a Korean query never matched the English trends at all.
    trends = [t for t in memory.trends if t.evidence]
    if not trends:
        return "No earlier trends recalled."
    today = datetime.now(ZoneInfo(config.aws.timezone)).date()
    trends.sort(key=lambda t: t.momentum(today, half_life), reverse=True)
    lines = [
        f"- {t.title} ({t.status.value}; first seen {t.first_seen or t.evidence[0].date}, latest evidence on "
        f"{', '.join(sorted({ev.date for ev in t.evidence}))}): {t.evidence[-1].summary[:gist_chars]}"
        for t in trends
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
    """Stage a source page's representative image (og:image) for the delivered report. A Threads
    delivery posts the first staged image whose page the thread cites.

    Args:
        source_url: The article/source URL whose representative image to attach.
    """
    delivery = current_delivery_context()
    if normalize_citation_url(source_url) not in delivery.seen_urls:
        # The image rides a public post, so it must come from a search hit or a page the agent read.
        return (
            f"NOT attached: {source_url} was not a search hit or a page you read. read_url it first, or attach "
            "an image from a source you found."
        )
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
async def deliver_report(report: str, channel: Literal["markdown", "threads"] = "markdown") -> str:
    """Hand over the finished research report. Call it with channel "markdown" once the report is
    complete; call it with channel "threads" only when this run is allowed to publish to Threads.

    Args:
        report: The final report. For "markdown" use Markdown; for "threads" use plain text.
        channel: "markdown" (default) or "threads".
    """
    from output.delivery import THREADS, deliver_research_report

    delivery = current_delivery_context()
    target = channel

    # Whether this run may publish to Threads is DATA the caller set (the CLI's --publish), not
    # something to infer from the request's wording. A self-correcting refusal: nothing is published
    # and the agent can re-call.
    allowed = delivery.allowed_channels
    if target not in allowed:
        permitted = ", ".join(sorted(allowed))
        logger.error("Refusing to deliver to '%s'; this run allows %s", target, permitted)
        return (
            f'NOT delivered. This run is not allowed to publish to "{target}". '
            f"Call deliver_report again with one of: {permitted}."
        )

    # A citation no tool ever returned is a fabricated source. Refuse it, and name the offending URLs
    # so the agent can drop or replace them; a re-call with a corrected report is the remedy.
    uncited = _uncited_urls(report)
    if uncited:
        logger.error("Refusing to deliver a report citing %d unverifiable URL(s): %s", len(uncited), uncited)
        return (
            f"NOT delivered. These URLs in your report were not a search hit or a page you read, so they "
            f"cannot be published as sources: {', '.join(uncited)}. read_url them first, remove them, or "
            "replace them with a URL a search returned, then call deliver_report again."
        )

    if target != THREADS:
        await deliver_research_report(report, channel=target, delivery=delivery)
        return "Recorded the Markdown report."
    async with delivery.threads_lock:
        return await _deliver_thread(report, delivery)


async def _deliver_thread(report: str, delivery: DeliveryContext) -> str:
    """The Threads half of deliver_report, run under the context's Threads lock."""
    from output.delivery import THREADS, DeliveryOutcome, deliver_research_report, sends_back_as_too_long

    target = THREADS
    already = delivery.delivered_channels | delivery.partial_channels
    # Not for a thread already live, nor one about to be sent back as too long: the check costs a model
    # call over every cited source, and its verdict would be on text that never posts.
    if target not in already and not sends_back_as_too_long(report, delivery):
        if not extract_urls(report):
            delivery.last_refusal = "no cited source"
            return (
                "NOT posted: the thread cites no source, so nothing in it can be checked. Cite the URL "
                "each claim comes from, then call deliver_report again."
            )
        # The same text sent back once is sent back with the same verdict, not re-judged: the check is
        # not deterministic, so asking again could wave it through. A rewrite is checked like the first
        # text: it posted unchecked before, new claims included.
        same = delivery.refused_for_facts == _same_text(report)
        issues = delivery.fact_issues if same else await _check_thread_facts(report, delivery)
        if issues:
            delivery.refused_for_facts = _same_text(report)
            delivery.fact_issues = issues
            delivery.last_refusal = "claims its sources do not support"
            logger.warning("Research thread sent back: %d claim(s) its sources do not support", len(issues))
            listed = "\n".join(
                f"- post {i.get('post', '?')}: \"{i.get('claim', '')}\" (the source says: {i.get('source_says', '')})"
                for i in issues
            )
            return (
                "NOT posted: these claims do not match the sources they cite:\n"
                f"{listed}\nCorrect or drop them, then call deliver_report again; the new text is checked the same way."
            )

    outcome = await deliver_research_report(report, channel=target, delivery=delivery)
    # Report what the reader actually got: half a thread dropped over the channel cap is not
    # "delivered", and the agent's final answer must not say it was.
    stats = delivery.threads_stats
    if outcome is DeliveryOutcome.NOT_POSTED:
        # The text of THIS call was not published — saying "delivered" here announced a revision
        # the reader never received.
        return (
            f"NOT posted: {target} already carries this run's report, so the text you just passed was "
            "never published. Do not call deliver_report for it again; say so in your final answer."
        )
    if outcome is DeliveryOutcome.TOO_LONG:
        max_posts = get_config().agent.research_max_threads_posts
        return (
            f"NOT posted: as written the thread would lose text to the channel's limits ({stats.summary()}). "
            f"Rewrite it as at most {max_posts} posts of at most {research_post_budget(max_posts)} characters "
            "each, then call deliver_report again; the new text is checked again before it posts."
        )
    if outcome is DeliveryOutcome.FAILED:
        return f"Failed to deliver the report to {target} ({stats.summary()})."
    if not stats.complete:
        return (
            f"Delivered the report to {target} INCOMPLETELY: {stats.summary()}. "
            "Do not re-send it (that would duplicate what did land); say so in your final answer."
        )
    return f"Delivered the report to {target} ({stats.summary()})."


def _same_text(report: str) -> str:
    """The report with whitespace collapsed: a resubmission that differs only in a trailing newline is
    the same text, not a rewrite."""
    return " ".join(report.split())


def _fact_check_factory():
    """A Bedrock factory on the configured profile where it resolves, else ambient credentials."""
    import boto3

    from shared import BedrockLanguageModelFactory
    from shared.utils import available_boto_profile

    config = get_config()
    session = boto3.Session(
        region_name=config.aws.bedrock_region, profile_name=available_boto_profile(config.aws.profile)
    )
    return BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)


async def _check_thread_facts(report: str, delivery: DeliveryContext) -> list[dict]:
    """Claims in the thread that contradict or overstate the sources it cites, read from what the
    tools returned for those URLs. Best-effort: a failed check returns [] and the thread posts as
    before, since the check must not be able to stop publishing on its own."""
    from langchain_core.output_parsers import StrOutputParser

    from output.renderers import render_threads_research
    from shared import ThreadFactCheckPrompt, parse_json_from_llm_output

    keys = (citation_key(url, delivery.seen_urls) for url in extract_urls(report))
    cited = [delivery.source_texts[key] for key in dict.fromkeys(keys) if key in delivery.source_texts]
    if not cited:
        return []
    try:
        config = get_config()
        root, replies, _, _ = render_threads_research(report, max_posts=config.agent.research_max_threads_posts)
        thread = "\n\n".join(f"POST {n}:\n{post}" for n, post in enumerate([root, *replies], start=1))
        # Cut by characters, not CountTokens: a character is at most about one token, so this bounds the
        # call, while token counting cost about 17 blocking round trips per long page.
        sources = "\n\n".join(
            f"[{n}]\n{text[: config.pipeline.item_text_max_tokens]}" for n, text in enumerate(cited, start=1)
        )
        llm = _fact_check_factory().get_model(config.pipeline.digest_model, stage="research-fact-check")
        raw = await (ThreadFactCheckPrompt.get_prompt() | llm | StrOutputParser()).ainvoke(
            {"thread": thread, "sources": sources}
        )
        issues = parse_json_from_llm_output(raw).get("issues", [])
        return [i for i in issues if isinstance(i, dict) and i.get("claim")]
    except Exception:
        logger.warning("Thread fact check failed; posting without it", exc_info=True)
        return []
