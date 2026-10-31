"""The deep-research run's rules, independent of the agent framework that drives it: which URLs a
report may cite (only ones a tool surfaced), the fact check a public thread must pass, and how a
report is delivered. The Strands tools (adapters/research_tools.py) are thin wrappers over these, and
every outside call goes through the run's ResearchPorts."""

from __future__ import annotations

import asyncio
import re
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum, auto
from typing import Literal
from zoneinfo import ZoneInfo

from application.ports import LanguageModels, MemoryStore, ResearchSources, StateStore, ThreadsPublisher
from domain.formatting import citation_key, extract_urls, normalize_citation_url
from domain.models import ImageAsset
from domain.renderers import render_threads_research, research_post_budget
from shared import THREADS_MAX_POST_CHARS, get_config, logger

# The two places a research report can go. MARKDOWN is the report handed back to the caller (the CLI
# prints it and saves it); THREADS publishes a plain-text thread to the public account and is allowed
# only when the caller asked for it.
MARKDOWN = "markdown"
THREADS = "threads"


@dataclass
class DeliveryStats:
    """What the last delivery actually put in front of the reader. `rendered` is the number of posts
    the report became, `delivered` how many landed, and dropped/trimmed how much content the renderer
    had to discard to fit the channel's limits. Carried back to the agent so its tool result states
    the truth: a thread whose second half was dropped is not "delivered"."""

    rendered: int = 0
    delivered: int = 0
    # The root may be live although nothing is known to have landed (its publish step failed).
    uncertain: bool = False
    dropped: int = 0
    trimmed: int = 0

    @property
    def complete(self) -> bool:
        return self.delivered >= self.rendered and not self.dropped and not self.trimmed

    def summary(self) -> str:
        parts = [f"{self.delivered}/{self.rendered} posts delivered"]
        if self.dropped:
            parts.append(f"{self.dropped} post(s) DROPPED (report too long for the channel)")
        if self.trimmed:
            parts.append(f"{self.trimmed} post(s) trimmed to fit the {THREADS_MAX_POST_CHARS}-char cap")
        return "; ".join(parts)


class DeliveryOutcome(Enum):
    """What ONE deliver_research_report call did. Deliberately not a bool: "nothing was posted
    because this channel already has the report" and "the post failed" are different answers, and
    reporting the first as success let a REVISED report be announced as delivered while the reader
    still had the old one."""

    POSTED = auto()
    NOT_POSTED = auto()
    FAILED = auto()
    # Nothing was posted: the thread would have lost posts to the channel's limits. Returned once.
    TOO_LONG = auto()


@dataclass
class ResearchPorts:
    """What a research run reaches outside through, built by entrypoints/bootstrap.py. A port left
    None makes its tools answer that it is unavailable: a test that binds none can reach nothing."""

    sources: ResearchSources | None = None
    threads: ThreadsPublisher | None = None
    # The thread's pre-publish fact check.
    models: LanguageModels | None = None
    # Built on first use: the stores' lookups cost a round trip each.
    state_store: Callable[[], StateStore] | None = None
    memory_store: Callable[[], MemoryStore] | None = None


@dataclass
class DeliveryContext:
    """Per-invocation state for one research run: which channels it may use, the OG images the agent
    staged via attach_image, the URLs its tools surfaced, and what each delivery did."""

    # Publishing to the PUBLIC Threads account is decided by the caller (the CLI's --publish), never
    # inferred from the request's wording.
    publish_threads: bool = False
    staged_images: list[ImageAsset] = field(default_factory=list)
    delivered_channels: set[str] = field(default_factory=set)
    # Channels where SOMETHING landed but the delivery was not usable as a whole (a Threads root with
    # no replies). A repeat deliver_report must not re-post over what did land.
    partial_channels: set[str] = field(default_factory=set)
    # Every URL a tool actually surfaced this invocation, in normalize_citation_url form.
    # deliver_report refuses a report citing a URL that appeared in no tool result — the
    # deterministic guard against a fabricated citation reaching the requester or a public post.
    seen_urls: set[str] = field(default_factory=set)
    # What each surfaced URL's tool result said (a search hit's snippet, or the page read in full), so
    # a thread can be checked against the sources it cites before it goes public.
    source_texts: dict[str, str] = field(default_factory=dict)
    # The thread last sent back for claims its sources do not support, whitespace collapsed: the same
    # text gets the same verdict instead of a second, non-deterministic check.
    refused_for_facts: str | None = None
    fact_issues: list[dict] = field(default_factory=list)
    # Why the last Threads attempt was sent back, for the CLI's warning when nothing was resubmitted.
    last_refusal: str = ""
    # Held across a Threads delivery, its fact check included: two identical calls in one turn each ran
    # the (non-deterministic) check, and one passing posted the text the other refused.
    threads_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    # The Markdown report the agent submitted (the latest one: a revision replaces it).
    report: str = ""
    # What the Threads delivery put in front of the reader (MARKDOWN has nothing to count).
    threads_stats: DeliveryStats = field(default_factory=DeliveryStats)
    # Whether a Threads post was actually attempted this run.
    threads_attempted: bool = False
    ports: ResearchPorts = field(default_factory=ResearchPorts)

    @property
    def allowed_channels(self) -> set[str]:
        return {MARKDOWN, THREADS} if self.publish_threads else {MARKDOWN}


_request_delivery: ContextVar[DeliveryContext | None] = ContextVar("research_request_delivery", default=None)


def current_delivery_context() -> DeliveryContext:
    # A fresh context when nothing is bound rather than a shared module-level singleton, so state
    # can never accumulate across invocations. Real entrypoints always bind via request_context.
    return _request_delivery.get() or DeliveryContext()


@contextmanager
def request_context(delivery: DeliveryContext):
    """Bind per-invocation delivery state so concurrent invocations don't share globals."""
    token = _request_delivery.set(delivery)
    try:
        yield
    finally:
        _request_delivery.reset(token)


async def _deliver_threads(report: str, delivery: DeliveryContext) -> bool:
    """Post a research report to Threads: a root + flat reply chain (each <=500 chars). At most one
    staged image rides the root (Threads media indexing is slow)."""
    threads = delivery.ports.threads
    if threads is None:
        logger.error("No Threads publisher is configured for this run; the thread was not posted")
        return False
    max_posts = get_config().agent.research_max_threads_posts
    root_text, replies, dropped, trimmed = render_threads_research(report, max_posts=max_posts)
    delivery.threads_stats = DeliveryStats(dropped=dropped, trimmed=trimmed)
    if not root_text.strip():
        # Empty report → nothing to post. An empty root would 400 the Threads API; skip cleanly.
        logger.warning("Research report rendered to empty Threads root; skipping Threads delivery.")
        return False

    # Only an image from a page the thread cites: one from an uncited page put a source in front of the
    # reader that the thread never names.
    keys = (citation_key(url, delivery.seen_urls) for post in (root_text, *replies) for url in extract_urls(post))
    cited = {key for key in keys if key}
    image = next((i for i in delivery.staged_images if normalize_citation_url(i.source_url) in cited), None)
    if delivery.staged_images and image is None:
        logger.warning("No staged image comes from a page the thread cites; posting text-only")

    outcome = await threads.post_thread(
        root_text=root_text,
        replies=replies,
        image_bytes=image.data if image else None,
        image_content_type=image.content_type if image else "image/png",
        image_alt=image.alt if image else "",
    )
    # Explicitly read the counts: the outcome is a NamedTuple and therefore always truthy, so
    # `return outcome` would report a 0-of-8 post as a success.
    delivery.threads_stats.rendered = outcome.expected
    delivery.threads_stats.delivered = outcome.posted
    delivery.threads_stats.uncertain = outcome.uncertain
    return outcome.published


def sends_back_as_too_long(report: str, delivery: DeliveryContext, *, dropped: int = -1, trimmed: int = 0) -> bool:
    """Whether this thread will be refused for losing text to the channel's limits. Every such text is,
    rewrites included: a rewrite that still overflowed went out cut, numbered "1/10 … 10/10" without the
    conclusion that was dropped, and read as complete. Not posting is the worst case now."""
    if dropped < 0:
        _, _, dropped, trimmed = render_threads_research(
            report, max_posts=get_config().agent.research_max_threads_posts
        )
    return bool(dropped or trimmed)


async def deliver_research_report(report: str, *, channel: str, delivery: DeliveryContext) -> DeliveryOutcome:
    """Hand a finished report to `channel`.

    MARKDOWN records the report for the caller; a later call replaces it (nothing external happened,
    so a revision is just a better answer). THREADS publishes, and is idempotent: a second call for a
    channel that already carries the report returns NOT_POSTED rather than success, because the text
    was not published — and the likely reason for the second call is a REVISED report, which must not
    be announced as delivered. A partially-delivered thread is NOT re-sent (that would duplicate what
    did land); the outcome says so instead."""
    if channel == MARKDOWN:
        delivery.report = report
        delivery.delivered_channels.add(MARKDOWN)
        return DeliveryOutcome.POSTED

    if channel in delivery.delivered_channels | delivery.partial_channels:
        logger.warning("Report already posted to '%s'; the text of this call was NOT published", channel)
        return DeliveryOutcome.NOT_POSTED
    root_text, replies, dropped, trimmed = render_threads_research(
        report, max_posts=get_config().agent.research_max_threads_posts
    )
    if sends_back_as_too_long(report, delivery, dropped=dropped, trimmed=trimmed):
        # Refuse before anything is public: the dropped tail is the conclusion and its sources, and
        # telling the agent after the post could only say "incomplete".
        delivery.last_refusal = "too long"
        delivery.threads_stats = DeliveryStats(rendered=len(replies) + 1, dropped=dropped, trimmed=trimmed)
        logger.warning("Research thread too long for Threads; sent back: %s", delivery.threads_stats.summary())
        return DeliveryOutcome.TOO_LONG
    # Reset before dispatch so the verdict below reads THIS attempt's counts, never a previous one's.
    delivery.threads_stats = DeliveryStats()
    delivery.threads_attempted = True
    ok = await _deliver_threads(report, delivery)
    if ok:
        delivery.delivered_channels.add(channel)
        if not delivery.threads_stats.complete:
            logger.warning("Research report delivered INCOMPLETE to %s: %s", channel, delivery.threads_stats.summary())
        return DeliveryOutcome.POSTED
    if delivery.threads_stats.delivered > 0 or delivery.threads_stats.uncertain:
        # Not usable as a delivery (a Threads root whose replies all failed, or one whose publish step
        # failed and may be live), but posts did or may have landed on a public account. Re-sending
        # would duplicate them, so the channel is spent.
        delivery.partial_channels.add(channel)
        logger.warning("Research report delivered INCOMPLETE to %s: %s", channel, delivery.threads_stats.summary())
        return DeliveryOutcome.POSTED
    return DeliveryOutcome.FAILED


# The line every search and paper result carries for its own URL (adapters/research_backends.py).
_RESULT_URL = re.compile(r"^\s*URL: (\S+)", re.MULTILINE)


def record_seen_urls(text: str, *extra: str) -> str:
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


def uncited_urls(report: str) -> list[str]:
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


def _sources() -> ResearchSources | None:
    sources = current_delivery_context().ports.sources
    if sources is None:
        logger.error("No research sources are configured for this run")
    return sources


async def web_search(
    query: str,
    *,
    topic: Literal["news"] | None = None,
    time_range: str | None = None,
    include_domains: list[str] | None = None,
) -> str:
    sources = _sources()
    if sources is None:
        return "Search is not available in this run."
    kwargs: dict = {"topic": topic, "time_range": time_range}
    if include_domains:
        kwargs["include_domains"] = include_domains
    return record_seen_urls(await sources.web_search(query, **kwargs))


async def community_search(query: str) -> str:
    return await web_search(query, include_domains=get_config().agent.community_search_domains)


async def search_papers(query: str) -> str:
    sources = _sources()
    if sources is None:
        return "Paper search is not available in this run."
    return record_seen_urls(await sources.paper_search(query))


async def read_url(url: str) -> str:
    sources = _sources()
    text = await sources.read_page(url) if sources else ""
    if not text:
        # A page that could not be read is not a source: recording it would let an invented URL
        # pass deliver_report's citation check after one failed read.
        return f"Could not read {url}; it is not a source you can cite."
    # Only the page itself counts as surfaced. A link inside it is the page's text, which anyone can
    # write: recording it let a planted URL pass the citation and image checks. Reading it makes it one.
    record_seen_urls("", url)
    current_delivery_context().source_texts[normalize_citation_url(url)] = text
    return text


async def recall_trends() -> str:
    from domain.models import TrendMemory
    from shared import TRENDS_KEY

    config = get_config()
    half_life = config.pipeline.trend_momentum_half_life_days
    gist_chars = config.agent.search_content_preview_chars
    store = current_delivery_context().ports.state_store

    def _load() -> TrendMemory:
        if store is None:
            raise RuntimeError("no trend store is configured for this run")
        raw = store().read(TRENDS_KEY)
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


async def recall_digest(digest_date: str) -> str:
    raw = digest_date.strip()
    try:
        day = date.fromisoformat(raw)
    except ValueError:
        return f'"{digest_date}" is not a YYYY-MM-DD date.'

    config = get_config()
    max_stories = config.pipeline.top_n
    max_chars = config.agent.search_content_preview_chars
    store = current_delivery_context().ports.memory_store

    def _load() -> dict | None:
        if store is None:
            raise RuntimeError("no digest store is configured for this run")
        return store().get_digest(day.isoformat())

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

    from domain.digest_snapshot import DigestStateManager

    content = DigestStateManager.load_from_dict(data).get_content()
    if content is None or not content.items:
        return f"The digest for {day.isoformat()} carries no stories."
    lines = [f"- {it.title[:max_chars]}" for it in content.items[:max_stories]]
    return f"Digest for {day.isoformat()}:\n{content.lead[:max_chars]}\n\n" + "\n".join(lines)


async def attach_image(source_url: str) -> str:
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
    sources = _sources()
    asset = await sources.fetch_image(source_url) if sources else None
    if not asset:
        return f"No usable image found for {source_url}."
    delivery.staged_images.append(asset)
    logger.info("Staged OG image from '%s' (%d bytes)", source_url, len(asset.data))
    return f"Attached image from {source_url} ({asset.content_type}, {len(asset.data)} bytes)."


async def deliver_report(report: str, channel: str = MARKDOWN) -> str:
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
    uncited = uncited_urls(report)
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
        return await deliver_thread(report, delivery)


async def deliver_thread(report: str, delivery: DeliveryContext) -> str:
    """The Threads half of deliver_report, run under the context's Threads lock."""
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
        issues = delivery.fact_issues if same else await check_thread_facts(report, delivery)
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


async def check_thread_facts(report: str, delivery: DeliveryContext) -> list[dict]:
    """Claims in the thread that contradict or overstate the sources it cites, read from what the
    tools returned for those URLs. Best-effort: a failed check returns [] and the thread posts as
    before, since the check must not be able to stop publishing on its own."""
    from langchain_core.output_parsers import StrOutputParser

    from application.prompts import ThreadFactCheckPrompt
    from shared import parse_json_from_llm_output

    keys = (citation_key(url, delivery.seen_urls) for url in extract_urls(report))
    cited = [delivery.source_texts[key] for key in dict.fromkeys(keys) if key in delivery.source_texts]
    if not cited:
        return []
    if delivery.ports.models is None:
        logger.warning("No model is configured for the thread fact check; posting without it")
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
        llm = delivery.ports.models.get_model(config.pipeline.digest_model, stage="research-fact-check")
        raw = await (ThreadFactCheckPrompt.get_prompt() | llm | StrOutputParser()).ainvoke(
            {"thread": thread, "sources": sources}
        )
        issues = parse_json_from_llm_output(raw).get("issues", [])
        return [i for i in issues if isinstance(i, dict) and i.get("claim")]
    except Exception:
        logger.warning("Thread fact check failed; posting without it", exc_info=True)
        return []
