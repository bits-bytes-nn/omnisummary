"""The research agent's Strands tools. Each one is a thin wrapper: its signature and docstring are the
tool spec the model reads, and its rules live in application/research.py."""

from __future__ import annotations

from typing import Literal

from strands import tool

from application import research


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
    return await research.web_search(query, topic="news" if news else None, time_range=time_range)


@tool
async def community_search(query: str) -> str:
    """Search community discussion (Reddit, X, Hacker News, Substack) for reactions, sentiment,
    and first-hand reports about a topic.

    Args:
        query: The search query.
    """
    return await research.community_search(query)


@tool
async def search_papers(query: str) -> str:
    """Search scholarly papers (OpenAlex, incl. arXiv preprints) for a research/technical claim.
    Results carry the publication date and citation count.

    Args:
        query: The search query, in natural language.
    """
    return await research.search_papers(query)


@tool
async def read_url(url: str) -> str:
    """Fetch and read the full text of a specific web page (a primary source you found via
    search) so you can ground your report in its actual content.

    Args:
        url: The page URL to read.
    """
    return await research.read_url(url)


@tool
async def recall_trends() -> str:
    """Every AI/ML trend tracked across earlier daily digests (cross-day memory), strongest first,
    with when it was first seen and the dates of its latest evidence (only the last few are kept, so
    they are not a count of every appearance). Judge yourself which ones bear on the topic; the
    trends are written in English. Use this for the "how has this evolved / what came before" angle."""
    return await research.recall_trends()


@tool
async def recall_digest(digest_date: str) -> str:
    """Read back what a specific day's daily digest actually carried (its lead and story titles),
    for a question about what was covered on that date.

    Args:
        digest_date: The digest's date as YYYY-MM-DD.
    """
    return await research.recall_digest(digest_date)


@tool
async def attach_image(source_url: str) -> str:
    """Stage a source page's representative image (og:image) for the delivered report. A Threads
    delivery posts the first staged image whose page the thread cites.

    Args:
        source_url: The article/source URL whose representative image to attach.
    """
    return await research.attach_image(source_url)


@tool
async def deliver_report(report: str, channel: Literal["markdown", "threads"] = "markdown") -> str:
    """Hand over the finished research report. Call it with channel "markdown" once the report is
    complete; call it with channel "threads" only when this run is allowed to publish to Threads.

    Args:
        report: The final report. For "markdown" use Markdown; for "threads" use plain text.
        channel: "markdown" (default) or "threads".
    """
    return await research.deliver_report(report, channel)


RESEARCH_TOOLS = [
    web_search,
    community_search,
    search_papers,
    read_url,
    recall_trends,
    recall_digest,
    attach_image,
    deliver_report,
]
