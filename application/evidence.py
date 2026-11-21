"""Background evidence for the editor's implications: per candidate, what a search finds on the established
theory, research, precedents and data the story touches. The editor may argue an implication from it, and
the grounding pass checks against it, so a cited theory or figure is one a search actually returned."""

from __future__ import annotations

import asyncio

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from application.ports import LanguageModels, ResearchSources
from domain.aggregator import normalize_url
from domain.models import CollectedItem
from shared import logger, parse_json_from_llm_output
from shared.config import PipelineConfig

_QUERY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You prepare background research for a {domain} digest editor. For each numbered story, write the "
            "search queries that would find the established theory, research findings, historical precedents "
            "and data that put the story in context: what the field already knows that bears on it. Search for "
            "the general phenomenon the story is an instance of, so leave out the story's own companies, products "
            "and event: those return more news about the same event. Aim the queries at named theories, published "
            "research and official statistics rather than commentary. Write the queries in English.",
        ),
        (
            "human",
            "{stories}\n\nAnswer ONLY with JSON: "
            '{{"queries": [{{"index": <story number>, "web": "<web search query>", '
            '"papers": "<scholarly search query, or empty when no research bears on it>"}}]}}',
        ),
    ]
)

_STORY_CHARS = 1500


async def gather_evidence(
    config: PipelineConfig, models: LanguageModels, sources: ResearchSources, items: list[CollectedItem]
) -> dict[str, str]:
    """Background evidence per story, keyed by normalized URL. A failed search leaves that story without
    evidence; it never fails the digest."""
    stories = "\n\n".join(f"[{i + 1}] {it.title}\n{it.text[:_STORY_CHARS]}" for i, it in enumerate(items))
    chain = _QUERY_PROMPT | models.get_model(config.digest_model, stage="evidence-queries") | StrOutputParser()
    try:
        data = parse_json_from_llm_output(await chain.ainvoke({"domain": config.domain, "stories": stories}))
        queries = {int(q["index"]): q for q in data.get("queries", [])}
    except Exception:
        logger.warning("Evidence queries failed; the digest is written without background evidence", exc_info=True)
        return {}

    async def one(i: int, item: CollectedItem) -> tuple[str, str]:
        q = queries.get(i + 1) or {}
        web, papers = str(q.get("web") or "").strip(), str(q.get("papers") or "").strip()
        found = await asyncio.gather(
            sources.web_search(web) if web else _empty(),
            sources.paper_search(papers) if papers else _empty(),
            return_exceptions=True,
        )
        blocks = [
            f"{label} ({query}):\n{text}"
            for label, query, text in zip(("Web", "Papers"), (web, papers), found, strict=True)
            if isinstance(text, str) and text and not _is_miss(text)
        ]
        return normalize_url(item.url), "\n\n".join(blocks)

    pairs = await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    evidence = {url: text for url, text in pairs if text}
    logger.info("Background evidence for %d of %d stories", len(evidence), len(items))
    return evidence


async def _empty() -> str:
    return ""


def _is_miss(text: str) -> bool:
    return text.startswith(("No results", "No related papers", "Search failed", "TAVILY_API_KEY"))
