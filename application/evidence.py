"""Background evidence for the closing remarks: per shipped story, what a search finds on the established theory,
research, precedents and data it touches. The remark is argued from it, and the grounding pass checks against
it, so a cited theory or figure is one a search actually returned."""

from __future__ import annotations

import asyncio

from langchain_core.output_parsers import StrOutputParser

from application.ports import LanguageModels, ResearchSources
from application.prompts import EvidenceQueryPrompt
from domain.aggregator import normalize_url
from domain.models import CollectedItem
from shared import logger, parse_json_from_llm_output
from shared.config import PipelineConfig


async def gather_evidence(
    config: PipelineConfig, models: LanguageModels, sources: ResearchSources, items: list[CollectedItem]
) -> dict[str, str]:
    """Background evidence per story, keyed by normalized URL. A failed search leaves that story without
    evidence; it never fails the digest."""
    try:
        stories = "\n\n".join(
            f"[{i + 1}] {it.title}\n{models.truncate_to_tokens(it.text, config.evidence_story_tokens)}"
            for i, it in enumerate(items)
        )
        chain = EvidenceQueryPrompt.get_prompt() | models.get_model(config.digest_model, stage="evidence-queries")
        data = parse_json_from_llm_output(
            await (chain | StrOutputParser()).ainvoke({"domain": config.domain, "stories": stories})
        )
        entries = data.get("queries", []) if isinstance(data, dict) else []
        if not isinstance(entries, list):
            entries = []
    except Exception:
        logger.warning("Evidence queries failed; the remarks stay the editor's", exc_info=True)
        return {}
    queries: dict[int, dict] = {}
    for entry in entries:
        try:
            queries[int(entry["index"])] = entry
        except (KeyError, TypeError, ValueError, OverflowError):
            continue

    async def one(i: int, item: CollectedItem) -> tuple[str, str]:
        q = queries.get(i + 1) or {}
        searches = [
            (label, query, search)
            for label, query, search in (
                ("Web", str(q.get("web") or "").strip(), sources.web_search),
                ("Papers", str(q.get("papers") or "").strip(), sources.paper_search),
            )
            if query
        ]
        found = await asyncio.gather(*(search(query) for _, query, search in searches), return_exceptions=True)
        # The port's results carry one "URL:" line each; a miss or a failure is text without one.
        blocks = [
            f"{label} ({query}):\n{text}"
            for (label, query, _), text in zip(searches, found, strict=True)
            if isinstance(text, str) and "URL:" in text
        ]
        return normalize_url(item.url), "\n\n".join(blocks)

    pairs = await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    evidence = {url: text for url, text in pairs if text}
    logger.info("Background evidence for %d of %d stories", len(evidence), len(items))
    return evidence
