from __future__ import annotations

import httpx
from tavily import AsyncTavilyClient

from shared import Config, logger, resolve_secret, retry_async

SEMANTIC_SCHOLAR_URL = "https://api.semanticscholar.org/graph/v1/paper/search"


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
    finally:
        await client.close()


async def extract_url(url: str) -> str:
    """Fetch and extract the full readable text of a single page via Tavily extract,
    capped at the configured research content limit. Used by the deep-research agent's
    read_url tool to pull a primary source it found via search."""
    client = _get_tavily_client()
    if not client:
        return "TAVILY_API_KEY not configured."

    cap = Config.load().agent.research_content_cap_chars
    try:
        response = await client.extract(urls=[url], format="text")
    except Exception as e:
        logger.warning("Tavily extract failed for '%s': %s", url, e)
        return f"Could not read URL: {e}"
    finally:
        await client.close()

    results = response.get("results", [])
    if not results:
        return f"No readable content extracted from {url}."
    content = results[0].get("raw_content") or results[0].get("content") or ""
    if not content:
        return f"No readable content extracted from {url}."
    return content[:cap]


async def _search_papers(query: str) -> str:
    agent_config = Config.load().agent
    async with httpx.AsyncClient(timeout=agent_config.search_request_timeout) as client:

        async def _fetch() -> httpx.Response:
            resp = await client.get(
                SEMANTIC_SCHOLAR_URL,
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
            authors = ", ".join(
                name
                for a in (p.get("authors") or [])[: agent_config.search_paper_max_authors]
                if (name := a.get("name"))
            )
            abstract = (p.get("abstract") or "")[: agent_config.search_paper_abstract_max_chars]
            results.append(
                f"- {p.get('title', 'N/A')} ({p.get('year', 'N/A')}) by {authors}\n"
                f"  URL: {p.get('url', '')}\n"
                f"  Abstract: {abstract}"
            )

        logger.info("Found %d papers for query '%s'", len(papers), query)
        return "\n\n".join(results)
