from __future__ import annotations

import httpx
from tavily import AsyncTavilyClient

from shared import get_config, logger, resolve_secret, retry_async

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


def _get_tavily_client() -> AsyncTavilyClient | None:
    # env first, then SSM SecureString — so search works in the Lambdas, which carry the key in SSM
    # rather than the environment.
    api_key = resolve_secret("TAVILY_API_KEY", "tavily-api-key")
    if not api_key:
        return None
    return AsyncTavilyClient(api_key=api_key)


def _format_search_results(results: list[dict], preview_chars: int) -> str:
    # `... or ''` (not just a .get default): Tavily returns an explicit null content/title for some
    # pages, and `None[:n]` would raise — failing the ENTIRE query and pushing the agent toward its
    # hallucination fallback. Coerce null to empty so one thin result doesn't sink the whole search.
    # The published date is kept when Tavily has one (news results carry it): "latest" questions are
    # the agent's main use, and without it every result read as undated.
    return "\n\n".join(
        f"- {r.get('title') or 'N/A'}\n  URL: {r.get('url') or ''}\n"
        + (f"  Published: {r['published_date']}\n" if r.get("published_date") else "")
        + f"  Content: {(r.get('content') or '')[:preview_chars]}"
        for r in results
    )


async def tavily_search(
    query: str,
    *,
    topic: str | None = None,
    time_range: str | None = None,
    include_domains: list[str] | None = None,
) -> str:
    client = _get_tavily_client()
    if not client:
        return "TAVILY_API_KEY not configured."

    config = get_config()
    agent_config = config.agent
    kwargs: dict = {"query": query, "max_results": agent_config.search_result_limit}
    # The digest's content-farm list applies here too, or copy sites reach the research reports.
    if config.collectors.web_search.exclude_domains:
        kwargs["exclude_domains"] = config.collectors.web_search.exclude_domains
    if topic:
        kwargs["topic"] = topic
    if time_range:
        kwargs["time_range"] = time_range
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

    cap = get_config().agent.research_content_cap_chars
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


def _openalex_abstract(inverted: dict[str, list[int]] | None) -> str:
    """OpenAlex ships abstracts as an inverted index (word -> positions); rebuild the text."""
    if not inverted:
        return ""
    words = sorted((pos, word) for word, positions in inverted.items() for pos in positions)
    return " ".join(word for _, word in words)


async def paper_search(query: str) -> str:
    """Scholarly works from OpenAlex (keyless, natural-language search). It replaced Semantic Scholar,
    whose keyless API answered 429 to every call from this machine, so papers never reached a report."""
    agent_config = get_config().agent
    params: dict[str, str | int] = {
        "search": query,
        # A work without an abstract gives the agent a title and nothing to judge it by.
        "filter": "has_abstract:true",
        "per-page": agent_config.search_result_limit,
        "select": "id,doi,title,publication_date,authorships,cited_by_count,primary_location,abstract_inverted_index",
    }
    async with httpx.AsyncClient(timeout=agent_config.search_request_timeout) as client:

        async def _fetch() -> httpx.Response:
            resp = await client.get(OPENALEX_WORKS_URL, params=params)
            if resp.status_code == 429 or resp.status_code >= 500:
                resp.raise_for_status()
            return resp

        try:
            response = await retry_async(
                _fetch,
                max_retries=agent_config.search_max_retries,
                backoff_sec=agent_config.search_retry_backoff_sec,
                retry_on=(httpx.HTTPStatusError,),
                description="OpenAlex paper search",
            )
        except httpx.HTTPError as e:
            logger.warning("OpenAlex paper search failed: %s", e)
            return f"SEARCH_FAILED: paper search unavailable ({e}). Could not retrieve papers."

        if response.status_code != 200:
            logger.warning("OpenAlex returned status %d", response.status_code)
            return f"SEARCH_FAILED: paper search returned status {response.status_code}."
        try:
            works = response.json().get("results", [])
        except ValueError:
            return "SEARCH_FAILED: invalid response from the paper search."
        if not works:
            return "No related papers found."

        results: list[str] = []
        for w in works:
            authors = ", ".join(
                name
                for a in (w.get("authorships") or [])[: agent_config.search_paper_max_authors]
                if (name := (a.get("author") or {}).get("display_name"))
            )
            url = (w.get("primary_location") or {}).get("landing_page_url") or w.get("doi") or w.get("id") or ""
            abstract = _openalex_abstract(w.get("abstract_inverted_index"))[
                : agent_config.search_paper_abstract_max_chars
            ]
            results.append(
                f"- {w.get('title') or 'N/A'} ({w.get('publication_date') or 'n.d.'}; cited by "
                f"{w.get('cited_by_count', 0)}) by {authors or 'unknown'}\n"
                f"  URL: {url}\n"
                f"  Abstract: {abstract}"
            )
        logger.info("Found %d papers for query '%s'", len(works), query)
        return "\n\n".join(results)
