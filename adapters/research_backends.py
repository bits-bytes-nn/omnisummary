from __future__ import annotations

import asyncio
import math
import re
import time
from datetime import UTC, datetime, timedelta, timezone, tzinfo
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from tavily import AsyncTavilyClient

from adapters.aws import resolve_secret
from adapters.og_image import fetch_og_image
from domain.models import ImageAsset
from shared import get_config, logger, retry_async
from shared.constants import BROWSER_USER_AGENT

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
        _result_block(
            r.get("title") or "N/A",
            r.get("url") or "",
            [("Published", r.get("published_date") or ""), ("Content", (r.get("content") or "")[:preview_chars])],
        )
        for r in results
    )


def _result_block(heading: str, url: str, fields: list[tuple[str, str]]) -> str:
    """One search or paper result. Every field is put on one line and the URL is its first token, so
    only this code can start a line with "URL:": a snippet, an author name or a date carrying its own
    "\nURL: ..." line read as a result's URL and passed the research agent's citation check."""
    lines = [f"- {_one_line(heading)}", f"  URL: {(url.split() or [''])[0]}"]
    lines += [f"  {label}: {_one_line(value)}" for label, value in fields if value]
    return "\n".join(lines)


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


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
        # The exception text is logged, not returned: it can carry the request URL, which the agent
        # would then be allowed to cite.
        return f"Search failed ({type(e).__name__})."
    finally:
        await client.close()


# A page that answers an ordinary reader this way has an access control in front of it: a bot check,
# a login or a rate limit.
_REFUSED_STATUS = frozenset({401, 403, 429})
# schema.org's isAccessibleForFree, the publisher's own declaration that the page is paid content.
_NOT_FREE = re.compile(r'"isAccessibleForFree"\s*:\s*"?false"?', re.IGNORECASE)
_ACCESS_CHECK_CONCURRENCY = 10
_ACCESS_CHECK_TIMEOUT_SEC = 10.0
# The declaration sits in the page's JSON-LD; a result that is a large PDF is not downloaded whole.
_ACCESS_CHECK_MAX_BYTES = 512_000


async def access_controlled(urls: list[str]) -> set[str]:
    """The URLs whose page refuses an ordinary reader (401/403/429) or declares itself paid. Tavily
    reads a page from its own servers, so extracting one of these would get around the control the
    publisher put in front of it (docs/design.md, 판). Checked per page, not per host: a host list only
    ever learned of a paywall after a story from it had shipped. For web search results and research
    reads, a few dozen pages over many hosts; RSS fill pages are not checked (hundreds per run on a few
    hosts, enough to trip a publisher's WAF). A page that cannot be reached at all is an outage, not a
    control, and passes."""
    if not urls:
        return set()
    semaphore = asyncio.Semaphore(_ACCESS_CHECK_CONCURRENCY)

    async def refused(client: httpx.AsyncClient, url: str) -> bool:
        async with semaphore:
            try:
                async with client.stream("GET", url) as resp:
                    if resp.status_code in _REFUSED_STATUS:
                        return True
                    head = b""
                    async for chunk in resp.aiter_bytes():
                        head += chunk
                        if len(head) >= _ACCESS_CHECK_MAX_BYTES:
                            break
            except Exception as e:
                # Unreachable, or a URL httpx cannot even form (a malformed IDN): not a control.
                logger.debug("Access check could not read '%s': %s", url, e)
                return False
        return bool(_NOT_FREE.search(head.decode("utf-8", "ignore")))

    async with httpx.AsyncClient(
        headers={"User-Agent": BROWSER_USER_AGENT}, timeout=_ACCESS_CHECK_TIMEOUT_SEC, follow_redirects=True
    ) as client:
        unique = list(dict.fromkeys(urls))
        verdicts = await asyncio.gather(*(refused(client, u) for u in unique))
    blocked = {u for u, no in zip(unique, verdicts, strict=True) if no}
    if blocked:
        logger.info(
            "Access check: %d of %d pages refuse a reader or declare themselves paid (%s)",
            len(blocked),
            len(unique),
            ", ".join(sorted({urlsplit(u).hostname or u for u in blocked})),
        )
    return blocked


async def extract_url(url: str) -> str:
    """Fetch and extract the full readable text of a single page via Tavily extract,
    capped at the configured research content limit. Used by the deep-research agent's
    read_url tool to pull a primary source it found via search. "" when nothing readable came
    back (the reason is logged), so the caller can tell a read page from a failed one."""
    client = _get_tavily_client()
    if not client:
        logger.warning("TAVILY_API_KEY not configured; cannot read '%s'", url)
        return ""

    cap = get_config().agent.research_content_cap_chars
    try:
        response = await client.extract(urls=[url], format="text")
    except Exception as e:
        logger.warning("Tavily extract failed for '%s': %s", url, e)
        return ""
    finally:
        await client.close()

    results = response.get("results", [])
    content = (results[0].get("raw_content") or results[0].get("content") or "") if results else ""
    if not content:
        logger.warning("No readable content extracted from '%s'", url)
    return content[:cap]


# Tavily extract takes at most 20 URLs per request.
EXTRACT_BATCH_SIZE = 20


async def extract_pages(urls: list[str]) -> dict[str, str]:
    """Page text for each URL Tavily could read, keyed by URL. Best-effort: a failed batch or a
    missing key yields fewer entries, never an error, so a caller keeps whatever text it had."""
    client = await asyncio.to_thread(_get_tavily_client)
    if not client or not urls:
        return {}
    pages: dict[str, str] = {}
    try:
        for start in range(0, len(urls), EXTRACT_BATCH_SIZE):
            try:
                response = await client.extract(urls=urls[start : start + EXTRACT_BATCH_SIZE], format="text")
            except Exception as e:
                logger.warning(
                    "Tavily extract failed for %d URL(s): %s", len(urls[start : start + EXTRACT_BATCH_SIZE]), e
                )
                continue
            for result in response.get("results", []):
                content = result.get("raw_content") or result.get("content") or ""
                if result.get("url") and content:
                    pages[result["url"]] = content
    finally:
        await client.close()
    return pages


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
            return f"Search failed: paper search unavailable ({type(e).__name__})."

        if response.status_code != 200:
            logger.warning("OpenAlex returned status %d", response.status_code)
            return f"Search failed: paper search returned status {response.status_code}."
        try:
            works = response.json().get("results", [])
        except ValueError:
            return "Search failed: invalid response from the paper search."
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
            heading = (
                f"{w.get('title') or 'N/A'} ({w.get('publication_date') or 'n.d.'}; cited by "
                f"{w.get('cited_by_count', 0)}) by {authors or 'unknown'}"
            )
            results.append(_result_block(heading, url, [("Abstract", abstract)]))
        logger.info("Found %d papers for query '%s'", len(works), query)
        return "\n\n".join(results)


class WebResearchSources:
    """The research agent's reading: Tavily for the web and community search and page reads, OpenAlex
    for papers, and a source page's og:image."""

    async def web_search(
        self,
        query: str,
        *,
        topic: str | None = None,
        time_range: str | None = None,
        include_domains: list[str] | None = None,
    ) -> str:
        return await tavily_search(query, topic=topic, time_range=time_range, include_domains=include_domains)

    async def paper_search(self, query: str) -> str:
        return await paper_search(query)

    async def read_page(self, url: str) -> str:
        if await access_controlled([url]):
            return ""
        return await extract_url(url)

    async def fetch_image(self, url: str) -> ImageAsset | None:
        return await fetch_og_image(url)

    async def market_quote(self, symbol: str, period: str = "5d") -> str:
        return await yahoo_quote(symbol, period)

    async def analyst_ratings(self, symbol: str) -> str:
        return await yahoo_analyst_ratings(symbol)


_QUOTE_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
_SUMMARY_URL = "https://query2.finance.yahoo.com/v10/finance/quoteSummary/{symbol}"
_CRUMB_URL = "https://query2.finance.yahoo.com/v1/test/getcrumb"
_RATING_DAYS = 90
_SHOWN_ACTIONS = 10


def _ticker(symbol: str) -> str:
    """The upper-cased ticker, or "" when it is not one (Yahoo tickers are short ASCII: 005930.KS, ^GSPC, KRW=X)."""
    symbol = symbol.strip().upper()
    ok = symbol and len(symbol) <= 20 and symbol.isascii() and all(c.isalnum() or c in "^.-=" for c in symbol)
    return symbol if ok else ""


def _price(value: float) -> str:
    """Thousands separators, cents below 1,000 and significant digits below 1 (225,000 / 344.60 / 0.0123)."""
    if abs(value) >= 1000:
        return f"{value:,.0f}"
    return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4g}"


# Yahoo's chart ranges. Up to a year the closes are DAILY, so the change and the period's high and low are
# the real ones (weekly bars were dated by the week's first day and missed a mid-week peak); 5y is monthly.
QUOTE_PERIODS = {"5d": "1d", "1mo": "1d", "6mo": "1d", "1y": "1d", "5y": "1mo"}
_SHOWN_CLOSES = 8


async def yahoo_quote(symbol: str, period: str = "5d") -> str:
    """Closes over `period` from Yahoo Finance's chart endpoint, with the period's high and low, for
    Korean (005930.KS), US (AAPL), index (^GSPC, ^KS11), FX (KRW=X) and crypto (BTC-USD) tickers alike.
    Dates are the exchange's local dates. A bar still trading (the Korean session, crypto at any hour) is
    labelled the latest price, not a close."""
    symbol = _ticker(symbol)
    if not symbol:
        return ""
    if period not in QUOTE_PERIODS:
        period = "5d"
    config = get_config().agent
    try:
        async with httpx.AsyncClient(
            timeout=config.search_request_timeout, headers={"User-Agent": "Mozilla/5.0"}
        ) as client:
            resp = await client.get(
                _QUOTE_URL.format(symbol=symbol), params={"range": period, "interval": QUOTE_PERIODS[period]}
            )
            resp.raise_for_status()
        result = ((resp.json() or {}).get("chart") or {}).get("result")
        first = result[0] if isinstance(result, list) and result else None
        return _format_quote(symbol, period, first) if isinstance(first, dict) else ""
    except (
        httpx.HTTPError,
        ValueError,
        TypeError,
        AttributeError,
        IndexError,
        KeyError,
        OverflowError,
        OSError,
        RecursionError,
    ) as e:
        # One odd payload must not cost the other tickers of the same call.
        logger.warning("Quote for '%s' failed: %s", symbol, e)
        return ""


def _format_quote(symbol: str, period: str, result: dict) -> str:
    meta = result.get("meta") or {}
    # Each bar dated in the exchange's own zone, daylight saving included: one offset for every bar put
    # summer bars stamped at local midnight (US monthly, FX and futures dailies) on the previous day.
    offset = timezone(timedelta(seconds=int(meta.get("gmtoffset") or 0)))
    try:
        name = meta.get("exchangeTimezoneName")
        zone: tzinfo = ZoneInfo(str(name)) if name else offset
    except (ZoneInfoNotFoundError, ValueError):
        zone = offset
    stamps = result.get("timestamp") or []
    closes = ((result.get("indicators") or {}).get("quote") or [{}])[0].get("close") or []
    days = [
        (datetime.fromtimestamp(int(t), zone).date().isoformat(), float(c))
        for t, c in zip(stamps, closes, strict=False)
        if isinstance(c, int | float) and math.isfinite(c)
    ]
    if not days:
        return ""
    last_day, last = days[-1]
    first_day, first = days[0]
    change = (last / first - 1) * 100 if first else 0.0
    high_day, high = max(days, key=lambda d: d[1])
    low_day, low = min(days, key=lambda d: d[1])
    step = -(-len(days) // _SHOWN_CLOSES)
    shown = dict(days[::step] + [days[-1]])
    # Still trading when the last trade falls inside the current regular session (crypto: always). On a
    # weekend the "current" session is the next one, which the last trade precedes.
    regular = (meta.get("currentTradingPeriod") or {}).get("regular") or {}
    now = time.time()
    live = period != "5y" and all(isinstance(v, int | float) for v in (regular.get("start"), regular.get("end")))
    live = (
        live
        and regular["start"] <= now < regular["end"]
        and last_day == datetime.fromtimestamp(now, zone).date().isoformat()
    )
    label = "Latest price (session open)" if live else "Latest (month to date)" if period == "5y" else "Last close"
    unit = "monthly" if QUOTE_PERIODS[period] == "1mo" else "daily"
    return _result_block(
        f"{meta.get('longName') or meta.get('shortName') or symbol} ({symbol})",
        f"https://finance.yahoo.com/quote/{symbol}",
        [
            (label, f"{_price(last)} {meta.get('currency') or ''} on {last_day}"),
            ("Closes", ", ".join(f"{d} {_price(c)}" for d, c in shown.items())),
            ("Change", f"{change:+.2f}% since {first_day} ({period})"),
            ("Period high / low", f"{_price(high)} on {high_day} / {_price(low)} on {low_day} ({unit} closes)"),
            ("Exchange", str(meta.get("exchangeName") or "")),
        ],
    )


async def yahoo_analyst_ratings(symbol: str) -> str:
    """Yahoo Finance's analyst consensus for one stock (analyst count, mean/high/low target, this month's
    rating counts) and its dated rating and target actions of the last _RATING_DAYS days, newest first.
    Korean tickers carry the consensus but no per-firm actions."""
    symbol = _ticker(symbol)
    if not symbol:
        return ""
    config = get_config().agent
    try:
        async with httpx.AsyncClient(
            timeout=config.search_request_timeout, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True
        ) as client:
            # quoteSummary wants the session cookie and its crumb, the same handshake the quote page does.
            await client.get("https://fc.yahoo.com")
            crumb = (await client.get(_CRUMB_URL)).text.strip()
            resp = await client.get(
                _SUMMARY_URL.format(symbol=symbol),
                params={"modules": "financialData,recommendationTrend,upgradeDowngradeHistory", "crumb": crumb},
            )
            resp.raise_for_status()
        result = ((resp.json() or {}).get("quoteSummary") or {}).get("result")
        first = result[0] if isinstance(result, list) and result else None
        return _format_ratings(symbol, first, time.time()) if isinstance(first, dict) else ""
    except (httpx.HTTPError, ValueError, TypeError, AttributeError, IndexError, KeyError, OverflowError) as e:
        logger.warning("Analyst ratings for '%s' failed: %s", symbol, e)
        return ""


def _raw(value: object) -> float | None:
    raw = value.get("raw") if isinstance(value, dict) else value
    return float(raw) if isinstance(raw, int | float) and math.isfinite(raw) else None


def _format_ratings(symbol: str, result: dict, now: float) -> str:
    data = result.get("financialData") or {}
    currency = str(data.get("financialCurrency") or "")
    count, mean, high, low = (
        _raw(data.get(k)) for k in ("numberOfAnalystOpinions", "targetMeanPrice", "targetHighPrice", "targetLowPrice")
    )
    rows = (result.get("recommendationTrend") or {}).get("trend") or []
    trend: dict = next((row for row in rows if isinstance(row, dict) and row.get("period") == "0m"), {})
    counts = ", ".join(
        f"{label} {int(trend[key])}"
        for key, label in (
            ("strongBuy", "strong buy"),
            ("buy", "buy"),
            ("hold", "hold"),
            ("sell", "sell"),
            ("strongSell", "strong sell"),
        )
        if isinstance(trend.get(key), int)
    )
    actions = []
    for row in (result.get("upgradeDowngradeHistory") or {}).get("history") or []:
        stamp = row.get("epochGradeDate")
        if not isinstance(stamp, int | float) or now - stamp > _RATING_DAYS * 86400:
            continue
        grade = (
            f"{row.get('fromGrade') or '-'} -> {row.get('toGrade') or '-'}"
            if row.get("fromGrade")
            else str(row.get("toGrade") or "")
        )
        target, prior = _raw(row.get("currentPriceTarget")), _raw(row.get("priorPriceTarget"))
        if target:
            grade += f", target {_price(prior)} -> {_price(target)}" if prior else f", target {_price(target)}"
        day = datetime.fromtimestamp(stamp, UTC).date().isoformat()
        actions.append((stamp, f"{day} {row.get('firm') or '?'}: {grade} ({row.get('action') or ''})"))
    actions.sort(reverse=True)
    if not (count or counts or actions):
        return ""
    target_text = f"mean {_price(mean)} {currency}" if mean else ""
    if mean and high and low:
        target_text += f" (high {_price(high)}, low {_price(low)})"
    return _result_block(
        f"Analyst ratings for {symbol} (Yahoo Finance, as of {datetime.fromtimestamp(now, UTC).date().isoformat()})",
        f"https://finance.yahoo.com/quote/{symbol}/analysis",
        [
            ("Analysts", str(int(count)) if count else ""),
            ("Price target", target_text),
            ("Ratings this month", counts),
            (
                f"Actions in the last {_RATING_DAYS} days (newest first)",
                "; ".join(text for _, text in actions[:_SHOWN_ACTIONS]),
            ),
        ],
    )
