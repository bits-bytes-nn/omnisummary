"""Korean broker research as Naver's 증권 research pages list it (stock.naver.com/research): every
broker's company, industry, strategy, economy, market and fixed-income notes, each with the broker, the
date, a summary and, for company notes, the stock, the rating and the target price."""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup
from pypdf import PdfReader

from domain.models import CollectedItem
from shared import SourceType, logger, retry_async
from shared.config import NaverResearchCollectorConfig

from .base import (
    FEED_FETCH_HEADERS,
    RETRIABLE_STATUS_CODES,
    BaseCollector,
    TransientStatusError,
    cutoff_datetime,
    gather_collector_results,
    in_collection_window,
)

API_BASE = "https://stock.naver.com/api/stockSecurity/researches/v2"
PAGE_URL = "https://stock.naver.com/research/{category}/{nid}"
# The API's own maximum page size.
_PAGE_SIZE = 50
# writeDate is a Seoul date whatever the deployment's zone.
_SEOUL = ZoneInfo("Asia/Seoul")
# A note's attached PDF (stock.pstatic.net, public) is read at most this large; a chart-heavy outlook runs
# to a few MB.
_NOTE_MAX_BYTES = 30_000_000


class NaverResearchCollector(BaseCollector):
    """One origin per broker (max_per_origin keeps one house from taking the pool), and the link is
    withheld from the post by default: the story names the broker instead."""

    config: NaverResearchCollectorConfig

    def __init__(self, config: NaverResearchCollectorConfig):
        self.config = config
        self._partial: list[str] = []

    async def collect(self) -> list[CollectedItem]:
        if not self.config.categories:
            return []
        end = self.config.reference_time or datetime.now(_SEOUL)
        start = cutoff_datetime(self.config.lookback_hours, end).astimezone(_SEOUL).date()
        last = (end - timedelta(seconds=1)).astimezone(_SEOUL).date()
        categories = [str(c) for c in self.config.categories]
        self._partial = []
        async with httpx.AsyncClient(
            headers=FEED_FETCH_HEADERS, timeout=self.config.request_timeout, follow_redirects=True
        ) as client:
            tasks = [self._collect_category(client, c, start.isoformat(), last.isoformat()) for c in categories]
            result = await gather_collector_results(tasks, labels=categories, raise_if_all_failed=True)
        logger.info(
            "Naver research gathered %d reports from %d/%d categories (%d failed)",
            len(result.items),
            result.total - result.failed - result.empty,
            result.total,
            result.failed,
        )
        self.record_run_health(total=result.total, failed=result.failed, empty=result.empty, what="categories")
        if self._partial:
            self.degraded_detail = "; ".join(filter(None, [self.degraded_detail, *self._partial]))
            logger.warning("Collector is DEGRADED: %s", self.degraded_detail)
        return _one_item_per_stock_and_day(result.items)

    async def _collect_category(
        self, client: httpx.AsyncClient, category: str, start: str, end: str
    ) -> list[CollectedItem]:
        items: list[CollectedItem] = []
        rows_seen = dated = attributed = 0
        for index in range(self.config.max_pages):
            try:
                payload = await self._page(client, category, index, start, end)
            except Exception as e:
                if not index:
                    raise
                # The pages already read stay: one failed page must not cost the category's newest notes.
                logger.warning(
                    "Naver research '%s' page %d failed; keeping %d earlier page(s): %s", category, index, index, e
                )
                self._partial.append(f"'{category}' stopped at page {index} of the window ({e})")
                break
            if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
                raise ValueError(f"Naver research '{category}' answered without an items list")
            rows = payload["items"]
            rows_seen += len(rows)
            dated += sum(1 for row in rows if _dated(row) is not None)
            attributed += sum(1 for row in rows if row.get("brokerName"))
            items += [item for row in rows if (item := self._to_item(category, row)) is not None]
            if not payload.get("hasNext"):
                break
            await asyncio.sleep(self.config.page_delay_sec)
        else:
            logger.warning(
                "Naver research '%s' still had more pages after %d; older reports in the window were not read",
                category,
                self.config.max_pages,
            )
            self._partial.append(f"'{category}' still had pages after max_pages={self.config.max_pages}")
        if rows_seen and not dated:
            # Rows came back and none had a readable date: the API changed shape. Fail the category rather
            # than report it empty, which is what a quiet Sunday looks like.
            raise ValueError(f"Naver research '{category}' returned {rows_seen} rows with no readable writeDate")
        if rows_seen and not attributed:
            # Every row without a broker: a renamed field, not a quiet day.
            raise ValueError(f"Naver research '{category}' returned {rows_seen} rows with no brokerName")
        return items

    async def _page(self, client: httpx.AsyncClient, category: str, index: int, start: str, end: str) -> dict:
        async def _call() -> dict:
            resp = await client.get(
                f"{API_BASE}/{category}",
                params={"index": index, "size": _PAGE_SIZE, "startDate": start, "endDate": end},
            )
            if resp.status_code in RETRIABLE_STATUS_CODES:
                raise TransientStatusError(f"Naver research '{category}' page {index} returned {resp.status_code}")
            resp.raise_for_status()
            return resp.json()

        return await retry_async(
            _call,
            max_retries=self.config.max_retries,
            backoff_sec=self.config.retry_backoff_sec,
            retry_on=(httpx.TransportError, TransientStatusError),
            description=f"Naver research '{category}' page {index}",
        )

    def _to_item(self, category: str, row: dict[str, Any]) -> CollectedItem | None:
        nid, title, broker = (
            str(row.get("nid") or ""),
            str(row.get("title") or "").strip(),
            str(row.get("brokerName") or ""),
        )
        day = _dated(row)
        # No broker, no attribution: the domain rules name every analyst view's source.
        if day is None or not (nid and title and broker):
            return None
        # The API dates a report by day only: stamped at the END of that day, so a report from the day
        # before a morning run is not cut by a window that starts mid-morning (unshipped repeats are fine;
        # the shipped-URL ledger keeps a story from running twice).
        published = datetime.combine(day, time(23, 59, 59), tzinfo=_SEOUL)
        if not in_collection_window(
            published, lookback_hours=self.config.lookback_hours, reference_time=self.config.reference_time
        ):
            return None
        stock = str(row.get("itemName") or "")
        header = [broker] if broker else []
        if row.get("analystName"):
            header.append(f"애널리스트 {row['analystName']}")
        if stock:
            header.append(stock)
        if row.get("opinionText"):
            header.append(f"투자의견 {row['opinionText']}")
        target = _won(row.get("goalPrice"))
        if target:
            header.append(f"목표주가 {target:,}원")
        if row.get("industryKoreanName"):
            header.append(f"업종 {row['industryKoreanName']}")
        header.append(day.isoformat())
        summary = BeautifulSoup(str(row.get("content") or ""), "html.parser").get_text("\n", strip=True)
        return CollectedItem(
            item_id=f"naver-research-{category}-{nid}",
            source_type=SourceType.RESEARCH,
            # Strategy and economy notes share generic titles ("Weekly"); the broker keeps them apart in
            # title dedup.
            title=f"{stock}: {title}" if stock else f"{broker}: {title}" if broker else title,
            url=PAGE_URL.format(category=category, nid=nid),
            text=f"{' · '.join(header)}\n{summary}",
            author=broker,
            published_at=published,
            # Never linked: the post names the broker (owner's call, docs/design.md).
            show_link=False,
            metadata={
                "research_category": category,
                "stock": stock,
                "day": day.isoformat(),
                "opinion": str(row.get("opinionText") or ""),
                "target": target,
                "broker": broker,
                "nid": int(nid) if nid.isdigit() else 0,
            },
        )


def _won(value: object) -> int:
    """A target price in won (580000, "580,000", 580000.0), or 0."""
    digits = str(value or "").split(".")[0].replace(",", "")
    return int(digits) if digits.isdigit() else 0


def _dated(row: dict[str, Any]):
    try:
        return datetime.strptime(str(row.get("writeDate")), "%Y-%m-%d").date()
    except ValueError:
        return None


def _one_item_per_stock_and_day(items: list[CollectedItem]) -> list[CollectedItem]:
    """Several brokers' notes on one stock on one day become one item that lists every rating and target.

    On an earnings day most company notes fall on a few stocks: each note took a ranking slot of its own,
    and the pool filled with copies of one day (docs/decisions.md). Together they are the consensus a reader wants."""
    groups: dict[tuple[str, str], list[CollectedItem]] = {}
    out: list[CollectedItem] = []
    for item in items:
        stock = item.metadata.get("stock")
        if item.metadata.get("research_category") == "company" and stock:
            groups.setdefault((stock, item.metadata["day"]), []).append(item)
        else:
            out.append(item)
    for (stock, day), notes in groups.items():
        if len(notes) == 1:
            out.append(notes[0])
            continue
        # The earliest note (lowest nid) stands for the group: its URL is what a run that saw only that
        # note shipped, so the shipped-URL ledger still recognises the story when more notes arrive.
        notes.sort(key=lambda n: (n.metadata.get("nid", 0), n.item_id))
        brokers = len({n.metadata.get("broker") or n.author for n in notes})
        targets = [n.metadata["target"] for n in notes if n.metadata.get("target")]
        calls = ", ".join(
            f"{n.author} {n.metadata.get('opinion') or '의견 없음'}"
            + (f" {n.metadata['target']:,}원" if n.metadata.get("target") else "")
            for n in notes
        )
        span = f", 목표주가 {min(targets):,}~{max(targets):,}원" if targets else ""
        first = notes[0]
        out.append(
            first.model_copy(
                update={
                    "item_id": f"naver-research-company-{stock}-{day}",
                    "title": f"{stock}: 증권사 {brokers}곳 리포트{span}",
                    "text": f"{stock} · {day} · {calls}\n\n" + "\n\n".join(n.text for n in notes),
                    "author": f"{first.author} 외 {brokers - 1}곳" if brokers > 1 else first.author,
                    "metadata": {**first.metadata, "notes": len(notes)},
                }
            )
        )
    return out


async def note_text(item: CollectedItem, *, timeout: float, max_chars: int) -> str:
    """The text of a research note's attached PDF, which the detail API names (`attachUrl`), at most
    `max_chars`; "" for a merged consensus item (no single note), a note without an attachment, or anything
    unreadable. The PDF is public; a refusal is not retried around."""
    category, nid = item.metadata.get("research_category"), item.metadata.get("nid")
    if not (category and nid) or item.metadata.get("notes"):
        return ""
    try:
        # `timeout` bounds the download and, separately, the parse: httpx's own timeout is per read, and a
        # pypdf parse in a thread cannot be cancelled, only left behind.
        async with (
            asyncio.timeout(timeout),
            httpx.AsyncClient(headers=FEED_FETCH_HEADERS, timeout=timeout, follow_redirects=True) as client,
        ):
            detail = await client.get(f"{API_BASE}/{category}/{nid}")
            detail.raise_for_status()
            url = detail.json().get("attachUrl")
            if not url:
                return ""
            body = bytearray()
            async with client.stream("GET", url) as resp:
                resp.raise_for_status()
                async for chunk in resp.aiter_bytes():
                    body += chunk
                    if len(body) > _NOTE_MAX_BYTES:
                        logger.warning("Research note PDF over %d bytes, not read: %s", _NOTE_MAX_BYTES, url)
                        return ""
        return await asyncio.wait_for(asyncio.to_thread(_pdf_text, bytes(body), max_chars), timeout)
    except Exception as e:
        logger.warning("Research note text unreadable for '%s' (%s)", item.title[:60], e)
        return ""


def _pdf_text(data: bytes, max_chars: int) -> str:
    """Pages in order until `max_chars`: a few kilobytes of shared, compressed page content can expand to
    millions of characters (a 14.5 KB file gave 18.4M in 50 s), and the editor reads only its budget."""
    text = ""
    for page in PdfReader(io.BytesIO(data)).pages:
        text += (page.extract_text() or "") + "\n"
        if len(text) >= max_chars:
            break
    return text[:max_chars].strip()
