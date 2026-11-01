"""Korean broker research as Naver's 증권 research pages list it (stock.naver.com/research): every
broker's company, industry, strategy, economy, market and fixed-income notes, each with the broker, the
date, a summary and, for company notes, the stock, the rating and the target price."""

from __future__ import annotations

import asyncio
from datetime import datetime, time, timedelta, tzinfo
from typing import Any
from zoneinfo import ZoneInfo

import httpx
from bs4 import BeautifulSoup

from domain.models import CollectedItem
from shared import SourceType, logger
from shared.config import NaverResearchCollectorConfig

from .base import FEED_FETCH_HEADERS, BaseCollector, cutoff_datetime, gather_collector_results, in_collection_window

API_BASE = "https://stock.naver.com/api/stockSecurity/researches/v2"
PAGE_URL = "https://stock.naver.com/research/{category}/{nid}"
# The API's own maximum page size.
_PAGE_SIZE = 50
_SEOUL = ZoneInfo("Asia/Seoul")


class NaverResearchCollector(BaseCollector):
    """One origin per broker (max_per_origin keeps one house from taking the pool), and the link is
    withheld from the post by default: the story names the broker and the analyst instead."""

    config: NaverResearchCollectorConfig

    def __init__(self, config: NaverResearchCollectorConfig):
        self.config = config

    async def collect(self) -> list[CollectedItem]:
        if not self.config.categories:
            return []
        zone = self.config.reference_time.tzinfo if self.config.reference_time else None
        zone = zone or _SEOUL
        end = self.config.reference_time or datetime.now(zone)
        start = cutoff_datetime(self.config.lookback_hours, end).astimezone(zone).date()
        last = (end - timedelta(seconds=1)).astimezone(zone).date()
        async with httpx.AsyncClient(
            headers=FEED_FETCH_HEADERS, timeout=self.config.request_timeout, follow_redirects=True
        ) as client:
            tasks = [
                self._collect_category(client, category, start.isoformat(), last.isoformat(), zone)
                for category in self.config.categories
            ]
            result = await gather_collector_results(
                tasks, labels=[str(c) for c in self.config.categories], raise_if_all_failed=True
            )
        logger.info(
            "Naver research gathered %d reports from %d/%d categories (%d failed)",
            len(result.items),
            result.total - result.failed - result.empty,
            result.total,
            result.failed,
        )
        self.record_run_health(total=result.total, failed=result.failed, empty=result.empty, what="categories")
        return result.items

    async def _collect_category(
        self, client: httpx.AsyncClient, category: str, start: str, end: str, zone: tzinfo
    ) -> list[CollectedItem]:
        items: list[CollectedItem] = []
        for index in range(self.config.max_pages):
            resp = await client.get(
                f"{API_BASE}/{category}",
                params={"index": index, "size": _PAGE_SIZE, "startDate": start, "endDate": end},
            )
            resp.raise_for_status()
            payload = resp.json()
            for row in payload.get("items") or []:
                item = self._to_item(category, row, zone)
                if item is not None:
                    items.append(item)
            if not payload.get("hasNext"):
                break
            await asyncio.sleep(self.config.page_delay_sec)
        return items

    def _to_item(self, category: str, row: dict[str, Any], zone: tzinfo) -> CollectedItem | None:
        nid, title, broker = (
            str(row.get("nid") or ""),
            str(row.get("title") or "").strip(),
            str(row.get("brokerName") or ""),
        )
        try:
            day = datetime.strptime(str(row.get("writeDate")), "%Y-%m-%d").date()
        except ValueError:
            return None
        # The API dates a report by day only: stamped at the END of that day, so a report from the day
        # before a morning run is not cut by a window that starts mid-morning (unshipped repeats are fine;
        # the shipped-URL ledger keeps a story from running twice).
        published = datetime.combine(day, time(23, 59, 59), tzinfo=zone)
        if not (nid and title) or not in_collection_window(
            published, lookback_hours=self.config.lookback_hours, reference_time=self.config.reference_time
        ):
            return None
        stock = str(row.get("itemName") or "")
        header = [broker]
        if row.get("analystName"):
            header.append(f"애널리스트 {row['analystName']}")
        if stock:
            header.append(stock)
        if row.get("opinionText"):
            header.append(f"투자의견 {row['opinionText']}")
        if str(row.get("goalPrice") or "").isdigit() and int(row["goalPrice"]) > 0:
            header.append(f"목표주가 {int(row['goalPrice']):,}원")
        if row.get("industryKoreanName"):
            header.append(f"업종 {row['industryKoreanName']}")
        summary = BeautifulSoup(str(row.get("content") or ""), "html.parser").get_text("\n", strip=True)
        return CollectedItem(
            item_id=f"naver-research-{category}-{nid}",
            source_type=SourceType.RSS,
            title=f"{stock}: {title}" if stock else title,
            url=PAGE_URL.format(category=category, nid=nid),
            text=f"{' · '.join(header)} · {day.isoformat()}\n{summary}",
            author=broker,
            published_at=published,
            metadata={
                # One origin per broker, and the broker as the source line ("신한투자증권 리서치").
                "feed_url": f"naver-research:{broker}",
                "feed_title": f"{broker} 리서치",
                "research_category": category,
                "hide_link": not self.config.show_links,
            },
        )
