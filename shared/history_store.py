from __future__ import annotations

from datetime import date
from typing import Any

from .logger import logger
from .state_store import StateStore


def published_urls_from_snapshots(snapshots: list[dict[str, Any]]) -> set[str]:
    """Extract the URLs that were actually published in past digests from their stored
    snapshots (digest_result.content.items[].url). Lets cross-day dedup self-heal from
    AgentCore Memory history rather than only from the ledger a future run populates."""
    urls: set[str] = set()
    for snap in snapshots or []:
        content = ((snap or {}).get("digest_result") or {}).get("content") or {}
        for item in content.get("items") or []:
            url = item.get("url") if isinstance(item, dict) else None
            if url:
                urls.add(url)
    return urls


PUBLISHED_URLS_KEY = "published_urls.json"
RECENT_LEADS_KEY = "recent_leads.json"
VISUAL_FORMATS_KEY = "visual_formats.json"


class PublishedUrlLedger:
    """Rolling map of normalized-URL -> last-published ISO date, persisted in the StateStore
    alongside trends.json. Lets the aggregator drop articles already published in the last N
    days so the same story isn't re-summarized days apart."""

    def __init__(self, store: StateStore, ttl_days: int) -> None:
        self.store = store
        self.ttl_days = ttl_days

    def recent_urls(self, today: date) -> set[str]:
        """URLs published on a STRICTLY EARLIER day within the window. Same-day (age 0) is
        excluded from the set so a same-day re-run reproduces today's digest rather than
        suppressing its own just-recorded stories; within-run duplicates are handled by the
        aggregator's own dedup, not here."""
        ledger = self.store.read_json(PUBLISHED_URLS_KEY, default={}) or {}
        if not isinstance(ledger, dict):
            return set()
        keep: set[str] = set()
        for url, iso in ledger.items():
            try:
                age = (today - date.fromisoformat(iso)).days
                if 0 < age < self.ttl_days:
                    keep.add(url)
            except (ValueError, TypeError):
                continue
        return keep

    def record(self, urls: list[str], today: date) -> None:
        """Merge newly published URLs (stamped today) into the ledger and prune entries
        older than the TTL window."""
        if not urls:
            return
        ledger = self.store.read_json(PUBLISHED_URLS_KEY, default={}) or {}
        if not isinstance(ledger, dict):
            ledger = {}
        today_iso = today.isoformat()
        for url in urls:
            if url:
                ledger[url] = today_iso
        pruned = {}
        for url, iso in ledger.items():
            try:
                if (today - date.fromisoformat(iso)).days < self.ttl_days:
                    pruned[url] = iso
            except (ValueError, TypeError):
                continue
        self.store.write_json(PUBLISHED_URLS_KEY, pruned)
        logger.info("Published-URL ledger: +%d new, %d retained", len(urls), len(pruned))


class RollingLog:
    """A capped, FIFO list of small records persisted as one JSON blob. Used for recent
    digest leads (anti-repetition) and recent visual formats (variation)."""

    def __init__(self, store: StateStore, key: str, max_entries: int) -> None:
        self.store = store
        self.key = key
        self.max_entries = max_entries

    def entries(self) -> list[dict]:
        data = self.store.read_json(self.key, default=[]) or []
        return data if isinstance(data, list) else []

    def append(self, record: dict) -> None:
        log = self.entries()
        log.append(record)
        self.store.write_json(self.key, log[-self.max_entries :])
