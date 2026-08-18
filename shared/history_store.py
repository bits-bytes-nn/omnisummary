from __future__ import annotations

from datetime import date
from typing import Any

from .logger import logger
from .state_store import StateReadError, StateStore


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
THREADS_POSTED_KEY = "threads_posted.json"


class ThreadsPostLedger:
    """Idempotency marker for the daily Threads post. Records the dates a digest has already
    been published to Threads so a re-run — a same-day manual `main.py`, or an automatic
    async retry of the visual Lambda after a timeout — doesn't post the whole root+replies
    set again. Persisted as a capped {date: owner_run_id} map in the StateStore.

    The caller marks the date BEFORE starting the (multi-minute) post and rolls back via
    unmark() if the post fails, so concurrent invocations racing the same date don't all
    pass the already_posted() check during the long publish window.

    unmark() is OWNERSHIP-SCOPED: it only releases a marker this run wrote (matching run_id).
    Without that, invocation B failing could delete the marker invocation A wrote for a post
    that SUCCEEDED — and the next scheduled run would then re-post a duplicate digest. The
    StateStore read-modify-write is still not atomic (no lock), but ownership scoping closes
    the specific 'a peer's failure erases my success' hole."""

    MAX_DATES = 30

    def __init__(self, store: StateStore) -> None:
        self.store = store

    def _marks(self) -> dict[str, str] | None:
        """Return {iso_date: owner_run_id}, tolerating the legacy list-of-dates format.
        None when the store could not be READ — the caller must then treat the ledger as unknown
        and write nothing (a blind write would erase every recorded date)."""
        try:
            data = self.store.read_json(THREADS_POSTED_KEY, default={})
        except StateReadError as e:
            logger.error("Threads post ledger unreadable (%s); treating post history as unknown", e)
            return None
        if isinstance(data, dict):
            return {k: v for k, v in data.items() if isinstance(k, str) and isinstance(v, str)}
        if isinstance(data, list):  # legacy: bare ISO dates, no owner
            return {d: "" for d in data if isinstance(d, str)}
        return {}

    def _write(self, marks: dict[str, str]) -> None:
        # Cap to the most recent MAX_DATES by ISO date (lexicographic == chronological).
        trimmed = dict(sorted(marks.items())[-self.MAX_DATES :])
        self.store.write_json(THREADS_POSTED_KEY, trimmed)

    def already_posted(self, today: date) -> bool:
        """False when the ledger can't be read: an unknown history must not block the publish
        (that would turn a state-store blip into a lost digest). A duplicate post is recoverable;
        a missing one is not."""
        marks = self._marks()
        return marks is not None and today.isoformat() in marks

    def mark(self, today: date, run_id: str = "") -> None:
        marks = self._marks()
        if marks is None:
            logger.error("Not marking %s as posted: the Threads ledger could not be read", today)
            return
        iso = today.isoformat()
        if iso in marks:
            return
        marks[iso] = run_id
        self._write(marks)

    def unmark(self, today: date, run_id: str = "") -> None:
        """Release a date marked optimistically before a post that then failed, so the post
        stays retryable. Only releases a marker THIS run owns (its run_id) — never a marker a
        concurrent run wrote for a post that may have succeeded. No-op if absent or not owned."""
        iso = today.isoformat()
        marks = self._marks()
        if marks is None:
            logger.error("Not releasing the Threads marker for %s: the ledger could not be read", iso)
            return
        owner = marks.get(iso)
        if owner is None:
            return
        # Own it if the run_id matches, or if the marker is unowned (legacy/empty) — the latter
        # preserves prior single-writer behavior when no run_id is threaded through.
        if owner and run_id and owner != run_id:
            logger.info("Not releasing Threads marker for %s: owned by another run", iso)
            return
        del marks[iso]
        self._write(marks)


class PublishedUrlLedger:
    """Rolling map of normalized-URL -> last-published ISO date, persisted in the StateStore
    alongside trends.json. Lets the aggregator drop articles already published in the last N
    days so the same story isn't re-summarized days apart."""

    def __init__(self, store: StateStore, ttl_days: int) -> None:
        self.store = store
        self.ttl_days = ttl_days

    def _ledger(self) -> dict | None:
        """The stored {url: iso_date} map, or None when the store could not be read (unknown
        history — the caller must not write, or the whole ledger is replaced by today's URLs)."""
        try:
            ledger = self.store.read_json(PUBLISHED_URLS_KEY, default={}) or {}
        except StateReadError as e:
            logger.error("Published-URL ledger unreadable (%s); treating publish history as unknown", e)
            return None
        return ledger if isinstance(ledger, dict) else {}

    def recent_urls(self, today: date) -> set[str]:
        """URLs published on a STRICTLY EARLIER day within the window. Same-day (age 0) is
        excluded from the set so a same-day re-run reproduces today's digest rather than
        suppressing its own just-recorded stories; within-run duplicates are handled by the
        aggregator's own dedup, not here. An unreadable ledger yields no exclusions (a repeat
        story is a far smaller cost than an aborted run)."""
        ledger = self._ledger()
        if ledger is None:
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
        ledger = self._ledger()
        if ledger is None:
            # Skip the write: a read-modify-write on an unknown ledger would replace weeks of
            # publish history with just today's URLs, re-opening every story for re-publication.
            logger.error("Not recording %d published URL(s): the ledger could not be read", len(urls))
            return
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

    def _log(self) -> list[dict] | None:
        """The stored records, or None when the store could not be read (unknown history)."""
        try:
            data = self.store.read_json(self.key, default=[]) or []
        except StateReadError as e:
            logger.error("Rolling log '%s' unreadable (%s); treating history as unknown", self.key, e)
            return None
        return data if isinstance(data, list) else []

    def entries(self) -> list[dict]:
        """The stored records; [] when unreadable. Called from the publish path (the visual reads
        its format window before posting), so it must never raise."""
        return self._log() or []

    def append(self, record: dict, dedup_key: str | None = None) -> None:
        """Append a record, capped FIFO. When dedup_key is given, first drop any existing entry
        whose value at that key equals this record's — so a same-day re-run (e.g.
        --force-republish) replaces its prior entry instead of pushing a duplicate that would
        crowd out the window (the ledger and trends paths dedup by date the same way).

        Skipped entirely when the log could not be read: writing [record] would discard the whole
        window and reset the variation/novelty signals it exists to provide."""
        log = self._log()
        if log is None:
            logger.error("Not appending to rolling log '%s': it could not be read", self.key)
            return
        if dedup_key is not None and dedup_key in record:
            log = [e for e in log if e.get(dedup_key) != record[dedup_key]]
        log.append(record)
        # max_entries == 0 disables the window; log[-0:] is the WHOLE list, so the cap silently
        # became "keep everything" — the one setting meant to keep nothing grew without bound.
        self.store.write_json(self.key, log[-self.max_entries :] if self.max_entries > 0 else [])
