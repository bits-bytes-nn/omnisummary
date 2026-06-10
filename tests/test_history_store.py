from datetime import date, timedelta

from shared.history_store import (
    PUBLISHED_URLS_KEY,
    THREADS_POSTED_KEY,
    PublishedUrlLedger,
    RollingLog,
    ThreadsPostLedger,
    published_urls_from_snapshots,
)
from shared.state_store import StateStore


class _MemStore(StateStore):
    def __init__(self) -> None:
        self.blobs: dict[str, str] = {}

    def read(self, key: str) -> str | None:
        return self.blobs.get(key)

    def write(self, key: str, content: str) -> None:
        self.blobs[key] = content

    def exists(self, key: str) -> bool:
        return key in self.blobs


class TestPublishedUrlLedger:
    def test_record_then_recent_within_ttl(self):
        store = _MemStore()
        ledger = PublishedUrlLedger(store, ttl_days=6)
        # Recorded yesterday → excluded on a later day within the window.
        ledger.record(["https://a.com/x", "https://b.com/y"], date(2026, 6, 8))
        assert ledger.recent_urls(date(2026, 6, 9)) == {"https://a.com/x", "https://b.com/y"}

    def test_same_day_not_excluded(self):
        # A same-day re-run must reproduce today's digest, not suppress its own just-recorded
        # stories — within-run dedup handles same-day duplicates, not the cross-day ledger.
        store = _MemStore()
        ledger = PublishedUrlLedger(store, ttl_days=6)
        today = date(2026, 6, 9)
        ledger.record(["https://a.com/x"], today)
        assert ledger.recent_urls(today) == set()

    def test_entries_outside_ttl_are_pruned(self):
        store = _MemStore()
        store.write_json(PUBLISHED_URLS_KEY, {"https://old.com": "2026-06-01", "https://new.com": "2026-06-08"})
        ledger = PublishedUrlLedger(store, ttl_days=6)
        today = date(2026, 6, 9)
        # old.com is 8 days back (>= ttl 6) → excluded; new.com is 1 day back → kept.
        assert ledger.recent_urls(today) == {"https://new.com"}

    def test_record_prunes_stale_on_write(self):
        store = _MemStore()
        store.write_json(PUBLISHED_URLS_KEY, {"https://old.com": "2026-05-01"})
        ledger = PublishedUrlLedger(store, ttl_days=6)
        today = date(2026, 6, 9)
        ledger.record(["https://fresh.com"], today)
        kept = store.read_json(PUBLISHED_URLS_KEY)
        assert "https://old.com" not in kept
        assert kept["https://fresh.com"] == "2026-06-09"

    def test_corrupt_ledger_degrades_to_empty(self):
        store = _MemStore()
        store.write(PUBLISHED_URLS_KEY, "{ not json")
        ledger = PublishedUrlLedger(store, ttl_days=6)
        assert ledger.recent_urls(date(2026, 6, 9)) == set()


class TestPublishedUrlsFromSnapshots:
    def test_extracts_item_urls_from_digest_snapshots(self):
        snaps = [
            {"digest_result": {"content": {"items": [{"url": "https://a.com/1"}, {"url": "https://b.com/2"}]}}},
            {"digest_result": {"content": {"items": [{"url": "https://c.com/3"}]}}},
        ]
        assert published_urls_from_snapshots(snaps) == {"https://a.com/1", "https://b.com/2", "https://c.com/3"}

    def test_tolerates_missing_or_malformed_snapshots(self):
        snaps = [
            {},
            {"digest_result": None},
            {"digest_result": {"content": {}}},
            {"digest_result": {"content": {"items": [{"no_url": 1}, None]}}},
        ]
        assert published_urls_from_snapshots(snaps) == set()
        assert published_urls_from_snapshots([]) == set()


class TestThreadsPostLedger:
    def test_mark_then_already_posted(self):
        ledger = ThreadsPostLedger(_MemStore())
        today = date(2026, 6, 10)
        assert ledger.already_posted(today) is False
        ledger.mark(today)
        assert ledger.already_posted(today) is True

    def test_other_day_not_posted(self):
        ledger = ThreadsPostLedger(_MemStore())
        ledger.mark(date(2026, 6, 10))
        assert ledger.already_posted(date(2026, 6, 11)) is False

    def test_mark_is_idempotent(self):
        store = _MemStore()
        ledger = ThreadsPostLedger(store)
        ledger.mark(date(2026, 6, 10))
        ledger.mark(date(2026, 6, 10))
        assert store.read_json(THREADS_POSTED_KEY) == ["2026-06-10"]

    def test_dates_are_capped(self):
        store = _MemStore()
        ledger = ThreadsPostLedger(store)
        for d in range(1, 1 + ThreadsPostLedger.MAX_DATES + 5):
            ledger.mark(date(2026, 1, 1) + timedelta(days=d))
        assert len(store.read_json(THREADS_POSTED_KEY)) == ThreadsPostLedger.MAX_DATES

    def test_corrupt_blob_degrades_to_not_posted(self):
        store = _MemStore()
        store.write(THREADS_POSTED_KEY, "{ not json")
        ledger = ThreadsPostLedger(store)
        assert ledger.already_posted(date(2026, 6, 10)) is False


class TestRollingLog:
    def test_append_caps_to_max_entries(self):
        store = _MemStore()
        log = RollingLog(store, "leads.json", max_entries=3)
        for i in range(5):
            log.append({"lead": f"L{i}"})
        leads = [e["lead"] for e in log.entries()]
        assert leads == ["L2", "L3", "L4"]

    def test_entries_empty_when_unset(self):
        assert RollingLog(_MemStore(), "k.json", max_entries=3).entries() == []
