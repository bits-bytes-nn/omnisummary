import json
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from shared.memory import AgentCoreMemoryStore, LocalMemoryStore, MemoryReadError, create_memory_store


def _store() -> AgentCoreMemoryStore:
    with patch("shared.memory.boto3.client"):
        return AgentCoreMemoryStore("m", actor_id="a")


class TestFitToLimit:
    def test_small_state_unchanged_and_valid(self):
        store = _store()
        state = {"ranked_items": [{"item": {"text": "hi"}, "score": 0.5}], "collected_items": {}, "digest_result": None}
        out = store._fit_to_limit(state)
        assert json.loads(out) == state  # valid JSON, unchanged

    def test_oversized_state_sheds_to_valid_json_under_limit(self):
        store = _store()
        big = "x" * 200_000
        state = {
            "ranked_items": [{"item": {"item_id": f"i{n}", "text": big}, "score": 0.5} for n in range(20)],
            "collected_items": {f"i{n}": {"text": big} for n in range(20)},
            "digest_result": {"digest_text": big, "content": {"lead": big, "headline_index": 1, "items": []}},
        }
        out = store._fit_to_limit(state)
        parsed = json.loads(out)  # must be valid JSON (never byte-sliced)
        assert len(out) <= store.MAX_EVENT_TEXT
        assert "ranked_items" in parsed  # still a well-formed snapshot dict

    def test_shedding_keeps_every_story_and_the_digest_content(self):
        # The visual Lambda publishes off THIS payload, so what survives the trim decides whether
        # the day has a digest. Assert the surviving SHAPE by parsing it (story count, content items
        # non-empty) — a substring check would pass on a snapshot that had shed all of them.
        store = _store()
        body = "x" * 30_000
        state = {
            "ranked_items": [{"item": {"item_id": f"i{n}", "text": body}, "score": 0.8} for n in range(4)],
            "collected_items": {f"i{n}": {"text": body} for n in range(4)},
            "digest_result": {
                "digest_text": "d",
                "content": {
                    "lead": "리드.",
                    "headline_index": 1,
                    "items": [{"title": f"스토리 {n}", "url": f"u{n}", "body": "b"} for n in range(5)],
                },
            },
        }
        parsed = json.loads(store._fit_to_limit(state))
        assert len(parsed["ranked_items"]) == 4
        assert [r["item"]["item_id"] for r in parsed["ranked_items"]] == ["i0", "i1", "i2", "i3"]
        assert len(parsed["digest_result"]["content"]["items"]) == 5


class TestLocalMemoryStore:
    def test_put_and_get_latest(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        store.put_digest("2026-06-01", {"a": 1})
        store.put_digest("2026-06-02", {"a": 2})
        latest = store.get_latest_digest()
        assert latest == {"a": 2}

    def test_get_latest_empty(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        assert store.get_latest_digest() is None

    def test_get_digest_by_date(self, tmp_path):
        # The visual Lambda must publish the date it was fired for, not whatever is newest:
        # 'load latest' published yesterday's stories when today's snapshot was missing.
        store = LocalMemoryStore(tmp_path)
        store.put_digest("2026-06-01", {"a": 1})
        store.put_digest("2026-06-02", {"a": 2})
        assert store.get_digest("2026-06-01") == {"a": 1}
        assert store.get_digest("2026-06-03") is None

    def test_get_recent_digests_newest_first_and_capped(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        for d in ("2026-06-05", "2026-06-06", "2026-06-07"):
            store.put_digest(d, {"d": d})
        recent = store.get_recent_digests(2)
        assert [r["d"] for r in recent] == ["2026-06-07", "2026-06-06"]

    def test_get_recent_digests_excludes_given_date(self, tmp_path):
        # A same-day re-run must not seed dedup with today's own snapshot.
        store = LocalMemoryStore(tmp_path)
        for d in ("2026-06-07", "2026-06-08", "2026-06-09"):
            store.put_digest(d, {"d": d})
        recent = store.get_recent_digests(6, exclude_date="2026-06-09")
        assert [r["d"] for r in recent] == ["2026-06-08", "2026-06-07"]

    def test_get_recent_digests_after_date_bounds_window(self, tmp_path):
        # after_date floors the seed to the TTL window so a stale snapshot outside it (here the
        # 06-01 one, before the 06-05 floor) can't suppress a legitimately-recurring story.
        store = LocalMemoryStore(tmp_path)
        for d in ("2026-06-01", "2026-06-06", "2026-06-08"):
            store.put_digest(d, {"d": d})
        recent = store.get_recent_digests(10, after_date="2026-06-05")
        assert [r["d"] for r in recent] == ["2026-06-08", "2026-06-06"]  # 06-01 excluded


class TestCreateMemoryStore:
    def test_local_when_no_memory_id(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEMORY_ID", raising=False)
        store = create_memory_store(tmp_path)
        assert isinstance(store, LocalMemoryStore)

    def test_agentcore_when_memory_id_set(self, monkeypatch):
        monkeypatch.setenv("MEMORY_ID", "mem-123")
        with patch("shared.memory.boto3.client") as mock_client:
            store = create_memory_store()
        assert isinstance(store, AgentCoreMemoryStore)
        assert store.memory_id == "mem-123"
        mock_client.assert_called_once()


class TestAgentCoreMemoryStore:
    def _store(self):
        with patch("shared.memory.boto3.client") as mock_client:
            client = MagicMock()
            mock_client.return_value = client
            store = AgentCoreMemoryStore("mem-1", region_name="us-west-2")
        return store, client

    def test_put_digest_creates_event(self):
        store, client = self._store()
        store.put_digest("2026-06-02", {"ranked_items": []})
        client.create_event.assert_called_once()
        kwargs = client.create_event.call_args.kwargs
        assert kwargs["memoryId"] == "mem-1"
        assert kwargs["sessionId"] == "digest-2026-06-02"
        payload = kwargs["payload"][0]["conversational"]
        assert payload["role"] == "ASSISTANT"
        assert "ranked_items" in payload["content"]["text"]

    def test_put_digest_trims_when_over_limit(self):
        store, client = self._store()
        big = {
            "collected_items": {"a": {"text": "x" * 200_000}},
            "ranked_items": [{"item": {"item_id": "a"}}],
            "digest_result": {"digest_text": "ok"},
        }
        store.put_digest("2026-06-02", big)
        text = client.create_event.call_args.kwargs["payload"][0]["conversational"]["content"]["text"]
        assert len(text) <= AgentCoreMemoryStore.MAX_EVENT_TEXT
        assert '"collected_items": {}' in text

    def test_put_digest_truncates_oversized_ranked_text(self):
        # Even after dropping collected_items, the ranked-item bodies alone exceed the
        # limit (this is what aborted the pipeline on 2026-06-04). Must still fit + store.
        store, client = self._store()
        big = {
            "collected_items": {},
            "ranked_items": [{"item": {"item_id": f"i{n}", "text": "y" * 30_000}, "score": 0.8} for n in range(5)],
            "digest_result": {"digest_text": "ok"},
        }
        store.put_digest("2026-06-02", big)
        text = client.create_event.call_args.kwargs["payload"][0]["conversational"]["content"]["text"]
        assert len(text) <= AgentCoreMemoryStore.MAX_EVENT_TEXT
        client.create_event.assert_called_once()  # stored, did not raise
        assert "ranked_items" in text

    def test_put_digest_minimal_fallback_when_still_too_large(self):
        # Pathological: many ranked items with huge text — falls back to metadata only.
        store, client = self._store()
        big = {
            "collected_items": {},
            "ranked_items": [{"item": {"item_id": f"i{n}", "text": "z" * 50_000}, "score": 0.8} for n in range(20)],
            "digest_result": {"digest_text": "ok"},
        }
        store.put_digest("2026-06-02", big)
        text = client.create_event.call_args.kwargs["payload"][0]["conversational"]["content"]["text"]
        assert len(text) <= AgentCoreMemoryStore.MAX_EVENT_TEXT
        assert "ranked_items" in text

    def test_get_latest_digest_picks_newest_session(self):
        store, client = self._store()
        client.list_sessions.return_value = {
            "sessionSummaries": [
                {"sessionId": "digest-2026-06-01"},
                {"sessionId": "digest-2026-06-02"},
                {"sessionId": "trend-2026-06-02"},
            ]
        }
        client.list_events.return_value = {
            "events": [{"payload": [{"conversational": {"content": {"text": '{"x": 9}'}}}]}]
        }
        result = store.get_latest_digest()
        assert result == {"x": 9}
        assert client.list_events.call_args.kwargs["sessionId"] == "digest-2026-06-02"

    def test_get_latest_digest_none_when_no_sessions(self):
        store, client = self._store()
        client.list_sessions.return_value = {"sessionSummaries": []}
        assert store.get_latest_digest() is None

    def test_get_latest_digest_follows_pagination(self):
        # The true latest session is on the SECOND page. A single-page (maxResults=100) lookup
        # would miss it and serve a stale snapshot; pagination must collect all pages first.
        store, client = self._store()
        page1 = {"sessionSummaries": [{"sessionId": "digest-2026-06-01"}], "nextToken": "tok"}
        page2 = {"sessionSummaries": [{"sessionId": "digest-2026-06-30"}]}
        client.list_sessions.side_effect = [page1, page2]
        client.list_events.return_value = {
            "events": [{"payload": [{"conversational": {"content": {"text": '{"x": 1}'}}}]}]
        }
        result = store.get_latest_digest()
        assert result == {"x": 1}
        assert client.list_events.call_args.kwargs["sessionId"] == "digest-2026-06-30"  # newest across pages
        assert client.list_sessions.call_count == 2

    def test_get_recent_digests_excludes_and_date_bounds(self):
        # Production path: newest-first, drops exclude_date (today) and anything before after_date.
        store, client = self._store()
        client.list_sessions.return_value = {
            "sessionSummaries": [
                {"sessionId": "digest-2026-06-02"},  # before floor → dropped
                {"sessionId": "digest-2026-06-07"},
                {"sessionId": "digest-2026-06-08"},
                {"sessionId": "digest-2026-06-09"},  # exclude_date → dropped
                {"sessionId": "trend-2026-06-08"},  # non-digest → ignored
            ]
        }
        client.list_events.side_effect = lambda **kw: {
            "events": [{"payload": [{"conversational": {"content": {"text": json.dumps({"sid": kw["sessionId"]})}}}]}]
        }
        recent = store.get_recent_digests(10, exclude_date="2026-06-09", after_date="2026-06-05")
        assert [r["sid"] for r in recent] == ["digest-2026-06-08", "digest-2026-06-07"]


class TestAgentCoreGetDigestByDate:
    def _store(self):
        with patch("shared.memory.boto3.client") as mock_client:
            client = MagicMock()
            mock_client.return_value = client
            store = AgentCoreMemoryStore("mem-1", region_name="us-west-2")
        return store, client

    def test_reads_the_session_for_that_date(self):
        store, client = self._store()
        client.list_events.return_value = {
            "events": [{"payload": [{"conversational": {"content": {"text": '{"a": 1}'}}}]}]
        }
        assert store.get_digest("2026-08-17") == {"a": 1}
        # Addressed directly by session id — never a list-then-compare-generated_at (that field is
        # UTC and disagrees with the KST digest date on every pre-09:00 KST run).
        assert client.list_events.call_args.kwargs["sessionId"] == "digest-2026-08-17"
        client.list_sessions.assert_not_called()

    def test_missing_session_is_none_not_a_stale_fallback(self):
        store, client = self._store()
        client.list_events.return_value = {"events": []}
        assert store.get_digest("2026-08-17") is None

    def test_api_failure_raises_instead_of_reading_as_an_empty_day(self):
        # A throttled/denied read used to come back as None — indistinguishable from "this day has
        # no digest" — so the visual Lambda skipped the day's only delivery and returned 200.
        store, client = self._store()
        client.list_events.side_effect = RuntimeError("throttled")
        with pytest.raises(MemoryReadError, match="digest-2026-08-17"):
            store.get_digest("2026-08-17")

    def test_a_genuinely_missing_snapshot_is_still_none(self):
        store, client = self._store()
        client.list_events.return_value = {"events": []}
        assert store.get_digest("2026-08-17") is None


class TestNewestEventWithinASession:
    """A session normally holds one event, but a same-day re-run appends another — and list_events
    promises no ordering, so reading one event could serve the day's FIRST (superseded) attempt."""

    def _store(self):
        with patch("shared.memory.boto3.client") as mock_client:
            client = MagicMock()
            mock_client.return_value = client
            store = AgentCoreMemoryStore("mem-1", region_name="us-west-2")
        return store, client

    @staticmethod
    def _event(text: str, stamp=None):
        event = {"payload": [{"conversational": {"content": {"text": text}}}]}
        if stamp is not None:
            event["eventTimestamp"] = stamp
        return event

    def test_picks_the_newest_event_regardless_of_listing_order(self):
        store, client = self._store()
        old = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)
        new = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        client.list_events.return_value = {
            "events": [self._event('{"attempt": 1}', old), self._event('{"attempt": 2}', new)]
        }
        assert store.get_digest("2026-08-17") == {"attempt": 2}
        # More than one event per session is read, but only a few — get_recent_digests pays this
        # per session, so the page size must not grow with history.
        assert client.list_events.call_args.kwargs["maxResults"] == AgentCoreMemoryStore.EVENTS_PER_SESSION
        assert AgentCoreMemoryStore.EVENTS_PER_SESSION <= 10

    def test_payload_generated_at_breaks_a_timestamp_tie_and_is_optional(self):
        store, client = self._store()
        same = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        first = self._event(json.dumps({"digest_result": {"generated_at": "2026-08-17T03:00:00+00:00"}}), same)
        second = self._event(json.dumps({"digest_result": {"generated_at": "2026-08-17T05:00:00+00:00"}}), same)
        client.list_events.return_value = {"events": [second, first]}
        picked = store.get_digest("2026-08-17")
        assert picked["digest_result"]["generated_at"] == "2026-08-17T05:00:00+00:00"

        # A snapshot stored WITHOUT that field must still load — the stamp only breaks ties.
        client.list_events.return_value = {"events": [self._event('{"ranked_items": []}')]}
        assert store.get_digest("2026-08-17") == {"ranked_items": []}

    def test_one_unparseable_event_does_not_hide_a_good_one(self):
        store, client = self._store()
        client.list_events.return_value = {
            "events": [
                self._event("{not json", datetime(2026, 8, 17, 12, 0, tzinfo=UTC)),
                self._event('{"ok": 1}', datetime(2026, 8, 17, 11, 0, tzinfo=UTC)),
            ]
        }
        assert store.get_digest("2026-08-17") == {"ok": 1}

    def test_only_unparseable_events_raise_rather_than_read_as_an_empty_day(self):
        store, client = self._store()
        client.list_events.return_value = {"events": [self._event("{not json")]}
        with pytest.raises(MemoryReadError, match="digest-2026-08-17"):
            store.get_digest("2026-08-17")
