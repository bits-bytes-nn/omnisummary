import json
from datetime import UTC, date, datetime
from unittest.mock import MagicMock, patch

import pytest

from shared.memory import AgentCoreMemoryStore, LocalMemoryStore, MemoryReadError, create_memory_store


def _store() -> AgentCoreMemoryStore:
    with patch("shared.memory.boto3.client"):
        return AgentCoreMemoryStore("m", actor_id="a")


class TestFitToLimit:
    def test_small_state_unchanged_and_valid(self):
        store = _store()
        state = {"ranked_items": [{"item": {"text": "hi"}, "score": 0.5}], "digest_result": None}
        out = store._fit_to_limit(state)
        assert json.loads(out) == state  # valid JSON, unchanged

    def test_oversized_state_sheds_to_valid_json_under_limit(self):
        store = _store()
        big = "x" * 200_000
        state = {
            "ranked_items": [{"item": {"item_id": f"i{n}", "text": big}, "score": 0.5} for n in range(20)],
            "digest_result": {"digest_text": big, "content": {"lead": big, "items": []}},
        }
        out = store._fit_to_limit(state)
        parsed = json.loads(out)  # must be valid JSON (never byte-sliced)
        assert len(out) <= store.MAX_EVENT_TEXT
        assert "ranked_items" in parsed  # still a well-formed snapshot dict

    def test_the_last_resort_sheds_the_candidates_before_the_published_digest(self):
        # digest_result is what the visual Lambda publishes; it used to go first.
        store = _store()
        state = {
            "ranked_items": [{"item": {"item_id": f"i{n}", "title": "t" * 2_000}, "score": 0.5} for n in range(60)],
            "digest_result": {"digest_text": "d", "content": {"lead": "리드.", "items": [{"title": "s"}]}},
        }
        parsed = json.loads(store._fit_to_limit(state))
        assert parsed["digest_result"]["content"]["lead"] == "리드."
        assert parsed["ranked_items"] == []

    def test_shedding_keeps_every_story_and_the_digest_content(self):
        # The visual Lambda publishes off THIS payload, so what survives the trim decides whether
        # the day has a digest. Assert the surviving SHAPE by parsing it (story count, content items
        # non-empty) — a substring check would pass on a snapshot that had shed all of them.
        store = _store()
        body = "x" * 30_000
        state = {
            "ranked_items": [{"item": {"item_id": f"i{n}", "text": body}, "score": 0.8} for n in range(4)],
            "digest_result": {
                "digest_text": "d",
                "content": {
                    "lead": "리드.",
                    "items": [{"title": f"스토리 {n}", "url": f"u{n}", "body": "b"} for n in range(5)],
                },
            },
        }
        parsed = json.loads(store._fit_to_limit(state))
        assert len(parsed["ranked_items"]) == 4
        assert [r["item"]["item_id"] for r in parsed["ranked_items"]] == ["i0", "i1", "i2", "i3"]
        assert len(parsed["digest_result"]["content"]["items"]) == 5


class TestLocalMemoryStore:
    def test_get_digest_by_date(self, tmp_path):
        # The visual Lambda must publish the date it was fired for, not whatever is newest:
        # 'load latest' published yesterday's stories when today's snapshot was missing.
        store = LocalMemoryStore(tmp_path)
        store.put_digest("2026-06-01", {"a": 1})
        store.put_digest("2026-06-02", {"a": 2})
        assert store.get_digest("2026-06-01") == {"a": 1}
        assert store.get_digest("2026-06-03") is None

    def test_get_recent_digests_reads_the_window_before_the_date_newest_first(self, tmp_path):
        # Today's own snapshot is excluded (a same-day re-run keeps its stories), and anything older
        # than the window can't suppress a legitimately-recurring story.
        store = LocalMemoryStore(tmp_path)
        for d in ("2026-06-01", "2026-06-06", "2026-06-08", "2026-06-09"):
            store.put_digest(d, {"d": d})
        recent = store.get_recent_digests(date(2026, 6, 9), 4)
        assert [r["d"] for r in recent] == ["2026-06-08", "2026-06-06"]

    def test_an_unreadable_day_is_skipped_not_fatal(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        store.put_digest("2026-06-08", {"d": "2026-06-08"})
        (tmp_path / "digest_2026-06-07.json").write_text("{not json", encoding="utf-8")
        assert [r["d"] for r in store.get_recent_digests(date(2026, 6, 9), 3)] == ["2026-06-08"]


def store_region() -> str:
    from shared import get_config

    return get_config().aws.region


class TestCreateMemoryStore:
    def test_local_when_no_memory_is_deployed(self, monkeypatch, tmp_path):
        monkeypatch.delenv("MEMORY_ID", raising=False)
        with patch("shared.memory._deployed_memory_id", return_value=""), patch("shared.memory.boto3.Session"):
            store = create_memory_store(tmp_path)
        assert isinstance(store, LocalMemoryStore)

    def test_a_failed_lookup_raises_instead_of_reading_the_local_store(self, monkeypatch, tmp_path):
        # Expired credentials fell back to the local store, so recall answered "nothing stored"
        # for a date production had written.
        monkeypatch.delenv("MEMORY_ID", raising=False)
        session = MagicMock()
        session.client.return_value.get_paginator.return_value.paginate.side_effect = RuntimeError("expired token")
        with patch("shared.memory.boto3.Session", return_value=session):
            with pytest.raises(MemoryReadError, match="expired token"):
                create_memory_store(tmp_path)

    def test_a_local_run_finds_the_stacks_memory_by_name(self, monkeypatch):
        # Without MEMORY_ID (set only on the Lambdas) a local run read none of production's
        # snapshots: cross-day dedup saw no yesterday and recall_digest found nothing.
        monkeypatch.delenv("MEMORY_ID", raising=False)
        session = MagicMock()
        session.client.return_value.get_paginator.return_value.paginate.return_value = [
            {"memories": [{"id": "other_dev_digest_state-AAA"}, {"id": "omnisummary_dev_digest_state-9GEp6f8MtL"}]}
        ]
        with patch("shared.memory.boto3.Session", return_value=session):
            store = create_memory_store()
        assert isinstance(store, AgentCoreMemoryStore)
        assert store.memory_id == "omnisummary_dev_digest_state-9GEp6f8MtL"
        # The READ client comes from the same profiled session as the lookup: a default client on the
        # shell's AWS_PROFILE (another account) found the memory by name, then "Memory not found".
        assert [c.args[0] for c in session.client.call_args_list] == ["bedrock-agentcore-control", "bedrock-agentcore"]

    def test_agentcore_when_memory_id_set(self, monkeypatch):
        monkeypatch.setenv("MEMORY_ID", "mem-123")
        session = MagicMock()
        with patch("shared.memory.boto3.Session", return_value=session):
            store = create_memory_store()
        assert isinstance(store, AgentCoreMemoryStore)
        assert store.memory_id == "mem-123"
        session.client.assert_called_once_with("bedrock-agentcore", region_name=store_region())


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

    def test_put_digest_truncates_oversized_ranked_text(self):
        # The ranked-item bodies alone can exceed the limit (this is what aborted the pipeline on
        # 2026-06-04). Must still fit + store.
        store, client = self._store()
        big = {
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
            "ranked_items": [{"item": {"item_id": f"i{n}", "text": "z" * 50_000}, "score": 0.8} for n in range(20)],
            "digest_result": {"digest_text": "ok"},
        }
        store.put_digest("2026-06-02", big)
        text = client.create_event.call_args.kwargs["payload"][0]["conversational"]["content"]["text"]
        assert len(text) <= AgentCoreMemoryStore.MAX_EVENT_TEXT
        assert "ranked_items" in text


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

    def test_reads_every_page_of_a_re_run_session(self):
        store, client = self._store()
        old = datetime(2026, 8, 17, 10, 0, tzinfo=UTC)
        new = datetime(2026, 8, 17, 12, 0, tzinfo=UTC)
        client.list_events.side_effect = [
            {"events": [self._event('{"attempt": 1}', old)], "nextToken": "t"},
            {"events": [self._event('{"attempt": 2}', new)]},
        ]
        assert store.get_digest("2026-08-17") == {"attempt": 2}
        assert client.list_events.call_args.kwargs["nextToken"] == "t"

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
