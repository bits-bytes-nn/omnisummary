import json
from unittest.mock import MagicMock, patch

from shared.memory import AgentCoreMemoryStore, LocalMemoryStore, create_memory_store


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
