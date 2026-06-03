from unittest.mock import MagicMock, patch

from shared.memory import AgentCoreMemoryStore, LocalMemoryStore, create_memory_store


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

    def test_record_and_recall_trends(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        store.record_trend("trend one", session_id="s1")
        store.record_trend("trend two", session_id="s2")
        recalled = store.recall("anything", top_k=5)
        assert recalled == ["trend one", "trend two"]

    def test_recall_respects_top_k(self, tmp_path):
        store = LocalMemoryStore(tmp_path)
        for i in range(10):
            store.record_trend(f"trend {i}", session_id=f"s{i}")
        recalled = store.recall("q", top_k=3)
        assert recalled == ["trend 7", "trend 8", "trend 9"]


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

    def test_recall_returns_record_text(self):
        store, client = self._store()
        # Real bedrock-agentcore RetrieveMemoryRecords returns 'memoryRecordSummaries'.
        client.retrieve_memory_records.return_value = {
            "memoryRecordSummaries": [{"content": {"text": "fact A"}}, {"content": {"text": "fact B"}}]
        }
        assert store.recall("query") == ["fact A", "fact B"]

    def test_recall_swallows_errors(self):
        store, client = self._store()
        client.retrieve_memory_records.side_effect = RuntimeError("boom")
        assert store.recall("query") == []
