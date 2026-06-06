from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import agent_tools


class TestFormatSearchResults:
    def test_formats_fields(self):
        results = [{"title": "T1", "url": "http://a", "content": "body one"}]
        out = agent_tools._format_search_results(results, preview_chars=300)
        assert "- T1" in out
        assert "URL: http://a" in out
        assert "Content: body one" in out

    def test_truncates_content(self):
        results = [{"title": "T", "url": "u", "content": "x" * 500}]
        out = agent_tools._format_search_results(results, preview_chars=300)
        assert out.count("x") == 300

    def test_preview_chars_is_configurable(self):
        results = [{"title": "T", "url": "u", "content": "x" * 500}]
        out = agent_tools._format_search_results(results, preview_chars=10)
        assert out.count("x") == 10


class TestTavilySearch:
    @pytest.mark.asyncio
    async def test_no_api_key(self, monkeypatch):
        monkeypatch.delenv("TAVILY_API_KEY", raising=False)
        result = await agent_tools._tavily_search("q")
        assert result == "TAVILY_API_KEY not configured."

    @pytest.mark.asyncio
    async def test_passes_topic_and_domains(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "key")
        client = AsyncMock()
        client.search.return_value = {"results": [{"title": "T", "url": "u", "content": "c"}]}
        with patch.object(agent_tools, "_get_tavily_client", return_value=client):
            await agent_tools._tavily_search("q", topic="news", include_domains=["x.com"])
        kwargs = client.search.call_args.kwargs
        assert kwargs["topic"] == "news"
        assert kwargs["include_domains"] == ["x.com"]
        assert kwargs["max_results"] == 5

    @pytest.mark.asyncio
    async def test_empty_results(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "key")
        client = AsyncMock()
        client.search.return_value = {"results": []}
        with patch.object(agent_tools, "_get_tavily_client", return_value=client):
            result = await agent_tools._tavily_search("q")
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_handles_exception(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "key")
        client = AsyncMock()
        client.search.side_effect = RuntimeError("boom")
        with patch.object(agent_tools, "_get_tavily_client", return_value=client):
            result = await agent_tools._tavily_search("q")
        assert "Search failed" in result

    @pytest.mark.asyncio
    async def test_community_search_uses_community_domains(self, monkeypatch):
        from shared import Config

        monkeypatch.setenv("TAVILY_API_KEY", "key")
        expected = Config.load().agent.community_search_domains
        with patch.object(agent_tools, "_tavily_search", new=AsyncMock(return_value="ok")) as mock:
            await agent_tools.search_community._tool_func("query")
        assert mock.call_args.kwargs["include_domains"] == expected
        assert "reddit.com" in expected

    @pytest.mark.asyncio
    async def test_news_search_uses_news_topic(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "key")
        with patch.object(agent_tools, "_tavily_search", new=AsyncMock(return_value="ok")) as mock:
            await agent_tools.search_related_news._tool_func("query")
        assert mock.call_args.kwargs["topic"] == "news"


class TestSearchPapers:
    def _client_returning(self, responses):
        client = AsyncMock()
        client.get.side_effect = responses
        client.__aenter__.return_value = client
        client.__aexit__.return_value = False
        return client

    @pytest.mark.asyncio
    async def test_retries_on_429_then_succeeds(self, monkeypatch):
        import httpx

        monkeypatch.setattr("shared.utils.asyncio.sleep", AsyncMock())

        rate_limited = MagicMock(status_code=429, request=MagicMock())
        ok = MagicMock(status_code=200)
        ok.json.return_value = {
            "data": [{"title": "Paper", "year": 2024, "authors": [{"name": "A"}], "url": "u", "abstract": "abs"}]
        }
        client = self._client_returning([rate_limited, ok])

        with patch.object(httpx, "AsyncClient", return_value=client):
            result = await agent_tools.search_papers._tool_func("transformers")

        assert "Paper" in result
        assert client.get.call_count == 2

    @pytest.mark.asyncio
    async def test_exhausts_retries_on_persistent_429(self, monkeypatch):
        import httpx

        monkeypatch.setattr("shared.utils.asyncio.sleep", AsyncMock())
        rate_limited = MagicMock(status_code=429, request=MagicMock())
        retries = agent_tools.Config.load().agent.search_max_retries
        client = self._client_returning([rate_limited] * retries)

        with patch.object(httpx, "AsyncClient", return_value=client):
            result = await agent_tools.search_papers._tool_func("q")

        assert "SEARCH_FAILED" in result
        assert client.get.call_count == retries

    @pytest.mark.asyncio
    async def test_caps_authors_and_abstract_from_config(self, monkeypatch):
        import httpx

        cfg = agent_tools.Config.load().agent
        ok = MagicMock(status_code=200)
        ok.json.return_value = {
            "data": [
                {
                    "title": "P",
                    "year": 2024,
                    "authors": [{"name": f"Author{i}"} for i in range(10)],
                    "url": "u",
                    "abstract": "x" * 500,
                }
            ]
        }
        client = self._client_returning([ok])
        with patch.object(httpx, "AsyncClient", return_value=client):
            result = await agent_tools.search_papers._tool_func("q")

        assert result.count("Author") == cfg.search_paper_max_authors
        assert "x" * cfg.search_paper_abstract_max_chars in result
        assert "x" * (cfg.search_paper_abstract_max_chars + 1) not in result


class TestRecallTrends:
    def _store_with(self, memory):
        from shared.state_store import StateStore

        class _S(StateStore):
            def read(self, key):
                return memory.model_dump_json() if key == "trends.json" else None

            def write(self, key, content):
                pass

            def exists(self, key):
                return key == "trends.json"

        return _S()

    @pytest.mark.asyncio
    async def test_matches_query_terms(self):
        from shared.models import Trend, TrendEvidence, TrendMemory

        memory = TrendMemory(
            trends=[
                Trend(
                    id="open-models",
                    title="Open Weight Models",
                    first_seen="2026-06-01",
                    last_seen="2026-06-05",
                    evidence=[TrendEvidence(date="2026-06-05", summary="Meta shipped a model")],
                ),
                Trend(
                    id="agents",
                    title="Agent Frameworks",
                    first_seen="2026-06-01",
                    last_seen="2026-06-05",
                    evidence=[TrendEvidence(date="2026-06-05", summary="new framework")],
                ),
            ]
        )
        with patch("shared.create_state_store", return_value=self._store_with(memory)):
            result = await agent_tools.recall_trends._tool_func("open models")
        assert "Open Weight Models" in result
        assert "Agent Frameworks" not in result

    @pytest.mark.asyncio
    async def test_empty_recall(self):
        from shared.models import TrendMemory

        with patch("shared.create_state_store", return_value=self._store_with(TrendMemory())):
            result = await agent_tools.recall_trends._tool_func("nothing")
        assert "No earlier trends" in result

    @pytest.mark.asyncio
    async def test_excludes_archived(self):
        from shared.models import Trend, TrendEvidence, TrendMemory, TrendStatus

        memory = TrendMemory(
            trends=[
                Trend(
                    id="old",
                    title="Old Topic",
                    status=TrendStatus.ARCHIVED,
                    first_seen="2026-01-01",
                    last_seen="2026-02-01",
                    evidence=[TrendEvidence(date="2026-02-01", summary="topic mention")],
                )
            ]
        )
        with patch("shared.create_state_store", return_value=self._store_with(memory)):
            result = await agent_tools.recall_trends._tool_func("topic")
        assert "No earlier trends" in result


class TestMakeVisual:
    @pytest.mark.asyncio
    async def test_disabled_without_openai_key(self):
        with patch("shared.resolve_secret", return_value=""):
            result = await agent_tools.make_visual._tool_func("a slide about X")
        assert "disabled" in result.lower()

    @pytest.mark.asyncio
    async def test_free_form_generate_and_post(self, monkeypatch):
        gen = MagicMock()

        from shared.models import VisualBrief

        async def fake_generate(instruction, source, context):
            assert instruction == "a 1-page presentation slide"
            assert context == "extra research"
            return b"PNG", VisualBrief(title="슬라이드", caption="요약", prompt="draw")

        gen.generate = fake_generate
        agent_tools.delivery_context.channel_id = "C1"
        agent_tools.delivery_context.thread_ts = "1.0"

        async def fake_upload(*a, **k):
            fake_upload.kwargs = k
            return True

        try:
            with patch("shared.resolve_secret", return_value="key"):
                with patch("agent.visuals.VisualGenerator", return_value=gen):
                    with patch("agent.agent_tools._build_llm_factory", return_value=(MagicMock(), MagicMock())):
                        with patch("output.slack_handler.send_image_to_slack", side_effect=fake_upload):
                            result = await agent_tools.make_visual._tool_func(
                                "a 1-page presentation slide", item_number=0, context="extra research"
                            )
            assert "슬라이드" in result
            assert fake_upload.kwargs["channel_id"] == "C1"
        finally:
            agent_tools.delivery_context.channel_id = ""
            agent_tools.delivery_context.thread_ts = ""

    @pytest.mark.asyncio
    async def test_unknown_item_number(self, monkeypatch):
        agent_tools.state_manager.clear()
        agent_tools.delivery_context.channel_id = "C1"  # past the early channel guard
        try:
            with patch("shared.resolve_secret", return_value="key"):
                result = await agent_tools.make_visual._tool_func("draw it", item_number=99)
            assert "not found" in result
        finally:
            agent_tools.delivery_context.channel_id = ""


class TestRequestContext:
    def test_context_overrides_globals_and_resets(self):
        from agent.agent_tools import (
            DeliveryContext,
            current_delivery_context,
            current_state_manager,
            request_context,
        )
        from agent.tool_state import DigestStateManager

        # Outside any request, accessors return the module defaults.
        assert current_delivery_context() is agent_tools.delivery_context
        assert current_state_manager() is agent_tools.state_manager

        scoped_state = DigestStateManager()
        scoped_delivery = DeliveryContext(channel_id="CSCOPED", thread_ts="9.9")
        with request_context(scoped_state, scoped_delivery):
            assert current_state_manager() is scoped_state
            assert current_delivery_context().channel_id == "CSCOPED"

        # Reset after the block — no leak into the global.
        assert current_delivery_context() is agent_tools.delivery_context
        assert agent_tools.delivery_context.channel_id == ""
