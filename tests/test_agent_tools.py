from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import agent_tools


class TestFormatSearchResults:
    def test_formats_fields(self):
        results = [{"title": "T1", "url": "http://a", "content": "body one"}]
        out = agent_tools._format_search_results(results)
        assert "- T1" in out
        assert "URL: http://a" in out
        assert "Content: body one" in out

    def test_truncates_content(self):
        results = [{"title": "T", "url": "u", "content": "x" * 500}]
        out = agent_tools._format_search_results(results)
        assert out.count("x") == 300


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
        monkeypatch.setenv("TAVILY_API_KEY", "key")
        with patch.object(agent_tools, "_tavily_search", new=AsyncMock(return_value="ok")) as mock:
            await agent_tools.search_community._tool_func("query")
        assert mock.call_args.kwargs["include_domains"] == agent_tools.COMMUNITY_SEARCH_DOMAINS

    @pytest.mark.asyncio
    async def test_news_search_uses_news_topic(self, monkeypatch):
        monkeypatch.setenv("TAVILY_API_KEY", "key")
        with patch.object(agent_tools, "_tavily_search", new=AsyncMock(return_value="ok")) as mock:
            await agent_tools.search_related_news._tool_func("query")
        assert mock.call_args.kwargs["topic"] == "news"


class TestRecallTrends:
    @pytest.mark.asyncio
    async def test_returns_recalled_trends(self):
        store = MagicMock()
        store.recall.return_value = ["trend A", "trend B"]
        with patch("shared.create_memory_store", return_value=store):
            result = await agent_tools.recall_trends._tool_func("open models")
        assert "trend A" in result and "trend B" in result
        store.recall.assert_called_once_with("open models", top_k=5)

    @pytest.mark.asyncio
    async def test_empty_recall(self):
        store = MagicMock()
        store.recall.return_value = []
        with patch("shared.create_memory_store", return_value=store):
            result = await agent_tools.recall_trends._tool_func("nothing")
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

        async def fake_generate(instruction, source, context):
            assert instruction == "a 1-page presentation slide"
            assert context == "extra research"
            return b"PNG", {"title": "슬라이드", "caption": "요약"}

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
        with patch("shared.resolve_secret", return_value="key"):
            result = await agent_tools.make_visual._tool_func("draw it", item_number=99)
        assert "not found" in result
