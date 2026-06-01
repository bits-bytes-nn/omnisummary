from unittest.mock import AsyncMock, patch

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
