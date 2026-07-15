from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from shared.research import research_backends as rb


class TestFormatSearchResults:
    def test_formats_with_preview(self):
        results = [{"title": "T1", "url": "u1", "content": "abc"}, {"title": "T2", "url": "u2", "content": "def"}]
        out = rb._format_search_results(results, preview_chars=300)
        assert "T1" in out and "u1" in out and "abc" in out
        assert "T2" in out and "u2" in out and "def" in out

    def test_missing_fields_use_defaults(self):
        out = rb._format_search_results([{}], preview_chars=300)
        assert "N/A" in out

    def test_explicit_null_fields_do_not_raise(self):
        # Tavily returns explicit null content/title/url for some pages. None[:n] would raise and
        # fail the ENTIRE query (a .get default only covers a MISSING key, not a null value).
        out = rb._format_search_results([{"title": None, "url": None, "content": None}], preview_chars=10)
        assert "N/A" in out  # null title coerced to the N/A placeholder, no exception

    def test_preview_truncates_content(self):
        out = rb._format_search_results([{"title": "T", "url": "u", "content": "0123456789ABC"}], preview_chars=10)
        assert "0123456789" in out
        assert "ABC" not in out


class TestTavilySearch:
    @pytest.mark.asyncio
    async def test_no_key_returns_message(self):
        with patch.object(rb, "_get_tavily_client", return_value=None):
            result = await rb._tavily_search("q")
        assert "TAVILY_API_KEY not configured" in result

    @pytest.mark.asyncio
    async def test_passes_topic_and_domains(self):
        client = MagicMock()
        client.search = AsyncMock(return_value={"results": [{"title": "t", "url": "u", "content": "c"}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            await rb._tavily_search("q", topic="news", include_domains=["x.com"])
        kwargs = client.search.call_args.kwargs
        assert kwargs["topic"] == "news"
        assert kwargs["include_domains"] == ["x.com"]

    @pytest.mark.asyncio
    async def test_closes_client_after_search(self):
        client = MagicMock()
        client.search = AsyncMock(return_value={"results": [{"title": "t", "url": "u", "content": "c"}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            await rb._tavily_search("q")
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_results(self):
        client = MagicMock()
        client.search = AsyncMock(return_value={"results": []})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb._tavily_search("q")
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_search_exception_is_caught(self):
        client = MagicMock()
        client.search = AsyncMock(side_effect=RuntimeError("down"))
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb._tavily_search("q")
        assert "Search failed" in result


class TestExtractUrl:
    @pytest.mark.asyncio
    async def test_no_key(self):
        with patch.object(rb, "_get_tavily_client", return_value=None):
            assert "TAVILY_API_KEY not configured" in await rb.extract_url("http://x")

    @pytest.mark.asyncio
    async def test_returns_capped_content(self):
        client = MagicMock()
        client.extract = AsyncMock(return_value={"results": [{"raw_content": "X" * 100}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            with patch.object(rb.Config, "load") as load:
                load.return_value.agent.research_content_cap_chars = 10
                result = await rb.extract_url("http://x")
        assert result == "X" * 10

    @pytest.mark.asyncio
    async def test_falls_back_to_content_field(self):
        # No raw_content → use the 'content' field.
        client = MagicMock()
        client.extract = AsyncMock(return_value={"results": [{"content": "Y" * 100}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            with patch.object(rb.Config, "load") as load:
                load.return_value.agent.research_content_cap_chars = 10
                result = await rb.extract_url("http://x")
        assert result == "Y" * 10

    @pytest.mark.asyncio
    async def test_both_fields_empty(self):
        client = MagicMock()
        client.extract = AsyncMock(return_value={"results": [{}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb.extract_url("http://x")
        assert "No readable content" in result

    @pytest.mark.asyncio
    async def test_empty_extract(self):
        client = MagicMock()
        client.extract = AsyncMock(return_value={"results": []})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb.extract_url("http://x")
        assert "No readable content" in result

    @pytest.mark.asyncio
    async def test_search_papers_author_missing_name(self):
        # Some Semantic Scholar author objects lack 'name' — must not KeyError.
        payload = {
            "data": [
                {
                    "title": "T",
                    "year": 2024,
                    "authors": [{"authorId": "1"}, {"name": "Real"}],
                    "url": "u",
                    "abstract": "a",
                }
            ]
        }
        resp = MagicMock(status_code=200)
        resp.json.return_value = payload
        with patch.object(rb, "retry_async", new=AsyncMock(return_value=resp)):
            result = await rb._search_papers("q")
        assert "Real" in result and "T" in result

    @pytest.mark.asyncio
    async def test_extract_exception(self):
        client = MagicMock()
        client.extract = AsyncMock(side_effect=RuntimeError("boom"))
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb.extract_url("http://x")
        assert "Could not read URL" in result


class TestSearchPapers:
    @pytest.mark.asyncio
    async def test_parses_papers(self):
        payload = {
            "data": [
                {
                    "title": "Attention Is All You Need",
                    "year": 2017,
                    "authors": [{"name": "Vaswani"}, {"name": "Shazeer"}],
                    "url": "http://paper",
                    "abstract": "Transformers.",
                }
            ]
        }
        resp = MagicMock(status_code=200)
        resp.json.return_value = payload
        with patch.object(rb, "retry_async", new=AsyncMock(return_value=resp)):
            result = await rb._search_papers("transformers")
        assert "Attention Is All You Need" in result
        assert "Vaswani" in result
        assert "http://paper" in result

    @pytest.mark.asyncio
    async def test_rate_limited_returns_message(self):
        with patch.object(
            rb,
            "retry_async",
            new=AsyncMock(side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=MagicMock())),
        ):
            result = await rb._search_papers("q")
        assert "SEARCH_FAILED" in result

    @pytest.mark.asyncio
    async def test_no_papers(self):
        resp = MagicMock(status_code=200)
        resp.json.return_value = {"data": []}
        with patch.object(rb, "retry_async", new=AsyncMock(return_value=resp)):
            result = await rb._search_papers("q")
        assert result == "No related papers found."
