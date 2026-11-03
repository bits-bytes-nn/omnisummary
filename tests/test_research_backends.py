from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from adapters import research_backends as rb


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
            result = await rb.tavily_search("q")
        assert "TAVILY_API_KEY not configured" in result

    @pytest.mark.asyncio
    async def test_passes_topic_and_domains(self):
        client = MagicMock()
        client.search = AsyncMock(return_value={"results": [{"title": "t", "url": "u", "content": "c"}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            await rb.tavily_search("q", topic="news", include_domains=["x.com"])
        kwargs = client.search.call_args.kwargs
        assert kwargs["topic"] == "news"
        assert kwargs["include_domains"] == ["x.com"]

    @pytest.mark.asyncio
    async def test_the_digests_content_farm_list_applies_to_research_search(self, monkeypatch):
        monkeypatch.setattr(rb.get_config().collectors.web_search, "exclude_domains", ["farm.example"])
        client = MagicMock()
        client.search = AsyncMock(return_value={"results": [{"title": "t", "url": "u", "content": "c"}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            await rb.tavily_search("q")
        assert (
            client.search.call_args.kwargs["exclude_domains"] == rb.get_config().collectors.web_search.exclude_domains
        )

    @pytest.mark.asyncio
    async def test_closes_client_after_search(self):
        client = MagicMock()
        client.search = AsyncMock(return_value={"results": [{"title": "t", "url": "u", "content": "c"}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            await rb.tavily_search("q")
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_results(self):
        client = MagicMock()
        client.search = AsyncMock(return_value={"results": []})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb.tavily_search("q")
        assert result == "No results found."

    @pytest.mark.asyncio
    async def test_search_exception_is_caught(self):
        client = MagicMock()
        client.search = AsyncMock(side_effect=RuntimeError("down"))
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb.tavily_search("q")
        assert "Search failed" in result


class TestExtractUrl:
    @pytest.mark.asyncio
    async def test_no_key(self):
        with patch.object(rb, "_get_tavily_client", return_value=None):
            assert await rb.extract_url("http://x") == ""

    @pytest.mark.asyncio
    async def test_returns_capped_content(self):
        client = MagicMock()
        client.extract = AsyncMock(return_value={"results": [{"raw_content": "X" * 100}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            with patch.object(rb, "get_config") as cfg:
                cfg.return_value.agent.research_content_cap_chars = 10
                result = await rb.extract_url("http://x")
        assert result == "X" * 10

    @pytest.mark.asyncio
    async def test_falls_back_to_content_field(self):
        # No raw_content → use the 'content' field.
        client = MagicMock()
        client.extract = AsyncMock(return_value={"results": [{"content": "Y" * 100}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            with patch.object(rb, "get_config") as cfg:
                cfg.return_value.agent.research_content_cap_chars = 10
                result = await rb.extract_url("http://x")
        assert result == "Y" * 10

    @pytest.mark.asyncio
    async def test_both_fields_empty(self):
        client = MagicMock()
        client.extract = AsyncMock(return_value={"results": [{}]})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb.extract_url("http://x")
        assert result == ""

    @pytest.mark.asyncio
    async def test_empty_extract(self):
        client = MagicMock()
        client.extract = AsyncMock(return_value={"results": []})
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb.extract_url("http://x")
        assert result == ""

    @pytest.mark.asyncio
    async def test_extract_exception(self):
        client = MagicMock()
        client.extract = AsyncMock(side_effect=RuntimeError("boom"))
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            result = await rb.extract_url("http://x")
        assert result == ""


class TestSearchPapers:
    WORK = {
        "id": "https://openalex.org/W1",
        "doi": "https://doi.org/10.48550/arxiv.2607.21519",
        "title": "Diffusion Language Model for Recommendation",
        "publication_date": "2026-07-23",
        "cited_by_count": 3,
        "authorships": [{"author": {"display_name": "Kim"}}, {"author": {}}, {"author": {"display_name": "Lee"}}],
        "primary_location": {"landing_page_url": "https://arxiv.org/abs/2607.21519"},
        "abstract_inverted_index": {"Diffusion": [0], "models": [1], "recommend.": [2]},
    }

    @staticmethod
    def _response(payload: dict, status: int = 200) -> MagicMock:
        resp = MagicMock(status_code=status)
        resp.json.return_value = payload
        return resp

    @pytest.mark.asyncio
    async def test_parses_works_with_date_citations_and_rebuilt_abstract(self):
        with patch.object(rb, "retry_async", new=AsyncMock(return_value=self._response({"results": [self.WORK]}))):
            result = await rb.paper_search("diffusion recommendation")
        assert "Diffusion Language Model for Recommendation (2026-07-23; cited by 3) by Kim, Lee" in result
        assert "URL: https://arxiv.org/abs/2607.21519" in result
        assert "Abstract: Diffusion models recommend." in result

    @pytest.mark.asyncio
    async def test_falls_back_to_the_doi_without_a_landing_page(self):
        work = {**self.WORK, "primary_location": None}
        with patch.object(rb, "retry_async", new=AsyncMock(return_value=self._response({"results": [work]}))):
            result = await rb.paper_search("q")
        assert "URL: https://doi.org/10.48550/arxiv.2607.21519" in result

    @pytest.mark.asyncio
    async def test_rate_limited_returns_message(self):
        with patch.object(
            rb,
            "retry_async",
            new=AsyncMock(side_effect=httpx.HTTPStatusError("429", request=MagicMock(), response=MagicMock())),
        ):
            result = await rb.paper_search("q")
        assert "Search failed" in result

    @pytest.mark.asyncio
    async def test_no_papers(self):
        with patch.object(rb, "retry_async", new=AsyncMock(return_value=self._response({"results": []}))):
            result = await rb.paper_search("q")
        assert result == "No related papers found."


class TestExtractPages:
    @pytest.mark.asyncio
    async def test_batches_and_keys_by_url_skipping_a_failed_batch(self):
        urls = [f"http://x/{i}" for i in range(rb.EXTRACT_BATCH_SIZE + 1)]
        client = MagicMock()
        client.extract = AsyncMock(
            side_effect=[
                {"results": [{"url": "http://x/0", "raw_content": "page"}, {"url": "http://x/1"}]},
                RuntimeError,
            ]
        )
        client.close = AsyncMock()
        with patch.object(rb, "_get_tavily_client", return_value=client):
            pages = await rb.extract_pages(urls)
        assert pages == {"http://x/0": "page"}
        assert [len(c.kwargs["urls"]) for c in client.extract.call_args_list] == [rb.EXTRACT_BATCH_SIZE, 1]
        client.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_no_key_reads_nothing(self):
        with patch.object(rb, "_get_tavily_client", return_value=None):
            assert await rb.extract_pages(["http://x"]) == {}


class TestWebResearchSources:
    """The adapter behind the research port: each port method reaches its own backend."""

    async def test_each_method_reaches_its_backend_with_its_arguments(self):
        sources = rb.WebResearchSources()
        with (
            patch.object(rb, "tavily_search", new=AsyncMock(return_value="web")) as web,
            patch.object(rb, "paper_search", new=AsyncMock(return_value="papers")) as papers,
            patch.object(rb, "extract_url", new=AsyncMock(return_value="page")) as page,
            patch.object(rb, "access_controlled", new=AsyncMock(return_value=set())),
            patch.object(rb, "fetch_og_image", new=AsyncMock(return_value=None)) as image,
        ):
            assert await sources.web_search("q", topic="news", time_range="week", include_domains=["x.com"]) == "web"
            assert await sources.paper_search("p") == "papers"
            assert await sources.read_page("https://a") == "page"
            assert await sources.fetch_image("https://b") is None
        web.assert_awaited_once_with("q", topic="news", time_range="week", include_domains=["x.com"])
        papers.assert_awaited_once_with("p")
        page.assert_awaited_once_with("https://a")
        image.assert_awaited_once_with("https://b")

    @pytest.mark.asyncio
    async def test_a_paid_or_bot_checked_page_is_not_read(self):
        sources = rb.WebResearchSources()
        with (
            patch.object(rb, "access_controlled", new=AsyncMock(return_value={"https://a"})),
            patch.object(rb, "extract_url", new=AsyncMock(return_value="page")) as page,
        ):
            assert await sources.read_page("https://a") == ""
        page.assert_not_awaited()


class TestAccessControlled:
    @pytest.mark.asyncio
    async def test_refusals_and_paid_declarations_are_caught_and_outages_pass(self, monkeypatch):
        import httpx

        def answer(request: httpx.Request) -> httpx.Response:
            path = request.url.path
            if path == "/down":
                raise httpx.ConnectError("unreachable")
            return {
                "/open": httpx.Response(
                    200, text='<script type="application/ld+json">{"isAccessibleForFree": true}</script>'
                ),
                "/paid": httpx.Response(200, text='{"@type":"NewsArticle","isAccessibleForFree":"False"}'),
                "/bot": httpx.Response(403, text="press and hold"),
                "/limited": httpx.Response(429),
                "/login": httpx.Response(401),
                "/gone": httpx.Response(404),
            }[path]

        real = httpx.AsyncClient
        monkeypatch.setattr(rb.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(answer), **kw))
        urls = [f"https://site.example/{p}" for p in ("open", "paid", "bot", "limited", "login", "gone", "down")]
        # A URL httpx cannot form (a malformed IDN) passes instead of failing the whole check.
        blocked = await rb.access_controlled(urls + ["https://xn--a.com/"])
        assert blocked == {f"https://site.example/{p}" for p in ("paid", "bot", "limited", "login")}

    @pytest.mark.asyncio
    async def test_no_urls_makes_no_request(self):
        assert await rb.access_controlled([]) == set()


class TestTavilyUsage:
    def test_credits_are_logged_and_a_missing_usage_says_so(self, monkeypatch):
        from adapters import research_backends

        logged = []
        monkeypatch.setattr(research_backends.logger, "info", lambda msg, *a: logged.append(msg % a))
        research_backends.record_tavily_usage("web-search", "search", {"results": [], "usage": {"credits": 2}})
        research_backends.record_tavily_usage("page-fill", "extract", {"results": []})
        assert logged == [
            "Search usage stage=web-search operation=search credits=2",
            "Search usage stage=page-fill operation=extract credits=unreported",
        ]
