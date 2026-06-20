from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import research_tools as rt
from agent.research_tools import DeliveryContext, current_delivery_context, request_context
from shared import ImageAsset


class TestSearchTools:
    @pytest.mark.asyncio
    async def test_web_search_general(self):
        with patch.object(rt, "_tavily_search", new=AsyncMock(return_value="ok")) as tav:
            await rt.web_search._tool_func("q")
        assert tav.await_args.kwargs.get("topic") is None

    @pytest.mark.asyncio
    async def test_web_search_news_maps_topic(self):
        with patch.object(rt, "_tavily_search", new=AsyncMock(return_value="ok")) as tav:
            await rt.web_search._tool_func("q", recency="news")
        assert tav.await_args.kwargs.get("topic") == "news"

    @pytest.mark.asyncio
    async def test_community_search_passes_domains(self):
        with patch.object(rt, "_tavily_search", new=AsyncMock(return_value="ok")) as tav:
            await rt.community_search._tool_func("q")
        domains = rt.Config.load().agent.community_search_domains
        assert tav.await_args.kwargs.get("include_domains") == domains

    @pytest.mark.asyncio
    async def test_search_papers_delegates(self):
        with patch.object(rt, "_search_papers", new=AsyncMock(return_value="papers")) as sp:
            result = await rt.search_papers._tool_func("transformers")
        sp.assert_awaited_once_with("transformers")
        assert result == "papers"

    @pytest.mark.asyncio
    async def test_read_url_delegates(self):
        with patch.object(rt, "extract_url", new=AsyncMock(return_value="text")) as ex:
            result = await rt.read_url._tool_func("http://x")
        ex.assert_awaited_once_with("http://x")
        assert result == "text"


class TestRecallTrends:
    @pytest.mark.asyncio
    async def test_no_match_message(self):
        store = MagicMock()
        store.exists.return_value = False
        memory = MagicMock()
        memory.search.return_value = []
        with patch("shared.create_state_store", return_value=store):
            with patch("shared.TrendMemory", return_value=memory):
                result = await rt.recall_trends._tool_func("open models")
        assert "No earlier trends recalled" in result

    @pytest.mark.asyncio
    async def test_formats_matched_trend(self):
        ev = MagicMock(date="2026-06-01", summary="GLM released")
        trend = MagicMock(title="open weights", evidence=[ev])
        trend.status.value = "accelerating"
        store = MagicMock()
        store.exists.return_value = True
        store.read.return_value = '{"trends": []}'
        memory = MagicMock()
        memory.search.return_value = [trend]
        with patch("shared.create_state_store", return_value=store):
            with patch("shared.TrendMemory") as tm:
                tm.model_validate_json.return_value = memory
                result = await rt.recall_trends._tool_func("open weights")
        assert "open weights" in result and "GLM released" in result

    @pytest.mark.asyncio
    async def test_store_error_yields_empty(self):
        with patch("shared.create_state_store", side_effect=RuntimeError("boom")):
            result = await rt.recall_trends._tool_func("x")
        assert "No earlier trends recalled" in result


class TestAttachImage:
    @pytest.mark.asyncio
    async def test_stages_image_on_context(self):
        delivery = DeliveryContext(channel_id="C")
        asset = ImageAsset(data=b"img", source_url="http://src", image_url="http://img")
        with request_context(delivery):
            with patch.object(rt, "fetch_og_image", new=AsyncMock(return_value=asset)):
                msg = await rt.attach_image._tool_func("http://src")
        assert delivery.staged_images == [asset]
        assert "Attached image" in msg

    @pytest.mark.asyncio
    async def test_no_image_found(self):
        delivery = DeliveryContext()
        with request_context(delivery):
            with patch.object(rt, "fetch_og_image", new=AsyncMock(return_value=None)):
                msg = await rt.attach_image._tool_func("http://src")
        assert delivery.staged_images == []
        assert "No usable image" in msg

    @pytest.mark.asyncio
    async def test_caps_staged_images(self):
        limit = rt.Config.load().agent.research_max_staged_images
        delivery = DeliveryContext(
            staged_images=[ImageAsset(data=b"x", source_url="u", image_url="i") for _ in range(limit)]
        )
        with request_context(delivery):
            with patch.object(rt, "fetch_og_image", new=AsyncMock()) as fetch:
                msg = await rt.attach_image._tool_func("http://src")
        fetch.assert_not_awaited()  # capped before any network fetch
        assert "maximum" in msg
        assert len(delivery.staged_images) == limit


class TestDeliverReport:
    @pytest.mark.asyncio
    async def test_routes_to_slack(self):
        delivery = DeliveryContext(channel_id="C")
        with request_context(delivery):
            with patch("output.delivery.deliver_research_report", new=AsyncMock(return_value=True)) as deliver:
                msg = await rt.deliver_report._tool_func("report body", channel="slack")
        assert deliver.await_args.kwargs["channel"] == "slack"
        assert "Delivered" in msg

    @pytest.mark.asyncio
    async def test_routes_to_threads(self):
        delivery = DeliveryContext(channel_id="C")
        with request_context(delivery):
            with patch("output.delivery.deliver_research_report", new=AsyncMock(return_value=True)) as deliver:
                await rt.deliver_report._tool_func("body", channel="threads")
        assert deliver.await_args.kwargs["channel"] == "threads"

    @pytest.mark.asyncio
    async def test_unknown_channel_returns_error_without_delivering(self):
        delivery = DeliveryContext(channel_id="C")
        with request_context(delivery):
            with patch("output.delivery.deliver_research_report", new=AsyncMock(return_value=True)) as deliver:
                msg = await rt.deliver_report._tool_func("body", channel="email")
        deliver.assert_not_awaited()  # invalid channel is rejected, not silently downgraded
        assert "Unknown channel" in msg

    @pytest.mark.asyncio
    async def test_failed_delivery_reports_failure(self):
        delivery = DeliveryContext(channel_id="C")
        with request_context(delivery):
            with patch("output.delivery.deliver_research_report", new=AsyncMock(return_value=False)):
                msg = await rt.deliver_report._tool_func("body", channel="slack")
        assert "Failed to deliver" in msg


class TestRequestContext:
    def test_binds_and_resets(self):
        custom = DeliveryContext(channel_id="X")
        # Unbound: a fresh, non-shared context (never a module-level singleton).
        unbound = current_delivery_context()
        assert isinstance(unbound, DeliveryContext)
        assert unbound.channel_id == ""
        assert current_delivery_context() is not unbound  # a new instance each call when unbound
        with request_context(custom):
            assert current_delivery_context() is custom
        assert current_delivery_context() is not custom
