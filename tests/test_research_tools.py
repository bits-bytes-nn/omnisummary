from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent import research_tools as rt
from agent.research_tools import DeliveryContext, current_delivery_context, request_context
from output.delivery import DeliveryOutcome
from shared import ImageAsset


class TestSearchTools:
    @pytest.mark.asyncio
    async def test_web_search_general(self):
        with patch.object(rt, "tavily_search", new=AsyncMock(return_value="ok")) as tav:
            await rt.web_search._tool_func("q")
        assert tav.await_args.kwargs.get("topic") is None

    @pytest.mark.asyncio
    async def test_web_search_news_maps_topic(self):
        with patch.object(rt, "tavily_search", new=AsyncMock(return_value="ok")) as tav:
            await rt.web_search._tool_func("q", recency="news")
        assert tav.await_args.kwargs.get("topic") == "news"

    @pytest.mark.asyncio
    async def test_community_search_passes_domains(self):
        with patch.object(rt, "tavily_search", new=AsyncMock(return_value="ok")) as tav:
            await rt.community_search._tool_func("q")
        domains = rt.get_config().agent.community_search_domains
        assert tav.await_args.kwargs.get("include_domains") == domains

    @pytest.mark.asyncio
    async def test_semantic_scholar_search_delegates(self):
        with patch.object(rt, "semantic_scholar_search", new=AsyncMock(return_value="papers")) as sp:
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
        store.read.return_value = None
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
        store.read.return_value = '{"trends": []}'
        memory = MagicMock()
        memory.search.return_value = [trend]
        with patch("shared.create_state_store", return_value=store):
            with patch("shared.TrendMemory") as tm:
                tm.model_validate_json.return_value = memory
                result = await rt.recall_trends._tool_func("open weights")
        assert "open weights" in result and "GLM released" in result

    @pytest.mark.asyncio
    async def test_a_store_failure_is_not_reported_as_no_history(self):
        # "S3 wouldn't tell me" is not "this topic never came up": returning TrendMemory() made a
        # throttled/denied read indistinguishable from an untracked topic, and the report then
        # asserted that absence. Same distinction recall_digest draws for the digest store.
        with patch("shared.create_state_store", side_effect=RuntimeError("boom")):
            result = await rt.recall_trends._tool_func("x")
        assert "could not be READ" in result
        assert "No earlier trends recalled" not in result

    @pytest.mark.asyncio
    async def test_an_empty_store_still_reports_no_history(self):
        store = MagicMock()
        store.read.return_value = None
        with patch("shared.create_state_store", return_value=store):
            result = await rt.recall_trends._tool_func("x")
        assert "No earlier trends recalled" in result


class TestRecallDigest:
    """One single-purpose tool: what did a SPECIFIC day's digest carry. It must never answer with a
    different day's stories, or the report cites the wrong day as that day's coverage."""

    @staticmethod
    def _snapshot() -> dict:
        return {
            "ranked_items": [],
            "digest_result": {
                "digest_text": "t",
                "content": {
                    "lead": "\uc624\ub298\uc758 \ub9ac\ub4dc.",
                    "headline_index": 1,
                    "items": [{"title": "\uc2a4\ud1a0\ub9ac 1", "url": "u1", "body": "b"}],
                },
            },
        }

    @pytest.mark.asyncio
    async def test_returns_the_lead_and_story_titles(self):
        store = MagicMock()
        store.get_digest.return_value = self._snapshot()
        with patch("shared.create_memory_store", return_value=store):
            result = await rt.recall_digest._tool_func("2026-08-17")
        store.get_digest.assert_called_once_with("2026-08-17")
        assert "2026-08-17" in result
        assert "\uc624\ub298\uc758 \ub9ac\ub4dc." in result
        assert "\uc2a4\ud1a0\ub9ac 1" in result

    @pytest.mark.asyncio
    async def test_missing_day_says_so_instead_of_serving_another_date(self):
        store = MagicMock()
        store.get_digest.return_value = None
        with patch("shared.create_memory_store", return_value=store):
            result = await rt.recall_digest._tool_func("2026-08-17")
        assert result == "No digest stored for 2026-08-17."

    @pytest.mark.asyncio
    async def test_an_unreadable_store_is_never_reported_as_an_uncovered_day(self):
        # A throttled/denied read used to return the same sentence as "that day has no digest", so a
        # report could assert the digest never covered a topic it in fact covered.
        store = MagicMock()
        store.get_digest.side_effect = RuntimeError("throttled")
        with patch("shared.create_memory_store", return_value=store):
            result = await rt.recall_digest._tool_func("2026-08-17")
        assert "could not be READ" in result
        assert "throttled" in result
        assert "No digest stored" not in result

    @pytest.mark.asyncio
    async def test_malformed_date_is_rejected(self):
        with patch("shared.create_memory_store") as store:
            result = await rt.recall_digest._tool_func("yesterday")
        store.assert_not_called()
        assert "not a YYYY-MM-DD date" in result

    @pytest.mark.asyncio
    async def test_output_is_bounded(self):
        from shared import get_config

        snapshot = self._snapshot()
        cap = get_config().pipeline.top_n
        snapshot["digest_result"]["content"]["items"] = [
            {"title": f"\uc2a4\ud1a0\ub9ac {i}" + "\uac00" * 500, "url": f"u{i}", "body": "b"} for i in range(cap + 5)
        ]
        store = MagicMock()
        store.get_digest.return_value = snapshot
        with patch("shared.create_memory_store", return_value=store):
            result = await rt.recall_digest._tool_func("2026-08-17")
        chars = get_config().agent.search_content_preview_chars
        lines = [ln for ln in result.splitlines() if ln.startswith("- ")]
        assert len(lines) == cap
        assert all(len(ln) <= chars + 2 for ln in lines)


class TestAttachImage:
    @pytest.mark.asyncio
    async def test_stages_image_on_context(self):
        delivery = DeliveryContext()
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
        limit = rt.get_config().agent.research_max_staged_images
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
    async def test_markdown_is_the_default_channel(self):
        delivery = DeliveryContext()
        with request_context(delivery):
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                msg = await rt.deliver_report._tool_func("# 보고서")
        assert deliver.await_args.kwargs["channel"] == "markdown"
        assert "Recorded" in msg

    @pytest.mark.asyncio
    async def test_routes_to_threads_when_the_run_allows_it(self):
        delivery = DeliveryContext(publish_threads=True)
        with request_context(delivery):
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                await rt.deliver_report._tool_func("body", channel="threads")
        assert deliver.await_args.kwargs["channel"] == "threads"

    @pytest.mark.asyncio
    async def test_unknown_channel_returns_error_without_delivering(self):
        delivery = DeliveryContext(publish_threads=True)
        with request_context(delivery):
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                msg = await rt.deliver_report._tool_func("body", channel="slack")
        deliver.assert_not_awaited()  # invalid channel is rejected, not silently downgraded
        assert "Unknown channel" in msg

    @pytest.mark.asyncio
    async def test_threads_is_refused_unless_the_caller_asked_for_it(self):
        # Publishing to the PUBLIC Threads account is the caller's flag, never the model's reading of
        # the request: a topic whose SUBJECT is Threads must not be able to publish.
        delivery = DeliveryContext()
        with request_context(delivery):
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                msg = await rt.deliver_report._tool_func("body", channel="threads")
        deliver.assert_not_awaited()
        assert "NOT delivered" in msg and "markdown" in msg

    @pytest.mark.asyncio
    async def test_a_repeat_call_is_reported_as_not_posted(self):
        # The likeliest reason for a second call is a REVISED report. Reporting it as "Delivered"
        # told the requester a correction had gone out while the reader still had the old text.
        delivery = DeliveryContext(publish_threads=True)
        with request_context(delivery):
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.NOT_POSTED)
            ):
                msg = await rt.deliver_report._tool_func("revised body", channel="threads")
        assert "NOT posted" in msg
        assert "Delivered" not in msg

    @pytest.mark.asyncio
    async def test_failed_delivery_reports_failure(self):
        delivery = DeliveryContext(publish_threads=True)
        with request_context(delivery):
            with patch("output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.FAILED)):
                msg = await rt.deliver_report._tool_func("body", channel="threads")
        assert "Failed to deliver" in msg

    @pytest.mark.asyncio
    async def test_partial_delivery_is_not_reported_as_a_clean_delivery(self):
        # Regression: posts dropped over the channel cap / trimmed mid-sentence, or replies that
        # never landed, all came back as "Delivered the report to threads." The agent then asserted
        # a complete delivery in its final answer.
        from output.delivery import DeliveryStats

        delivery = DeliveryContext(publish_threads=True)

        async def _deliver(report, *, channel, delivery):
            delivery.threads_stats = DeliveryStats(rendered=8, delivered=6, dropped=3, trimmed=1)
            return DeliveryOutcome.POSTED

        with request_context(delivery):
            with patch("output.delivery.deliver_research_report", new=_deliver):
                msg = await rt.deliver_report._tool_func("body", channel="threads")
        assert "INCOMPLETELY" in msg
        assert "6/8 posts delivered" in msg
        assert "3 post(s) DROPPED" in msg
        assert "trimmed" in msg
        assert "Do not re-send" in msg  # a second deliver_report is a no-op, so don't invite one

    @pytest.mark.asyncio
    async def test_complete_delivery_states_the_counts(self):
        from output.delivery import DeliveryStats

        delivery = DeliveryContext(publish_threads=True)

        async def _deliver(report, *, channel, delivery):
            delivery.threads_stats = DeliveryStats(rendered=6, delivered=6)
            return DeliveryOutcome.POSTED

        with request_context(delivery):
            with patch("output.delivery.deliver_research_report", new=_deliver):
                msg = await rt.deliver_report._tool_func("body", channel="threads")
        assert "INCOMPLETE" not in msg
        assert "6/6 posts delivered" in msg


class TestRequestContext:
    def test_binds_and_resets(self):
        custom = DeliveryContext(publish_threads=True)
        # Unbound: a fresh, non-shared context (never a module-level singleton).
        unbound = current_delivery_context()
        assert isinstance(unbound, DeliveryContext)
        assert unbound.publish_threads is False
        assert current_delivery_context() is not unbound  # a new instance each call when unbound
        with request_context(custom):
            assert current_delivery_context() is custom
        assert current_delivery_context() is not custom


class TestCitationGuard:
    """The URLs a tool actually surfaced are recorded on the DeliveryContext and compared against the
    report's, so a fabricated citation is refused before it reaches the requester or a public post."""

    @pytest.mark.asyncio
    async def test_a_url_no_tool_returned_blocks_delivery(self):
        delivery = DeliveryContext(publish_threads=True)
        with request_context(delivery):
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                msg = await rt.deliver_report._tool_func("근거: https://invented.example/paper", channel="threads")
        deliver.assert_not_awaited()
        assert "NOT delivered" in msg
        assert "https://invented.example/paper" in msg

    @pytest.mark.asyncio
    async def test_a_url_a_search_tool_surfaced_is_allowed(self):
        delivery = DeliveryContext(publish_threads=True)
        with request_context(delivery):
            with patch(
                "agent.research_tools.tavily_search",
                new=AsyncMock(return_value="- T\n  URL: https://real.example/a\n  Content: x"),
            ):
                await rt.web_search._tool_func("q")
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                msg = await rt.deliver_report._tool_func("근거: https://real.example/a", channel="threads")
        deliver.assert_awaited_once()
        assert "Delivered" in msg

    @pytest.mark.asyncio
    async def test_an_http_to_https_rewrite_or_a_parenthesised_url_still_matches(self):
        # The comparison must refuse a FABRICATED citation, never a real one the model reformatted.
        delivery = DeliveryContext()
        with request_context(delivery):
            with patch(
                "agent.research_tools.tavily_search",
                new=AsyncMock(return_value="URL: http://www.real.example/a/"),
            ):
                await rt.web_search._tool_func("q")
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                await rt.deliver_report._tool_func("본문 (https://real.example/a).")
        deliver.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_read_page_counts_as_surfaced(self):
        delivery = DeliveryContext()
        with request_context(delivery):
            with patch("agent.research_tools.extract_url", new=AsyncMock(return_value="page text, no urls")):
                await rt.read_url._tool_func("https://primary.example/post")
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                await rt.deliver_report._tool_func("출처 https://primary.example/post")
        deliver.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_a_report_citing_nothing_is_delivered(self):
        delivery = DeliveryContext()
        with request_context(delivery):
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                await rt.deliver_report._tool_func("URL 없는 리포트다.")
        deliver.assert_awaited_once()

    @pytest.mark.parametrize(
        "citation",
        [
            "[Real Example](https://real.example/a)",
            "[출처](https://real.example/a)",
            "<https://real.example/a>",
        ],
    )
    @pytest.mark.asyncio
    async def test_the_markdown_link_form_the_system_prompt_asks_for_is_allowed(self, citation):
        # URL_RE does not stop at ')', so matching the raw pattern would read "[label](url)" with the
        # closing paren glued on and refuse every well-formed Markdown report.
        delivery = DeliveryContext()
        with request_context(delivery):
            with patch(
                "agent.research_tools.tavily_search",
                new=AsyncMock(return_value="- T\n  URL: https://real.example/a\n  Content: x"),
            ):
                await rt.web_search._tool_func("q")
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                msg = await rt.deliver_report._tool_func(f"근거: {citation}")
        deliver.assert_awaited_once()
        assert "NOT delivered" not in msg

    @pytest.mark.asyncio
    async def test_a_fabricated_url_in_markdown_link_form_is_still_refused(self):
        delivery = DeliveryContext()
        with request_context(delivery):
            with patch(
                "output.delivery.deliver_research_report", new=AsyncMock(return_value=DeliveryOutcome.POSTED)
            ) as deliver:
                msg = await rt.deliver_report._tool_func("근거: [그럴듯한 논문](https://invented.example/paper)")
        deliver.assert_not_awaited()
        assert "NOT delivered" in msg
        assert "https://invented.example/paper" in msg
        assert "그럴듯한" not in msg
