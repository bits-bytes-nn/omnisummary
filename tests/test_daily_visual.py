from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pipeline.daily_visual import DailyVisualMaker
from shared.config import Config
from shared.constants import SourceType
from shared.models import CollectedItem, RankedItem, VisualBrief


def _maker() -> DailyVisualMaker:
    config = Config()
    factory = MagicMock()
    factory.get_model.return_value = MagicMock()
    maker = DailyVisualMaker(config, factory)
    return maker


def _items(n: int = 3) -> list[RankedItem]:
    return [
        RankedItem(
            item=CollectedItem(
                item_id=f"i{k}", source_type=SourceType.WEB, title=f"Story {k}", url=f"http://e.com/{k}", text="body"
            ),
            score=0.8,
        )
        for k in range(1, n + 1)
    ]


class TestDailyVisualMaker:
    @pytest.mark.asyncio
    async def test_skips_without_openai_key(self):
        maker = _maker()
        with patch("pipeline.daily_visual.resolve_secret", return_value=""):
            assert await maker.run(_items()) is False

    @pytest.mark.asyncio
    async def test_skips_on_empty_items(self):
        assert await _maker().run([]) is False

    @pytest.mark.asyncio
    async def test_editor_skip_returns_false(self):
        maker = _maker()
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value={"skip": True})):
                assert await maker.run(_items()) is False

    @pytest.mark.asyncio
    async def test_invalid_item_number_returns_false(self):
        maker = _maker()
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value={"skip": False, "item_number": 99})):
                assert await maker.run(_items()) is False

    @pytest.mark.asyncio
    async def test_happy_path_posts(self):
        maker = _maker()
        plan = {"skip": False, "item_number": 2, "research": [], "instruction": "a 4-panel cartoon"}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    result = await maker.run(_items())
        assert result is True
        # the chosen source must be item #2
        args, kwargs = maker.generator.generate.call_args
        assert "Story 2" in args[1]

    @pytest.mark.asyncio
    async def test_slack_disabled_skips_upload(self):
        maker = _maker()
        maker.config.pipeline.enable_slack_post = False
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)) as up:
                    result = await maker.run(_items())
        assert result is False
        up.assert_not_called()

    @pytest.mark.asyncio
    async def test_threads_enabled_fans_out_with_content(self):
        from shared.models import DigestContent, DigestItem

        maker = _maker()
        maker.config.pipeline.enable_threads_post = True
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        content = DigestContent(
            lead="오늘의 리드.",
            headline_index=1,
            items=[DigestItem(title="스토리", url="http://e.com/1", body="본문.")],
        )
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=True)) as th:
                        await maker.run(_items(), content)
        th.assert_awaited_once()
        # root carries the digest lead; replies carry the per-item story
        assert th.await_args.kwargs["root_text"] == "오늘의 리드."
        assert any("스토리" in r for r in th.await_args.kwargs["replies"])
        assert th.await_args.kwargs["image_bytes"] == b"PNG"

    @pytest.mark.asyncio
    async def test_threads_disabled_by_default(self):
        maker = _maker()
        plan = {"skip": False, "item_number": 1, "research": [], "instruction": "x"}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(
                    return_value=(b"PNG", VisualBrief(title="T", caption="C", prompt="draw"))
                )
                with patch("output.slack_handler.send_image_to_slack", new=AsyncMock(return_value=True)):
                    with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=True)) as th:
                        await maker.run(_items())
        th.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_generation_failure_returns_false(self):
        maker = _maker()
        plan = {"skip": False, "item_number": 1, "instruction": "draw"}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(side_effect=RuntimeError("boom"))
                assert await maker.run(_items()) is False

    @pytest.mark.asyncio
    async def test_gather_context_dispatches_by_source(self):
        # The editor agentically picks a source per research step; _gather_context routes
        # each to the matching backend (papers -> Semantic Scholar, community/news -> Tavily).
        maker = _maker()
        research = [
            {"source": "papers", "query": "diffusion scaling"},
            {"source": "community", "query": "reactions"},
            {"source": "news", "query": "launch"},
        ]
        with patch("agent.agent_tools._search_papers", new=AsyncMock(return_value="PAPERS")) as papers:
            with patch("agent.agent_tools._tavily_search", new=AsyncMock(side_effect=["COMMUNITY", "NEWS"])) as tav:
                context = await maker._gather_context(research)

        assert "PAPERS" in context and "COMMUNITY" in context and "NEWS" in context
        papers.assert_awaited_once_with("diffusion scaling")
        # community step must pass the configured community domains; news step uses topic=news
        community_call, news_call = tav.await_args_list
        assert community_call.kwargs.get("include_domains") == maker.config.agent.community_search_domains
        assert news_call.kwargs.get("topic") == "news"

    @pytest.mark.asyncio
    async def test_gather_context_empty_research_returns_empty(self):
        assert await _maker()._gather_context([]) == ""

    @pytest.mark.asyncio
    async def test_gather_context_skips_failed_step(self):
        # A backend that raises must be skipped, not abort the whole gather.
        maker = _maker()
        research = [{"source": "papers", "query": "q1"}, {"source": "news", "query": "q2"}]
        with patch("agent.agent_tools._search_papers", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with patch("agent.agent_tools._tavily_search", new=AsyncMock(return_value="NEWS")):
                context = await maker._gather_context(research)
        assert context == "NEWS"

    @pytest.mark.asyncio
    async def test_pick_story_parses_prose_wrapped_json(self):
        # Real path: the editor LLM returns prose-wrapped JSON; _pick_story must extract it.
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _maker()
        maker.llm = RunnableLambda(lambda _: AIMessage(content='Here:\n{"skip": false, "item_number": 1}\ndone'))
        plan = await maker._pick_story(_items())
        assert plan == {"skip": False, "item_number": 1}

    @pytest.mark.asyncio
    async def test_pick_story_malformed_returns_empty(self):
        from langchain_core.messages import AIMessage
        from langchain_core.runnables import RunnableLambda

        maker = _maker()
        maker.llm = RunnableLambda(lambda _: AIMessage(content="no json here at all"))
        assert await maker._pick_story(_items()) == {}
