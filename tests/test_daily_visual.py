import json
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
        plan = {"skip": False, "item_number": 2, "search_query": "", "instruction": "a 4-panel cartoon"}
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
    async def test_generation_failure_returns_false(self):
        maker = _maker()
        plan = {"skip": False, "item_number": 1, "instruction": "draw"}
        with patch("pipeline.daily_visual.resolve_secret", return_value="key"):
            with patch.object(maker, "_pick_story", new=AsyncMock(return_value=plan)):
                maker.generator.generate = AsyncMock(side_effect=RuntimeError("boom"))
                assert await maker.run(_items()) is False

    def test_pick_story_parses_embedded_json(self):
        # _pick_story tolerates prose around the JSON object
        raw = 'Here:\n{"skip": false, "item_number": 1}\ndone'
        start, end = raw.find("{"), raw.rfind("}") + 1
        assert json.loads(raw[start:end]) == {"skip": False, "item_number": 1}
