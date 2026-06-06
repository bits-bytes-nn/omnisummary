from unittest.mock import AsyncMock, MagicMock, patch

from lambda_handlers import visual_handler


class TestVisualHandler:
    def test_handler_returns_200_on_success(self):
        with patch("lambda_handlers.visual_handler.asyncio.run") as run:
            result = visual_handler.handler({}, None)
        run.assert_called_once()
        assert result["statusCode"] == 200

    def test_handler_returns_500_on_exception(self):
        with patch("lambda_handlers.visual_handler.asyncio.run", side_effect=RuntimeError("boom")):
            result = visual_handler.handler({}, None)
        assert result["statusCode"] == 500
        assert "boom" in result["body"]


class TestVisualRun:
    async def test_skips_when_disabled(self):
        config = MagicMock()
        config.pipeline.enable_daily_visual = False
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store") as store:
                await visual_handler._run()
        store.assert_not_called()

    async def test_skips_when_no_digest_state(self):
        config = MagicMock()
        config.pipeline.enable_daily_visual = True
        store = MagicMock()
        store.get_latest_digest.return_value = None
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store", return_value=store):
                with patch("lambda_handlers.visual_handler.DailyVisualMaker") as maker:
                    await visual_handler._run()
        maker.assert_not_called()

    async def test_runs_maker_with_ranked_items(self):
        config = MagicMock()
        config.pipeline.enable_daily_visual = True
        store = MagicMock()
        store.get_latest_digest.return_value = {"some": "state"}
        ranked = [MagicMock()]
        mgr = MagicMock()
        mgr.get_ranked_items.return_value = ranked
        mgr.get_digest_text.return_value = "DIGEST"
        maker_instance = MagicMock()
        maker_instance.run = AsyncMock(return_value=True)
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store", return_value=store):
                with patch("lambda_handlers.visual_handler.DigestStateManager.load_from_dict", return_value=mgr):
                    with patch("lambda_handlers.visual_handler.boto3.Session"):
                        with patch("lambda_handlers.visual_handler.BedrockLanguageModelFactory"):
                            with patch("lambda_handlers.visual_handler.DailyVisualMaker", return_value=maker_instance):
                                await visual_handler._run()
        maker_instance.run.assert_awaited_once_with(ranked, "DIGEST")
