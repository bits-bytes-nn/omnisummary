from unittest.mock import AsyncMock, MagicMock, patch

from lambda_handlers import digest_handler
from shared.models import HealthReport, SourceHealth, SourceStatus


class TestHandler:
    def test_returns_200_on_success(self):
        with patch("lambda_handlers.digest_handler.asyncio.run") as run:
            result = digest_handler.handler({}, None)
        run.assert_called_once()
        assert result["statusCode"] == 200

    def test_returns_500_on_exception(self):
        with patch("lambda_handlers.digest_handler.asyncio.run", side_effect=RuntimeError("boom")):
            result = digest_handler.handler({}, None)
        assert result["statusCode"] == 500
        assert "boom" in result["body"]


def _config() -> MagicMock:
    config = MagicMock()
    config.aws.timezone = "Asia/Seoul"
    config.aws.bedrock_region = "us-west-2"
    return config


class TestRun:
    async def test_exits_early_when_no_items(self):
        config = _config()
        health = HealthReport(sources=[SourceHealth(name="rss", item_count=0, status=SourceStatus.EMPTY)])
        with patch("lambda_handlers.digest_handler.Config.load", return_value=config):
            with patch("lambda_handlers.digest_handler.boto3.Session"):
                with patch("lambda_handlers.digest_handler.BedrockLanguageModelFactory"):
                    with patch(
                        "lambda_handlers.digest_handler.run_collectors_with_health",
                        new=AsyncMock(return_value=([], health)),
                    ):
                        with patch("lambda_handlers.digest_handler._maybe_alert") as alert:
                            with patch("lambda_handlers.digest_handler.run_pipeline", new=AsyncMock()) as pipeline:
                                await digest_handler._run()
        alert.assert_called_once_with(health)
        pipeline.assert_not_called()

    async def test_full_flow_persists_and_triggers_visual(self):
        config = _config()
        items = [MagicMock()]
        health = HealthReport(sources=[SourceHealth(name="rss", item_count=1, status=SourceStatus.OK)])
        result = (items, [MagicMock()], MagicMock())
        with patch("lambda_handlers.digest_handler.Config.load", return_value=config):
            with patch("lambda_handlers.digest_handler.boto3.Session"):
                with patch("lambda_handlers.digest_handler.BedrockLanguageModelFactory"):
                    with patch(
                        "lambda_handlers.digest_handler.run_collectors_with_health",
                        new=AsyncMock(return_value=(items, health)),
                    ):
                        with patch("lambda_handlers.digest_handler._maybe_alert"):
                            with patch(
                                "lambda_handlers.digest_handler.run_pipeline",
                                new=AsyncMock(return_value=result),
                            ):
                                with patch("lambda_handlers.digest_handler.persist_digest") as persist:
                                    with patch("lambda_handlers.digest_handler._trigger_visual") as trigger:
                                        await digest_handler._run()
        persist.assert_called_once()
        trigger.assert_called_once()

    async def test_rsshub_base_url_override_from_env(self, monkeypatch):
        monkeypatch.setenv("RSSHUB_BASE_URL", "http://example.local:1200")
        config = _config()
        health = HealthReport(sources=[SourceHealth(name="rss", item_count=0, status=SourceStatus.EMPTY)])
        with patch("lambda_handlers.digest_handler.Config.load", return_value=config):
            with patch("lambda_handlers.digest_handler.boto3.Session"):
                with patch("lambda_handlers.digest_handler.BedrockLanguageModelFactory"):
                    with patch(
                        "lambda_handlers.digest_handler.run_collectors_with_health",
                        new=AsyncMock(return_value=([], health)),
                    ):
                        with patch("lambda_handlers.digest_handler._maybe_alert"):
                            await digest_handler._run()
        assert config.collectors.rsshub.base_url == "http://example.local:1200"


class TestTriggerVisual:
    def test_no_function_name_skips(self, monkeypatch):
        monkeypatch.delenv("VISUAL_FUNCTION_NAME", raising=False)
        with patch("lambda_handlers.digest_handler.boto3.client") as client:
            digest_handler._trigger_visual()
        client.assert_not_called()

    def test_invokes_visual_lambda(self, monkeypatch):
        monkeypatch.setenv("VISUAL_FUNCTION_NAME", "omnisummary-dev-visual")
        lambda_client = MagicMock()
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=lambda_client):
            digest_handler._trigger_visual()
        lambda_client.invoke.assert_called_once()
        kwargs = lambda_client.invoke.call_args.kwargs
        assert kwargs["FunctionName"] == "omnisummary-dev-visual"
        assert kwargs["InvocationType"] == "Event"

    def test_invoke_error_is_swallowed(self, monkeypatch):
        monkeypatch.setenv("VISUAL_FUNCTION_NAME", "fn")
        lambda_client = MagicMock()
        lambda_client.invoke.side_effect = Exception("network down")
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=lambda_client):
            digest_handler._trigger_visual()
