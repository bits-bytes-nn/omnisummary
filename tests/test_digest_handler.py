import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lambda_handlers import digest_handler
from shared.models import HealthReport, SourceHealth, SourceStatus


class TestHandler:
    def test_returns_200_on_success(self):
        with patch("lambda_handlers.digest_handler.asyncio.run") as run:
            result = digest_handler.handler({}, None)
        run.assert_called_once()
        assert result["statusCode"] == 200

    def test_reraises_on_exception_so_alarms_and_dlq_fire(self):
        # A returned 500 body counts as a SUCCESSFUL invocation to Lambda: neither the Errors
        # alarm nor the async DLQ would ever see a broken digest. The failure must propagate
        # (retry_attempts=0 means it can't re-post).
        with patch("lambda_handlers.digest_handler.asyncio.run", side_effect=RuntimeError("boom")):
            with patch("lambda_handlers.digest_handler.logger") as log:
                with pytest.raises(RuntimeError, match="boom"):
                    digest_handler.handler({}, None)
        assert log.error.called  # still logged (with the correlation id set) before re-raising


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

    async def test_metric_counts_digest_stories_not_ranker_candidates(self):
        # Regression (2026-08-13 / 08-17): the digest shipped ZERO stories while this metric
        # reported the full ranker candidate count, so the EmptyDigestAlarm never fired.
        from shared.models import DigestContent, DigestItem

        config = _config()
        items = [MagicMock()]
        health = HealthReport(sources=[SourceHealth(name="rss", item_count=1, status=SourceStatus.OK)])
        digest = MagicMock()
        digest.content = DigestContent(
            lead="l", headline_index=1, items=[DigestItem(title="t", url="http://e/1", body="b")]
        )
        result = (items, [MagicMock() for _ in range(8)], digest)  # 8 ranked candidates, 1 story
        with patch("lambda_handlers.digest_handler.Config.load", return_value=config):
            with patch("lambda_handlers.digest_handler.boto3.Session"):
                with patch("lambda_handlers.digest_handler.BedrockLanguageModelFactory"):
                    with patch(
                        "lambda_handlers.digest_handler.run_collectors_with_health",
                        new=AsyncMock(return_value=(items, health)),
                    ):
                        with patch("lambda_handlers.digest_handler._maybe_alert"):
                            with patch(
                                "lambda_handlers.digest_handler.run_pipeline", new=AsyncMock(return_value=result)
                            ):
                                with patch("lambda_handlers.digest_handler.persist_digest"):
                                    with patch("lambda_handlers.digest_handler._trigger_visual"):
                                        with patch("lambda_handlers.digest_handler._emit_digest_items_metric") as emit:
                                            await digest_handler._run()
        emit.assert_called_once_with(1)

    async def test_story_less_digest_emits_zero(self):
        config = _config()
        items = [MagicMock()]
        health = HealthReport(sources=[SourceHealth(name="rss", item_count=1, status=SourceStatus.OK)])
        digest = MagicMock()
        digest.content = None  # the 08-17 failure mode: content that could not be parsed
        result = (items, [MagicMock()], digest)
        with patch("lambda_handlers.digest_handler.Config.load", return_value=config):
            with patch("lambda_handlers.digest_handler.boto3.Session"):
                with patch("lambda_handlers.digest_handler.BedrockLanguageModelFactory"):
                    with patch(
                        "lambda_handlers.digest_handler.run_collectors_with_health",
                        new=AsyncMock(return_value=(items, health)),
                    ):
                        with patch("lambda_handlers.digest_handler._maybe_alert"):
                            with patch(
                                "lambda_handlers.digest_handler.run_pipeline", new=AsyncMock(return_value=result)
                            ):
                                with patch("lambda_handlers.digest_handler.persist_digest"):
                                    with patch("lambda_handlers.digest_handler._trigger_visual"):
                                        with patch("lambda_handlers.digest_handler._emit_digest_items_metric") as emit:
                                            await digest_handler._run()
        emit.assert_called_once_with(0)

    async def test_persist_failure_is_loud_and_does_not_trigger_the_visual(self):
        # The visual Lambda publishes off the snapshot and is the only Threads delivery path, so a
        # failed persist means zero output. Triggering it anyway would publish an OLDER date's
        # stories; staying quiet would hide a fully-lost day. Skip the trigger and re-raise.
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
                                "lambda_handlers.digest_handler.run_pipeline", new=AsyncMock(return_value=result)
                            ):
                                with patch(
                                    "lambda_handlers.digest_handler.persist_digest",
                                    side_effect=RuntimeError("memory full"),
                                ):
                                    with patch("lambda_handlers.digest_handler._trigger_visual") as trigger:
                                        with pytest.raises(RuntimeError, match="persist failed"):
                                            await digest_handler._run()
        trigger.assert_not_called()

    def test_emit_metric_writes_emf_doc(self, capsys):
        digest_handler._emit_digest_items_metric(3)
        doc = json.loads(capsys.readouterr().out.strip())
        assert doc["DigestItemsPublished"] == 3
        assert doc["_aws"]["CloudWatchMetrics"][0]["Namespace"] == "OmniSummary"

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
            digest_handler._trigger_visual(date(2026, 8, 18))
        client.assert_not_called()

    def test_invokes_visual_lambda(self, monkeypatch):
        monkeypatch.setenv("VISUAL_FUNCTION_NAME", "omnisummary-dev-visual")
        lambda_client = MagicMock()
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=lambda_client):
            digest_handler._trigger_visual(date(2026, 8, 18))
        lambda_client.invoke.assert_called_once()
        kwargs = lambda_client.invoke.call_args.kwargs
        assert kwargs["FunctionName"] == "omnisummary-dev-visual"
        assert kwargs["InvocationType"] == "Event"
        # The date is passed EXPLICITLY so the visual publishes this run's snapshot, and a DLQ
        # replay of the failed invoke carries the same date instead of being re-dated to today.
        assert json.loads(kwargs["Payload"]) == {"digest_date": "2026-08-18"}

    def test_invoke_error_is_swallowed(self, monkeypatch):
        monkeypatch.setenv("VISUAL_FUNCTION_NAME", "fn")
        lambda_client = MagicMock()
        lambda_client.invoke.side_effect = Exception("network down")
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=lambda_client):
            digest_handler._trigger_visual(date(2026, 8, 18))
