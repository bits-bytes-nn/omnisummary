import json
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lambda_handlers import digest_handler
from shared.models import HealthReport, SourceHealth, SourceStatus


class TestHandler:
    def test_runs_the_pipeline_and_returns_200(self):
        # Patch _run, not asyncio.run: patching the runner asserted only that SOMETHING was awaited.
        with patch("lambda_handlers.digest_handler._run", new=AsyncMock()) as run:
            result = digest_handler.handler({}, None)
        run.assert_awaited_once_with()
        assert result["statusCode"] == 200

    def test_reraises_on_exception_so_alarms_and_dlq_fire(self):
        # A returned 500 body counts as a SUCCESSFUL invocation to Lambda: neither the Errors
        # alarm nor the async DLQ would ever see a broken digest. The failure must propagate
        # (retry_attempts=0 means it can't re-post).
        with patch("lambda_handlers.digest_handler._run", new=AsyncMock(side_effect=RuntimeError("boom"))):
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
        alert.assert_called_once_with(health, config.collectors.alert_on_empty)
        pipeline.assert_not_called()

    async def test_full_flow_persists_the_stories_and_triggers_the_visual_for_that_date(self):
        # Assert the SHAPE that is handed to the only delivery path — the snapshot's ranked count
        # and non-empty digest content — and the date the visual is fired for. A story-less digest
        # that still triggered the visual is exactly how 2026-08-13/08-17 published a broken post.
        from datetime import datetime
        from zoneinfo import ZoneInfo

        from shared.models import DigestContent, DigestItem

        config = _config()
        items = [MagicMock()]
        health = HealthReport(sources=[SourceHealth(name="rss", item_count=1, status=SourceStatus.OK)])
        digest = MagicMock()
        digest.content = DigestContent(
            lead="리드.", headline_index=1, items=[DigestItem(title="t", url="http://e/1", body="b")]
        )
        ranked = [MagicMock(), MagicMock(), MagicMock()]
        result = (items, ranked, digest)
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
                                with patch("lambda_handlers.digest_handler.persist_digest") as persist:
                                    with patch("lambda_handlers.digest_handler._trigger_visual") as trigger:
                                        await digest_handler._run()
        today = datetime.now(ZoneInfo("Asia/Seoul")).date()
        persisted_items, persisted_ranked, persisted_digest, persisted_date = persist.call_args.args
        assert persisted_items == items
        assert len(persisted_ranked) == 3
        assert persisted_digest.content.items  # the stories actually reach the snapshot
        assert persisted_date == today
        assert persist.call_args.kwargs["base_dir"] is None  # AgentCore-backed store in AWS
        trigger.assert_called_once_with(today)

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

    def test_invoke_error_raises_so_the_undelivered_day_is_visible(self, monkeypatch):
        # The visual Lambda is the ONLY Threads delivery path. A swallowed invoke error meant the
        # digest run returned 200 while the day was never published, with no alarm and no DLQ entry.
        # The snapshot is already persisted at this point, so raising loses nothing.
        monkeypatch.setenv("VISUAL_FUNCTION_NAME", "fn")
        lambda_client = MagicMock()
        lambda_client.invoke.side_effect = Exception("network down")
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=lambda_client):
            with pytest.raises(RuntimeError, match="Could not trigger visual delivery"):
                digest_handler._trigger_visual(date(2026, 8, 18))

    def test_missing_function_name_raises_in_aws(self, monkeypatch):
        # Locally the visual runs inline from main.py, so an unset name is normal (test above). In
        # AWS it is the only link to delivery, so it must fail the run rather than exit green.
        monkeypatch.delenv("VISUAL_FUNCTION_NAME", raising=False)
        monkeypatch.setattr("lambda_handlers.digest_handler.is_running_in_aws", lambda: True)
        with pytest.raises(RuntimeError, match="VISUAL_FUNCTION_NAME"):
            digest_handler._trigger_visual(date(2026, 8, 18))
