from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from lambda_handlers import visual_handler
from output.threads_handler import ThreadsDelivery


class TestVisualHandler:
    def test_handler_returns_200_on_success(self):
        with patch("lambda_handlers.visual_handler.asyncio.run") as run:
            result = visual_handler.handler({}, None)
        run.assert_called_once()
        assert result["statusCode"] == 200

    def test_handler_reraises_on_exception_so_alarms_and_dlq_fire(self):
        # Returning a 500 body made Lambda record a success — no Errors alarm, no DLQ message.
        with patch("lambda_handlers.visual_handler.asyncio.run", side_effect=RuntimeError("boom")):
            with patch("lambda_handlers.visual_handler.logger") as log:
                with pytest.raises(RuntimeError, match="boom"):
                    visual_handler.handler({}, None)
        assert log.error.called


class TestRequestedDate:
    """The digest Lambda passes the digest date explicitly, so the visual publishes the day it was
    fired for. 'Load the latest snapshot' published yesterday's stories when today's was missing,
    and comparing digest_result.generated_at is not an option: it is UTC, so it disagrees with the
    KST digest date on every pre-09:00 KST run."""

    tz = ZoneInfo("Asia/Seoul")

    def test_explicit_date_from_the_payload(self):
        assert visual_handler._requested_date({"digest_date": "2026-08-17"}, self.tz) == date(2026, 8, 17)

    def test_dlq_replay_envelope_is_honoured(self):
        # A DLQ message wraps the original payload under requestPayload; a replay must publish the
        # date the FAILED run was for, not today's.
        event = {"version": "1.0", "requestPayload": {"digest_date": "2026-08-13"}}
        assert visual_handler._requested_date(event, self.tz) == date(2026, 8, 13)

    def test_missing_and_malformed_fall_back_to_today(self):
        from datetime import datetime

        today = datetime.now(self.tz).date()
        assert visual_handler._requested_date({}, self.tz) == today
        assert visual_handler._requested_date({"digest_date": "not-a-date"}, self.tz) == today


class TestVisualRun:
    async def test_skips_when_disabled(self):
        config = MagicMock()
        config.pipeline.enable_daily_visual = False
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store") as store:
                await visual_handler._run()
        store.assert_not_called()

    async def test_skips_when_no_digest_state_for_that_date(self):
        config = MagicMock()
        config.pipeline.enable_daily_visual = True
        config.aws.timezone = "Asia/Seoul"
        store = MagicMock()
        store.get_digest.return_value = None
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store", return_value=store):
                with patch("lambda_handlers.visual_handler.DailyVisualMaker") as maker:
                    await visual_handler._run({"digest_date": "2026-08-17"})
        # Loaded BY date, and no stale fallback to whatever snapshot is newest.
        store.get_digest.assert_called_once_with("2026-08-17")
        store.get_latest_digest.assert_not_called()
        maker.assert_not_called()

    async def test_runs_maker_for_the_requested_date(self):
        config = MagicMock()
        config.pipeline.enable_daily_visual = True
        config.aws.timezone = "Asia/Seoul"
        store = MagicMock()
        store.get_digest.return_value = {"some": "state"}
        ranked = [MagicMock()]
        mgr = MagicMock()
        mgr.get_ranked_items.return_value = ranked
        content = MagicMock()
        mgr.get_content.return_value = content
        maker_instance = MagicMock()
        maker_instance.run = AsyncMock(return_value=True)
        maker_instance.threads_outcome = ThreadsDelivery(6, 6)
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store", return_value=store):
                with patch("lambda_handlers.visual_handler.DigestStateManager.load_from_dict", return_value=mgr):
                    with patch("lambda_handlers.visual_handler.boto3.Session"):
                        with patch("lambda_handlers.visual_handler.BedrockLanguageModelFactory"):
                            with patch("lambda_handlers.visual_handler.DailyVisualMaker", return_value=maker_instance):
                                await visual_handler._run({"digest_date": "2026-08-18"})
        store.get_digest.assert_called_once_with("2026-08-18")
        args, kwargs = maker_instance.run.call_args
        assert args == (ranked, content)
        assert kwargs["today"] == date(2026, 8, 18)


class TestThreadsOutcomeAlert:
    """A digest that posted its root and only some replies used to be indistinguishable from a
    complete one. The publish path is this Lambda, so the notice belongs here."""

    def test_alerts_on_a_partial_reply_chain(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:ap-northeast-2:1:alerts")
        sns = MagicMock()
        with patch("lambda_handlers.visual_handler.boto3.client", return_value=sns):
            visual_handler._maybe_alert_threads_outcome(ThreadsDelivery(4, 6), date(2026, 8, 18))
        sns.publish.assert_called_once()
        message = sns.publish.call_args.kwargs["Message"]
        assert "4/6 posts" in message
        assert "2026-08-18" in message

    def test_alerts_on_total_failure(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:ap-northeast-2:1:alerts")
        sns = MagicMock()
        with patch("lambda_handlers.visual_handler.boto3.client", return_value=sns):
            visual_handler._maybe_alert_threads_outcome(ThreadsDelivery(0, 6), date(2026, 8, 18))
        assert "FAILED" in sns.publish.call_args.kwargs["Subject"]

    def test_silent_on_a_complete_post(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:ap-northeast-2:1:alerts")
        sns = MagicMock()
        with patch("lambda_handlers.visual_handler.boto3.client", return_value=sns):
            visual_handler._maybe_alert_threads_outcome(ThreadsDelivery(6, 6), date(2026, 8, 18))
        sns.publish.assert_not_called()

    def test_no_op_without_the_topic_env(self, monkeypatch):
        monkeypatch.delenv("ALERT_SNS_TOPIC_ARN", raising=False)
        with patch("lambda_handlers.visual_handler.boto3.client") as client:
            visual_handler._maybe_alert_threads_outcome(ThreadsDelivery(0, 6), date(2026, 8, 18))
        client.assert_not_called()

    def test_no_op_without_an_outcome(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:ap-northeast-2:1:alerts")
        with patch("lambda_handlers.visual_handler.boto3.client") as client:
            visual_handler._maybe_alert_threads_outcome(None, date(2026, 8, 18))
        client.assert_not_called()
