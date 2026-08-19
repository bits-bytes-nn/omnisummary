import json
import time
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from lambda_handlers import visual_handler
from output.threads_handler import ThreadsDelivery
from shared.memory import MemoryReadError


class TestVisualHandler:
    def test_handler_forwards_the_event_and_returns_200(self):
        # Patch _run (not asyncio.run) so the EVENT the handler forwards is asserted: patching the
        # runner made the test pass no matter what payload — or none — reached the pipeline.
        event = {"digest_date": "2026-08-18"}
        with patch("lambda_handlers.visual_handler._run", new=AsyncMock()) as run:
            result = visual_handler.handler(event, None)
        assert run.await_args.args[0] == event
        assert run.await_args.kwargs["deadline"] is None  # no Lambda context in this test
        assert result["statusCode"] == 200

    def test_handler_forwards_the_context_deadline(self):
        context = MagicMock()
        context.get_remaining_time_in_millis.return_value = 600_000
        with patch("lambda_handlers.visual_handler._run", new=AsyncMock()) as run:
            visual_handler.handler({}, context)
        assert run.await_args.kwargs["deadline"] is not None

    def test_handler_reraises_on_exception_so_alarms_and_dlq_fire(self):
        # Returning a 500 body made Lambda record a success — no Errors alarm, no DLQ message.
        with patch("lambda_handlers.visual_handler._run", new=AsyncMock(side_effect=RuntimeError("boom"))):
            with patch("lambda_handlers.visual_handler.logger") as log:
                with pytest.raises(RuntimeError, match="boom"):
                    visual_handler.handler({}, None)
        assert log.error.called


class TestCorrelationId:
    """The digest run's correlation id travels with the invoke, so both halves of one digest — the
    pipeline and its only delivery path — appear under one id in the logs."""

    def test_prefers_the_id_the_digest_run_passed(self):
        with patch("lambda_handlers.visual_handler._run", new=AsyncMock()):
            with patch("lambda_handlers.visual_handler.set_correlation_id") as set_id:
                visual_handler.handler({"digest_date": "2026-08-18", "correlation_id": "abc123"}, MagicMock())
        assert set_id.call_args.args[0] == "abc123"

    def test_dlq_replay_envelope_is_honoured(self):
        event = {"version": "1.0", "requestPayload": {"correlation_id": "abc123"}}
        assert visual_handler._requested_correlation_id(event) == "abc123"

    def test_falls_back_to_the_request_id_when_the_invoke_carries_none(self):
        context = MagicMock()
        context.aws_request_id = "req-987"
        with patch("lambda_handlers.visual_handler._run", new=AsyncMock()):
            with patch("lambda_handlers.visual_handler.set_correlation_id") as set_id:
                visual_handler.handler({"digest_date": "2026-08-18"}, context)
        assert set_id.call_args.args[0] == "req-987"


class TestRequestedDate:
    """The digest Lambda passes the digest date explicitly, so the visual publishes the day it was
    fired for. 'Load the latest snapshot' published yesterday's stories when today's was missing,
    and comparing digest_result.generated_at is not an option: it is UTC, so it disagrees with the
    KST digest date on every pre-09:00 KST run."""

    tz = ZoneInfo("Asia/Seoul")

    def test_explicit_date_from_the_payload(self):
        # The flag says the invoke NAMED its date, which is what makes a missing snapshot a failure.
        assert visual_handler._requested_date({"digest_date": "2026-08-17"}, self.tz) == (date(2026, 8, 17), True)

    def test_dlq_replay_envelope_is_honoured(self):
        # A DLQ message wraps the original payload under requestPayload; a replay must publish the
        # date the FAILED run was for, not today's.
        event = {"version": "1.0", "requestPayload": {"digest_date": "2026-08-13"}}
        assert visual_handler._requested_date(event, self.tz) == (date(2026, 8, 13), True)

    def test_missing_and_malformed_fall_back_to_today(self):
        from datetime import datetime

        today = datetime.now(self.tz).date()
        assert visual_handler._requested_date({}, self.tz) == (today, False)
        assert visual_handler._requested_date({"digest_date": "not-a-date"}, self.tz) == (today, False)


class TestVisualRun:
    async def test_skips_when_disabled(self):
        config = MagicMock()
        config.pipeline.enable_daily_visual = False
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store") as store:
                await visual_handler._run()
        store.assert_not_called()

    async def test_missing_snapshot_for_a_named_date_raises(self):
        # The digest Lambda names the date it just persisted, so a miss means this — the only
        # Threads delivery path — published nothing. Returning 200 hid that on 2026-08-13/08-17.
        config = MagicMock()
        config.pipeline.enable_daily_visual = True
        config.aws.timezone = "Asia/Seoul"
        store = MagicMock()
        store.get_digest.return_value = None
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store", return_value=store):
                with patch("lambda_handlers.visual_handler.DailyVisualMaker") as maker:
                    with pytest.raises(RuntimeError, match="No digest state for 2026-08-17"):
                        await visual_handler._run({"digest_date": "2026-08-17"})
        # Loaded BY date, and no stale fallback to whatever snapshot is newest.
        store.get_digest.assert_called_once_with("2026-08-17")
        store.get_latest_digest.assert_not_called()
        maker.assert_not_called()

    async def test_missing_snapshot_without_a_named_date_stays_quiet(self):
        # A today-fallback invoke (local/manual) may legitimately run before any digest exists;
        # that must not raise, or every such run would fail loudly for nothing.
        config = MagicMock()
        config.pipeline.enable_daily_visual = True
        config.aws.timezone = "Asia/Seoul"
        store = MagicMock()
        store.get_digest.return_value = None
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store", return_value=store):
                with patch("lambda_handlers.visual_handler.DailyVisualMaker") as maker:
                    await visual_handler._run({})
        maker.assert_not_called()

    async def test_read_failure_propagates_instead_of_reading_as_an_empty_day(self):
        config = MagicMock()
        config.pipeline.enable_daily_visual = True
        config.aws.timezone = "Asia/Seoul"
        store = MagicMock()
        store.get_digest.side_effect = MemoryReadError("throttled")
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store", return_value=store):
                with pytest.raises(MemoryReadError):
                    await visual_handler._run({"digest_date": "2026-08-17"})

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
        # No caller deadline → None, so the publish path behaves exactly as it did before.
        assert kwargs["deadline"] is None

    async def test_publish_outcome_is_emitted_as_a_metric(self, capsys):
        config = MagicMock()
        config.pipeline.enable_daily_visual = True
        config.aws.timezone = "Asia/Seoul"
        store = MagicMock()
        store.get_digest.return_value = {"some": "state"}
        maker_instance = MagicMock()
        maker_instance.run = AsyncMock(return_value=False)
        maker_instance.threads_outcome = ThreadsDelivery(0, 6)
        with patch("lambda_handlers.visual_handler.Config.load", return_value=config):
            with patch("lambda_handlers.visual_handler.create_memory_store", return_value=store):
                with patch("lambda_handlers.visual_handler.DigestStateManager.load_from_dict"):
                    with patch("lambda_handlers.visual_handler.boto3.Session"):
                        with patch("lambda_handlers.visual_handler.BedrockLanguageModelFactory"):
                            with patch("lambda_handlers.visual_handler.DailyVisualMaker", return_value=maker_instance):
                                await visual_handler._run({"digest_date": "2026-08-18"})
        emitted = [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{") and visual_handler.THREADS_POSTS_METRIC in line
        ]
        assert emitted and emitted[-1][visual_handler.THREADS_POSTS_METRIC] == 0


class TestRemainingDeadline:
    """The Lambda's remaining time is converted to a plain monotonic float HERE; the context object
    itself is never threaded into the pipeline."""

    def test_none_without_a_context(self):
        assert visual_handler._remaining_deadline(None) is None

    def test_derived_from_the_remaining_millis(self):
        context = MagicMock()
        context.get_remaining_time_in_millis.return_value = 600_000
        deadline = visual_handler._remaining_deadline(context)
        assert deadline is not None
        assert 500 < deadline - time.monotonic() <= 600


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
            visual_handler._maybe_alert_threads_outcome(ThreadsDelivery(6, 6, True), date(2026, 8, 18))
        sns.publish.assert_not_called()

    def test_alerts_on_a_text_only_day(self, monkeypatch):
        # `expected` counts POSTS, so a day whose visual never made it was complete success by that
        # measure and said nothing: the image is dropped on a render failure, a missing OpenAI key or
        # an unreadable secret, and the only trace was one log line inside the maker.
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:ap-northeast-2:1:alerts")
        sns = MagicMock()
        with patch("lambda_handlers.visual_handler.boto3.client", return_value=sns):
            visual_handler._maybe_alert_threads_outcome(ThreadsDelivery(6, 6, False), date(2026, 8, 18))
        sns.publish.assert_called_once()
        assert "TEXT-ONLY" in sns.publish.call_args.kwargs["Message"]

    def test_a_failed_day_alerts_once_not_twice(self, monkeypatch):
        # A total failure has no image either; it must not produce a second, redundant notice.
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:ap-northeast-2:1:alerts")
        sns = MagicMock()
        with patch("lambda_handlers.visual_handler.boto3.client", return_value=sns):
            visual_handler._maybe_alert_threads_outcome(ThreadsDelivery(0, 6), date(2026, 8, 18))
        sns.publish.assert_called_once()

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


class TestThreadsMetrics:
    """The only delivery path must leave a numeric trace of what it produced. A missing datapoint
    reads as "no data" in CloudWatch, not as a zero — so the record is emitted unconditionally."""

    @staticmethod
    def _records(capsys) -> list[dict]:
        return [
            json.loads(line)
            for line in capsys.readouterr().out.splitlines()
            if line.startswith("{") and visual_handler.THREADS_POSTS_METRIC in line
        ]

    def test_both_metrics_ride_one_record_with_a_utc_timestamp(self, capsys):
        from datetime import UTC, datetime

        visual_handler._emit_threads_metrics(ThreadsDelivery(6, 6, with_image=True))
        record = self._records(capsys)[-1]
        assert record[visual_handler.THREADS_POSTS_METRIC] == 6
        assert record[visual_handler.THREADS_IMAGE_METRIC] == 1
        names = {m["Name"] for m in record["_aws"]["CloudWatchMetrics"][0]["Metrics"]}
        assert names == {visual_handler.THREADS_POSTS_METRIC, visual_handler.THREADS_IMAGE_METRIC}
        # UTC epoch ms, never the naive local clock (which files every datapoint at the wrong time).
        drift = abs(record["_aws"]["Timestamp"] / 1000 - datetime.now(UTC).timestamp())
        assert drift < 60

    def test_a_text_only_post_reports_no_image(self, capsys):
        visual_handler._emit_threads_metrics(ThreadsDelivery(6, 6))
        record = self._records(capsys)[-1]
        assert record[visual_handler.THREADS_POSTS_METRIC] == 6
        assert record[visual_handler.THREADS_IMAGE_METRIC] == 0

    def test_a_run_with_no_outcome_still_emits_zeroes(self, capsys):
        visual_handler._emit_threads_metrics(None)
        record = self._records(capsys)[-1]
        assert record[visual_handler.THREADS_POSTS_METRIC] == 0
        assert record[visual_handler.THREADS_IMAGE_METRIC] == 0

    def test_the_record_is_dimensioned_by_project_and_stage(self, capsys, monkeypatch):
        # Every deployment publishing into one datapoint means a dev run masks a prod outage.
        monkeypatch.setenv("PROJECT_NAME", "omnisummary")
        monkeypatch.setenv("STAGE", "dev")
        visual_handler._emit_threads_metrics(ThreadsDelivery(6, 6))
        record = self._records(capsys)[-1]
        assert record["_aws"]["CloudWatchMetrics"][0]["Dimensions"] == [["Project", "Stage"]]
        assert record["Project"] == "omnisummary" and record["Stage"] == "dev"
