from unittest.mock import MagicMock, patch

from lambda_handlers import digest_handler
from shared.models import HealthReport, SourceHealth, SourceStatus


def _report_with_failure() -> HealthReport:
    return HealthReport(
        sources=[
            SourceHealth(name="rss", item_count=5, status=SourceStatus.OK),
            SourceHealth(name="reddit", item_count=0, status=SourceStatus.FAILED, detail="403"),
        ]
    )


class TestMaybeAlert:
    def test_no_topic_arn_skips(self, monkeypatch):
        monkeypatch.delenv("ALERT_SNS_TOPIC_ARN", raising=False)
        with patch("lambda_handlers.digest_handler.boto3.client") as mock_client:
            digest_handler._maybe_alert(_report_with_failure())
        mock_client.assert_not_called()

    def test_no_failures_skips(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        healthy = HealthReport(sources=[SourceHealth(name="rss", item_count=5, status=SourceStatus.OK)])
        with patch("lambda_handlers.digest_handler.boto3.client") as mock_client:
            digest_handler._maybe_alert(healthy)
        mock_client.assert_not_called()

    def test_publishes_on_failure(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        sns = MagicMock()
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=sns):
            digest_handler._maybe_alert(_report_with_failure())
        sns.publish.assert_called_once()
        kwargs = sns.publish.call_args.kwargs
        assert kwargs["TopicArn"] == "arn:aws:sns:::topic"
        # Unified project alarm format: "[omnisummary] Source Health — ALERT".
        assert kwargs["Subject"] == "[omnisummary] Source Health — ALERT"
        assert "reddit" in kwargs["Message"]
        assert "[FAILED] reddit" in kwargs["Message"]

    def test_subject_names_the_real_project_and_stage(self, monkeypatch):
        # With the hardcoded default a dev-stage and a prod-stage alert were byte-identical, so a
        # second deployment alerted under the wrong name.
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        monkeypatch.setenv("PROJECT_NAME", "omnisummary")
        monkeypatch.setenv("STAGE", "prod")
        sns = MagicMock()
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=sns):
            digest_handler._maybe_alert(_report_with_failure())
        assert sns.publish.call_args.kwargs["Subject"] == "[omnisummary/prod] Source Health — ALERT"

    def test_message_carries_the_correlation_id(self, monkeypatch):
        # The id is set on every invocation but appeared in no alert, so an operator could not get
        # from the mail to the matching JSON log lines.
        from shared.logger import set_correlation_id

        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        set_correlation_id("abc123def456")
        sns = MagicMock()
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=sns):
            digest_handler._maybe_alert(_report_with_failure())
        assert "abc123def456" in sns.publish.call_args.kwargs["Message"]
        assert "Correlation id" in sns.publish.call_args.kwargs["Message"]

    def test_publishes_on_stale_source(self, monkeypatch):
        # A STALE source (items served off a park file whose local sync stopped) must alert too —
        # it isn't a FAILURE, so the has_failures-only gate stayed silent for days.
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        report = HealthReport(
            sources=[
                SourceHealth(name="rss", item_count=5, status=SourceStatus.OK),
                SourceHealth(name="youtube", item_count=3, status=SourceStatus.STALE, detail="72.0h old"),
            ]
        )
        sns = MagicMock()
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=sns):
            digest_handler._maybe_alert(report)
        sns.publish.assert_called_once()
        message = sns.publish.call_args.kwargs["Message"]
        assert "Stale sources" in message and "youtube" in message
        assert "Failed sources" not in message  # nothing failed

    def test_publish_error_is_swallowed(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        sns = MagicMock()
        sns.publish.side_effect = Exception("network down")
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=sns):
            digest_handler._maybe_alert(_report_with_failure())


class TestEmptySourceAlert:
    """A source that ran clean and returned NOTHING is dark: no exception, no stale park file, no
    failure ratio. Only the sources config NAMES may alert, so reddit/x quiet days can't page daily."""

    @staticmethod
    def _report_with_empty() -> HealthReport:
        return HealthReport(
            sources=[
                SourceHealth(name="rss", item_count=5, status=SourceStatus.OK),
                SourceHealth(name="reddit", item_count=0, status=SourceStatus.EMPTY),
                SourceHealth(name="web_search", item_count=0, status=SourceStatus.EMPTY),
            ]
        )

    def test_an_unwatched_empty_source_does_not_alert(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        with patch("lambda_handlers.digest_handler.boto3.client") as mock_client:
            digest_handler._maybe_alert(self._report_with_empty(), [])
        mock_client.assert_not_called()

    def test_a_watched_empty_source_alerts(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        sns = MagicMock()
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=sns):
            digest_handler._maybe_alert(self._report_with_empty(), ["web_search"])
        sns.publish.assert_called_once()
        message = sns.publish.call_args.kwargs["Message"]
        assert "Empty sources" in message and "web_search" in message
        # reddit is empty too, but it is not on the watch list — it must not appear as an incident.
        assert "reddit" not in message.split("Report:")[0]


class TestRankingHealthAlert:
    """A ranking batch that fails every retry deletes ~40 candidates from the day; the digest that
    follows looks entirely normal. Published separately from the collector alert (which runs BEFORE
    the pipeline), so a pipeline exception can never swallow the collector notice."""

    def test_publishes_when_candidates_were_lost(self, monkeypatch):
        from datetime import date

        from shared.models import RankingHealth

        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        sns = MagicMock()
        health = RankingHealth(batches_total=3, batches_failed=1, items_total=90, items_scored=60, items_lost=30)
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=sns):
            digest_handler._maybe_alert_ranking(health, date(2026, 8, 18))
        sns.publish.assert_called_once()
        assert "Ranking Health" in sns.publish.call_args.kwargs["Subject"]
        assert "30 of 90 candidates" in sns.publish.call_args.kwargs["Message"]

    def test_silent_on_a_complete_ranking_pass(self, monkeypatch):
        from datetime import date

        from shared.models import RankingHealth

        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        with patch("lambda_handlers.digest_handler.boto3.client") as client:
            digest_handler._maybe_alert_ranking(RankingHealth(batches_total=2, items_total=60), date(2026, 8, 18))
            digest_handler._maybe_alert_ranking(None, date(2026, 8, 18))
        client.assert_not_called()
