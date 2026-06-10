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

    def test_publish_error_is_swallowed(self, monkeypatch):
        monkeypatch.setenv("ALERT_SNS_TOPIC_ARN", "arn:aws:sns:::topic")
        sns = MagicMock()
        sns.publish.side_effect = Exception("network down")
        with patch("lambda_handlers.digest_handler.boto3.client", return_value=sns):
            digest_handler._maybe_alert(_report_with_failure())
