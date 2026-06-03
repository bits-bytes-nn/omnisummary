from shared.models import HealthReport, SourceHealth, SourceStatus


class TestHealthReport:
    def test_empty_report_has_no_failures(self):
        assert HealthReport().has_failures is False

    def test_has_failures_true_when_any_failed(self):
        report = HealthReport(
            sources=[
                SourceHealth(name="rss", item_count=10, status=SourceStatus.OK),
                SourceHealth(name="reddit", item_count=0, status=SourceStatus.FAILED, detail="403"),
            ]
        )
        assert report.has_failures is True

    def test_has_failures_false_when_only_empty(self):
        report = HealthReport(
            sources=[
                SourceHealth(name="rss", item_count=10, status=SourceStatus.OK),
                SourceHealth(name="reddit", item_count=0, status=SourceStatus.EMPTY),
            ]
        )
        assert report.has_failures is False

    def test_summary_includes_all_sources(self):
        report = HealthReport(
            sources=[
                SourceHealth(name="rss", item_count=12, status=SourceStatus.OK),
                SourceHealth(name="youtube", item_count=0, status=SourceStatus.FAILED, detail="boom"),
            ]
        )
        summary = report.summary()
        assert "[OK] rss: 12 items" in summary
        assert "[FAILED] youtube: 0 items — boom" in summary

    def test_summary_omits_detail_when_absent(self):
        report = HealthReport(sources=[SourceHealth(name="rss", item_count=3, status=SourceStatus.OK)])
        assert report.summary() == "[OK] rss: 3 items"

    def test_status_enum_values(self):
        assert SourceStatus.OK.value == "ok"
        assert SourceStatus.EMPTY.value == "empty"
        assert SourceStatus.FAILED.value == "failed"
