from shared.models import HealthReport, SourceHealth, SourceStatus


class TestHealthReport:
    def test_empty_report_has_no_failures(self):
        assert HealthReport().failed_sources == []

    def test_failed_sources_names_only_the_failed_source(self):
        report = HealthReport(
            sources=[
                SourceHealth(name="rss", item_count=10, status=SourceStatus.OK),
                SourceHealth(name="reddit", item_count=0, status=SourceStatus.FAILED, detail="403"),
            ]
        )
        assert report.failed_sources == ["reddit"]

    def test_empty_source_is_not_a_failure(self):
        report = HealthReport(
            sources=[
                SourceHealth(name="rss", item_count=10, status=SourceStatus.OK),
                SourceHealth(name="reddit", item_count=0, status=SourceStatus.EMPTY),
            ]
        )
        assert report.failed_sources == []

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
        assert SourceStatus.STALE.value == "stale"

    def test_stale_is_reported_but_is_not_a_failure(self):
        # A STALE source produced items off a park file whose sync has stopped: it must be listed
        # for alerting, but must NOT land in failed_sources (which drives the FAILED escalation).
        report = HealthReport(
            sources=[
                SourceHealth(name="rss", item_count=10, status=SourceStatus.OK),
                SourceHealth(name="youtube", item_count=3, status=SourceStatus.STALE, detail="72.0h old"),
            ]
        )
        assert report.failed_sources == []
        assert report.stale_sources == ["youtube"]
        assert "[STALE] youtube: 3 items — 72.0h old" in report.summary()

    def test_stale_sources_empty_when_all_healthy(self):
        report = HealthReport(sources=[SourceHealth(name="rss", item_count=1, status=SourceStatus.OK)])
        assert report.stale_sources == []
