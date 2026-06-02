from datetime import date

from pipeline.trend_tracker import ARCHIVED_MARKER, MAX_EVIDENCE_PER_TREND, TrendTracker


class TestStripCodeFences:
    def test_strips_json_fence(self):
        assert TrendTracker._strip_code_fences("```markdown\n# Trends\n```") == "# Trends"

    def test_strips_bare_fence(self):
        assert TrendTracker._strip_code_fences("```\nhi\n```") == "hi"

    def test_no_fence_unchanged(self):
        assert TrendTracker._strip_code_fences("# Trends\nbody") == "# Trends\nbody"


class TestTrimForLlm:
    def test_empty_returns_empty(self):
        assert TrendTracker._trim_for_llm("", "2026-06-02") == ("", "")

    def test_splits_archived_section(self):
        content = f"# Active\n- **Trend**: A\n\n{ARCHIVED_MARKER}\n- old archived entry"
        active, archived = TrendTracker._trim_for_llm(content, "2026-06-02")
        assert ARCHIVED_MARKER in archived
        assert "old archived entry" in archived
        assert ARCHIVED_MARKER not in active

    def test_truncates_oversized(self):
        big = "x" * 20000
        active, _ = TrendTracker._trim_for_llm(big, "2026-06-02")
        assert "truncated for size" in active

    def test_invalid_date_skips_cutoff(self):
        content = "# Active\n- **Trend**: A"
        active, _ = TrendTracker._trim_for_llm(content, "not-a-date")
        assert "Trend" in active


class TestTrimEvidence:
    def test_drops_entries_before_cutoff(self):
        content = (
            "- **Trend**: X\n"
            "- **Evidence**:\n"
            "- [2026-05-01] old item\n"
            "- [2026-06-01] new item\n"
            "- **Impact**: high\n"
        )
        result = TrendTracker._trim_evidence(content, date(2026, 5, 15))
        assert "new item" in result
        assert "old item" not in result
        assert "1 earlier entries omitted" in result

    def test_caps_evidence_count(self):
        recent = "\n".join(f"- [2026-06-0{i}] item{i}" for i in range(1, 9))
        content = f"- **Trend**: X\n- **Evidence**:\n{recent}\n- **Impact**: y\n"
        result = TrendTracker._trim_evidence(content, date(2026, 1, 1))
        kept = [ln for ln in result.split("\n") if ln.strip().startswith("- [")]
        assert len(kept) == MAX_EVIDENCE_PER_TREND

    def test_keeps_undated_evidence(self):
        content = "- **Trend**: X\n- **Evidence**:\n- [bad-date] keep me\n- **Impact**: y\n"
        result = TrendTracker._trim_evidence(content, date(2026, 6, 1))
        assert "keep me" in result

    def test_non_evidence_lines_untouched(self):
        content = "# Header\n- **Trend**: X\n- **Impact**: high\nplain line"
        result = TrendTracker._trim_evidence(content, date(2026, 6, 1))
        assert result == content  # no evidence block -> content passes through verbatim


class TestMergeArchived:
    def test_no_old_archived_returns_updated(self):
        assert TrendTracker._merge_archived("# Active", "") == "# Active"

    def test_appends_when_no_new_archived_section(self):
        updated = "# Active trends"
        old = f"{ARCHIVED_MARKER}\n- archived A"
        merged = TrendTracker._merge_archived(updated, old)
        assert "Active trends" in merged
        assert "archived A" in merged

    def test_merges_unique_old_entries(self):
        updated = f"# Active\n\n{ARCHIVED_MARKER}\n- new entry"
        old = f"{ARCHIVED_MARKER}\n- new entry\n- unique old entry"
        merged = TrendTracker._merge_archived(updated, old)
        assert merged.count("- new entry") == 1
        assert "unique old entry" in merged
