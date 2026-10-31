from datetime import date

import pytest
from pydantic import ValidationError

from domain.models import CollectedItem, RankedItem, Trend, TrendEvidence, TrendMemory, VisualBrief, generate_item_id
from shared.constants import SourceType


class TestCollectedItem:
    def test_fallback_item_id_from_url(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://example.com")
        assert item.item_id == generate_item_id("http://example.com")

    def test_explicit_item_id_preserved(self):
        item = CollectedItem(item_id="my_id", source_type=SourceType.REDDIT, title="T", url="http://example.com")
        assert item.item_id == "my_id"

    def test_hash_by_url(self):
        a = CollectedItem(source_type=SourceType.REDDIT, title="A", url="http://a.com")
        b = CollectedItem(source_type=SourceType.REDDIT, title="B", url="http://a.com")
        assert hash(a) == hash(b)

    def test_eq_by_url(self):
        a = CollectedItem(source_type=SourceType.REDDIT, title="A", url="http://a.com")
        b = CollectedItem(source_type=SourceType.REDDIT, title="B", url="http://a.com")
        assert a == b

    def test_not_eq_different_url(self):
        a = CollectedItem(source_type=SourceType.REDDIT, title="A", url="http://a.com")
        b = CollectedItem(source_type=SourceType.REDDIT, title="A", url="http://b.com")
        assert a != b


class TestRankedItem:
    def test_score_in_range(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://a.com")
        ranked = RankedItem(item=item, score=0.5)
        assert ranked.score == 0.5

    def test_score_below_zero_rejected(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://a.com")
        with pytest.raises(ValidationError):
            RankedItem(item=item, score=-0.1)

    def test_score_above_one_rejected(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://a.com")
        with pytest.raises(ValidationError):
            RankedItem(item=item, score=1.1)

    def test_boundary_scores(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://a.com")
        assert RankedItem(item=item, score=0.0).score == 0.0
        assert RankedItem(item=item, score=1.0).score == 1.0


class TestVisualBrief:
    def test_valid_brief(self):
        brief = VisualBrief(title="T", prompt="P", orientation="portrait")
        assert brief.title == "T"

    def test_a_long_title_does_not_cost_the_visual(self):
        # The title only feeds a log line; a length cap could only fail an otherwise good brief.
        assert VisualBrief(title="x" * 300, prompt="P", orientation="portrait")


class TestTrendMomentum:
    def test_today_evidence_full_weight(self):
        today = date(2026, 6, 5)
        trend = Trend(id="t", title="T", evidence=[TrendEvidence(date="2026-06-05", summary="s")])
        assert trend.momentum(today, half_life_days=7.0) == pytest.approx(1.0)

    def test_half_life_decay(self):
        today = date(2026, 6, 8)
        # 7 days old at half_life 7 -> 0.5; 0 days old -> 1.0
        trend = Trend(
            id="t",
            title="T",
            evidence=[
                TrendEvidence(date="2026-06-08", summary="fresh"),
                TrendEvidence(date="2026-06-01", summary="week-old"),
            ],
        )
        assert trend.momentum(today, half_life_days=7.0) == pytest.approx(1.5)

    def test_recent_outranks_stale(self):
        today = date(2026, 6, 30)
        recent = Trend(id="r", title="R", evidence=[TrendEvidence(date="2026-06-29", summary="s")])
        stale = Trend(id="s", title="S", evidence=[TrendEvidence(date="2026-06-01", summary="s")])
        assert recent.momentum(today, 7.0) > stale.momentum(today, 7.0)


    def test_bad_date_contributes_zero(self):
        today = date(2026, 6, 5)
        trend = Trend(id="t", title="T", evidence=[TrendEvidence(date="not-a-date", summary="s")])
        assert trend.momentum(today, half_life_days=7.0) == 0.0

    def test_future_date_clamps_to_full_weight(self):
        today = date(2026, 6, 5)
        trend = Trend(id="t", title="T", evidence=[TrendEvidence(date="2026-06-10", summary="s")])
        assert trend.momentum(today, half_life_days=7.0) == pytest.approx(1.0)


class TestTrendMemory:
    def test_by_id_lookup(self):
        memory = TrendMemory(trends=[Trend(id="x", title="X"), Trend(id="y", title="Y")])
        assert memory.by_id("y").title == "Y"
        assert memory.by_id("z") is None
