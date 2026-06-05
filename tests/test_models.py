import hashlib
from datetime import date

import pytest
from pydantic import ValidationError

from shared.constants import SourceType
from shared.models import CollectedItem, RankedItem, Trend, TrendEvidence, TrendMemory, VisualBrief


class TestCollectedItem:
    def test_fallback_item_id_from_url(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://example.com")
        expected = hashlib.sha256(b"http://example.com").hexdigest()[:16]
        assert item.item_id == expected

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
        brief = VisualBrief(title="T", caption="C", prompt="P")
        assert brief.title == "T"

    def test_overlong_title_rejected(self):
        with pytest.raises(ValidationError):
            VisualBrief(title="x" * 101, caption="C", prompt="P")

    def test_overlong_caption_rejected(self):
        with pytest.raises(ValidationError):
            VisualBrief(title="T", caption="x" * 301, prompt="P")


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

    def test_nonpositive_half_life_falls_back_to_count(self):
        today = date(2026, 6, 5)
        trend = Trend(
            id="t",
            title="T",
            evidence=[TrendEvidence(date="2026-01-01", summary="a"), TrendEvidence(date="2026-06-05", summary="b")],
        )
        assert trend.momentum(today, half_life_days=0) == 2.0

    def test_bad_date_contributes_zero(self):
        today = date(2026, 6, 5)
        trend = Trend(id="t", title="T", evidence=[TrendEvidence(date="not-a-date", summary="s")])
        assert trend.momentum(today, half_life_days=7.0) == 0.0

    def test_future_date_clamps_to_full_weight(self):
        today = date(2026, 6, 5)
        trend = Trend(id="t", title="T", evidence=[TrendEvidence(date="2026-06-10", summary="s")])
        assert trend.momentum(today, half_life_days=7.0) == pytest.approx(1.0)


class TestTrendMemorySearch:
    def _mem(self):
        from shared.models import TrendMemory, TrendStatus

        return TrendMemory(
            trends=[
                Trend(
                    id="kv",
                    title="KV Cache Efficiency",
                    status=TrendStatus.ACTIVE,
                    evidence=[TrendEvidence(date="2026-06-05", summary="text diffusion bypasses kv cache")],
                ),
                Trend(
                    id="ipo",
                    title="Lab IPO Race",
                    status=TrendStatus.ACTIVE,
                    evidence=[TrendEvidence(date="2026-06-01", summary="anthropic files s-1")],
                ),
                Trend(
                    id="old",
                    title="Archived Topic",
                    status=TrendStatus.ARCHIVED,
                    evidence=[TrendEvidence(date="2026-05-01", summary="kv cache old note")],
                ),
            ]
        )

    def test_excludes_archived(self):
        out = self._mem().search("kv cache", today=date(2026, 6, 5), half_life_days=7.0, top_k=5)
        assert [t.id for t in out] == ["kv"]  # archived 'old' excluded despite matching

    def test_ranks_by_term_hits_then_momentum(self):
        out = self._mem().search("ipo race", today=date(2026, 6, 5), half_life_days=7.0, top_k=5)
        assert out[0].id == "ipo"

    def test_empty_query_returns_by_momentum(self):
        out = self._mem().search("", today=date(2026, 6, 5), half_life_days=7.0, top_k=5)
        assert out[0].id == "kv"  # most recent evidence -> highest momentum

    def test_top_k_caps(self):
        out = self._mem().search("", today=date(2026, 6, 5), half_life_days=7.0, top_k=1)
        assert len(out) == 1

    def test_zero_half_life_falls_back_to_count(self):
        today = date(2026, 6, 5)
        trend = Trend(
            id="t",
            title="T",
            evidence=[TrendEvidence(date="2026-06-01", summary="a"), TrendEvidence(date="2026-05-01", summary="b")],
        )
        assert trend.momentum(today, half_life_days=0.0) == 2.0

    def test_invalid_evidence_date_skipped(self):
        today = date(2026, 6, 5)
        trend = Trend(id="t", title="T", evidence=[TrendEvidence(date="not-a-date", summary="s")])
        assert trend.momentum(today, 7.0) == 0.0

    def test_by_id_lookup(self):
        memory = TrendMemory(trends=[Trend(id="x", title="X"), Trend(id="y", title="Y")])
        assert memory.by_id("y").title == "Y"
        assert memory.by_id("z") is None
