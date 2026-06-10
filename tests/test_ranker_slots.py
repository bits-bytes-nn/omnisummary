from shared.config import PipelineConfig
from shared.constants import SourceType
from shared.models import CollectedItem, RankedItem


def _ranked(
    score: float, source: SourceType, *, item_id: str, channel: str = "", author: str = "", sub: str = ""
) -> RankedItem:
    metadata = {}
    if channel:
        metadata["channel_url"] = channel
    if sub:
        metadata["subreddit"] = sub
    item = CollectedItem(
        item_id=item_id,
        source_type=source,
        title=f"title-{item_id}",
        url=f"http://example.com/{item_id}",
        author=author or None,
        metadata=metadata,
    )
    return RankedItem(item=item, score=score)


def _ranker(**overrides):
    from unittest.mock import MagicMock

    from pipeline.ranker import ContentRanker

    config = PipelineConfig(**overrides)
    factory = MagicMock()
    factory.get_model.return_value = MagicMock()
    return ContentRanker(config, factory)


class TestOriginCap:
    def test_single_channel_cannot_monopolize_overflow(self):
        ranker = _ranker(
            top_n=3,
            min_score=0.5,
            source_slots={"youtube": 1},
            source_cap_multiplier=2,
            max_per_origin=1,
        )
        # 3 high-scoring videos all from the same channel
        items = [
            _ranked(0.9, SourceType.YOUTUBE, item_id="v1", channel="chanA"),
            _ranked(0.88, SourceType.YOUTUBE, item_id="v2", channel="chanA"),
            _ranked(0.86, SourceType.YOUTUBE, item_id="v3", channel="chanA"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        # No distinct origins to diversify into, so the fallback fills up to the SOURCE cap
        # (1 slot x 2 multiplier = 2) — bounded monopoly, never all 3.
        assert len(selected) == 2
        assert {r.item.item_id for r in selected} == {"v1", "v2"}

    def test_distinct_channels_fill_slots(self):
        ranker = _ranker(
            top_n=3,
            min_score=0.5,
            source_slots={"youtube": 1},
            source_cap_multiplier=5,
            max_per_origin=1,
        )
        items = [
            _ranked(0.9, SourceType.YOUTUBE, item_id="v1", channel="chanA"),
            _ranked(0.88, SourceType.YOUTUBE, item_id="v2", channel="chanB"),
            _ranked(0.86, SourceType.YOUTUBE, item_id="v3", channel="chanC"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        channels = {r.item.metadata["channel_url"] for r in selected}
        assert len(selected) == 3
        assert channels == {"chanA", "chanB", "chanC"}

    def test_higher_cap_allows_more_per_origin(self):
        # top_n=2 so the digest is filled before the fallback pass — isolates the
        # max_per_origin=2 behavior: the diversity pass alone admits 2 from one channel.
        ranker = _ranker(
            top_n=2,
            min_score=0.5,
            source_slots={"youtube": 1},
            source_cap_multiplier=5,
            max_per_origin=2,
        )
        items = [
            _ranked(0.9, SourceType.YOUTUBE, item_id="v1", channel="chanA"),
            _ranked(0.88, SourceType.YOUTUBE, item_id="v2", channel="chanA"),
            _ranked(0.86, SourceType.YOUTUBE, item_id="v3", channel="chanA"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert len(selected) == 2
        assert {r.item.item_id for r in selected} == {"v1", "v2"}

    def test_items_without_origin_not_capped(self):
        ranker = _ranker(
            top_n=3,
            min_score=0.5,
            source_slots={"web": 1},
            source_cap_multiplier=5,
            max_per_origin=1,
        )
        # web items have no origin key -> not subject to origin cap
        items = [
            _ranked(0.9, SourceType.WEB, item_id="w1"),
            _ranked(0.88, SourceType.WEB, item_id="w2"),
            _ranked(0.86, SourceType.WEB, item_id="w3"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert len(selected) == 3

    def test_fallback_fills_top_n_when_origins_exhausted(self):
        # Only one X author has items; without the relaxation pass the digest would stop
        # at 1 (origin cap) even though top_n=3 and the source cap allows more.
        ranker = _ranker(
            top_n=3,
            min_score=0.5,
            source_slots={"x": 1},
            source_cap_multiplier=5,
            max_per_origin=1,
        )
        items = [
            _ranked(0.9, SourceType.X, item_id="t1", author="alice"),
            _ranked(0.88, SourceType.X, item_id="t2", author="alice"),
            _ranked(0.86, SourceType.X, item_id="t3", author="alice"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert len(selected) == 3  # fallback relaxes origin cap (source cap 1x5=5 allows it)


class TestSourceSlotGrace:
    def test_admits_best_below_threshold_item_within_grace(self):
        # YouTube has a slot but nothing clears 0.6; its best (0.55) is within grace 0.1 → admitted.
        ranker = _ranker(
            min_score=0.6,
            source_slot_score_grace=0.1,
            source_slots={"youtube": 1, "rss": 1},
        )
        ranked = [
            _ranked(0.55, SourceType.YOUTUBE, item_id="y1", channel="c"),
            _ranked(0.40, SourceType.YOUTUBE, item_id="y2", channel="c"),
            _ranked(0.80, SourceType.RSS, item_id="r1"),
        ]
        extra = ranker._grace_candidates(ranked, [r for r in ranked if r.score >= 0.6])
        assert [r.item.item_id for r in extra] == ["y1"]  # best within grace, only one

    def test_no_grace_when_source_already_above_threshold(self):
        ranker = _ranker(min_score=0.6, source_slot_score_grace=0.1, source_slots={"youtube": 1})
        ranked = [
            _ranked(0.70, SourceType.YOUTUBE, item_id="y1", channel="c"),
            _ranked(0.55, SourceType.YOUTUBE, item_id="y2", channel="c"),
        ]
        extra = ranker._grace_candidates(ranked, [r for r in ranked if r.score >= 0.6])
        assert extra == []  # already has an above-threshold item

    def test_below_grace_floor_not_admitted(self):
        ranker = _ranker(min_score=0.6, source_slot_score_grace=0.1, source_slots={"youtube": 1})
        ranked = [_ranked(0.40, SourceType.YOUTUBE, item_id="y1", channel="c")]  # 0.40 < floor 0.50
        assert ranker._grace_candidates(ranked, []) == []

    def test_grace_disabled_returns_nothing(self):
        ranker = _ranker(min_score=0.6, source_slot_score_grace=0.0, source_slots={"youtube": 1})
        ranked = [_ranked(0.55, SourceType.YOUTUBE, item_id="y1", channel="c")]
        assert ranker._grace_candidates(ranked, []) == []


class TestOriginWeights:
    def test_named_origin_weight_is_additive_nudge(self):
        # weight 1.5 with nudge 0.1 -> +0.05 (NOT multiplicative 0.5*1.5=0.75)
        ranker = _ranker(origin_weights={"chanA": 1.5}, origin_weight_default=1.0, origin_weight_nudge=0.1)
        items = [_ranked(0.5, SourceType.YOUTUBE, item_id="v1", channel="chanA")]
        ranker._apply_origin_weights(items)
        assert abs(items[0].score - 0.55) < 1e-9

    def test_default_weight_nudges_unlisted_origin(self):
        # default 0.8 -> (0.8-1.0)*0.1 = -0.02
        ranker = _ranker(origin_weights={}, origin_weight_default=0.8, origin_weight_nudge=0.1)
        items = [_ranked(0.8, SourceType.YOUTUBE, item_id="v1", channel="chanB")]
        ranker._apply_origin_weights(items)
        assert abs(items[0].score - 0.78) < 1e-9

    def test_no_op_when_default_one_and_no_weights(self):
        ranker = _ranker(origin_weights={}, origin_weight_default=1.0)
        items = [_ranked(0.8, SourceType.YOUTUBE, item_id="v1", channel="chanB")]
        ranker._apply_origin_weights(items)
        assert items[0].score == 0.8

    def test_nudge_clamped_to_unit_range(self):
        ranker = _ranker(origin_weights={"chanA": 5.0}, origin_weight_default=1.0, origin_weight_nudge=0.1)
        items = [_ranked(0.95, SourceType.YOUTUBE, item_id="v1", channel="chanA")]
        ranker._apply_origin_weights(items)
        assert items[0].score == 1.0  # 0.95 + 0.4 clamped to 1.0
