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
            source_cap_multiplier=5,
            max_per_origin=1,
        )
        # 3 high-scoring videos all from the same channel
        items = [
            _ranked(0.9, SourceType.YOUTUBE, item_id="v1", channel="chanA"),
            _ranked(0.88, SourceType.YOUTUBE, item_id="v2", channel="chanA"),
            _ranked(0.86, SourceType.YOUTUBE, item_id="v3", channel="chanA"),
        ]
        selected = ranker._apply_source_slots(items)
        # max_per_origin=1 means only one chanA video survives
        assert len(selected) == 1
        assert selected[0].item.item_id == "v1"

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
        selected = ranker._apply_source_slots(items)
        channels = {r.item.metadata["channel_url"] for r in selected}
        assert len(selected) == 3
        assert channels == {"chanA", "chanB", "chanC"}

    def test_higher_cap_allows_more_per_origin(self):
        ranker = _ranker(
            top_n=3,
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
        selected = ranker._apply_source_slots(items)
        assert len(selected) == 2

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
        selected = ranker._apply_source_slots(items)
        assert len(selected) == 3


class TestOriginWeights:
    def test_named_origin_weight_applied(self):
        ranker = _ranker(origin_weights={"chanA": 1.5}, origin_weight_default=1.0)
        items = [_ranked(0.5, SourceType.YOUTUBE, item_id="v1", channel="chanA")]
        ranker._apply_origin_weights(items)
        assert items[0].score == 0.75

    def test_default_weight_applied_to_unlisted_origin(self):
        ranker = _ranker(origin_weights={}, origin_weight_default=0.5)
        items = [_ranked(0.8, SourceType.YOUTUBE, item_id="v1", channel="chanB")]
        ranker._apply_origin_weights(items)
        assert items[0].score == 0.4

    def test_no_op_when_default_one_and_no_weights(self):
        ranker = _ranker(origin_weights={}, origin_weight_default=1.0)
        items = [_ranked(0.8, SourceType.YOUTUBE, item_id="v1", channel="chanB")]
        ranker._apply_origin_weights(items)
        assert items[0].score == 0.8

    def test_score_capped_at_one(self):
        ranker = _ranker(origin_weights={"chanA": 2.0}, origin_weight_default=1.0)
        items = [_ranked(0.8, SourceType.YOUTUBE, item_id="v1", channel="chanA")]
        ranker._apply_origin_weights(items)
        assert items[0].score == 1.0
