from domain.models import CollectedItem, RankedItem
from shared.config import PipelineConfig
from shared.constants import SourceType


def _ranked(
    score: float,
    source: SourceType,
    *,
    item_id: str,
    channel: str = "",
    author: str = "",
    sub: str = "",
    feed: str = "",
    url: str = "",
) -> RankedItem:
    metadata = {}
    if channel:
        metadata["channel_url"] = channel
    if sub:
        metadata["subreddit"] = sub
    if feed:
        metadata["feed_url"] = feed
    item = CollectedItem(
        item_id=item_id,
        source_type=source,
        title=f"title-{item_id}",
        url=url or f"http://example.com/{item_id}",
        author=author or None,
        metadata=metadata,
    )
    return RankedItem(item=item, score=score)


def _ranker(**overrides):
    from unittest.mock import MagicMock

    from application.ranker import ContentRanker

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
            max_per_origin=1,
        )
        # 3 high-scoring videos all from the same channel
        items = [
            _ranked(0.9, SourceType.YOUTUBE, item_id="v1", channel="chanA"),
            _ranked(0.88, SourceType.YOUTUBE, item_id="v2", channel="chanA"),
            _ranked(0.86, SourceType.YOUTUBE, item_id="v3", channel="chanA"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        # No distinct origins to diversify into, so the relaxed pass fills the pool rather than
        # leaving it short for a diversity that cannot happen.
        assert len(selected) == 3
        assert {r.item.item_id for r in selected} == {"v1", "v2", "v3"}

    def test_distinct_channels_fill_slots(self):
        ranker = _ranker(
            top_n=3,
            min_score=0.5,
            source_slots={"youtube": 1},
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

    def test_an_item_missing_its_source_metadata_is_capped_by_its_host(self):
        # An origin-less item used to escape max_per_origin entirely. Here three RSS entries carry no
        # feed_url but share one host, so exactly one of them takes a slot and the lower-scored,
        # genuinely distinct-origin r4 gets the next — instead of one site filling the whole digest.
        ranker = _ranker(
            top_n=2,
            min_score=0.5,
            source_slots={"rss": 3},
            max_per_origin=1,
        )
        items = [
            _ranked(0.95, SourceType.RSS, item_id="r1"),
            _ranked(0.94, SourceType.RSS, item_id="r2"),
            _ranked(0.93, SourceType.RSS, item_id="r3"),
            _ranked(0.60, SourceType.RSS, item_id="r4", feed="https://other.example/feed"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert {r.item.item_id for r in selected} == {"r1", "r4"}

    def test_an_author_less_x_item_is_capped_by_its_host(self):
        # Observed live on 2026-08-18: DOMAIN_TO_SOURCE relabels a web-search hit or a pinned x.com
        # URL as SourceType.X, whose origin is item.author — which those items never carry. RSSHub was
        # unreachable that day, yet "Source x: 1 items" was filled by an author-less scrape that had no
        # origin key, no Origin line in the ranking prompt, and no max_per_origin at all.
        ranker = _ranker(top_n=2, min_score=0.5, source_slots={"x": 3}, max_per_origin=1)
        items = [
            _ranked(0.95, SourceType.X, item_id="s1", url="https://x.com/a/status/1"),
            _ranked(0.94, SourceType.X, item_id="s2", url="https://x.com/b/status/2"),
            _ranked(0.70, SourceType.X, item_id="s3", author="karpathy"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        # s2 shares s1's host, so the lower-scored but genuinely distinct s3 takes the second slot.
        assert {r.item.item_id for r in selected} == {"s1", "s3"}

    def test_one_web_site_cannot_take_two_slots(self):
        # Web items used to resolve to no origin at all, so a single outlet could occupy several
        # digest slots; the host is now the origin key.
        ranker = _ranker(
            top_n=3,
            min_score=0.5,
            source_slots={"web": 1, "rss": 1},
            max_per_origin=1,
        )
        items = [
            _ranked(0.95, SourceType.WEB, item_id="w1", url="https://site-a.example/1"),
            _ranked(0.94, SourceType.WEB, item_id="w2", url="https://www.site-a.example/2"),  # same host
            _ranked(0.93, SourceType.WEB, item_id="w3", url="https://site-b.example/3"),
            _ranked(0.60, SourceType.RSS, item_id="r1", feed="https://f.example/feed"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        ids = {r.item.item_id for r in selected}
        assert "w2" not in ids  # second story from site-a (www. normalized) is capped out
        assert ids == {"w1", "w3", "r1"}


class TestOriginCapWithoutSourceSlots:
    """`source_slots: {}` is a legitimate config (no per-source guarantees). It used to short-circuit
    before ANY origin accounting, so max_per_origin vanished with the slots and one feed could take
    every story — the exact failure the origin cap exists to prevent."""

    def test_the_origin_cap_still_applies(self):
        ranker = _ranker(top_n=2, min_score=0.5, source_slots={}, max_per_origin=1)
        items = [
            _ranked(0.95, SourceType.RSS, item_id="r1", feed="https://one.example/feed"),
            _ranked(0.94, SourceType.RSS, item_id="r2", feed="https://one.example/feed"),
            _ranked(0.60, SourceType.RSS, item_id="r3", feed="https://two.example/feed"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert {r.item.item_id for r in selected} == {"r1", "r3"}

    def test_the_digest_still_fills_when_only_one_origin_has_candidates(self):
        # The relaxed passes must still run: a reader must not lose a story to a diversity that has
        # no candidates to spend on.
        ranker = _ranker(top_n=2, min_score=0.5, source_slots={}, max_per_origin=1)
        items = [
            _ranked(0.95, SourceType.RSS, item_id="r1", feed="https://one.example/feed"),
            _ranked(0.94, SourceType.RSS, item_id="r2", feed="https://one.example/feed"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert {r.item.item_id for r in selected} == {"r1", "r2"}


class TestPinnedCaps:
    def test_pinned_item_counts_toward_the_origin_cap(self):
        # Pinned items are prepended by rank() and never entered this fill, so their origin went
        # uncounted and a same-origin item landed alongside the pin.
        ranker = _ranker(
            top_n=3,
            min_score=0.5,
            source_slots={"web": 1, "rss": 1},
            max_per_origin=1,
        )
        pinned = [_ranked(0.99, SourceType.WEB, item_id="p1", url="https://site-a.example/pin")]
        items = [
            _ranked(0.95, SourceType.WEB, item_id="w1", url="https://site-a.example/1"),  # pin's host
            _ranked(0.94, SourceType.WEB, item_id="w2", url="https://site-b.example/2"),
            _ranked(0.93, SourceType.RSS, item_id="r1", feed="https://f.example/feed"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n - len(pinned), pinned)
        ids = {r.item.item_id for r in selected}
        assert "w1" not in ids
        assert ids == {"w2", "r1"}

    def test_fallback_fills_top_n_when_origins_exhausted(self):
        # Only one X author has items; without the relaxation pass the digest would stop
        # at 1 (origin cap) even though top_n=3.
        ranker = _ranker(
            top_n=3,
            min_score=0.5,
            source_slots={"x": 1},
            max_per_origin=1,
        )
        items = [
            _ranked(0.9, SourceType.X, item_id="t1", author="alice"),
            _ranked(0.88, SourceType.X, item_id="t2", author="alice"),
            _ranked(0.86, SourceType.X, item_id="t3", author="alice"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert len(selected) == 3  # the relaxed pass lifts the origin cap


class TestFillPassOrder:
    def test_past_its_seat_a_source_competes_on_score_alone(self):
        # A per-source cap of 2 kept a 0.80 RSS story out of the pool while 0.60 items got in on 9 of
        # 16 days; max_per_origin already keeps one publisher from flooding it.
        r = _ranker(source_slots={"rss": 1, "youtube": 1}, max_per_origin=1)
        items = [
            _ranked(0.9, SourceType.RSS, item_id="a", feed="f1"),
            _ranked(0.85, SourceType.RSS, item_id="b", feed="f2"),
            _ranked(0.8, SourceType.RSS, item_id="c", feed="f3"),
            _ranked(0.6, SourceType.YOUTUBE, item_id="y", channel="c1"),
        ]
        assert {s.item.item_id for s in r._apply_source_slots(items, 3)} == {"a", "b", "y"}
        assert {s.item.item_id for s in r._apply_source_slots(items, 4)} == {"a", "b", "c", "y"}

    def test_the_pool_comes_back_in_score_order(self):
        r = _ranker(source_slots={"youtube": 1, "rss": 1}, max_per_origin=1)
        items = [
            _ranked(0.9, SourceType.RSS, item_id="a", feed="f1"),
            _ranked(0.8, SourceType.RSS, item_id="b", feed="f2"),
            _ranked(0.6, SourceType.YOUTUBE, item_id="y", channel="c1"),
        ]
        scores = [s.score for s in r._apply_source_slots(items, 3)]
        assert scores == sorted(scores, reverse=True)
