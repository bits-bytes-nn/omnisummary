from shared.config import PipelineConfig
from shared.constants import SourceType
from shared.models import CollectedItem, RankedItem


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
    grace: bool = False,
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
    return RankedItem(item=item, score=score, grace=grace)


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
        # No distinct origins to diversify into and no other source with candidates, so after the
        # origin cap and then the source cap are exhausted the LAST-RESORT pass fills the digest
        # rather than shipping it short: with a collector outage the caps have nothing left to
        # spend diversity on, and a reader loses a story for a diversity that cannot happen.
        assert len(selected) == 3
        assert {r.item.item_id for r in selected} == {"v1", "v2", "v3"}

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

    def test_an_item_missing_its_source_metadata_is_capped_by_its_host(self):
        # An origin-less item used to escape max_per_origin entirely. Here three RSS entries carry no
        # feed_url but share one host, so exactly one of them takes a slot and the lower-scored,
        # genuinely distinct-origin r4 gets the next — instead of one site filling the whole digest.
        ranker = _ranker(
            top_n=2,
            min_score=0.5,
            source_slots={"rss": 3},
            source_cap_multiplier=1,
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
        ranker = _ranker(top_n=2, min_score=0.5, source_slots={"x": 3}, source_cap_multiplier=1, max_per_origin=1)
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
            source_cap_multiplier=5,
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


class TestSlotOrder:
    def test_short_limit_gives_slots_to_the_strongest_sources(self):
        # limit(2) < sum(source_slots)(3): the guaranteed pass used to walk the config key order, so
        # the LAST-listed source lost its slot no matter how strong its candidate was.
        ranker = _ranker(
            top_n=2,
            min_score=0.5,
            source_slots={"reddit": 1, "rss": 1, "youtube": 1},
            source_cap_multiplier=1,
            max_per_origin=1,
        )
        items = [
            _ranked(0.95, SourceType.YOUTUBE, item_id="y1", channel="chanA"),
            _ranked(0.90, SourceType.RSS, item_id="r1", feed="https://f.example/feed"),
            _ranked(0.60, SourceType.REDDIT, item_id="d1", sub="ml"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert {r.item.item_id for r in selected} == {"y1", "r1"}

    def test_full_limit_selection_is_order_independent(self):
        # With limit >= sum(source_slots) (the live config) every source fills its own slot, so the
        # strength ordering must not change the outcome — whatever order the YAML lists.
        slots = {"reddit": 1, "rss": 1, "youtube": 1}
        items = [
            _ranked(0.95, SourceType.YOUTUBE, item_id="y1", channel="chanA"),
            _ranked(0.90, SourceType.RSS, item_id="r1", feed="https://f.example/feed"),
            _ranked(0.60, SourceType.REDDIT, item_id="d1", sub="ml"),
        ]
        selected_ids = set()
        for keys in (("reddit", "rss", "youtube"), ("youtube", "rss", "reddit")):
            ranker = _ranker(
                top_n=3,
                min_score=0.5,
                source_slots={k: slots[k] for k in keys},
                source_cap_multiplier=1,
                max_per_origin=1,
            )
            selected_ids.add(frozenset(r.item.item_id for r in ranker._apply_source_slots(items, 3)))
        assert selected_ids == {frozenset({"y1", "r1", "d1"})}

    def test_ties_break_on_source_key_for_determinism(self):
        ranker = _ranker(source_slots={"rss": 1, "reddit": 1, "youtube": 1})
        items = [
            _ranked(0.8, SourceType.RSS, item_id="r1"),
            _ranked(0.8, SourceType.REDDIT, item_id="d1"),
        ]
        assert [src for src, _ in ranker._slot_order(items, ranker.config.source_slots)] == [
            "reddit",
            "rss",
            "youtube",
        ]


class TestPinnedCaps:
    def test_pinned_item_counts_toward_origin_and_source_caps(self):
        # Pinned items are prepended by rank() and never entered this fill, so their origin went
        # uncounted and a same-origin item landed alongside the pin.
        ranker = _ranker(
            top_n=3,
            min_score=0.5,
            source_slots={"web": 1, "rss": 1},
            source_cap_multiplier=5,
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
        extra = ranker._grace_candidates(ranked, [r for r in ranked if r.score >= 0.6], [])
        assert [r.item.item_id for r in extra] == ["y1"]  # best within grace, only one

    def test_no_grace_when_source_already_above_threshold(self):
        ranker = _ranker(min_score=0.6, source_slot_score_grace=0.1, source_slots={"youtube": 1})
        ranked = [
            _ranked(0.70, SourceType.YOUTUBE, item_id="y1", channel="c"),
            _ranked(0.55, SourceType.YOUTUBE, item_id="y2", channel="c"),
        ]
        extra = ranker._grace_candidates(ranked, [r for r in ranked if r.score >= 0.6], [])
        assert extra == []  # already has an above-threshold item

    def test_no_grace_when_source_covered_by_pinned_item(self):
        # A source whose only above-threshold item is pinned (stripped from above_threshold) is
        # NOT empty — a pin covers it, so grace must not admit a below-threshold filler for it.
        ranker = _ranker(min_score=0.6, source_slot_score_grace=0.1, source_slots={"youtube": 1})
        ranked = [
            _ranked(0.85, SourceType.YOUTUBE, item_id="y_pin", channel="c"),
            _ranked(0.55, SourceType.YOUTUBE, item_id="y_weak", channel="c"),
        ]
        pinned = [r for r in ranked if r.item.item_id == "y_pin"]
        # above_threshold excludes the pinned item (as rank() does), so youtube looks empty here.
        extra = ranker._grace_candidates(ranked, [], pinned)
        assert extra == []  # covered by the pin → no weak filler, and the pin isn't re-admitted

    def test_below_grace_floor_not_admitted(self):
        ranker = _ranker(min_score=0.6, source_slot_score_grace=0.1, source_slots={"youtube": 1})
        ranked = [_ranked(0.40, SourceType.YOUTUBE, item_id="y1", channel="c")]  # 0.40 < floor 0.50
        assert ranker._grace_candidates(ranked, [], []) == []

    def test_grace_disabled_returns_nothing(self):
        ranker = _ranker(min_score=0.6, source_slot_score_grace=0.0, source_slots={"youtube": 1})
        ranked = [_ranked(0.55, SourceType.YOUTUBE, item_id="y1", channel="c")]
        assert ranker._grace_candidates(ranked, [], []) == []

    def test_a_score_tie_breaks_deterministically_on_item_id(self):
        # plain max() kept the FIRST equal-scoring candidate, i.e. the LLM's response order within a
        # batch — not stable run to run, unlike every other selection path here.
        ranker = _ranker(min_score=0.6, source_slot_score_grace=0.1, source_slots={"youtube": 1})
        tied = [
            _ranked(0.55, SourceType.YOUTUBE, item_id="y_b", channel="c"),
            _ranked(0.55, SourceType.YOUTUBE, item_id="y_a", channel="c"),
        ]
        assert [r.item.item_id for r in ranker._grace_candidates(tied, [], [])] == ["y_a"]
        assert [r.item.item_id for r in ranker._grace_candidates(list(reversed(tied)), [], [])] == ["y_a"]


class TestLastResortSourceCapRelaxation:
    """When a collector outage leaves every remaining candidate on one source, the source cap alone
    would ship a short digest. The relaxation runs LAST, only while the digest is short, and prefers
    candidates that still satisfy max_per_origin."""

    def test_not_entered_when_the_caps_already_fill_the_limit(self):
        ranker = _ranker(
            top_n=2,
            min_score=0.5,
            source_slots={"rss": 1, "web": 1},
            source_cap_multiplier=1,
            max_per_origin=1,
        )
        items = [
            _ranked(0.9, SourceType.RSS, item_id="r1", feed="f1"),
            _ranked(0.8, SourceType.WEB, item_id="w1", url="http://a.com/1"),
            _ranked(0.7, SourceType.RSS, item_id="r2", feed="f2"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert {r.item.item_id for r in selected} == {"r1", "w1"}

    def test_prefers_a_distinct_origin_over_a_repeat_one(self):
        ranker = _ranker(
            top_n=2,
            min_score=0.5,
            source_slots={"rss": 1},
            source_cap_multiplier=1,
            max_per_origin=1,
        )
        items = [
            _ranked(0.9, SourceType.RSS, item_id="r1", feed="f1"),
            _ranked(0.88, SourceType.RSS, item_id="same-origin", feed="f1"),
            _ranked(0.86, SourceType.RSS, item_id="other-origin", feed="f2"),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        # r1 takes the guaranteed slot (cap 1x1 exhausted); the relaxed pass then prefers the item
        # from a NEW feed over the higher-scoring one that repeats r1's origin.
        assert {r.item.item_id for r in selected} == {"r1", "other-origin"}

    def test_grace_items_are_never_used_as_filler(self):
        ranker = _ranker(
            top_n=3,
            min_score=0.6,
            source_slots={"rss": 1},
            source_cap_multiplier=1,
            max_per_origin=1,
        )
        items = [
            _ranked(0.9, SourceType.RSS, item_id="r1", feed="f1"),
            _ranked(0.55, SourceType.YOUTUBE, item_id="g1", channel="c1", grace=True),
        ]
        selected = ranker._apply_source_slots(items, ranker.config.top_n)
        assert {r.item.item_id for r in selected} == {"r1"}


class TestBackfillHonoursTheOriginCap:
    """The backfill candidates used to ignore max_per_origin outright, and _audit_shipped_diversity is
    detection-only by design — so a backfill item the editor used as an ADDITION rather than a
    replacement shipped a second story from an origin already at its cap. digest_2026-07-12 carried
    two r/LocalLLaMA GPU benchmarks against max_per_origin=1."""

    def _ranker(self):
        return _ranker(top_n=1, min_score=0.5, max_per_origin=1, source_slots={})

    def test_a_candidate_whose_origin_is_already_at_cap_is_skipped(self):
        ranker = self._ranker()
        core = [_ranked(0.95, SourceType.REDDIT, item_id="a", sub="LocalLLaMA")]
        candidates = [
            core[0],
            _ranked(0.94, SourceType.REDDIT, item_id="b", sub="LocalLLaMA"),
            _ranked(0.70, SourceType.REDDIT, item_id="c", sub="MachineLearning"),
        ]
        extras = ranker._backfill_candidates(candidates, core, room=2)
        # The merge-topup purpose is intact: the freed slot is filled by the next DISTINCT origin.
        assert [r.item.item_id for r in extras] == ["c"]
        assert all(r.backfill for r in extras)

    def test_the_cap_also_applies_among_the_backfill_items_themselves(self):
        ranker = self._ranker()
        core: list = []
        candidates = [
            _ranked(0.95, SourceType.REDDIT, item_id="a", sub="LocalLLaMA"),
            _ranked(0.94, SourceType.REDDIT, item_id="b", sub="LocalLLaMA"),
        ]
        extras = ranker._backfill_candidates(candidates, core, room=2)
        assert [r.item.item_id for r in extras] == ["a"]
