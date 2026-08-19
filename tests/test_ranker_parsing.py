import json

from pipeline.ranker import ContentRanker
from shared.constants import SourceType
from shared.models import CollectedItem


def _items(n: int = 3) -> list[CollectedItem]:
    return [
        CollectedItem(item_id=f"item_{i}", source_type=SourceType.REDDIT, title=f"Title {i}", url=f"http://{i}.com")
        for i in range(1, n + 1)
    ]


class TestParseRankings:
    def _parse(self, raw: str, items: list[CollectedItem] | None = None):
        items = items or _items()

        ranker = ContentRanker.__new__(ContentRanker)
        return ranker._parse_rankings(raw, items)

    def test_valid_json(self):
        raw = json.dumps(
            {
                "rankings": [
                    {"item_id": "item_1", "score": 0.9, "reasoning": "Important", "categories": ["AI"]},
                    {"item_id": "item_2", "score": 0.5, "reasoning": "Moderate"},
                ]
            }
        )
        result = self._parse(raw)
        assert len(result) == 2
        assert result[0].score == 0.9
        assert result[0].item.item_id == "item_1"
        assert result[1].reasoning == "Moderate"

    def test_markdown_wrapped_json(self):
        inner = json.dumps({"rankings": [{"item_id": "item_1", "score": 0.8}]})
        raw = f"```json\n{inner}\n```"
        result = self._parse(raw)
        assert len(result) == 1
        assert result[0].score == 0.8

    def test_json_with_surrounding_text(self):
        inner = json.dumps({"rankings": [{"item_id": "item_1", "score": 0.7}]})
        raw = f"Here are the rankings:\n{inner}\nDone."
        result = self._parse(raw)
        assert len(result) == 1

    def test_unknown_item_id_skipped(self):
        raw = json.dumps(
            {
                "rankings": [
                    {"item_id": "item_1", "score": 0.9},
                    {"item_id": "nonexistent", "score": 0.8},
                ]
            }
        )
        result = self._parse(raw)
        assert len(result) == 1
        assert result[0].item.item_id == "item_1"

    def test_malformed_entry_skipped(self):
        raw = json.dumps(
            {
                "rankings": [
                    {"item_id": "item_1", "score": 0.9},
                    {"score": 0.8},
                ]
            }
        )
        result = self._parse(raw)
        assert len(result) == 1

    def test_omitted_item_ids_are_silently_absent(self):
        # The root cause _rank_batch's coverage reconciliation exists for: a response that simply
        # omits ids parses cleanly, so those candidates vanish unless the caller reconciles.
        raw = json.dumps({"rankings": [{"item_id": "item_1", "score": 0.9}]})
        result = self._parse(raw, _items(3))
        assert [r.item.item_id for r in result] == ["item_1"]

    def test_repeated_item_id_collapsed_to_the_first_entry(self):
        # A repeat used to make len(ranked) == len(items) even though item_3 was never scored, so
        # the coverage reconciliation saw 1.0 and the editor got the same story twice.
        raw = json.dumps(
            {
                "rankings": [
                    {"item_id": "item_1", "score": 0.9, "reasoning": "first"},
                    {"item_id": "item_2", "score": 0.8},
                    {"item_id": "item_1", "score": 0.4, "reasoning": "second"},
                ]
            }
        )
        result = self._parse(raw, _items(3))
        assert [r.item.item_id for r in result] == ["item_1", "item_2"]
        assert result[0].score == 0.9
        assert result[0].reasoning == "first"

    def test_invalid_json_returns_empty(self):
        result = self._parse("not valid json at all")
        assert result == []

    def test_out_of_range_score_dropped(self):
        # score > 1.0 violates RankedItem's le=1.0 → Pydantic raises → entry dropped
        # (NOT clamped to 1.0). Documents the intended drop-vs-clamp behavior.
        raw = json.dumps({"rankings": [{"item_id": "item_1", "score": 1.5}]})

        result = self._parse(raw)
        assert len(result) == 0

    def test_leading_json_token_not_stripped_into_content(self):
        # removeprefix('json') must not corrupt a value — guard against the old
        # lstrip('json') char-set bug that would eat leading j/s/o/n characters.
        inner = json.dumps({"rankings": [{"item_id": "item_1", "score": 0.7, "reasoning": "sonnet json notes"}]})
        raw = f"```json\n{inner}\n```"
        result = self._parse(raw)
        assert len(result) == 1
        assert result[0].reasoning == "sonnet json notes"


class TestEngagementGuidance:
    """`view_count` is set by the YouTube collector alone, so on a batch with no such item the
    *Engagement Signal* block described a bonus nothing in the batch could receive — and for the one
    medium that does carry it, that bonus stacked on the medium-neutrality paragraph and the
    source-slot score grace."""

    def _ranker(self) -> ContentRanker:
        from shared.config import PipelineConfig

        ranker = ContentRanker.__new__(ContentRanker)
        ranker.config = PipelineConfig()
        return ranker

    def test_no_block_when_the_batch_carries_no_engagement_data(self):
        assert self._ranker()._engagement_guidance(_items()) == ""

    def test_the_block_and_the_configured_tiers_appear_when_it_does(self):
        video = CollectedItem(
            item_id="v",
            source_type=SourceType.YOUTUBE,
            title="t",
            url="https://y/v",
            metadata={"view_count": 20000},
        )
        block = self._ranker()._engagement_guidance([*_items(), video])
        assert "*Engagement Signal*" in block
        assert "10,000+ views" in block


class TestItemIdResolution:
    """The `=== Item N ===` header the prompt builder emits invites the model to answer with the
    DISPLAY ORDINAL, and production logs show it doing exactly that ('30', '27', '18', '9', '5'),
    plus a truncated 14-char id and full URLs. Each was discarded with a warning — and since the
    coverage re-ask only fires below ranking_min_coverage_ratio, one run logged 'Unknown item_id 30'
    then '38/40 scored, coverage 0.95' and never re-asked. Those candidates left the pool silently."""

    def _parse(self, raw_id: str, items: list[CollectedItem] | None = None):
        items = items or _items()
        ranker = ContentRanker.__new__(ContentRanker)
        return ranker._parse_rankings(json.dumps({"rankings": [{"item_id": raw_id, "score": 0.8}]}), items)

    def test_a_display_ordinal_resolves_to_that_position(self):
        result = self._parse("2")
        assert [r.item.item_id for r in result] == ["item_2"]

    def test_an_out_of_range_ordinal_is_still_discarded(self):
        assert self._parse("99") == []

    def test_a_real_id_is_never_reinterpreted_as_an_ordinal(self):
        # Exact match runs first, so a numeric item_id keeps its own identity.
        numeric = [
            CollectedItem(item_id="3", source_type=SourceType.RSS, title="numeric", url="http://n/1"),
            CollectedItem(item_id="item_b", source_type=SourceType.RSS, title="b", url="http://n/2"),
        ]
        assert [r.item.item_id for r in self._parse("3", numeric)] == ["3"]

    def test_a_truncated_id_resolves_by_unique_prefix(self):
        assert [r.item.item_id for r in self._parse("item_3")] == ["item_3"]
        long_ids = [
            CollectedItem(item_id="abcdef0123456789", source_type=SourceType.RSS, title="a", url="http://n/1"),
            CollectedItem(item_id="zzzz", source_type=SourceType.RSS, title="z", url="http://n/2"),
        ]
        assert [r.item.item_id for r in self._parse("abcdef01234567", long_ids)] == ["abcdef0123456789"]

    def test_an_ambiguous_prefix_is_discarded(self):
        ambiguous = [
            CollectedItem(item_id="abc1", source_type=SourceType.RSS, title="a", url="http://n/1"),
            CollectedItem(item_id="abc2", source_type=SourceType.RSS, title="b", url="http://n/2"),
        ]
        assert self._parse("abc", ambiguous) == []

    def test_a_full_url_resolves_to_its_item(self):
        assert [r.item.item_id for r in self._parse("http://2.com")] == ["item_2"]

    def test_a_url_differing_only_by_scheme_or_slash_still_resolves(self):
        assert [r.item.item_id for r in self._parse("https://2.com/")] == ["item_2"]

    def test_nothing_matching_is_still_discarded(self):
        assert self._parse("totally-made-up") == []
