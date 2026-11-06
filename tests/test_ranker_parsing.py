import pytest
from pydantic import ValidationError

from application.ranker import ContentRanker
from domain.models import CollectedItem, RankingEntry, Rankings
from shared.constants import SourceType


def _items(n: int = 3) -> list[CollectedItem]:
    return [
        CollectedItem(item_id=f"item_{i}", source_type=SourceType.REDDIT, title=f"Title {i}", url=f"http://{i}.com")
        for i in range(1, n + 1)
    ]


def _convert(entries: list[dict], items: list[CollectedItem] | None = None):
    return ContentRanker._rankings_to_items([RankingEntry(**e) for e in entries], items or _items())


class TestRankingsToItems:
    def test_entries_map_to_their_numbered_items(self):
        result = _convert(
            [{"item": 1, "score": 0.9, "reasoning": "Important"}, {"item": 2, "score": 0.5, "reasoning": "Moderate"}]
        )
        assert [r.item.item_id for r in result] == ["item_1", "item_2"]
        assert result[0].score == 0.9
        assert result[1].reasoning == "Moderate"

    def test_omitted_items_are_silently_absent(self):
        # The root cause _rank_batch's coverage reconciliation exists for: a response that simply
        # omits items is valid, so those candidates vanish unless the caller reconciles.
        assert [r.item.item_id for r in _convert([{"item": 1, "score": 0.9}])] == ["item_1"]

    def test_repeated_item_collapsed_to_the_first_entry(self):
        # A repeat used to make len(ranked) == len(items) even though item_3 was never scored, so
        # the coverage reconciliation saw 1.0 and the editor got the same story twice.
        result = _convert(
            [
                {"item": 1, "score": 0.9, "reasoning": "first"},
                {"item": 2, "score": 0.8},
                {"item": 1, "score": 0.4, "reasoning": "second"},
            ]
        )
        assert [r.item.item_id for r in result] == ["item_1", "item_2"]
        assert result[0].reasoning == "first"

    def test_a_zero_based_answer_is_discarded_whole(self):
        # Zero-based shifts every score onto the item before; the re-ask scores the batch again.
        assert _convert([{"item": 0, "score": 0.9}, {"item": 1, "score": 0.5}, {"item": 2, "score": 0.1}]) == []

    @pytest.mark.parametrize("bad", [-1, 4, 99])
    def test_an_out_of_range_number_is_skipped_and_the_rest_kept(self, bad):
        assert [r.item.item_id for r in _convert([{"item": 1, "score": 0.9}, {"item": bad, "score": 0.5}])] == [
            "item_1"
        ]

    def test_out_of_range_score_dropped(self):
        # The schema cannot bound a number, so RankedItem's 0..1 check drops the entry (not clamped).
        assert _convert([{"item": 1, "score": 1.5}, {"item": 2, "score": 0.7}])[0].item.item_id == "item_2"

    @pytest.mark.parametrize("bad", [True, 2.7, "2"])
    def test_the_schema_takes_only_an_integer_item_number(self, bad):
        with pytest.raises(ValidationError):
            Rankings.model_validate({"rankings": [{"item": bad, "score": 0.5}]})
