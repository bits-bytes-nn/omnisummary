import json

from application.ranker import ContentRanker
from domain.models import CollectedItem
from shared.constants import SourceType


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
                    {"item": 1, "score": 0.9, "reasoning": "Important"},
                    {"item": 2, "score": 0.5, "reasoning": "Moderate"},
                ]
            }
        )
        result = self._parse(raw)
        assert len(result) == 2
        assert result[0].score == 0.9
        assert result[0].item.item_id == "item_1"
        assert result[1].reasoning == "Moderate"

    def test_markdown_wrapped_json(self):
        inner = json.dumps({"rankings": [{"item": 1, "score": 0.8}]})
        raw = f"```json\n{inner}\n```"
        result = self._parse(raw)
        assert len(result) == 1
        assert result[0].score == 0.8

    def test_json_with_surrounding_text(self):
        inner = json.dumps({"rankings": [{"item": 1, "score": 0.7}]})
        raw = f"Here are the rankings:\n{inner}\nDone."
        result = self._parse(raw)
        assert len(result) == 1

    def test_an_out_of_range_number_is_skipped(self):
        raw = json.dumps(
            {
                "rankings": [
                    {"item": 1, "score": 0.9},
                    {"item": 99, "score": 0.8},
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
                    {"item": 1, "score": 0.9},
                    {"score": 0.8},
                ]
            }
        )
        result = self._parse(raw)
        assert len(result) == 1

    def test_omitted_items_are_silently_absent(self):
        # The root cause _rank_batch's coverage reconciliation exists for: a response that simply
        # omits items parses cleanly, so those candidates vanish unless the caller reconciles.
        raw = json.dumps({"rankings": [{"item": 1, "score": 0.9}]})
        result = self._parse(raw, _items(3))
        assert [r.item.item_id for r in result] == ["item_1"]

    def test_repeated_item_collapsed_to_the_first_entry(self):
        # A repeat used to make len(ranked) == len(items) even though item_3 was never scored, so
        # the coverage reconciliation saw 1.0 and the editor got the same story twice.
        raw = json.dumps(
            {
                "rankings": [
                    {"item": 1, "score": 0.9, "reasoning": "first"},
                    {"item": 2, "score": 0.8},
                    {"item": 1, "score": 0.4, "reasoning": "second"},
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
        raw = json.dumps({"rankings": [{"item": 1, "score": 1.5}]})

        result = self._parse(raw)
        assert len(result) == 0
