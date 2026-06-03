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
