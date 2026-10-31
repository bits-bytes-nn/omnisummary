from unittest.mock import MagicMock, patch

import pytest

import recall_cli
from domain.models import Trend, TrendEvidence, TrendMemory


def _snapshot() -> dict:
    return {
        "ranked_items": [],
        "digest_result": {
            "digest_text": "",
            "ranked_items": [],
            "content": {
                "lead": "리드.",
                "items": [{"title": "스토리", "url": "http://e.com/1", "body": "본문.", "implication": "함의."}],
            },
        },
    }


class TestDigest:
    def test_a_stored_day_returns_its_lead_and_stories(self):
        store = MagicMock()
        store.get_digest.return_value = _snapshot()
        with patch.object(recall_cli.bootstrap, "memory_store", return_value=store):
            result, code = recall_cli.run(["digest", "2026-10-11"])
        assert code == recall_cli.EXIT_OK and result["found"] is True
        assert result["stories"][0] == {
            "title": "스토리",
            "url": "http://e.com/1",
            "body": "본문.",
            "implication": "함의.",
        }
        store.get_digest.assert_called_once_with("2026-10-11")

    def test_an_unstored_day_succeeds_with_nothing_found(self):
        store = MagicMock()
        store.get_digest.return_value = None
        with patch.object(recall_cli.bootstrap, "memory_store", return_value=store):
            result, code = recall_cli.run(["digest", "2026-10-11"])
        assert code == recall_cli.EXIT_OK and result["found"] is False

    def test_an_unreadable_store_fails_instead_of_reading_as_empty(self):
        with patch.object(recall_cli.bootstrap, "memory_store", side_effect=RuntimeError("denied")):
            result, code = recall_cli.run(["digest", "2026-10-11"])
        assert code == recall_cli.EXIT_FAILED and "denied" in result["error"]

    def test_a_bad_date_is_rejected(self):
        result, code = recall_cli.run(["digest", "2026-13-01"])
        assert code == recall_cli.EXIT_REJECTED and result["status"] == "rejected"


class TestTrends:
    def test_trends_come_back_strongest_first(self):
        memory = TrendMemory(
            trends=[
                Trend(id="old", title="Old", evidence=[TrendEvidence(date="2026-01-01", summary="s")]),
                Trend(id="new", title="New", evidence=[TrendEvidence(date="2026-10-10", summary="s")]),
                Trend(id="none", title="No Evidence"),
            ]
        )
        store = MagicMock()
        store.read.return_value = memory.model_dump_json()
        with patch.object(recall_cli.bootstrap, "state_store", return_value=store):
            result, code = recall_cli.run(["trends"])
        assert code == recall_cli.EXIT_OK
        assert [t["title"] for t in result["trends"]] == ["New", "Old"]
        assert result["trends"][0]["latest"] == "s"


class TestBadArgumentsAreRejectedAsJson:
    # Plain argparse exited 2 with an empty stdout (usage on stderr), and -h printed help to stdout.
    @pytest.mark.parametrize("argv", [[], ["bogus"], ["trends", "extra"], ["-h"], ["digest", "2026-10-26", "x"]])
    def test_bad_arguments_come_back_rejected(self, argv, capsys):
        result, code = recall_cli.run(argv)
        assert (result["status"], code) == ("rejected", recall_cli.EXIT_REJECTED)
        assert result["error"]
        assert capsys.readouterr().out == ""
