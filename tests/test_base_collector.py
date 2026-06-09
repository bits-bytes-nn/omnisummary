import json
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from collectors.base import gather_collector_results, load_items_from_s3
from shared.constants import SourceType
from shared.models import CollectedItem


def _item(item_id: str) -> CollectedItem:
    return CollectedItem(item_id=item_id, source_type=SourceType.WEB, title="t", url=f"http://e.com/{item_id}")


async def _ok(item_id: str) -> list[CollectedItem]:
    return [_item(item_id)]


async def _fail() -> list[CollectedItem]:
    raise RuntimeError("boom")


class TestGatherCollectorResults:
    @pytest.mark.asyncio
    async def test_partial_failure_passes_through(self):
        items = await gather_collector_results([_ok("a"), _fail()], raise_if_all_failed=True)
        assert {i.item_id for i in items} == {"a"}

    @pytest.mark.asyncio
    async def test_all_failed_raises_when_flagged(self):
        with pytest.raises(RuntimeError, match="All 2 collector tasks failed"):
            await gather_collector_results([_fail(), _fail()], raise_if_all_failed=True)

    @pytest.mark.asyncio
    async def test_all_failed_silent_when_not_flagged(self):
        items = await gather_collector_results([_fail(), _fail()])
        assert items == []

    @pytest.mark.asyncio
    async def test_no_tasks_does_not_raise(self):
        items = await gather_collector_results([], raise_if_all_failed=True)
        assert items == []


class TestLoadItemsFromS3:
    def test_returns_none_without_bucket(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        assert load_items_from_s3("youtube_items.json") is None

    def test_reads_items_from_parent_prefix(self, monkeypatch):
        # S3_PREFIX is '<root>/digest_state'; parked items live one level up at '<root>/'.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        body = MagicMock()
        body.read.return_value = json.dumps(
            [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "https://y/v1", "text": "x"}]
        ).encode("utf-8")
        client = MagicMock()
        client.get_object.return_value = {"Body": body}
        with patch("collectors.base.boto3.client", return_value=client):
            items = load_items_from_s3("youtube_items.json")
        assert items is not None
        assert [i.item_id for i in items] == ["v1"]
        assert client.get_object.call_args.kwargs["Key"] == "omnisummary/youtube_items.json"

    def test_missing_object_returns_none(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        client = MagicMock()
        client.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        with patch("collectors.base.boto3.client", return_value=client):
            assert load_items_from_s3("youtube_items.json") is None
