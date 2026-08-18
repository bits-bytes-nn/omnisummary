import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from collectors.base import (
    ParkOutcome,
    dump_items_envelope,
    gather_collector_results,
    load_items_from_s3,
    park_file_key,
    park_root_prefix,
)
from shared.constants import SourceType
from shared.models import CollectedItem


def _s3_client_returning(body_bytes: bytes) -> MagicMock:
    body = MagicMock()
    body.read.return_value = body_bytes
    client = MagicMock()
    client.get_object.return_value = {"Body": body}
    return client


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
    def test_absent_without_bucket(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.ABSENT
        assert parked.usable is False and parked.degraded is False

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
            parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.FRESH
        assert [i.item_id for i in parked.items] == ["v1"]
        assert client.get_object.call_args.kwargs["Key"] == "omnisummary/youtube_items.json"

    def test_reads_root_level_key_when_prefix_is_bare(self, monkeypatch):
        # With no configured root prefix the CDK sets S3_PREFIX='digest_state', and the sync
        # scripts write the park file at the bucket root ('<file>'). The reader used to look under
        # 'digest_state/<file>' and never found it, silently falling back to live collection.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "digest_state")
        client = _s3_client_returning(b"[]")
        with patch("collectors.base.boto3.client", return_value=client):
            load_items_from_s3("youtube_items.json")
        assert client.get_object.call_args.kwargs["Key"] == "youtube_items.json"

    def test_park_key_matches_between_writer_and_reader(self):
        # The sync scripts key off the config's aws.s3_prefix (bucket ROOT); the Lambda reader
        # derives that root from S3_PREFIX ('<root>/digest_state'). Both must land on one key.
        for root in ("omnisummary", ""):
            state_prefix = f"{root}/digest_state" if root else "digest_state"
            assert park_file_key("rsshub_items.json", root) == park_file_key(
                "rsshub_items.json", park_root_prefix(state_prefix)
            )
        assert park_file_key("rsshub_items.json", "omnisummary") == "omnisummary/rsshub_items.json"
        assert park_file_key("rsshub_items.json", "") == "rsshub_items.json"

    def test_missing_object_is_absent(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        client = MagicMock()
        client.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        with patch("collectors.base.boto3.client", return_value=client):
            parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.ABSENT
        assert parked.degraded is False  # a missing park file is routine, not a stale sync

    def test_unexpected_client_error_is_error_not_absent(self, monkeypatch):
        # An AccessDenied read used to be logged at info as "no items found" and looked identical
        # to an absent file. It must still fall through to live collection (never raise) but be
        # reported as a degraded park so the misconfiguration surfaces.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        client = MagicMock()
        client.get_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        with patch("collectors.base.boto3.client", return_value=client):
            parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.ERROR
        assert parked.usable is False and parked.degraded is True
        assert "AccessDenied" in parked.detail

    def test_reads_envelope_shape(self, monkeypatch):
        # The newer {"generated_at", "items"} envelope must load like the legacy bare list.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        fresh = datetime.now(UTC).isoformat()
        body = json.dumps(
            {
                "generated_at": fresh,
                "items": [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "https://y/v1"}],
            }
        ).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            parked = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in parked.items] == ["v1"]
        assert parked.outcome == ParkOutcome.FRESH

    def test_stale_envelope_still_loads_but_warns(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        old = (datetime.now(UTC) - timedelta(hours=72)).isoformat()
        body = json.dumps(
            {"generated_at": old, "items": [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "u"}]}
        ).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            with patch("collectors.base.logger.warning") as warn:
                parked = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in parked.items] == ["v1"]  # stale beats empty
        assert parked.outcome == ParkOutcome.STALE
        assert parked.usable is True and parked.degraded is True  # used, but reported STALE
        assert parked.age_hours is not None and parked.age_hours > 36
        assert any("stalled" in str(c.args) for c in warn.call_args_list)

    def test_park_age_budget_is_configurable(self, monkeypatch):
        # A source whose sync runs less often can widen the window instead of alerting daily.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        old = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        body = json.dumps(
            {"generated_at": old, "items": [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "u"}]}
        ).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            parked = load_items_from_s3("youtube_items.json", max_age_hours=72)
        assert parked.outcome == ParkOutcome.FRESH  # 48h is inside a 72h budget

    def test_stale_empty_envelope_is_treated_as_absent(self, monkeypatch):
        # A park file that is BOTH empty and stale means the local sync stopped producing; falling
        # through to live collection lets a real outage report FAILED instead of silent EMPTY.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        old = (datetime.now(UTC) - timedelta(hours=72)).isoformat()
        body = json.dumps({"generated_at": old, "items": []}).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            parked = load_items_from_s3("rsshub_items.json")
        assert parked.outcome == ParkOutcome.ABSENT
        assert parked.usable is False  # -> live collection, so a real outage can report FAILED

    def test_fresh_empty_envelope_is_returned_not_absent(self, monkeypatch):
        # A legitimately quiet sync day must NOT fall through to live collection (which would
        # raise a false FAILED from a Lambda IP that the source blocks).
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        body = json.dumps({"generated_at": datetime.now(UTC).isoformat(), "items": []}).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            parked = load_items_from_s3("rsshub_items.json")
        assert parked.outcome == ParkOutcome.FRESH
        assert parked.usable is True and parked.items == []
        assert parked.degraded is False  # a quiet sync day is not a stale sync

    def test_unstamped_empty_list_is_returned_not_absent(self, monkeypatch):
        # Legacy bare list carries no age, so it can't be proven stale → keep prior behavior.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(b"[]")):
            parked = load_items_from_s3("rsshub_items.json")
        assert parked.outcome == ParkOutcome.FRESH
        assert parked.items == [] and parked.age_hours is None

    def test_malformed_json_is_error(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(b"{not json")):
            parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.ERROR
        assert parked.usable is False and parked.degraded is True


class TestDumpItemsEnvelope:
    def test_roundtrips_through_loader(self, monkeypatch):
        items = [_item("v1"), _item("v2")]
        payload = dump_items_envelope(items).encode("utf-8")
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(payload)):
            loaded = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in loaded.items] == ["v1", "v2"]
        assert loaded.outcome == ParkOutcome.FRESH

    def test_meta_block_roundtrips_and_is_optional(self, monkeypatch):
        # The writer/reader contract for the OPTIONAL meta block: what a sync recorded about how it
        # went must survive the round trip, and a payload without it must still load.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        with_meta = dump_items_envelope([_item("v1")], meta={"accounts_total": 40, "accounts_failed": 30}).encode()
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(with_meta)):
            loaded = load_items_from_s3("rsshub_items.json")
        assert loaded.meta == {"accounts_total": 40, "accounts_failed": 30}

        without_meta = dump_items_envelope([_item("v1")]).encode()
        assert "meta" not in json.loads(without_meta)
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(without_meta)):
            assert load_items_from_s3("rsshub_items.json").meta == {}

    def test_legacy_shapes_still_load_with_an_empty_meta(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        legacy_list = json.dumps(
            [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "https://y/v1"}]
        ).encode()
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(legacy_list)):
            parked = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in parked.items] == ["v1"]
        assert parked.meta == {}

        stamped_no_meta = json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "items": [{"item_id": "v2", "source_type": "youtube", "title": "T", "url": "https://y/v2"}],
            }
        ).encode()
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(stamped_no_meta)):
            parked = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in parked.items] == ["v2"]
        assert parked.meta == {}
