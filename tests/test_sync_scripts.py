import asyncio
import importlib.util
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"

# The two syncs are the ONLY supply of X posts and of transcript-carrying YouTube items: the digest
# Lambda reads their park files because a datacenter IP cannot fetch either. Both encode fixes for
# documented stale-data incidents in code nothing exercised.
_SYNCS = {"rsshub": "rsshub_items.json", "youtube": "youtube_items.json"}


class _StubCollector:
    """Stands in for the real collector: records the environment its collect() ran under, so the
    'never read back the park file you are about to write' guarantee can be asserted."""

    items: list = []
    raises: bool = False
    seen_state_bucket: str | None = "unset-marker"
    degraded_detail: str = ""

    def __init__(self, config) -> None:
        self.config = config
        self.run_meta = {"accounts_total": 12, "accounts_failed": 0, "accounts_empty": 0}

    async def collect(self) -> list:
        type(self).seen_state_bucket = os.environ.get("STATE_BUCKET")
        if type(self).raises:
            raise RuntimeError("upstream down")
        return type(self).items


def _load(filename: str):
    spec = importlib.util.spec_from_file_location(filename.removesuffix(".py"), _SCRIPTS / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _config(*, source: str, bucket: str = "state-bkt", prefix: str = "omni") -> MagicMock:
    config = MagicMock()
    for name in ("rsshub", "youtube"):
        getattr(config.collectors, name).enabled = name == source
    config.aws.state_bucket_name = bucket
    config.aws.s3_prefix = prefix
    config.aws.region = "ap-northeast-2"
    config.aws.profile = ""
    return config


@pytest.fixture(params=sorted(_SYNCS), ids=sorted(_SYNCS))
def sync(request, monkeypatch, tmp_path):
    """The sync script loaded once per source, with the process environment restored afterwards.

    The scripts pop STATE_BUCKET at IMPORT time (that is the point: the collector
    they then run must not read the very file this run is about to overwrite), and they call
    load_dotenv() — so the import mutates the test process's environment."""
    source = request.param
    park_name = _SYNCS[source]
    before = dict(os.environ)
    # Set the vars whose REMOVAL is the guarantee under test, so the assertion is not vacuous.
    os.environ["STATE_BUCKET"] = "state-bkt"
    # The script loads .env at import; the real one carries live sync keys.
    monkeypatch.setattr("dotenv.load_dotenv", lambda *args, **kwargs: False)
    module = _load("sync_to_s3.py")
    uploads: list[dict] = []
    s3 = MagicMock()
    s3.put_object.side_effect = lambda **kwargs: uploads.append(kwargs)
    # The sync's own stamp file, never the repo's digest_state/.
    monkeypatch.setattr(module, "_sync_stamp_path", lambda name: tmp_path / f"last_park_sync_{name}.txt")
    boto3 = MagicMock()
    boto3.Session.return_value.client.return_value = s3
    monkeypatch.setattr(module, "boto3", boto3)
    monkeypatch.setitem(module._SOURCES, source, (_StubCollector, park_name))
    monkeypatch.setattr(module.Config, "load", staticmethod(lambda: _config(source=source)))
    _StubCollector.items = []
    _StubCollector.raises = False
    _StubCollector.seen_state_bucket = "unset-marker"
    yield module, uploads, source, park_name, boto3
    os.environ.clear()
    os.environ.update(before)


def _item(source: str):
    from domain.models import CollectedItem
    from shared.constants import SourceType

    kind = SourceType.X if source == "rsshub" else SourceType.YOUTUBE
    return CollectedItem(item_id="i1", source_type=kind, title="t", url="https://e.com/1", text="body")


class TestTheSyncNeverReadsBackTheFileItIsAboutToWrite:
    """With STATE_BUCKET set (it is, in .env) the collector returns YESTERDAY's park file and the
    script re-uploads it — producing a file that is permanently FRESH and permanently frozen, which
    park_max_age_hours cannot detect by construction."""

    def test_state_bucket_is_unset_while_collecting(self, sync):
        module, _, source, _, _ = sync
        _StubCollector.items = [_item(source)]
        asyncio.run(module.main(source))
        assert _StubCollector.seen_state_bucket is None


class TestWhatGetsParked:
    def test_the_envelope_carries_the_items_and_the_run_meta(self, sync):
        # A fresh, on-time park file says nothing about a sync that reached 2 of 12 inputs; the meta
        # block is what lets the Lambda-side reader report that as DEGRADED.
        module, uploads, source, park_name, _ = sync
        _StubCollector.items = [_item(source)]
        asyncio.run(module.main(source))
        assert len(uploads) == 1
        assert uploads[0]["Bucket"] == "state-bkt"
        assert uploads[0]["Key"] == f"omni/{park_name}"
        payload = json.loads(uploads[0]["Body"].decode("utf-8"))
        assert len(payload["items"]) == 1
        assert payload["meta"] == {"accounts_total": 12, "accounts_failed": 0, "accounts_empty": 0}
        assert payload["generated_at"]

    def test_a_quiet_day_still_stamps_the_sync_time(self, sync):
        # Returning early on zero items left YESTERDAY's file in place, so the digest re-ingested
        # stale items and reported OK. The empty envelope's stamp is the only proof the sync RAN.
        module, uploads, source, _, _ = sync
        _StubCollector.items = []
        asyncio.run(module.main(source))
        assert len(uploads) == 1
        payload = json.loads(uploads[0]["Body"].decode("utf-8"))
        assert payload["items"] == []
        assert payload["generated_at"]

    def test_a_failed_collection_uploads_nothing(self, sync):
        # A collector exception must propagate: overwriting the previous (good) park file with
        # nothing would turn one upstream outage into a day with no items at all.
        module, uploads, source, _, _ = sync
        _StubCollector.raises = True
        with pytest.raises(RuntimeError, match="upstream down"):
            asyncio.run(module.main(source))
        assert uploads == []

    def test_a_disabled_collector_uploads_nothing(self, sync, monkeypatch):
        module, uploads, source, _, _ = sync
        monkeypatch.setattr(module.Config, "load", staticmethod(lambda: _config(source="none")))
        asyncio.run(module.main(source))
        assert uploads == []

    def test_no_bucket_configured_fails_instead_of_reporting_a_sync(self, sync, monkeypatch):
        # A local file nothing reads let the sync exit 0 while the digest got no park file.
        module, uploads, source, _, _ = sync
        monkeypatch.setattr(module.Config, "load", staticmethod(lambda: _config(source=source, bucket="")))
        _StubCollector.items = [_item(source)]
        with pytest.raises(SystemExit):
            asyncio.run(module.main(source))
        assert uploads == []

    def test_explicit_sync_credentials_are_preferred_over_the_profile(self, sync, monkeypatch):
        module, _, source, _, boto3 = sync
        monkeypatch.setenv("S3_SYNC_ACCESS_KEY_ID", "AKIA")
        monkeypatch.setenv("S3_SYNC_SECRET_ACCESS_KEY", "secret")
        _StubCollector.items = [_item(source)]
        asyncio.run(module.main(source))
        assert boto3.Session.call_args.kwargs["aws_access_key_id"] == "AKIA"
        assert "profile_name" not in boto3.Session.call_args.kwargs


class TestTheParkCoversASkippedDay:
    def test_the_sync_collects_lookback_plus_one_digest_period(self, sync, monkeypatch):
        # With a plain lookback window, a skipped sync's day was in no park file: the Lambda read the
        # stale file (up to park_max_age_hours late) and the next sync no longer reached back to it.
        from datetime import UTC, datetime, timedelta

        from adapters.collectors.base import in_collection_window
        from shared.config import RSSHubCollectorConfig, YouTubeCollectorConfig

        module, _, source, _, _ = sync
        config_cls = RSSHubCollectorConfig if source == "rsshub" else YouTubeCollectorConfig
        source_config = config_cls(lookback_hours=30, park_max_age_hours=24)
        config = _config(source=source)
        setattr(config.collectors, source, source_config)
        monkeypatch.setattr(module.Config, "load", staticmethod(lambda: config))
        seen: list = []
        monkeypatch.setattr(_StubCollector, "collect", _record_config(seen))
        asyncio.run(module.main(source))

        assert seen[0].lookback_hours == 54
        assert source_config.lookback_hours == 30
        now = datetime.now(UTC)
        last_good_sync = now - timedelta(hours=48)
        published_after_it = last_good_sync + timedelta(hours=1)
        assert in_collection_window(published_after_it, lookback_hours=seen[0].lookback_hours, reference_time=now)


class TestTheReaderReachesBackToThePreviousSync:
    def test_the_next_park_reaches_back_to_the_sync_the_last_digest_read(self, sync, monkeypatch):
        # A sync that ran late (after the 10:00 UTC digest) parked a file no digest read before the next
        # sync replaced it: anchoring on it lost the day before it. Anchor on the last CONSUMED sync.
        from datetime import UTC, datetime

        module, _, source, _, _ = sync
        config = MagicMock()
        config.aws.digest_cron_hour, config.aws.digest_cron_minute = "10", "0"
        syncs = [
            ("2026-10-16T09:30:00+00:00", "2026-10-16T09:35:00+00:00"),
            ("2026-10-17T09:30:00+00:00", "2026-10-17T09:35:00+00:00"),
            ("2026-10-17T12:00:00+00:00", "2026-10-17T12:05:00+00:00"),  # late: after the 10:00 digest
        ]
        for start, uploaded in syncs:
            module._record_healthy_sync(source, datetime.fromisoformat(start), datetime.fromisoformat(uploaded))
        now = datetime(2026, 10, 18, 9, 31, tzinfo=UTC)
        assert module._previous_sync_at(source, config, now) == "2026-10-17T09:30:00+00:00"
        assert module._previous_sync_at(source, MagicMock(), now) == "2026-10-17T12:00:00+00:00"  # no schedule

    def test_a_sync_that_uploads_after_the_digest_was_not_read_by_it(self, sync):
        # Started 09:58, uploaded 10:03: the 10:00 digest read the file before it.
        from datetime import UTC, datetime

        module, _, source, _, _ = sync
        config = MagicMock()
        config.aws.digest_cron_hour, config.aws.digest_cron_minute = "10", "0"
        for start, uploaded in (
            ("2026-10-16T09:30:00", "2026-10-16T09:35:00"),
            ("2026-10-17T09:58:00", "2026-10-17T10:03:00"),
        ):
            module._record_healthy_sync(
                source,
                datetime.fromisoformat(start).replace(tzinfo=UTC),
                datetime.fromisoformat(uploaded).replace(tzinfo=UTC),
            )
        now = datetime(2026, 10, 18, 9, 31, tzinfo=UTC)
        assert module._previous_sync_at(source, config, now) == "2026-10-16T09:30:00+00:00"

    def test_the_one_time_format_still_reads(self, sync):
        module, _, source, _, _ = sync
        module._sync_stamp_path(source).write_text("2026-10-17T09:30:00+00:00\n")
        start, uploaded = module._healthy_syncs(source)[0]
        assert start == uploaded and start.isoformat() == "2026-10-17T09:30:00+00:00"

    def test_each_park_records_its_healthy_start(self, sync):
        module, uploads, source, _, _ = sync
        asyncio.run(module.main(source))
        stamps = module._healthy_syncs(source)
        assert len(stamps) == 1
        assert stamps[0][0].isoformat() <= json.loads(uploads[-1]["Body"])["generated_at"]  # the START

    def test_a_degraded_or_failed_upload_is_not_a_healthy_sync(self, sync, monkeypatch):
        module, uploads, source, _, boto3 = sync
        asyncio.run(module.main(source))
        healthy = module._healthy_syncs(source)
        monkeypatch.setattr(_StubCollector, "degraded_detail", "10 of 12 accounts failed")
        asyncio.run(module.main(source))
        assert module._healthy_syncs(source) == healthy
        monkeypatch.setattr(_StubCollector, "degraded_detail", "")
        healthy_init = _StubCollector.__init__
        monkeypatch.setattr(_StubCollector, "__init__", _with_failed_inputs)
        asyncio.run(module.main(source))  # 1 of 12 failed, under the threshold: still a healthy sync
        assert len(module._healthy_syncs(source)) == 2
        monkeypatch.setattr(_StubCollector, "__init__", healthy_init)
        boto3.Session.return_value.client.return_value.put_object.side_effect = RuntimeError("s3 down")
        with pytest.raises(RuntimeError):
            asyncio.run(module.main(source))
        assert len(module._healthy_syncs(source)) == 2

    def test_a_skipped_sync_widens_the_read_back_to_the_previous_one(self):
        from datetime import UTC, datetime, timedelta

        from adapters.collectors.base import ParkedItems, ParkOutcome, park_read_lookback_hours

        now = datetime(2026, 10, 28, 15, 0, tzinfo=UTC)

        def lookback(previous_sync):
            meta = {"previous_sync_at": previous_sync.isoformat()} if previous_sync else {}
            parked = ParkedItems(outcome=ParkOutcome.FRESH, meta=meta)
            return park_read_lookback_hours(parked, lookback_hours=30, reference_time=now)

        assert lookback(None) == 30
        assert lookback(now - timedelta(hours=29.5)) == 30  # an ordinary day: yesterday's sync
        assert lookback(now - timedelta(hours=53.5)) == 54  # one skipped day: back to the sync before it
        assert lookback(now - timedelta(days=10)) == 54  # never wider than the sync parks


def _with_failed_inputs(self, config) -> None:
    self.config = config
    self.run_meta = {"accounts_total": 12, "accounts_failed": 1, "accounts_empty": 0}


def _record_config(seen: list):
    async def collect(self) -> list:
        seen.append(self.config)
        return []

    return collect
