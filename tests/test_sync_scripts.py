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
_SYNCS = {
    "rsshub": ("sync_rsshub_to_s3.py", "RSSHubCollector", "rsshub", "rsshub_items.json"),
    "youtube": ("sync_youtube_to_s3.py", "YouTubeCollector", "youtube", "youtube_items.json"),
}


class _StubCollector:
    """Stands in for the real collector: records the environment its collect() ran under, so the
    'never read back the park file you are about to write' guarantee can be asserted."""

    items: list = []
    raises: bool = False
    seen_state_bucket: str | None = "unset-marker"
    seen_proxy: str | None = "unset-marker"

    def __init__(self, config) -> None:
        self.config = config
        self.run_meta = {"accounts_total": 12, "accounts_failed": 10, "accounts_empty": 0}

    async def collect(self) -> list:
        type(self).seen_state_bucket = os.environ.get("STATE_BUCKET")
        type(self).seen_proxy = os.environ.get("CLOUDFLARE_PROXY_URL")
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
def sync(request, monkeypatch):
    """One loaded sync script per parametrization, with the process environment restored afterwards.

    The scripts pop STATE_BUCKET / the proxy vars at IMPORT time (that is the point: the collector
    they then run must not read the very file this run is about to overwrite), and they call
    load_dotenv() — so the import mutates the test process's environment."""
    filename, collector_name, source, park_name = _SYNCS[request.param]
    before = dict(os.environ)
    # Set the vars whose REMOVAL is the guarantee under test, so the assertion is not vacuous.
    os.environ["STATE_BUCKET"] = "state-bkt"
    os.environ["CLOUDFLARE_PROXY_URL"] = "https://proxy.example.com"
    module = _load(filename)
    uploads: list[dict] = []
    s3 = MagicMock()
    s3.put_object.side_effect = lambda **kwargs: uploads.append(kwargs)
    boto3 = MagicMock()
    boto3.Session.return_value.client.return_value = s3
    monkeypatch.setattr(module, "boto3", boto3)
    monkeypatch.setattr(module, collector_name, _StubCollector)
    monkeypatch.setattr(module.Config, "load", staticmethod(lambda: _config(source=source)))
    _StubCollector.items = []
    _StubCollector.raises = False
    _StubCollector.seen_state_bucket = "unset-marker"
    yield module, uploads, source, park_name, boto3
    os.environ.clear()
    os.environ.update(before)


def _item(source: str):
    from shared.constants import SourceType
    from shared.models import CollectedItem

    kind = SourceType.X if source == "rsshub" else SourceType.YOUTUBE
    return CollectedItem(item_id="i1", source_type=kind, title="t", url="https://e.com/1", text="body")


class TestTheSyncNeverReadsBackTheFileItIsAboutToWrite:
    """With STATE_BUCKET set (it is, in .env) the collector returns YESTERDAY's park file and the
    script re-uploads it — producing a file that is permanently FRESH and permanently frozen, which
    park_max_age_hours cannot detect by construction."""

    def test_state_bucket_is_unset_while_collecting(self, sync):
        module, _, source, _, _ = sync
        _StubCollector.items = [_item(source)]
        asyncio.run(module.main())
        assert _StubCollector.seen_state_bucket is None

    def test_the_proxy_is_unset_while_collecting(self, sync):
        # The whole point is to collect from THIS residential IP, not through a datacenter one.
        module, _, source, _, _ = sync
        _StubCollector.items = [_item(source)]
        asyncio.run(module.main())
        assert _StubCollector.seen_proxy is None


class TestWhatGetsParked:
    def test_the_envelope_carries_the_items_and_the_run_meta(self, sync):
        # A fresh, on-time park file says nothing about a sync that reached 2 of 12 inputs; the meta
        # block is what lets the Lambda-side reader report that as DEGRADED.
        module, uploads, source, park_name, _ = sync
        _StubCollector.items = [_item(source)]
        asyncio.run(module.main())
        assert len(uploads) == 1
        assert uploads[0]["Bucket"] == "state-bkt"
        assert uploads[0]["Key"] == f"omni/{park_name}"
        payload = json.loads(uploads[0]["Body"].decode("utf-8"))
        assert len(payload["items"]) == 1
        assert payload["meta"] == {"accounts_total": 12, "accounts_failed": 10, "accounts_empty": 0}
        assert payload["generated_at"]

    def test_a_quiet_day_still_stamps_the_sync_time(self, sync):
        # Returning early on zero items left YESTERDAY's file in place, so the digest re-ingested
        # stale items and reported OK. The empty envelope's stamp is the only proof the sync RAN.
        module, uploads, _, _, _ = sync
        _StubCollector.items = []
        asyncio.run(module.main())
        assert len(uploads) == 1
        payload = json.loads(uploads[0]["Body"].decode("utf-8"))
        assert payload["items"] == []
        assert payload["generated_at"]

    def test_a_failed_collection_uploads_nothing(self, sync):
        # A collector exception must propagate: overwriting the previous (good) park file with
        # nothing would turn one upstream outage into a day with no items at all.
        module, uploads, _, _, _ = sync
        _StubCollector.raises = True
        with pytest.raises(RuntimeError, match="upstream down"):
            asyncio.run(module.main())
        assert uploads == []

    def test_a_disabled_collector_uploads_nothing(self, sync, monkeypatch):
        module, uploads, _, _, _ = sync
        monkeypatch.setattr(module.Config, "load", staticmethod(lambda: _config(source="none")))
        asyncio.run(module.main())
        assert uploads == []

    def test_no_bucket_configured_writes_the_envelope_locally(self, sync, monkeypatch, tmp_path):
        module, uploads, source, park_name, _ = sync
        monkeypatch.setattr(module.Config, "load", staticmethod(lambda: _config(source=source, bucket="")))
        monkeypatch.chdir(tmp_path)
        _StubCollector.items = [_item(source)]
        asyncio.run(module.main())
        assert uploads == []
        payload = json.loads((tmp_path / "digest_state" / park_name).read_text(encoding="utf-8"))
        assert len(payload["items"]) == 1

    def test_explicit_sync_credentials_are_preferred_over_the_profile(self, sync, monkeypatch):
        module, _, source, _, boto3 = sync
        monkeypatch.setenv("S3_SYNC_ACCESS_KEY_ID", "AKIA")
        monkeypatch.setenv("S3_SYNC_SECRET_ACCESS_KEY", "secret")
        _StubCollector.items = [_item(source)]
        asyncio.run(module.main())
        assert boto3.Session.call_args.kwargs["aws_access_key_id"] == "AKIA"
        assert "profile_name" not in boto3.Session.call_args.kwargs
