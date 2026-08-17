import threading
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from collectors.rsshub import RSSHubCollector
from shared.config import RSSHubAccount, RSSHubCollectorConfig
from shared.constants import SourceType
from shared.models import CollectedItem


def _item(item_id: str) -> CollectedItem:
    return CollectedItem(item_id=item_id, source_type=SourceType.X, title=item_id, url=f"https://x.com/{item_id}")


def _config(**kwargs) -> RSSHubCollectorConfig:
    base = {
        "base_url": "http://localhost:1200",
        "accounts": [RSSHubAccount(username="karpathy", platform="x")],
        # Retries still happen; they just don't sleep, so the suite stays hermetic and fast.
        "retry_backoff_sec": 0,
    }
    base.update(kwargs)
    cfg = RSSHubCollectorConfig(**base)
    cfg.reference_time = datetime(2026, 6, 3, tzinfo=UTC)
    cfg.lookback_hours = 24
    return cfg


class TestReachability:
    @pytest.mark.asyncio
    async def test_unreachable_service_raises(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.load_items_from_s3", return_value=None):
            with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
                with pytest.raises(RuntimeError, match="unreachable"):
                    await c.collect()

    @pytest.mark.asyncio
    async def test_5xx_raises(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config())
        resp = MagicMock(status_code=503)
        with patch("collectors.rsshub.load_items_from_s3", return_value=None):
            with patch("httpx.get", return_value=resp):
                with pytest.raises(RuntimeError, match="503"):
                    await c.collect()

    @pytest.mark.asyncio
    async def test_reachable_proceeds(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config())
        resp = MagicMock(status_code=200)
        with patch("collectors.rsshub.load_items_from_s3", return_value=None):
            with patch("httpx.get", return_value=resp):
                with patch.object(c, "_parse_feed", return_value=[]):
                    items = await c.collect()
        assert items == []  # reachable but no recent posts -> empty (not raised)

    @pytest.mark.asyncio
    async def test_hung_feed_times_out_and_is_skipped(self, monkeypatch):
        # A feed host that never returns must not block its worker forever: the per-feed timeout
        # skips THAT feed while the healthy account still delivers — collect() neither hangs nor
        # raises. A second account is what keeps this a partial failure (an all-hung run raises,
        # see test_all_accounts_failing_raises).
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(
            _config(
                accounts=[RSSHubAccount(username="hangs", platform="x"), RSSHubAccount(username="ok", platform="x")],
                request_timeout=1,
            )
        )
        resp = MagicMock(status_code=200)
        release = threading.Event()

        def _parse(feed_url, username, platform):
            if username == "hangs":
                release.wait(10)  # blocks past request_timeout=1; released after collect() returns
                return []
            return [_item("t1")]

        with patch("collectors.rsshub.load_items_from_s3", return_value=None):
            with patch("httpx.get", return_value=resp):
                with patch.object(c, "_parse_feed", side_effect=_parse):
                    with patch("collectors.rsshub.logger") as log:
                        items = await c.collect()
        release.set()  # let the parked worker thread exit instead of stalling teardown
        assert [i.item_id for i in items] == ["t1"]  # hung feed skipped, healthy one kept
        assert any("timed out" in str(c.args) for c in log.warning.call_args_list)  # a real timeout

    @pytest.mark.asyncio
    async def test_all_accounts_failing_raises(self, monkeypatch):
        # Reachable service but nothing parseable on ANY account is an outage, not a quiet day:
        # it must surface as FAILED instead of a silent empty result.
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(
            _config(accounts=[RSSHubAccount(username="a", platform="x"), RSSHubAccount(username="b", platform="x")])
        )
        resp = MagicMock(status_code=200)
        with patch("collectors.rsshub.load_items_from_s3", return_value=None):
            with patch("httpx.get", return_value=resp):
                with patch.object(c, "_parse_feed", side_effect=RuntimeError("bozo")):
                    with pytest.raises(RuntimeError, match="All 2 RSSHub feeds failed"):
                        await c.collect()

    @pytest.mark.asyncio
    async def test_partial_failure_keeps_succeeding_accounts(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(
            _config(accounts=[RSSHubAccount(username="bad", platform="x"), RSSHubAccount(username="ok", platform="x")])
        )
        resp = MagicMock(status_code=200)

        def _parse(feed_url, username, platform):
            if username == "bad":
                raise RuntimeError("bozo")
            return [_item("t1")]

        with patch("collectors.rsshub.load_items_from_s3", return_value=None):
            with patch("httpx.get", return_value=resp):
                with patch.object(c, "_parse_feed", side_effect=_parse):
                    items = await c.collect()
        assert [i.item_id for i in items] == ["t1"]


class TestFanOutBound:
    @pytest.mark.asyncio
    async def test_concurrency_is_bounded_by_config(self, monkeypatch):
        # 40+ accounts each park a worker thread; unbounded fan-out oversubscribed the executor
        # so a feed's timeout could expire before its parse even started.
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(
            _config(
                accounts=[RSSHubAccount(username=f"u{i}", platform="x") for i in range(6)],
                max_concurrency=2,
            )
        )
        resp = MagicMock(status_code=200)
        lock = threading.Lock()
        state = {"active": 0, "peak": 0}

        def _parse(feed_url, username, platform):
            with lock:
                state["active"] += 1
                state["peak"] = max(state["peak"], state["active"])
            time.sleep(0.05)
            with lock:
                state["active"] -= 1
            return [_item(username)]

        with patch("collectors.rsshub.load_items_from_s3", return_value=None):
            with patch("httpx.get", return_value=resp):
                with patch.object(c, "_parse_feed", side_effect=_parse):
                    items = await c.collect()
        assert len(items) == 6  # every account still collected
        assert state["peak"] <= 2
