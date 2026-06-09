from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from collectors.rsshub import RSSHubCollector
from shared.config import RSSHubAccount, RSSHubCollectorConfig


def _config(**kwargs) -> RSSHubCollectorConfig:
    base = {
        "base_url": "http://localhost:1200",
        "accounts": [RSSHubAccount(username="karpathy", platform="x")],
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
