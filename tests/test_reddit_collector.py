from datetime import UTC, datetime
from unittest.mock import patch

import httpx
import pytest
import respx

from collectors.reddit import OAUTH_BASE, TOKEN_URL, RedditCollector, _resolve_reddit_credentials
from shared.config import RedditCollectorConfig
from shared.constants import SourceType


def _config(**kwargs) -> RedditCollectorConfig:
    base = {"subreddits": ["LocalLLaMA"], "sort": "hot", "limit": 5}
    base.update(kwargs)
    cfg = RedditCollectorConfig(**base)
    cfg.reference_time = datetime(2026, 6, 2, tzinfo=UTC)
    cfg.lookback_hours = 24
    return cfg


def _listing(created_utc: float, **post) -> dict:
    data = {
        "id": "abc",
        "title": "Test Post",
        "selftext": "body",
        "author": "alice",
        "permalink": "/r/LocalLLaMA/comments/abc/test/",
        "is_self": True,
        "score": 123,
        "num_comments": 45,
        "created_utc": created_utc,
    }
    data.update(post)
    return {"data": {"children": [{"data": data}]}}


class TestResolveCredentials:
    def test_prefers_env(self, monkeypatch):
        monkeypatch.setenv("REDDIT_CLIENT_ID", "envid")
        monkeypatch.setenv("REDDIT_CLIENT_SECRET", "envsecret")
        assert _resolve_reddit_credentials() == ("envid", "envsecret")

    def test_returns_none_on_ssm_failure(self, monkeypatch):
        monkeypatch.delenv("REDDIT_CLIENT_ID", raising=False)
        monkeypatch.delenv("REDDIT_CLIENT_SECRET", raising=False)
        with patch("shared.utils.boto3.client", side_effect=Exception("no ssm")):
            assert _resolve_reddit_credentials() is None


class TestRedditCollect:
    @pytest.mark.asyncio
    async def test_skips_when_no_credentials(self):
        collector = RedditCollector(_config())
        with patch("collectors.reddit._resolve_reddit_credentials", return_value=None):
            items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_collects_with_engagement_metadata(self):
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"access_token": "tok"}))
        recent = datetime(2026, 6, 2, tzinfo=UTC).timestamp() - 3600
        respx.get(f"{OAUTH_BASE}/r/LocalLLaMA/hot").mock(return_value=httpx.Response(200, json=_listing(recent)))
        collector = RedditCollector(_config())
        with patch("collectors.reddit._resolve_reddit_credentials", return_value=("id", "secret")):
            items = await collector.collect()

        assert len(items) == 1
        item = items[0]
        assert item.source_type == SourceType.REDDIT
        assert item.metadata["score"] == 123
        assert item.metadata["num_comments"] == 45
        assert item.metadata["subreddit"] == "LocalLLaMA"
        assert item.url == "https://www.reddit.com/r/LocalLLaMA/comments/abc/test/"

    @pytest.mark.asyncio
    @respx.mock
    async def test_filters_old_posts(self):
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"access_token": "tok"}))
        old = datetime(2026, 5, 1, tzinfo=UTC).timestamp()
        respx.get(f"{OAUTH_BASE}/r/LocalLLaMA/hot").mock(return_value=httpx.Response(200, json=_listing(old)))
        collector = RedditCollector(_config())
        with patch("collectors.reddit._resolve_reddit_credentials", return_value=("id", "secret")):
            items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_token_skips(self):
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"access_token": ""}))
        collector = RedditCollector(_config())
        with patch("collectors.reddit._resolve_reddit_credentials", return_value=("id", "secret")):
            items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_token_fetch_failure_degrades(self):
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(401, json={"error": "unauthorized"}))
        collector = RedditCollector(_config())
        with patch("collectors.reddit._resolve_reddit_credentials", return_value=("id", "secret")):
            items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    @respx.mock
    async def test_top_sort_adds_time_param(self):
        respx.post(TOKEN_URL).mock(return_value=httpx.Response(200, json={"access_token": "tok"}))
        route = respx.get(f"{OAUTH_BASE}/r/LocalLLaMA/top").mock(
            return_value=httpx.Response(200, json={"data": {"children": []}})
        )
        collector = RedditCollector(_config(sort="top"))
        with patch("collectors.reddit._resolve_reddit_credentials", return_value=("id", "secret")):
            await collector.collect()
        assert route.called
        assert "t=day" in str(route.calls.last.request.url)
