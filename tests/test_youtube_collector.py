from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collectors.youtube import YouTubeCollector
from shared.config import YouTubeCollectorConfig
from shared.constants import SourceType


def _config(**kwargs) -> YouTubeCollectorConfig:
    base = {"channels": ["https://www.youtube.com/@example"], "max_videos_per_channel": 3}
    base.update(kwargs)
    cfg = YouTubeCollectorConfig(**base)
    cfg.reference_time = datetime(2026, 6, 3, tzinfo=UTC)
    cfg.lookback_hours = 24
    return cfg


def _resp(status: int, payload: dict) -> MagicMock:
    r = MagicMock()
    r.status_code = status
    r.json.return_value = payload
    return r


def _playlist_payload(*video_ids: str) -> dict:
    return {"items": [{"snippet": {"resourceId": {"videoId": vid}}} for vid in video_ids]}


def _videos_payload(video_id: str, *, published: str = "2026-06-03T00:00:00Z", views: int = 1234) -> dict:
    return {
        "items": [
            {
                "id": video_id,
                "snippet": {
                    "title": "Test Video",
                    "description": "desc",
                    "channelTitle": "Example Channel",
                    "publishedAt": published,
                },
                "statistics": {"viewCount": str(views)},
            }
        ]
    }


class TestApiPath:
    @pytest.mark.asyncio
    async def test_api_happy_path(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())

        client = AsyncMock()
        client.get.side_effect = [
            _resp(200, _playlist_payload("vid00000001")),
            _resp(200, _videos_payload("vid00000001")),
        ]
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                with patch.object(collector, "_get_transcript", return_value="full transcript"):
                    items = await collector.collect()

        assert len(items) == 1
        item = items[0]
        assert item.source_type == SourceType.YOUTUBE
        assert item.item_id == "vid00000001"
        assert item.text == "full transcript"
        assert item.metadata["view_count"] == 1234

    @pytest.mark.asyncio
    async def test_non_200_returns_empty(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())

        client = AsyncMock()
        client.get.return_value = _resp(403, {})
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                items = await collector.collect()

        assert items == []

    @pytest.mark.asyncio
    async def test_malformed_playlist_json_returns_empty(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())

        bad = MagicMock(status_code=200)
        bad.json.side_effect = ValueError("truncated body")
        client = AsyncMock()
        client.get.return_value = bad
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    async def test_malformed_videos_json_returns_empty(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())

        bad_details = MagicMock(status_code=200)
        bad_details.json.side_effect = ValueError("truncated body")
        client = AsyncMock()
        client.get.side_effect = [
            _resp(200, _playlist_payload("vid00000001")),
            bad_details,
        ]
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    async def test_videos_non_200_returns_empty(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())

        client = AsyncMock()
        client.get.side_effect = [
            _resp(200, _playlist_payload("vid00000001")),
            _resp(500, {}),
        ]
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    async def test_unresolvable_channel_returns_empty(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        with patch.object(collector, "_resolve_channel_id", return_value=""):
            items = await collector.collect()
        assert items == []


class TestRssFallback:
    @pytest.mark.asyncio
    async def test_rss_fallback_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        collector = YouTubeCollector(_config())

        class _Feed:
            entries = [
                {
                    "yt_videoid": "rssvid00001",
                    "title": "RSS Video",
                    "author": "Example",
                    "summary": "rss summary",
                    "published_parsed": (2026, 6, 3, 0, 0, 0, 0, 0, 0),
                }
            ]

        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.feedparser.parse", return_value=_Feed()):
                with patch.object(collector, "_get_transcript", return_value=""):
                    items = await collector.collect()

        assert len(items) == 1
        assert items[0].item_id == "rssvid00001"
        assert items[0].text == "rss summary"  # falls back to summary when transcript empty


class TestResolveChannelId:
    def test_resolves_from_channel_id_quoted(self):
        collector = YouTubeCollector(_config())
        resp = MagicMock(text='...{"channelId":"UC1234567890abcdef"}...')
        with patch("collectors.youtube.httpx.get", return_value=resp):
            assert collector._resolve_channel_id("https://youtube.com/@x") == "UC1234567890abcdef"

    def test_resolves_from_channel_id_param(self):
        collector = YouTubeCollector(_config())
        resp = MagicMock(text='<link href="...channel_id=UCfromparam">')
        with patch("collectors.youtube.httpx.get", return_value=resp):
            assert collector._resolve_channel_id("https://youtube.com/@x") == "UCfromparam"

    def test_returns_empty_on_no_match(self):
        collector = YouTubeCollector(_config())
        resp = MagicMock(text="no ids here")
        with patch("collectors.youtube.httpx.get", return_value=resp):
            assert collector._resolve_channel_id("https://youtube.com/@x") == ""


class TestTranscript:
    def test_transcript_failure_returns_empty(self):
        collector = YouTubeCollector(_config())
        with patch("collectors.youtube.is_proxy_configured", return_value=False):
            with patch("collectors.youtube.YouTubeTranscriptApi", side_effect=RuntimeError("boom")):
                assert collector._get_transcript("vid") == ""

    @pytest.mark.asyncio
    async def test_fetch_transcript_times_out_and_skips(self):
        collector = YouTubeCollector(_config(transcript_timeout=1))

        async def timeout(awaitable, timeout):
            awaitable.close()  # avoid an un-awaited to_thread coroutine warning
            raise TimeoutError

        with patch.object(collector, "_get_transcript", return_value="never"):
            with patch("collectors.youtube.asyncio.wait_for", side_effect=timeout):
                result = await collector._fetch_transcript("vid")
        assert result == ""
