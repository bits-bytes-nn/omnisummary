from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collectors.youtube import YouTubeCollector
from shared.config import YouTubeCollectorConfig
from shared.constants import SourceType
from shared.models import CollectedItem


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

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
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

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
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

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
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

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
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

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    async def test_unresolvable_channel_raises_for_health(self, monkeypatch):
        # An unresolvable channel must register as a FAILURE (not silent EMPTY) so a
        # blackholed channel is distinguishable from one with no recent uploads. With a
        # single configured channel, gather(raise_if_all_failed=True) propagates.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="")):
            with patch.object(collector, "_resolve_channel_id", return_value=""):
                with pytest.raises(RuntimeError, match="resolve canonical channel ID"):
                    await collector.collect()


class TestS3Preload:
    @pytest.mark.asyncio
    async def test_prefers_s3_items_when_present(self, monkeypatch):
        # When a local sync has parked transcript-bearing items in S3, AWS reads those and
        # skips live collection entirely (which would yield transcript-less metadata).
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        parked = [
            CollectedItem(
                item_id="vS3",
                source_type=SourceType.YOUTUBE,
                title="From S3",
                url="https://y/v",
                text="full transcript",
            )
        ]
        with patch("collectors.youtube.load_items_from_s3", return_value=parked):
            with patch.object(collector, "_collect_channel", new=AsyncMock()) as live:
                items = await collector.collect()
        assert [i.item_id for i in items] == ["vS3"]
        live.assert_not_called()  # S3 hit → no live collection

    @pytest.mark.asyncio
    async def test_live_collection_when_no_s3(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        with patch("collectors.youtube.load_items_from_s3", return_value=None):
            with patch.object(collector, "_collect_channel", new=AsyncMock(return_value=[])):
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
    def test_resolves_canonical_channel_id(self):
        collector = YouTubeCollector(_config())
        cid = "UC" + "a1b2c3d4e5f6g7h8i9j0k1"  # UC + exactly 22 base64url chars
        resp = MagicMock(text=f'...{{"channelId":"{cid}"}}...')
        with patch.object(collector._sync_client, "get", return_value=resp):
            assert collector._resolve_channel_id("https://youtube.com/@x") == cid

    def test_rejects_noncanonical_channel_id(self):
        # too-short UC ids and the loose channel_id= param no longer resolve — they
        # would have produced a malformed UU... uploads playlist and a silent empty result.
        collector = YouTubeCollector(_config())
        for text in ('...{"channelId":"UC1234567890abcdef"}...', '<link href="...channel_id=UCfromparam">'):
            resp = MagicMock(text=text)
            with patch.object(collector._sync_client, "get", return_value=resp):
                assert collector._resolve_channel_id("https://youtube.com/@x") == ""

    def test_returns_empty_on_no_match(self):
        collector = YouTubeCollector(_config())
        resp = MagicMock(text="no ids here")
        with patch.object(collector._sync_client, "get", return_value=resp):
            assert collector._resolve_channel_id("https://youtube.com/@x") == ""


class TestResolveChannelIdViaApi:
    @pytest.mark.asyncio
    async def test_resolves_handle_via_data_api(self, monkeypatch):
        # The API forHandle lookup works from datacenter IPs where the page scrape is blocked.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        client = AsyncMock()
        client.get.return_value = _resp(200, {"items": [{"id": "UCabc123"}]})
        cid = await collector._resolve_channel_id_via_api("https://www.youtube.com/@AndrejKarpathy", client)
        assert cid == "UCabc123"
        assert client.get.call_args.kwargs["params"]["forHandle"] == "AndrejKarpathy"

    @pytest.mark.asyncio
    async def test_returns_empty_when_no_handle(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        client = AsyncMock()
        cid = await collector._resolve_channel_id_via_api("https://www.youtube.com/channel/UCx", client)
        assert cid == ""
        client.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_empty_on_empty_items(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        client = AsyncMock()
        client.get.return_value = _resp(200, {"items": []})
        assert await collector._resolve_channel_id_via_api("https://www.youtube.com/@x", client) == ""


class TestTranscript:
    def test_transcript_failure_returns_empty(self):
        collector = YouTubeCollector(_config())
        with patch("collectors.youtube.YouTubeTranscriptApi", side_effect=RuntimeError("boom")):
            assert collector._get_transcript("vid") == ""

    def test_fetch_uses_configured_language_first(self):
        collector = YouTubeCollector(_config(transcript_language="ko"))
        api = MagicMock()
        api.fetch.return_value = MagicMock(snippets=[MagicMock(text="안녕")])
        with patch("collectors.youtube.YouTubeTranscriptApi", return_value=api):
            out = collector._get_transcript("vid")
        assert out == "안녕"
        assert api.fetch.call_args.kwargs["languages"] == ("ko",)

    def test_falls_back_to_any_available_language(self):
        from youtube_transcript_api._errors import YouTubeTranscriptApiException

        collector = YouTubeCollector(_config(transcript_language="en"))
        api = MagicMock()
        # Configured 'en' missing → raise, then fall back to listed languages.
        api.fetch.side_effect = YouTubeTranscriptApiException("no en")
        track = MagicMock(language_code="ko")
        listing = MagicMock()
        listing.__iter__ = lambda self: iter([track])
        listing.find_transcript.return_value.fetch.return_value = MagicMock(snippets=[MagicMock(text="대체")])
        api.list.return_value = listing
        with patch("collectors.youtube.YouTubeTranscriptApi", return_value=api):
            out = collector._get_transcript("vid")
        assert out == "대체"
        assert api.list.return_value.find_transcript.call_args.args[0] == ["ko"]

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


class TestResolveChannelIdTimeout:
    @pytest.mark.asyncio
    async def test_resolve_channel_id_times_out_and_skips(self):
        collector = YouTubeCollector(_config(resolve_timeout=1))

        async def timeout(awaitable, timeout):
            awaitable.close()  # avoid an un-awaited to_thread coroutine warning
            raise TimeoutError

        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.asyncio.wait_for", side_effect=timeout):
                result = await collector._resolve_channel_id_async("https://youtube.com/@x")
        assert result == ""


class TestLifecycle:
    def test_del_closes_pooled_client(self):
        collector = YouTubeCollector(_config())
        with patch.object(collector._sync_client, "close") as close:
            collector.__del__()
        close.assert_called_once()
