import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from collectors.base import ParkedItems, ParkOutcome
from collectors.youtube import YouTubeCollector
from shared.config import YouTubeCollectorConfig
from shared.constants import SourceType
from shared.models import CollectedItem


def _absent_park() -> ParkedItems:
    """No S3 park file → the collector must fall through to live collection."""
    return ParkedItems(outcome=ParkOutcome.ABSENT)


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
    async def test_keeps_fresh_video_listed_below_a_stale_one(self, monkeypatch):
        # Regression: the uploads playlist is NOT reliably newest-first. With max_per_channel=1,
        # taking only the top row dropped a fresh in-window video that sat below a stale one
        # (the Dwarkesh miss). Over-fetch + sort-by-date must surface the fresh video instead.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config(max_videos_per_channel=1))

        # Playlist order: STALE first (out of the 24h window), FRESH second (in window).
        videos_payload = {
            "items": [
                {
                    "id": "stale000001",
                    "snippet": {
                        "title": "Old Pinned Talk",
                        "description": "d",
                        "channelTitle": "Ch",
                        "publishedAt": "2026-05-01T00:00:00Z",
                    },
                    "statistics": {"viewCount": "5"},
                },
                {
                    "id": "fresh000001",
                    "snippet": {
                        "title": "Brand New Episode",
                        "description": "d",
                        "channelTitle": "Ch",
                        "publishedAt": "2026-06-02T18:00:00Z",
                    },
                    "statistics": {"viewCount": "9"},
                },
            ]
        }
        client = AsyncMock()
        client.get.side_effect = [
            _resp(200, _playlist_payload("stale000001", "fresh000001")),
            _resp(200, videos_payload),
        ]
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                with patch.object(collector, "_get_transcript", return_value=""):
                    items = await collector.collect()

        assert len(items) == 1
        assert items[0].item_id == "fresh000001"  # the fresh one, not the stale top-of-playlist row

    @pytest.mark.asyncio
    async def test_non_200_raises_for_health(self, monkeypatch):
        # An API rejection (quota exhausted / revoked key) is a FAILURE, not an empty channel:
        # with a single configured channel the all-failed check propagates it so the source is
        # reported FAILED instead of looking like a day with no uploads.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())

        client = AsyncMock()
        client.get.return_value = _resp(403, {})
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                with pytest.raises(RuntimeError, match="returned 403"):
                    await collector.collect()

    @pytest.mark.asyncio
    async def test_malformed_playlist_json_raises_for_health(self, monkeypatch):
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
                with pytest.raises(RuntimeError, match="malformed JSON"):
                    await collector.collect()

    @pytest.mark.asyncio
    async def test_malformed_videos_json_raises_for_health(self, monkeypatch):
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
                with pytest.raises(RuntimeError, match="malformed JSON"):
                    await collector.collect()

    @pytest.mark.asyncio
    async def test_videos_persistent_5xx_raises_for_health(self, monkeypatch):
        # A 5xx is retriable, but an exhausted retry chain is still a channel FAILURE.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config(max_retries=2, retry_backoff_sec=0))

        client = AsyncMock()
        client.get.side_effect = lambda *a, **k: (
            _resp(200, _playlist_payload("vid00000001")) if client.get.await_count == 1 else _resp(503, {})
        )
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                with pytest.raises(RuntimeError, match="returned 503"):
                    await collector.collect()
        assert client.get.await_count == 3  # playlist + 2 attempts at the details call

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

    @pytest.mark.asyncio
    async def test_one_failing_channel_among_many_is_tolerated(self, monkeypatch):
        # Partial tolerance: only an ALL-channels failure escalates to FAILED.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config(channels=["https://www.youtube.com/@a", "https://www.youtube.com/@b"]))

        async def _collect(channel_url):
            if channel_url.endswith("@a"):
                raise RuntimeError("quota exceeded")
            return [CollectedItem(item_id="ok", source_type=SourceType.YOUTUBE, title="t", url="https://y/ok")]

        with patch("collectors.youtube.load_items_from_s3", return_value=_absent_park()):
            with patch.object(collector, "_collect_channel", side_effect=_collect):
                items = await collector.collect()
        assert [i.item_id for i in items] == ["ok"]


class TestRetriableStatuses:
    @pytest.mark.asyncio
    async def test_transient_5xx_is_retried_then_succeeds(self, monkeypatch):
        # A 503 on the playlist call used to fail the channel outright; it must be retried.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config(retry_backoff_sec=0))

        client = AsyncMock()
        client.get.side_effect = [
            _resp(503, {}),
            _resp(200, _playlist_payload("vid00000001")),
            _resp(200, _videos_payload("vid00000001")),
        ]
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                with patch.object(collector, "_get_transcript", return_value=""):
                    items = await collector.collect()
        assert [i.item_id for i in items] == ["vid00000001"]

    @pytest.mark.asyncio
    async def test_permanent_403_is_not_retried(self, monkeypatch):
        # 403 (quota exhausted / revoked key) is a verdict, not a hiccup: fail on the first response
        # instead of burning the channel's whole time budget on retries.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config(max_retries=3, retry_backoff_sec=0))

        client = AsyncMock()
        client.get.return_value = _resp(403, {})
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                with pytest.raises(RuntimeError, match="returned 403"):
                    await collector.collect()
        assert client.get.await_count == 1

    @pytest.mark.asyncio
    async def test_retry_after_is_honoured_and_capped(self, monkeypatch):
        # A 429's Retry-After is respected, but clamped to the per-request timeout so an absurd
        # value can't outlive the channel's budget.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config(request_timeout=7, retry_backoff_sec=0))

        throttled = _resp(429, {})
        throttled.headers = {"Retry-After": "3600"}
        ok = _resp(200, _playlist_payload("vid00000001"))
        ok.headers = {}
        details = _resp(200, _videos_payload("vid00000001"))
        details.headers = {}

        client = AsyncMock()
        client.get.side_effect = [throttled, ok, details]
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)

        slept: list[float] = []

        async def _sleep(delay):
            slept.append(delay)

        with patch.object(collector, "_resolve_channel_id_via_api", AsyncMock(return_value="UCabcdef")):
            with patch("collectors.youtube.httpx.AsyncClient", return_value=ctx):
                with patch("collectors.youtube.asyncio.sleep", new=_sleep):
                    with patch.object(collector, "_get_transcript", return_value=""):
                        items = await collector.collect()
        assert [i.item_id for i in items] == ["vid00000001"]
        # 3600s clamped to request_timeout (the retry backoff's own 0s sleep may follow).
        assert slept[0] == 7.0 and all(s <= 7.0 for s in slept)

    def test_retry_after_parsing(self):
        from collectors.youtube import _retry_after_delay

        resp = MagicMock()
        resp.headers = {}
        assert _retry_after_delay(resp, 10) == 0.0
        resp.headers = {"Retry-After": "4"}
        assert _retry_after_delay(resp, 10) == 4.0
        resp.headers = {"Retry-After": "not-a-date"}
        assert _retry_after_delay(resp, 10) == 0.0
        resp.headers = {"Retry-After": "Wed, 21 Oct 2015 07:28:00 GMT"}  # in the past → no wait
        assert _retry_after_delay(resp, 10) == 0.0


class TestChannelFanOut:
    @pytest.mark.asyncio
    async def test_fan_out_is_bounded_by_config(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(
            _config(channels=[f"https://www.youtube.com/@c{n}" for n in range(6)], max_concurrency=2)
        )
        state = {"active": 0, "peak": 0}

        async def _collect(channel_url):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return []

        with patch("collectors.youtube.load_items_from_s3", return_value=_absent_park()):
            with patch.object(collector, "_collect_channel", side_effect=_collect):
                await collector.collect()
        assert state["peak"] <= 2, f"fan-out exceeded max_concurrency: peak={state['peak']}"

    @pytest.mark.asyncio
    async def test_hung_channel_times_out_as_a_failure(self, monkeypatch):
        # A wedged channel must fail its task (so an all-channels hang reports FAILED) rather than
        # holding the digest until the Lambda itself times out.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())

        async def _hang(channel_url):
            await asyncio.sleep(10)
            return []

        with patch("collectors.youtube.load_items_from_s3", return_value=_absent_park()):
            with patch.object(collector, "_collect_channel", side_effect=_hang):
                with patch.object(type(collector.config), "channel_budget_sec", property(lambda self: 0)):
                    with pytest.raises(RuntimeError, match="timed out"):
                        await collector.collect()

    def test_channel_budget_derives_from_the_step_timeouts(self):
        cfg = _config(
            resolve_timeout=15,
            request_timeout=30,
            retry_backoff_sec=5,
            max_retries=3,
            transcript_timeout=15,
            max_videos_per_channel=3,
        )
        assert cfg.channel_budget_sec == 15 + (30 + 5) * 3 * 2 + 15 * 3


class TestTranscriptSocketTimeout:
    """asyncio.wait_for cannot cancel an asyncio.to_thread worker, so the wait_for around the
    transcript fetch only stopped the CALLER waiting — youtube_transcript_api's bare Session has no
    socket timeout, so the thread kept its pool slot for the rest of the process's life and
    asyncio.run blocked on the executor join at shutdown. The bound has to be at the socket."""

    def test_the_transcript_api_gets_a_session_carrying_the_configured_timeout(self):
        from collectors import youtube

        with patch.object(youtube, "YouTubeTranscriptApi") as api:
            api.return_value.fetch.return_value = MagicMock(snippets=[])
            youtube.fetch_youtube_transcript("vid", "en", timeout_sec=7)
        session = api.call_args.kwargs["http_client"]
        assert isinstance(session, youtube._TimeoutSession)
        assert session._timeout_sec == 7

    def test_the_session_injects_the_timeout_into_every_request(self):
        from collectors.youtube import _TimeoutSession

        session = _TimeoutSession(3)
        with patch("requests.Session.request", return_value="resp") as request:
            session.request("GET", "https://example.com")
        assert request.call_args.kwargs["timeout"] == 3

    def test_an_explicit_timeout_is_not_overridden(self):
        from collectors.youtube import _TimeoutSession

        session = _TimeoutSession(3)
        with patch("requests.Session.request", return_value="resp") as request:
            session.request("GET", "https://example.com", timeout=1)
        assert request.call_args.kwargs["timeout"] == 1

    def test_a_socket_timeout_degrades_to_an_empty_transcript(self):
        # The timeout surfaces as a requests exception the library does not wrap; the fetch is
        # best-effort, so it must not fail the whole channel collect.
        import requests

        from collectors import youtube

        with patch.object(youtube, "YouTubeTranscriptApi") as api:
            api.return_value.fetch.side_effect = requests.exceptions.ReadTimeout("hung")
            api.return_value.list.side_effect = requests.exceptions.ReadTimeout("hung")
            assert youtube.fetch_youtube_transcript("vid", "en", timeout_sec=1) == ""

    @pytest.mark.asyncio
    async def test_the_collector_passes_its_transcript_timeout_through(self):
        collector = YouTubeCollector(_config(transcript_timeout=9))
        with patch("collectors.youtube.fetch_youtube_transcript", return_value="") as fetch:
            collector._get_transcript("vid")
        assert fetch.call_args.kwargs["timeout_sec"] == 9


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
        with patch(
            "collectors.youtube.load_items_from_s3",
            return_value=ParkedItems(outcome=ParkOutcome.FRESH, items=parked),
        ):
            with patch.object(collector, "_collect_channel", new=AsyncMock()) as live:
                items = await collector.collect()
        assert [i.item_id for i in items] == ["vS3"]
        live.assert_not_called()  # S3 hit → no live collection

    @pytest.mark.asyncio
    async def test_stale_park_items_are_used_and_recorded_for_health(self, monkeypatch):
        # Stale beats empty: the parked items are still returned, but park_status must record the
        # staleness so run_collectors_with_health can report STALE instead of OK.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        parked = [CollectedItem(item_id="vOld", source_type=SourceType.YOUTUBE, title="t", url="https://y/v")]
        stale = ParkedItems(outcome=ParkOutcome.STALE, items=parked, age_hours=72.0, detail="72.0h old")
        with patch("collectors.youtube.load_items_from_s3", return_value=stale):
            with patch.object(collector, "_collect_channel", new=AsyncMock()) as live:
                items = await collector.collect()
        assert [i.item_id for i in items] == ["vOld"]
        live.assert_not_called()
        assert collector.park_status is not None and collector.park_status.degraded is True

    @pytest.mark.asyncio
    async def test_parked_items_outside_the_lookback_window_are_dropped(self, monkeypatch):
        # Both live branches drop entries published before the cutoff; the park branch returned the
        # file's items verbatim, so a stale park file re-ingested days-old videos every run.
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        parked = [
            CollectedItem(
                item_id="vFresh",
                source_type=SourceType.YOUTUBE,
                title="fresh",
                url="https://y/fresh",
                published_at=datetime(2026, 6, 2, 12, tzinfo=UTC),
            ),
            CollectedItem(
                item_id="vOld",
                source_type=SourceType.YOUTUBE,
                title="old",
                url="https://y/old",
                published_at=datetime(2026, 5, 20, tzinfo=UTC),
            ),
        ]
        stale = ParkedItems(outcome=ParkOutcome.STALE, items=parked, age_hours=72.0, detail="72.0h old")
        with patch("collectors.youtube.load_items_from_s3", return_value=stale):
            items = await collector.collect()
        assert [i.item_id for i in items] == ["vFresh"]

    @pytest.mark.asyncio
    async def test_a_backfill_run_does_not_ingest_todays_parked_items(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        parked = [
            CollectedItem(
                item_id="vLater",
                source_type=SourceType.YOUTUBE,
                title="later",
                url="https://y/later",
                published_at=datetime(2026, 6, 5, tzinfo=UTC),
            )
        ]
        with patch(
            "collectors.youtube.load_items_from_s3",
            return_value=ParkedItems(outcome=ParkOutcome.FRESH, items=parked),
        ):
            with patch.object(collector, "_collect_channel", new=AsyncMock()) as live:
                assert await collector.collect() == []
        live.assert_not_called()

    @pytest.mark.asyncio
    async def test_park_age_budget_comes_from_config(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config(park_max_age_hours=72))
        with patch("collectors.youtube.load_items_from_s3", return_value=_absent_park()) as load:
            with patch.object(collector, "_collect_channel", new=AsyncMock(return_value=[])):
                await collector.collect()
        assert load.call_args.kwargs["max_age_hours"] == 72

    @pytest.mark.asyncio
    async def test_live_collection_when_no_s3(self, monkeypatch):
        monkeypatch.setenv("YOUTUBE_API_KEY", "k")
        collector = YouTubeCollector(_config())
        with patch("collectors.youtube.load_items_from_s3", return_value=_absent_park()):
            with patch.object(collector, "_collect_channel", new=AsyncMock(return_value=[])):
                items = await collector.collect()
        assert items == []


class TestApiKeyResolution:
    @pytest.mark.asyncio
    async def test_key_resolved_once_off_the_event_loop(self, monkeypatch):
        # resolve_secret falls back to a BLOCKING SSM call. It used to run lazily from a property
        # read inside the async fan-out — on the loop thread, once per channel. It must now happen
        # exactly once per run, in a worker thread, before any channel task starts.
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        collector = YouTubeCollector(_config(channels=[f"https://www.youtube.com/@c{n}" for n in range(4)]))
        calls: list[tuple] = []

        def _resolve(*args):
            calls.append(args)
            return "key-from-ssm"

        with patch("collectors.youtube.load_items_from_s3", return_value=_absent_park()):
            with patch("collectors.youtube.resolve_secret", side_effect=_resolve):
                with patch.object(collector, "_collect_via_api", new=AsyncMock(return_value=[])) as via_api:
                    await collector.collect()
        assert len(calls) == 1  # once for the whole run, not once per channel
        assert collector.api_key == "key-from-ssm"
        assert via_api.await_count == 4  # the resolved key routed every channel to the API path

    @pytest.mark.asyncio
    async def test_parked_run_never_resolves_the_key(self, monkeypatch):
        # The S3 park path short-circuits before any HTTP, so it must not pay for an SSM lookup.
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        collector = YouTubeCollector(_config())
        parked = [CollectedItem(item_id="v", source_type=SourceType.YOUTUBE, title="t", url="https://y/v")]
        with patch(
            "collectors.youtube.load_items_from_s3",
            return_value=ParkedItems(outcome=ParkOutcome.FRESH, items=parked),
        ):
            with patch("collectors.youtube.resolve_secret") as resolve:
                await collector.collect()
        resolve.assert_not_called()


class _FeedDict(dict):
    """Stands in for feedparser's FeedParserDict, which supports both key and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class TestRssFallback:
    @staticmethod
    def _feed(*entries):
        return _FeedDict(entries=list(entries), bozo=False, feed=_FeedDict(title="Example"))

    @staticmethod
    def _entry(video_id: str = "rssvid00001", **overrides):
        entry = _FeedDict(
            yt_videoid=video_id,
            title="RSS Video",
            link=f"https://www.youtube.com/watch?v={video_id}",
            author="Example",
            summary="rss summary",
            published_parsed=(2026, 6, 3, 0, 0, 0, 0, 0, 0),
        )
        entry.update(overrides)
        return entry

    @pytest.mark.asyncio
    async def test_rss_fallback_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        collector = YouTubeCollector(_config())

        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch(
                "collectors.youtube.fetch_feed_with_retry",
                new=AsyncMock(return_value=self._feed(self._entry())),
            ):
                with patch.object(collector, "_get_transcript", return_value=""):
                    items = await collector.collect()

        assert len(items) == 1
        assert items[0].item_id == "rssvid00001"
        assert items[0].url == "https://www.youtube.com/watch?v=rssvid00001"
        assert items[0].text == "rss summary"  # falls back to summary when transcript empty

    @pytest.mark.asyncio
    async def test_the_fallback_feed_goes_through_the_shared_retrying_fetch(self, monkeypatch):
        # A raw feedparser.parse(url) has no socket timeout, no retry and no bozo check: one 5xx or
        # DNS blip lost the channel for the day, a truncated body reported EMPTY instead of FAILED,
        # and the un-cancellable worker thread outlived the channel's wait_for budget.
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        config = _config()
        collector = YouTubeCollector(config)
        fetch = AsyncMock(return_value=self._feed(self._entry()))
        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.fetch_feed_with_retry", new=fetch):
                with patch.object(collector, "_get_transcript", return_value=""):
                    await collector.collect()
        kwargs = fetch.await_args.kwargs
        assert kwargs["timeout"] == config.request_timeout
        assert kwargs["max_retries"] == config.max_retries
        assert kwargs["backoff_sec"] == config.retry_backoff_sec
        assert kwargs["proxy_fallback"] is True

    @pytest.mark.asyncio
    async def test_an_entry_with_no_readable_video_id_is_dropped(self, monkeypatch):
        # The transcript lookup and the canonical watch URL are both derived from the video id, so a
        # hashed fallback id would be useless downstream.
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        collector = YouTubeCollector(_config())
        feed = self._feed(self._entry(), self._entry("", link="https://www.youtube.com/@x"))
        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.fetch_feed_with_retry", new=AsyncMock(return_value=feed)):
                with patch.object(collector, "_get_transcript", return_value=""):
                    items = await collector.collect()
        assert [item.item_id for item in items] == ["rssvid00001"]

    @pytest.mark.asyncio
    async def test_the_video_id_is_read_off_the_link_when_the_feed_omits_it(self, monkeypatch):
        monkeypatch.delenv("YOUTUBE_API_KEY", raising=False)
        collector = YouTubeCollector(_config())
        feed = self._feed(self._entry("", link="https://www.youtube.com/watch?v=fromlink001"))
        with patch.object(collector, "_resolve_channel_id", return_value="UCabcdef"):
            with patch("collectors.youtube.fetch_feed_with_retry", new=AsyncMock(return_value=feed)):
                with patch.object(collector, "_get_transcript", return_value=""):
                    items = await collector.collect()
        assert [item.item_id for item in items] == ["fromlink001"]


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

    def test_sync_client_not_created_until_used(self):
        # The S3-parked path short-circuits before any HTTP, so the sync client must not be
        # eagerly opened at construction — and __del__ must be a safe no-op when it never was.
        collector = YouTubeCollector(_config())
        assert collector._sync_client_instance is None
        collector.__del__()  # no client created → must not raise
        assert collector._sync_client_instance is None
