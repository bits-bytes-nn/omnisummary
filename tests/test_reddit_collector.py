from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from collectors.reddit import RedditCollector
from shared.config import RedditCollectorConfig
from shared.constants import SourceType


def _config(**kwargs) -> RedditCollectorConfig:
    # retry_backoff_sec=0 keeps retries and inter-subreddit spacing instant in tests (jitter and
    # spacing both scale by it), so a retriable-status test doesn't sleep for real.
    base = {"subreddits": ["LocalLLaMA"], "sort": "hot", "limit": 5, "retry_backoff_sec": 0}
    base.update(kwargs)
    cfg = RedditCollectorConfig(**base)
    cfg.reference_time = datetime(2026, 6, 2, tzinfo=UTC)
    cfg.lookback_hours = 24
    return cfg


class _Feed(dict):
    """Mimics feedparser's parse result (attribute + .get access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _feed(entries, *, bozo=False, bozo_exception=None, status=200):
    return _Feed(entries=entries, bozo=bozo, bozo_exception=bozo_exception, status=status)


class _Entry(dict):
    """Mimics feedparser's FeedParserDict (attribute + key access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _entry(title="Test Post", link="https://www.reddit.com/r/LocalLLaMA/comments/abc123/test/", **kw):
    e = _Entry(
        {
            "title": title,
            "link": link,
            "id": "t3_abc123",
            "summary": "post body",
            "author": "alice",
            "published_parsed": (2026, 6, 2, 0, 0, 0, 0, 0, 0),
        }
    )
    e.update(kw)
    return e


class TestRedditCollect:
    @pytest.mark.asyncio
    async def test_no_subreddits_returns_empty(self):
        collector = RedditCollector(_config(subreddits=[]))
        assert await collector.collect() == []

    @pytest.mark.asyncio
    async def test_collects_via_rss(self):
        collector = RedditCollector(_config())
        with patch("collectors.reddit.parse_feed_with_fallback", return_value=_feed([_entry()])):
            items = await collector.collect()
        assert len(items) == 1
        item = items[0]
        assert item.source_type == SourceType.REDDIT
        assert item.metadata["subreddit"] == "LocalLLaMA"
        assert item.item_id == "abc123"
        assert "score" not in item.metadata

    @pytest.mark.asyncio
    async def test_filters_old_posts(self):
        old = _entry(published_parsed=(2026, 5, 1, 0, 0, 0, 0, 0, 0))
        collector = RedditCollector(_config())
        with patch("collectors.reddit.parse_feed_with_fallback", return_value=_feed([old])):
            items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    async def test_total_outage_raises_for_health_alert(self):
        # All subreddits failing (e.g. proxy/upstream error) must surface as a failure
        # so the health check marks Reddit FAILED rather than a silent empty day.
        bad = _feed([], bozo=True, bozo_exception=Exception("parse error"))
        collector = RedditCollector(_config())
        with patch("collectors.reddit.parse_feed_with_fallback", return_value=bad):
            with pytest.raises(RuntimeError):
                await collector.collect()

    @pytest.mark.asyncio
    async def test_http_error_status_raises(self):
        collector = RedditCollector(_config())
        with patch("collectors.reddit.parse_feed_with_fallback", return_value=_feed([], status=503)):
            with pytest.raises(RuntimeError):
                await collector.collect()

    @pytest.mark.asyncio
    async def test_bozo_with_entries_still_parses(self):
        # feedparser sets bozo on minor XML issues but still yields entries — must parse them.
        feed = _feed([_entry()], bozo=True, bozo_exception=Exception("minor xml warning"))
        collector = RedditCollector(_config())
        with patch("collectors.reddit.parse_feed_with_fallback", return_value=feed):
            items = await collector.collect()
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_partial_failure_keeps_succeeding_subreddits(self):
        good = _feed([_entry()])
        bad = _feed([], bozo=True, bozo_exception=Exception("boom"))
        collector = RedditCollector(_config(subreddits=["LocalLLaMA", "MachineLearning"]))
        with patch("collectors.reddit.parse_feed_with_fallback", side_effect=[good, bad]):
            items = await collector.collect()
        assert len(items) == 1  # one subreddit failed, the other survived

    @pytest.mark.asyncio
    async def test_retries_429_then_succeeds(self):
        # A rate-limited (429) fetch must be retried, not dropped on the first hit. Second attempt
        # returns a good feed → the subreddit is collected instead of lost.
        rate_limited = _feed([], status=429)
        good = _feed([_entry()])
        collector = RedditCollector(_config(max_retries=3))
        with patch("collectors.reddit.parse_feed_with_fallback", side_effect=[rate_limited, good]):
            items = await collector.collect()
        assert len(items) == 1

    @pytest.mark.asyncio
    async def test_429_exhausts_retries_then_fails(self):
        # Persistent 429 across all attempts surfaces as a failure (single subreddit → total outage
        # raises for the health alert).
        collector = RedditCollector(_config(max_retries=2))
        with patch("collectors.reddit.parse_feed_with_fallback", return_value=_feed([], status=429)) as mock_parse:
            with pytest.raises(RuntimeError):
                await collector.collect()
        assert mock_parse.call_count == 2  # retried up to max_retries

    @pytest.mark.asyncio
    async def test_404_is_not_retried(self):
        # A permanent 4xx (e.g. 404) must NOT be retried — fail fast.
        collector = RedditCollector(_config(max_retries=3))
        with patch("collectors.reddit.parse_feed_with_fallback", return_value=_feed([], status=404)) as mock_parse:
            with pytest.raises(RuntimeError):
                await collector.collect()
        assert mock_parse.call_count == 1  # not retried

    @pytest.mark.asyncio
    async def test_builds_correct_rss_url(self):
        collector = RedditCollector(_config(sort="top"))
        with patch("collectors.reddit.parse_feed_with_fallback", return_value=_feed([])) as mock_parse:
            await collector.collect()
        called_url = mock_parse.call_args.args[0]
        assert "/r/LocalLLaMA/top/.rss" in called_url
        assert "limit=5" in called_url
        assert "t=day" in called_url  # sort=top must request the daily window


class TestExtractPostId:
    def test_from_comments_link(self):
        link = "https://www.reddit.com/r/LocalLLaMA/comments/xyz789/title/"
        assert RedditCollector._extract_post_id("t3_xyz789", link) == "xyz789"

    def test_from_entry_id_fallback(self):
        assert RedditCollector._extract_post_id("t3_abc", "https://example.com/no-match") == "abc"

    def test_generated_when_no_id(self):
        out = RedditCollector._extract_post_id("", "https://example.com/x")
        assert out and len(out) == 16
