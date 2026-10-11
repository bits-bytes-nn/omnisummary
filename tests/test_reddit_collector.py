from datetime import UTC, datetime
from unittest.mock import patch

import pytest

from collectors.base import TransientStatusError
from collectors.reddit import RedditCollector
from shared.config import RedditCollectorConfig
from shared.constants import SourceType


def _config(**kwargs) -> RedditCollectorConfig:
    # retry_backoff_sec=0 keeps retries instant in tests, so a retriable-status test doesn't sleep.
    base = {"subreddits": ["LocalLLaMA"], "sort": "hot", "limit": 5, "retry_backoff_sec": 0, "lookback_hours": 24}
    base.update(kwargs)
    cfg = RedditCollectorConfig(**base)
    cfg.reference_time = datetime(2026, 6, 2, tzinfo=UTC)
    return cfg


class _Feed(dict):
    """Mimics feedparser's parse result (attribute + .get access)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _feed(entries):
    return _Feed(entries=entries, bozo=False)


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


def _fetch(*outcomes):
    """Stand-in for collectors.base.fetch_feed: yields the given outcomes in order (the last one
    repeats), raising any that is an exception so the collector's retry chain still runs."""
    calls: list[str] = []

    async def _do(url, **kwargs):
        calls.append(url)
        outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return _do, calls


class TestRedditCollect:
    @pytest.mark.asyncio
    async def test_no_subreddits_returns_empty(self):
        collector = RedditCollector(_config(subreddits=[]))
        assert await collector.collect() == []

    @pytest.mark.asyncio
    async def test_collects_via_rss(self):
        collector = RedditCollector(_config())
        fetch, _ = _fetch(_feed([_entry()]))
        with patch("collectors.base.fetch_feed", side_effect=fetch):
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
        fetch, _ = _fetch(_feed([old]))
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            items = await collector.collect()
        assert items == []

    @pytest.mark.asyncio
    async def test_total_outage_raises_for_health_alert(self):
        # All subreddits failing (an upstream error) must surface as a failure
        # so the health check marks Reddit FAILED rather than a silent empty day.
        collector = RedditCollector(_config())
        fetch, _ = _fetch(RuntimeError("Failed to parse Reddit feed"))
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            with pytest.raises(RuntimeError):
                await collector.collect()

    @pytest.mark.asyncio
    async def test_one_request_serves_every_subreddit_with_its_own_attribution_and_cap(self):
        # A second request seconds after the first is what Reddit 429s, so every subreddit rides on
        # ONE multi-subreddit request; each keeps at most `limit` posts so a busy one can't crowd out
        # a quiet one.
        collector = RedditCollector(_config(subreddits=["LocalLLaMA", "MachineLearning"], limit=2))
        entries = [
            _entry(title=f"llama {i}", link=f"https://www.reddit.com/r/LocalLLaMA/comments/l{i}/x/") for i in range(4)
        ] + [_entry(title="ml", link="https://www.reddit.com/r/MachineLearning/comments/m1/x/")]
        fetch, calls = _fetch(_feed(entries))
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            items = await collector.collect()
        assert len(calls) == 1
        assert "/r/LocalLLaMA+MachineLearning/" in calls[0]
        assert [(i.title, i.metadata["subreddit"]) for i in items] == [
            ("llama 0", "LocalLLaMA"),
            ("llama 1", "LocalLLaMA"),
            ("ml", "MachineLearning"),
        ]

    @pytest.mark.asyncio
    async def test_retries_429_then_succeeds(self):
        # A rate-limited (429) fetch must be retried, not dropped on the first hit. Second attempt
        # returns a good feed -> the subreddit is collected instead of lost.
        collector = RedditCollector(_config(max_retries=3))
        fetch, calls = _fetch(TransientStatusError("returned HTTP 429"), _feed([_entry()]))
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            items = await collector.collect()
        assert len(items) == 1
        assert len(calls) == 2

    @pytest.mark.asyncio
    async def test_429_exhausts_retries_then_fails(self):
        # Persistent 429 across all attempts surfaces as a failure (single subreddit -> total outage
        # raises for the health alert).
        collector = RedditCollector(_config(max_retries=2))
        fetch, calls = _fetch(TransientStatusError("returned HTTP 429"))
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            with pytest.raises(RuntimeError):
                await collector.collect()
        assert len(calls) == 2  # retried up to max_retries

    @pytest.mark.asyncio
    async def test_404_is_not_retried(self):
        # A permanent 4xx (e.g. 404) must NOT be retried - fail fast.
        collector = RedditCollector(_config(max_retries=3))
        fetch, calls = _fetch(RuntimeError("returned HTTP 404"))
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            with pytest.raises(RuntimeError):
                await collector.collect()
        assert len(calls) == 1  # not retried

    @pytest.mark.asyncio
    async def test_builds_correct_rss_url(self):
        collector = RedditCollector(_config(sort="top"))
        fetch, calls = _fetch(_feed([]))
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            await collector.collect()
        assert "/r/LocalLLaMA/top/.rss" in calls[0]
        assert "limit=5" in calls[0]
        assert "t=day" in calls[0]

    @pytest.mark.asyncio
    async def test_top_asks_for_the_day_whatever_the_lookback(self):
        # Reddit's window is relative to the fetch and runs are a day apart; t=week is a different
        # ranking whose older viral posts pushed today's out of the capped list.
        collector = RedditCollector(_config(sort="top", lookback_hours=30))
        fetch, calls = _fetch(_feed([]))
        with patch("collectors.base.fetch_feed", side_effect=fetch):
            await collector.collect()
        assert "t=day" in calls[0]


class TestExtractPostId:
    def test_from_comments_link(self):
        link = "https://www.reddit.com/r/LocalLLaMA/comments/xyz789/title/"
        assert RedditCollector._extract_post_id("t3_xyz789", link) == "xyz789"

    def test_from_entry_id_fallback(self):
        assert RedditCollector._extract_post_id("t3_abc", "https://example.com/no-match") == "abc"

    def test_generated_when_no_id(self):
        out = RedditCollector._extract_post_id("", "https://example.com/x")
        assert out and len(out) == 16
