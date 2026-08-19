import threading
import time
from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import httpx
import pytest

from collectors.base import ParkedItems, ParkOutcome
from collectors.rsshub import RSSHubCollector
from shared.config import RSSHubAccount, RSSHubCollectorConfig
from shared.constants import SourceType
from shared.models import CollectedItem


def _absent_park() -> ParkedItems:
    """No S3 park file → the collector must fall through to live collection."""
    return ParkedItems(outcome=ParkOutcome.ABSENT)


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
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch("httpx.get", side_effect=httpx.ConnectError("connection refused")):
                with pytest.raises(RuntimeError, match="unreachable"):
                    await c.collect()

    @pytest.mark.asyncio
    async def test_5xx_raises(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config())
        resp = MagicMock(status_code=503)
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch("httpx.get", return_value=resp):
                with pytest.raises(RuntimeError, match="503"):
                    await c.collect()

    @pytest.mark.asyncio
    async def test_reachable_proceeds(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config())
        resp = MagicMock(status_code=200)
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
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

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
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
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
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

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch("httpx.get", return_value=resp):
                with patch.object(c, "_parse_feed", side_effect=_parse):
                    items = await c.collect()
        assert [i.item_id for i in items] == ["t1"]


class TestAccountRetries:
    """An account feed used to get exactly ONE attempt: with ~41 accounts on the largest source, a
    single transient blip dropped that author for the day and could push the source past
    error_rate_threshold."""

    @pytest.mark.asyncio
    async def test_transient_status_is_retried_then_succeeds(self, monkeypatch):
        from collectors.base import TransientStatusError

        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config(max_retries=3))
        outcomes: list = [TransientStatusError("returned HTTP 502"), [_item("t1")]]

        def _parse(feed_url, username, platform):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch("httpx.get", return_value=MagicMock(status_code=200)):
                with patch.object(c, "_parse_feed", side_effect=_parse):
                    items = await c.collect()
        assert [i.item_id for i in items] == ["t1"]
        assert outcomes == []  # both the failed attempt and the retry ran

    @pytest.mark.asyncio
    async def test_permanent_failure_is_not_retried(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config(max_retries=3))
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch("httpx.get", return_value=MagicMock(status_code=200)):
                with patch.object(c, "_parse_feed", side_effect=RuntimeError("bozo")) as parse:
                    with pytest.raises(RuntimeError, match="All 1 RSSHub feeds failed"):
                        await c.collect()
        assert parse.call_count == 1  # a verdict, not a blip

    @pytest.mark.asyncio
    async def test_hung_feed_is_retried_with_a_full_timeout_each_attempt(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config(max_retries=3))
        timeouts: list[int] = []
        attempts = {"n": 0}

        async def _timeout_then_pass(awaitable, timeout):
            timeouts.append(timeout)
            attempts["n"] += 1
            if attempts["n"] == 1:
                awaitable.close()
                raise TimeoutError
            return await awaitable

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch("httpx.get", return_value=MagicMock(status_code=200)):
                with patch.object(c, "_parse_feed", return_value=[_item("t1")]):
                    with patch("collectors.rsshub.asyncio.wait_for", side_effect=_timeout_then_pass):
                        items = await c.collect()
        assert [i.item_id for i in items] == ["t1"]
        assert timeouts == [c.config.request_timeout, c.config.request_timeout]


class TestFailureHint:
    """The cookie hint used to be asserted unconditionally, sending ops to a Twitter container
    setting even when the failing feeds were on another platform."""

    @pytest.mark.asyncio
    async def test_hint_names_twitter_cookies_when_x_feeds_fail(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        accounts = [RSSHubAccount(username="a", platform="x"), RSSHubAccount(username="b", platform="x")]
        c = RSSHubCollector(_config(accounts=accounts, error_rate_threshold=40.0))

        def _parse(feed_url, username, platform):
            if username == "a":
                return [_item("a")]
            raise RuntimeError("feed down")

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch("httpx.get", return_value=MagicMock(status_code=200)):
                with patch.object(c, "_parse_feed", side_effect=_parse):
                    await c.collect()
        assert "TWITTER_AUTH_TOKEN" in c.degraded_detail

    @pytest.mark.asyncio
    async def test_no_twitter_hint_when_the_failing_feeds_are_another_platform(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        accounts = [
            RSSHubAccount(username="a", platform="mastodon"),
            RSSHubAccount(username="b", platform="mastodon"),
        ]
        c = RSSHubCollector(_config(accounts=accounts, error_rate_threshold=40.0))

        def _parse(feed_url, username, platform):
            if username == "a":
                return [_item("a")]
            raise RuntimeError("feed down")

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch("httpx.get", return_value=MagicMock(status_code=200)):
                with patch.object(c, "_parse_feed", side_effect=_parse):
                    await c.collect()
        assert "1/2 account feeds failed" in c.degraded_detail
        assert "TWITTER" not in c.degraded_detail


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

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch("httpx.get", return_value=resp):
                with patch.object(c, "_parse_feed", side_effect=_parse):
                    items = await c.collect()
        assert len(items) == 6  # every account still collected
        assert state["peak"] <= 2


class _Feed(dict):
    """feedparser-style dict with attribute access (mirrors tests/test_rss_collector.py)."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


def _entry(**kwargs) -> _Feed:
    base = {
        "title": "a post",
        "link": "https://x.com/karpathy/status/1",
        "id": "tweet-1",
        "published": "Tue, 02 Jun 2026 12:00:00 GMT",
        "summary": "post body",
    }
    base.update(kwargs)
    return _Feed(base)


def _feed(entries, *, bozo=False) -> _Feed:
    return _Feed(entries=entries, bozo=bozo, bozo_exception=Exception("x") if bozo else None)


class TestFeedParsing:
    def test_parses_entry_into_x_item(self):
        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.feedparser.parse", return_value=_feed([_entry()])):
            items = c._parse_feed("http://localhost:1200/twitter/user/karpathy", "karpathy", "x")
        assert len(items) == 1
        item = items[0]
        assert item.source_type == SourceType.X
        assert item.item_id == "tweet-1"
        assert item.author == "karpathy"
        assert item.text == "post body"
        assert item.metadata == {
            "rsshub_feed": "http://localhost:1200/twitter/user/karpathy",
            "platform": "x",
        }

    def test_prefers_content_over_summary(self):
        c = RSSHubCollector(_config())
        entry = _entry(content=[{"value": "rich body"}])
        with patch("collectors.rsshub.feedparser.parse", return_value=_feed([entry])):
            items = c._parse_feed("u", "karpathy", "x")
        assert items[0].text == "rich body"

    def test_entry_outside_the_window_is_dropped(self):
        c = RSSHubCollector(_config())
        old = _entry(published="Mon, 01 Jun 2026 00:00:00 GMT")  # reference 2026-06-03, lookback 24h
        with patch("collectors.rsshub.feedparser.parse", return_value=_feed([old])):
            assert c._parse_feed("u", "karpathy", "x") == []

    def test_entry_without_id_falls_back_to_a_url_hash(self):
        c = RSSHubCollector(_config())
        entry = _entry(id="")
        with patch("collectors.rsshub.feedparser.parse", return_value=_feed([entry])):
            items = c._parse_feed("u", "karpathy", "x")
        assert items[0].item_id  # generated from the link, never empty

    def test_unparseable_feed_raises_for_health(self):
        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.feedparser.parse", return_value=_feed([], bozo=True)):
            with pytest.raises(RuntimeError, match="Failed to parse RSSHub feed"):
                c._parse_feed("u", "karpathy", "x")

    def test_rsshub_own_error_status_is_classified(self):
        # RSSHub's own status was never inspected, so a 502 with an empty body surfaced as a generic
        # unparseable-feed error and was never retried.
        from collectors.base import TransientStatusError

        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.feedparser.parse", return_value=_Feed(entries=[], bozo=False, status=502)):
            with pytest.raises(TransientStatusError, match="returned HTTP 502"):
                c._parse_feed("u", "karpathy", "x")

    def test_permanent_status_is_not_transient(self):
        from collectors.base import TransientStatusError

        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.feedparser.parse", return_value=_Feed(entries=[], bozo=False, status=404)):
            with pytest.raises(RuntimeError, match="returned HTTP 404") as exc:
                c._parse_feed("u", "karpathy", "x")
        assert not isinstance(exc.value, TransientStatusError)

    def test_bozo_with_entries_is_still_parsed(self):
        # feedparser sets bozo on minor XML issues but still returns entries.
        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.feedparser.parse", return_value=_feed([_entry()], bozo=True)):
            assert len(c._parse_feed("u", "karpathy", "x")) == 1

    def test_malformed_entry_is_skipped_not_fatal(self):
        # A structurally broken entry (content that isn't feedparser's list-of-dicts) is skipped;
        # the healthy sibling in the same feed still lands.
        c = RSSHubCollector(_config())
        broken = _entry(id="broken", content="not-a-list")
        with patch("collectors.rsshub.feedparser.parse", return_value=_feed([broken, _entry()])):
            items = c._parse_feed("u", "karpathy", "x")
        assert [i.item_id for i in items] == ["tweet-1"]

    def test_field_less_entry_still_yields_an_item(self):
        # Pins current behavior: a title/link-less entry is NOT dropped here — the aggregator's
        # "missing a url or title" guard is what removes it before ranking.
        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.feedparser.parse", return_value=_feed([_Feed()])):
            items = c._parse_feed("u", "karpathy", "x")
        assert len(items) == 1 and items[0].url == "" and items[0].title == ""


class TestFeedRouting:
    def test_twitter_platforms_map_to_the_twitter_route(self):
        assert RSSHubCollector._build_feed_path("karpathy", "x") == "twitter/user/karpathy"
        assert RSSHubCollector._build_feed_path("karpathy", "Twitter") == "twitter/user/karpathy"

    def test_other_platform_keeps_its_own_route(self):
        assert RSSHubCollector._build_feed_path("someone", "Mastodon") == "mastodon/user/someone"

    def test_source_type_is_x_only_for_twitter_platforms(self):
        assert RSSHubCollector._detect_source_type("x") == SourceType.X
        assert RSSHubCollector._detect_source_type("mastodon") == SourceType.WEB


class TestParkedItems:
    @pytest.mark.asyncio
    async def test_park_file_short_circuits_live_collection(self):
        # The AWS path: X is collected locally and parked in S3, so the Lambda must not fetch live
        # (a datacenter IP gets blocked) and must not even probe RSSHub for reachability.
        c = RSSHubCollector(_config())
        parked = ParkedItems(outcome=ParkOutcome.FRESH, items=[_item("parked")])
        with patch("collectors.rsshub.load_items_from_s3", return_value=parked):
            with patch.object(c, "_check_reachable") as reach:
                items = await c.collect()
        assert [i.item_id for i in items] == ["parked"]
        reach.assert_not_called()
        assert c.park_status is not None and c.park_status.degraded is False

    @pytest.mark.asyncio
    async def test_stale_park_items_are_used_and_flagged(self):
        c = RSSHubCollector(_config())
        stale = ParkedItems(outcome=ParkOutcome.STALE, items=[_item("old")], age_hours=72.0, detail="72h")
        with patch("collectors.rsshub.load_items_from_s3", return_value=stale):
            items = await c.collect()
        assert [i.item_id for i in items] == ["old"]
        assert c.park_status is not None and c.park_status.degraded is True

    @pytest.mark.asyncio
    async def test_park_age_budget_comes_from_config(self):
        c = RSSHubCollector(_config(park_max_age_hours=72))
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()) as load:
            with patch.object(c, "_check_reachable"):
                with patch.object(c, "_parse_feed", return_value=[]):
                    await c.collect()
        assert load.call_args.kwargs["max_age_hours"] == 72

    @pytest.mark.asyncio
    async def test_disabled_collector_short_circuits(self):
        c = RSSHubCollector(_config(enabled=False))
        with patch("collectors.rsshub.load_items_from_s3") as load:
            assert await c.collect() == []
        load.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_accounts_configured_returns_empty(self):
        c = RSSHubCollector(_config(accounts=[]))
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch.object(c, "_check_reachable") as reach:
                assert await c.collect() == []
        reach.assert_not_called()


class TestDegradedReporting:
    """A source can shrink to a fraction of its feeds and still return items on time. Reported with
    the SAME error_rate_threshold the live warning already uses — no second ratio knob."""

    @pytest.mark.asyncio
    async def test_park_meta_over_the_threshold_flags_degraded(self):
        c = RSSHubCollector(_config(error_rate_threshold=50.0))
        parked = ParkedItems(
            outcome=ParkOutcome.FRESH,
            items=[_item("parked")],
            meta={"accounts_total": 40, "accounts_failed": 30},
        )
        with patch("collectors.rsshub.load_items_from_s3", return_value=parked):
            items = await c.collect()
        # The items are untouched — DEGRADED changes reporting only.
        assert [i.item_id for i in items] == ["parked"]
        assert "10/40" in c.degraded_detail

    @pytest.mark.asyncio
    async def test_park_meta_under_the_threshold_is_clean(self):
        c = RSSHubCollector(_config(error_rate_threshold=50.0))
        parked = ParkedItems(
            outcome=ParkOutcome.FRESH,
            items=[_item("parked")],
            meta={"accounts_total": 40, "accounts_failed": 2},
        )
        with patch("collectors.rsshub.load_items_from_s3", return_value=parked):
            await c.collect()
        assert c.degraded_detail == ""

    @pytest.mark.asyncio
    async def test_legacy_park_file_without_meta_is_not_flagged(self):
        c = RSSHubCollector(_config())
        parked = ParkedItems(outcome=ParkOutcome.FRESH, items=[_item("parked")])
        with patch("collectors.rsshub.load_items_from_s3", return_value=parked):
            await c.collect()
        assert c.degraded_detail == ""

    @pytest.mark.asyncio
    async def test_live_run_records_meta_and_flags_a_mostly_failed_fan_out(self):
        accounts = [RSSHubAccount(username=f"u{i}", platform="x") for i in range(4)]
        c = RSSHubCollector(_config(accounts=accounts, error_rate_threshold=50.0))

        def _parse(feed_url: str, username: str, platform: str):
            if username == "u0":
                return [_item(username)]
            raise RuntimeError("feed down")

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch.object(c, "_check_reachable"):
                with patch.object(c, "_parse_feed", side_effect=_parse):
                    items = await c.collect()
        assert [i.item_id for i in items] == ["u0"]
        assert c.run_meta == {"accounts_total": 4, "accounts_failed": 3, "accounts_empty": 0}
        assert "3/4" in c.degraded_detail
