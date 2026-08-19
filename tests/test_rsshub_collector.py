import asyncio
from datetime import UTC, datetime
from unittest.mock import patch

import httpx
import pytest

from collectors.base import ParkedItems, ParkOutcome, TransientStatusError
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


class _FakeClient:
    """Stands in for httpx.AsyncClient in the reachability probe."""

    def __init__(self, outcome, **kwargs):
        self.outcome = outcome

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _probe(outcome):
    """Patch target for the reachability probe's httpx client."""
    return patch("collectors.rsshub.httpx.AsyncClient", lambda **kwargs: _FakeClient(outcome, **kwargs))


def _reachable():
    return _probe(httpx.Response(status_code=200))


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


def _feed(entries) -> _Feed:
    return _Feed(entries=entries, bozo=False)


def _fetch_by_username(mapping):
    """Stand-in for collectors.base.fetch_feed keyed by the account in the RSSHub route path."""

    async def _fetch(url, **kwargs):
        username = url.rsplit("/", 1)[-1]
        outcome = mapping[username]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return _fetch


class TestReachability:
    @pytest.mark.asyncio
    async def test_unreachable_service_raises(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _probe(httpx.ConnectError("connection refused")):
                with pytest.raises(RuntimeError, match="unreachable"):
                    await c.collect()

    @pytest.mark.asyncio
    async def test_5xx_raises(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _probe(httpx.Response(status_code=503)):
                with pytest.raises(RuntimeError, match="503"):
                    await c.collect()

    @pytest.mark.asyncio
    async def test_reachable_proceeds(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config())
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _reachable():
                with patch("collectors.base.fetch_feed", return_value=_feed([])):
                    items = await c.collect()
        assert items == []  # reachable but no recent posts -> empty (not raised)

    @pytest.mark.asyncio
    async def test_hung_feed_times_out_and_is_skipped(self, monkeypatch):
        # A feed host that never returns must not take the source down: its own timeout fails THAT
        # feed while the healthy account still delivers, and collect() neither hangs nor raises.
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(
            _config(
                accounts=[RSSHubAccount(username="hangs", platform="x"), RSSHubAccount(username="ok", platform="x")],
                request_timeout=1,
                max_retries=1,
            )
        )
        fetch = _fetch_by_username(
            {"hangs": TransientStatusError("RSSHub feed timed out after 1s"), "ok": _feed([_entry()])}
        )
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _reachable():
                with patch("collectors.base.fetch_feed", side_effect=fetch):
                    with patch("collectors.rsshub.logger") as log:
                        items = await c.collect()
        assert [i.item_id for i in items] == ["tweet-1"]  # hung feed skipped, healthy one kept
        assert any("timed out" in str(call.args) for call in log.warning.call_args_list)

    @pytest.mark.asyncio
    async def test_all_accounts_failing_raises(self, monkeypatch):
        # Reachable service but nothing parseable on ANY account is an outage, not a quiet day:
        # it must surface as FAILED instead of a silent empty result.
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(
            _config(accounts=[RSSHubAccount(username="a", platform="x"), RSSHubAccount(username="b", platform="x")])
        )
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _reachable():
                with patch("collectors.base.fetch_feed", side_effect=RuntimeError("bozo")):
                    with pytest.raises(RuntimeError, match="All 2 RSSHub feeds failed"):
                        await c.collect()

    @pytest.mark.asyncio
    async def test_partial_failure_keeps_succeeding_accounts(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(
            _config(accounts=[RSSHubAccount(username="bad", platform="x"), RSSHubAccount(username="ok", platform="x")])
        )
        fetch = _fetch_by_username({"bad": RuntimeError("bozo"), "ok": _feed([_entry()])})
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _reachable():
                with patch("collectors.base.fetch_feed", side_effect=fetch):
                    items = await c.collect()
        assert [i.item_id for i in items] == ["tweet-1"]


class TestAccountRetries:
    """An account feed used to get exactly ONE attempt: with ~41 accounts on the largest source, a
    single transient blip dropped that author for the day and could push the source past
    error_rate_threshold."""

    @pytest.mark.asyncio
    async def test_transient_status_is_retried_then_succeeds(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config(max_retries=3))
        outcomes: list = [TransientStatusError("returned HTTP 502"), _feed([_entry()])]

        async def _fetch(url, **kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _reachable():
                with patch("collectors.base.fetch_feed", side_effect=_fetch):
                    items = await c.collect()
        assert [i.item_id for i in items] == ["tweet-1"]
        assert outcomes == []  # both the failed attempt and the retry ran

    @pytest.mark.asyncio
    async def test_permanent_failure_is_not_retried(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(_config(max_retries=3))
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _reachable():
                with patch("collectors.base.fetch_feed", side_effect=RuntimeError("bozo")) as fetch:
                    with pytest.raises(RuntimeError, match="All 1 RSSHub feeds failed"):
                        await c.collect()
        assert fetch.call_count == 1  # a verdict, not a blip

    @pytest.mark.asyncio
    async def test_every_attempt_carries_the_configured_timeout(self):
        c = RSSHubCollector(_config(max_retries=3, request_timeout=9))
        timeouts: list[float] = []
        outcomes: list = [TransientStatusError("returned HTTP 502"), _feed([_entry()])]

        async def _fetch(url, *, description, timeout):
            timeouts.append(timeout)
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch.object(c, "_check_reachable"):
                with patch("collectors.base.fetch_feed", side_effect=_fetch):
                    items = await c.collect()
        assert [i.item_id for i in items] == ["tweet-1"]
        assert timeouts == [9, 9]


class TestFailureHint:
    """The cookie hint used to be asserted unconditionally, sending ops to a Twitter container
    setting even when the failing feeds were on another platform."""

    @pytest.mark.asyncio
    async def test_hint_names_twitter_cookies_when_x_feeds_fail(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        accounts = [RSSHubAccount(username="a", platform="x"), RSSHubAccount(username="b", platform="x")]
        c = RSSHubCollector(_config(accounts=accounts, error_rate_threshold=40.0))
        fetch = _fetch_by_username({"a": _feed([_entry()]), "b": RuntimeError("feed down")})
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _reachable():
                with patch("collectors.base.fetch_feed", side_effect=fetch):
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
        fetch = _fetch_by_username({"a": _feed([_entry()]), "b": RuntimeError("feed down")})
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _reachable():
                with patch("collectors.base.fetch_feed", side_effect=fetch):
                    await c.collect()
        assert "1/2 account feeds failed" in c.degraded_detail
        assert "TWITTER" not in c.degraded_detail


class TestFanOutBound:
    @pytest.mark.asyncio
    async def test_concurrency_is_bounded_by_config(self, monkeypatch):
        # Unbounded fan-out let a feed's timeout expire before its fetch had even started.
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        c = RSSHubCollector(
            _config(
                accounts=[RSSHubAccount(username=f"u{i}", platform="x") for i in range(6)],
                max_concurrency=2,
            )
        )
        state = {"active": 0, "peak": 0}

        async def _fetch(url, **kwargs):
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            await asyncio.sleep(0.01)
            state["active"] -= 1
            return _feed([_entry()])

        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with _reachable():
                with patch("collectors.base.fetch_feed", side_effect=_fetch):
                    items = await c.collect()
        assert len(items) == 6  # every account still collected
        assert state["peak"] <= 2


class TestFeedParsing:
    """What the shared entry parser produces for an RSSHub route (the loop itself is covered in
    tests/test_base_collector.py)."""

    async def _collect_one(self, feed, **config_kwargs):
        c = RSSHubCollector(_config(**config_kwargs))
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch.object(c, "_check_reachable"):
                with patch("collectors.base.fetch_feed", return_value=feed):
                    return await c.collect()

    @pytest.mark.asyncio
    async def test_parses_entry_into_x_item(self):
        items = await self._collect_one(_feed([_entry()]))
        assert len(items) == 1
        item = items[0]
        assert item.source_type == SourceType.X
        assert item.item_id == "tweet-1"
        assert item.author == "karpathy"  # the ACCOUNT, not the entry's own author field
        assert item.text == "post body"
        assert item.metadata == {
            "rsshub_feed": "http://localhost:1200/twitter/user/karpathy",
            "platform": "x",
        }

    @pytest.mark.asyncio
    async def test_entry_outside_the_window_is_dropped(self):
        old = _entry(published="Mon, 01 Jun 2026 00:00:00 GMT")  # reference 2026-06-03, lookback 24h
        assert await self._collect_one(_feed([old])) == []

    @pytest.mark.asyncio
    async def test_mastodon_account_is_a_web_item(self):
        items = await self._collect_one(
            _feed([_entry()]), accounts=[RSSHubAccount(username="someone", platform="mastodon")]
        )
        assert items[0].source_type == SourceType.WEB


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
                with patch("collectors.base.fetch_feed", return_value=_feed([])):
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
    the SAME thresholds the live warning uses — no second set of knobs."""

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
        assert "30/40 account feeds failed" in c.degraded_detail

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
    async def test_a_park_file_whose_every_feed_came_back_empty_is_flagged(self):
        # Expired X cookies make every account feed answer 200 with no entries: no failure rate
        # trips, and the fresh park file used to read as perfectly healthy.
        c = RSSHubCollector(_config(empty_rate_threshold=90.0))
        parked = ParkedItems(
            outcome=ParkOutcome.FRESH,
            items=[_item("parked")],
            meta={"accounts_total": 40, "accounts_failed": 0, "accounts_empty": 40},
        )
        with patch("collectors.rsshub.load_items_from_s3", return_value=parked):
            await c.collect()
        assert "40/40 account feeds returned nothing" in c.degraded_detail
        assert "TWITTER_AUTH_TOKEN" in c.degraded_detail  # the actionable cause for an X deployment

    @pytest.mark.asyncio
    async def test_live_run_records_meta_and_flags_a_mostly_failed_fan_out(self):
        accounts = [RSSHubAccount(username=f"u{i}", platform="x") for i in range(4)]
        c = RSSHubCollector(_config(accounts=accounts, error_rate_threshold=50.0))
        fetch = _fetch_by_username(
            {
                "u0": _feed([_entry()]),
                "u1": RuntimeError("down"),
                "u2": RuntimeError("down"),
                "u3": RuntimeError("down"),
            }
        )
        with patch("collectors.rsshub.load_items_from_s3", return_value=_absent_park()):
            with patch.object(c, "_check_reachable"):
                with patch("collectors.base.fetch_feed", side_effect=fetch):
                    items = await c.collect()
        assert [i.item_id for i in items] == ["tweet-1"]
        assert c.run_meta == {"accounts_total": 4, "accounts_failed": 3, "accounts_empty": 0}
        assert "3/4" in c.degraded_detail
