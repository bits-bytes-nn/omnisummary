import json
from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock, patch

import httpx
import pytest
from botocore.exceptions import ClientError, NoCredentialsError, ReadTimeoutError

from collectors.base import (
    FEED_FETCH_HEADERS,
    BaseCollector,
    ParkedItems,
    ParkOutcome,
    TransientStatusError,
    collection_window_dates,
    dump_items_envelope,
    fetch_feed,
    fetch_feed_with_retry,
    gather_collector_results,
    load_items_from_s3,
    park_file_key,
    park_root_prefix,
    parse_feed_entries,
    recency_bucket,
)
from shared.constants import SourceType
from shared.models import CollectedItem

RSS_BODY = b"""<?xml version="1.0"?><rss version="2.0"><channel><title>Example</title>
<item><title>Post</title><link>https://example.com/p/1</link><guid>p1</guid>
<description>body</description><author>alice@example.com (alice)</author>
<pubDate>Tue, 02 Jun 2026 00:00:00 GMT</pubDate></item></channel></rss>"""


def _s3_client_returning(body_bytes: bytes) -> MagicMock:
    body = MagicMock()
    body.read.return_value = body_bytes
    client = MagicMock()
    client.get_object.return_value = {"Body": body}
    return client


def _item(item_id: str) -> CollectedItem:
    return CollectedItem(item_id=item_id, source_type=SourceType.WEB, title="t", url=f"http://e.com/{item_id}")


async def _ok(item_id: str) -> list[CollectedItem]:
    return [_item(item_id)]


async def _fail() -> list[CollectedItem]:
    raise RuntimeError("boom")


async def _empty() -> list[CollectedItem]:
    return []


class TestGatherCollectorResults:
    @pytest.mark.asyncio
    async def test_partial_failure_passes_through(self):
        result = await gather_collector_results([_ok("a"), _fail()], raise_if_all_failed=True)
        assert {i.item_id for i in result.items} == {"a"}

    @pytest.mark.asyncio
    async def test_all_failed_raises_when_flagged(self):
        with pytest.raises(RuntimeError, match="All 2 collector tasks failed"):
            await gather_collector_results([_fail(), _fail()], raise_if_all_failed=True)

    @pytest.mark.asyncio
    async def test_all_failed_silent_when_not_flagged(self):
        result = await gather_collector_results([_fail(), _fail()])
        assert result.items == []

    @pytest.mark.asyncio
    async def test_no_tasks_does_not_raise(self):
        result = await gather_collector_results([], raise_if_all_failed=True)
        assert result.items == [] and result.total == 0

    @pytest.mark.asyncio
    async def test_counts_are_returned_so_a_partial_outage_is_visible(self):
        # A source that answered from 1 of 4 inputs is indistinguishable from a healthy one by item
        # count alone — the counts are what let the caller report it DEGRADED.
        result = await gather_collector_results([_ok("a"), _fail(), _fail(), _empty()])
        assert (result.total, result.failed, result.empty) == (4, 2, 1)
        assert [i.item_id for i in result.items] == ["a"]


class _Collector(BaseCollector):
    async def collect(self):
        return []


class TestDegradationReporting:
    """degraded_detail/run_meta are shared by every collector so one source can't stay silent while
    another reports. Reporting only — nothing here filters items."""

    def test_run_health_records_meta_and_flags_a_majority_failure(self):
        c = _Collector()
        c.record_run_health(total=40, failed=30, empty=2, threshold=50.0, what="feeds", hint="cookies expired")
        assert c.run_meta == {"accounts_total": 40, "accounts_failed": 30, "accounts_empty": 2}
        assert "30/40 feeds failed" in c.degraded_detail
        assert "cookies expired" in c.degraded_detail

    def test_run_health_stays_quiet_below_the_threshold(self):
        c = _Collector()
        c.record_run_health(total=40, failed=5, empty=0, threshold=50.0, what="feeds")
        assert c.degraded_detail == ""
        assert c.run_meta["accounts_failed"] == 5  # still parked, so the sync's state is recorded

    def test_park_meta_written_by_a_half_dead_sync_reports_degraded(self):
        c = _Collector()
        parked = ParkedItems(
            outcome=ParkOutcome.FRESH,
            items=[_item("v1")],
            meta={"accounts_total": 12, "accounts_failed": 10},
        )
        c.flag_degraded_park(parked, threshold=50.0, what="channels")
        assert "10/12 channels failed" in c.degraded_detail

    def test_park_without_meta_is_never_reported_degraded(self):
        c = _Collector()
        c.flag_degraded_park(ParkedItems(outcome=ParkOutcome.FRESH, items=[_item("v1")]), threshold=50.0, what="x")
        assert c.degraded_detail == ""


class TestAMissingParkFileWhereItIsThePrimaryPath:
    """ParkOutcome.ABSENT is excluded from `degraded` by design — locally it means "collect live".
    In AWS, for YouTube, live collection yields metadata with NO transcripts, so a wrong S3_PREFIX, a
    deleted object or a sync that never ran produced items, reported OK, and dropped every transcript
    on every day."""

    ABSENT = ParkedItems(outcome=ParkOutcome.ABSENT, detail="no object at 's3://b/k/youtube_items.json'")

    def test_in_aws_an_absent_required_park_is_degraded_and_names_the_key(self):
        c = _Collector()
        with patch("collectors.base.is_running_in_aws", return_value=True):
            c.flag_missing_park(self.ABSENT, required=True, hint="NO transcripts")
        assert "park file required but absent" in c.degraded_detail
        assert "s3://b/k/youtube_items.json" in c.degraded_detail
        assert "NO transcripts" in c.degraded_detail

    def test_locally_it_stays_the_normal_path(self):
        c = _Collector()
        with patch("collectors.base.is_running_in_aws", return_value=False):
            c.flag_missing_park(self.ABSENT, required=True)
        assert c.degraded_detail == ""

    def test_a_source_that_does_not_require_a_park_file_is_unaffected(self):
        c = _Collector()
        with patch("collectors.base.is_running_in_aws", return_value=True):
            c.flag_missing_park(self.ABSENT, required=False)
        assert c.degraded_detail == ""

    def test_a_usable_park_file_is_not_a_missing_one(self):
        c = _Collector()
        with patch("collectors.base.is_running_in_aws", return_value=True):
            c.flag_missing_park(ParkedItems(outcome=ParkOutcome.FRESH, items=[_item("v1")]), required=True)
        assert c.degraded_detail == ""

    def test_the_absent_outcome_carries_why_there_was_no_file(self, monkeypatch):
        # A bare "absent" is not actionable; the detail is what lands in the health report.
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        assert "STATE_BUCKET" in load_items_from_s3("youtube_items.json").detail


class TestEmptyRateDegradation:
    """A source whose inputs all answer 200 with ZERO entries (expired RSSHub cookies, a paywalled
    200, a playlist that resolves to nothing) trips no failure rate at all — and reported a clean OK
    as long as ONE input still produced an item. And a rate cannot express the verdict for a source
    with few inputs: 1 of 2 subreddits is exactly 50%, 2 of 2 already raises FAILED."""

    def test_mostly_empty_inputs_are_degraded_when_configured(self):
        c = _Collector()
        c.record_run_health(total=40, failed=0, empty=39, threshold=50.0, empty_threshold=90.0, what="account feeds")
        assert "39/40 account feeds returned nothing" in c.degraded_detail

    def test_empty_inputs_are_ignored_by_default(self):
        # Off by default: many RSS blogs legitimately publish nothing on a given day, so the knob is
        # a per-source opt-in rather than a new daily alert for everyone.
        c = _Collector()
        c.record_run_health(total=40, failed=0, empty=39, threshold=50.0, what="feeds")
        assert c.degraded_detail == ""
        assert c.run_meta["accounts_empty"] == 39

    def test_absolute_count_is_disabled_by_default(self):
        c = _Collector()
        c.record_run_health(total=2, failed=1, empty=0, threshold=50.0, what="subreddits")
        assert c.degraded_detail == ""

    def test_park_path_sees_empty_inputs_too(self):
        # The park file records accounts_empty, which flag_degraded_park ignored entirely — so a
        # fresh park file written by a sync whose every feed came back empty read as healthy.
        c = _Collector()
        parked = ParkedItems(
            outcome=ParkOutcome.FRESH,
            items=[_item("v1")],
            meta={"accounts_total": 40, "accounts_failed": 0, "accounts_empty": 39},
        )
        c.flag_degraded_park(parked, threshold=50.0, empty_threshold=90.0, what="account feeds")
        assert "39/40 account feeds returned nothing" in c.degraded_detail
        assert c.degraded_detail.startswith("parked sync: ")


class _FakeClient:
    """Stands in for httpx.AsyncClient: records the kwargs it was constructed with and answers every
    GET with a fixed response or exception."""

    constructed: list[dict] = []

    def __init__(self, outcome, **kwargs):
        self.outcome = outcome
        type(self).constructed.append(kwargs)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get(self, url):
        self.requested = url
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def _client_factory(outcome):
    _FakeClient.constructed = []
    return lambda **kwargs: _FakeClient(outcome, **kwargs)


def _response(status: int = 200, content: bytes = RSS_BODY) -> httpx.Response:
    return httpx.Response(status_code=status, content=content)


class TestFetchFeed:
    """feedparser.parse(url) fetches through urllib with NO socket timeout, and asyncio.wait_for
    cannot cancel the asyncio.to_thread worker that ran it — so every timed-out attempt leaked a
    thread for the rest of the process's life. The body is fetched with httpx instead."""

    @pytest.mark.asyncio
    async def test_body_is_fetched_with_the_configured_timeout_then_parsed(self):
        with patch("collectors.base.httpx.AsyncClient", _client_factory(_response())):
            feed = await fetch_feed("https://example.com/feed", description="RSS feed", timeout=7)
        assert [e.get("title") for e in feed.entries] == ["Post"]
        assert _FakeClient.constructed[0]["timeout"] == 7
        assert _FakeClient.constructed[0]["headers"] == FEED_FETCH_HEADERS

    @pytest.mark.asyncio
    async def test_a_timeout_is_transient(self):
        with patch("collectors.base.httpx.AsyncClient", _client_factory(httpx.ReadTimeout("hung"))):
            with pytest.raises(TransientStatusError, match="timed out"):
                await fetch_feed("https://hang.example/feed", description="RSS feed", timeout=1)

    @pytest.mark.asyncio
    async def test_a_transport_failure_is_transient_not_a_dead_feed(self):
        # A DNS hiccup used to lose the feed for the whole day while an HTTP 503 got three attempts.
        with patch("collectors.base.httpx.AsyncClient", _client_factory(httpx.ConnectError("dns"))):
            with pytest.raises(TransientStatusError, match="fetch failed"):
                await fetch_feed("https://gone.example/feed", description="RSS feed", timeout=5)

    @pytest.mark.asyncio
    async def test_5xx_is_transient_and_4xx_is_a_verdict(self):
        with patch("collectors.base.httpx.AsyncClient", _client_factory(_response(status=503))):
            with pytest.raises(TransientStatusError, match="HTTP 503"):
                await fetch_feed("https://x.example/feed", description="RSS feed", timeout=5)
        with patch("collectors.base.httpx.AsyncClient", _client_factory(_response(status=403))):
            with pytest.raises(RuntimeError, match="HTTP 403") as exc:
                await fetch_feed("https://x.example/feed", description="RSS feed", timeout=5)
        assert not isinstance(exc.value, TransientStatusError)

    @pytest.mark.asyncio
    async def test_an_unparseable_body_is_a_permanent_failure(self):
        with patch("collectors.base.httpx.AsyncClient", _client_factory(_response(content=b"not xml at all"))):
            with pytest.raises(RuntimeError, match="Failed to parse") as exc:
                await fetch_feed("https://x.example/feed", description="RSS feed", timeout=5)
        assert not isinstance(exc.value, TransientStatusError)


class TestFetchFeedWithRetry:
    @pytest.mark.asyncio
    async def test_transient_failure_is_retried_then_succeeds(self):
        good = MagicMock(entries=[{"title": "t"}])
        outcomes = [TransientStatusError("HTTP 503"), good]

        async def _fetch(url, **kwargs):
            outcome = outcomes.pop(0)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        with patch("collectors.base.fetch_feed", side_effect=_fetch):
            feed = await fetch_feed_with_retry(
                "https://x.example/feed", description="RSS feed", timeout=1, max_retries=3, backoff_sec=0
            )
        assert feed is good
        assert outcomes == []

    @pytest.mark.asyncio
    async def test_a_permanent_failure_burns_no_retry_budget(self):
        with patch("collectors.base.fetch_feed", side_effect=RuntimeError("HTTP 404")) as fetch:
            with pytest.raises(RuntimeError, match="404"):
                await fetch_feed_with_retry(
                    "https://x.example/feed", description="RSS feed", timeout=1, max_retries=3, backoff_sec=0
                )
        assert fetch.call_count == 1

    @pytest.mark.asyncio
    async def test_retries_are_jittered_per_url(self):
        # Plain linear backoff resynchronises dozens of concurrent feed retries into exactly the
        # burst the upstream rate-limited, so the delay carries a per-URL offset.
        delays: list[float] = []

        async def _sleep(seconds):
            delays.append(seconds)

        with patch("collectors.base.fetch_feed", side_effect=TransientStatusError("HTTP 429")):
            with patch("shared.utils.asyncio.sleep", side_effect=_sleep):
                with pytest.raises(TransientStatusError):
                    await fetch_feed_with_retry(
                        "https://a.example/feed", description="a", timeout=1, max_retries=2, backoff_sec=5
                    )
                with pytest.raises(TransientStatusError):
                    await fetch_feed_with_retry(
                        "https://b.example/feed", description="b", timeout=1, max_retries=2, backoff_sec=5
                    )
        assert len(delays) == 2
        assert delays[0] != delays[1]
        assert all(5 <= d <= 10 for d in delays)

    @pytest.mark.asyncio
    async def test_proxy_fallback_is_opt_in(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_PROXY_URL", "https://proxy.example.com")
        monkeypatch.setenv("CLOUDFLARE_PROXY_TOKEN", "tok")
        good = MagicMock(entries=[{"title": "t"}])
        seen: list[str] = []

        async def _fetch(url, **kwargs):
            seen.append(url)
            if "proxy.example.com" not in url:
                raise RuntimeError("HTTP 403")
            return good

        with patch("collectors.base.fetch_feed", side_effect=_fetch):
            feed = await fetch_feed_with_retry(
                "https://www.reddit.com/r/x/.rss",
                description="Reddit feed",
                timeout=1,
                max_retries=1,
                backoff_sec=0,
                proxy_fallback=True,
            )
        assert feed is good
        assert len(seen) == 2


class TestUpstreamRecencyDerivation:
    """Both derivations exist so a source cannot ask its upstream for a NARROWER window than the
    pipeline believes it has: web_search truncated 30 hours to `days=1`, and Reddit pinned `t=day`."""

    def test_the_window_is_the_runs_own_and_anchored_to_the_reference_time(self):
        # `days`-style parameters are anchored to now, so a --date backfill searched today.
        assert collection_window_dates(30, datetime(2026, 6, 3, tzinfo=UTC)) == ("2026-06-01", "2026-06-03")

    def test_day_granularity_only_ever_widens_the_request(self):
        start, end = collection_window_dates(1, datetime(2026, 6, 3, 12, tzinfo=UTC))
        assert (start, end) == ("2026-06-03", "2026-06-03")

    @pytest.mark.parametrize(
        ("lookback_hours", "expected"),
        [(1, "hour"), (24, "day"), (30, "week"), (168, "week"), (200, "month"), (10_000, "year")],
    )
    def test_the_bucket_is_the_narrowest_one_that_covers_the_window(self, lookback_hours, expected):
        assert recency_bucket(lookback_hours) == expected


class TestParseFeedEntries:
    """RSS, RSSHub and Reddit carried byte-for-byte identical entry loops, so a fix to one silently
    left the other two behind."""

    # Window [2026-06-01, 2026-06-03]: the reference time is midnight at the END of the digest date,
    # so the upper bound is real and an entry past it belongs to a later digest.
    REFERENCE_TIME = datetime(2026, 6, 3, tzinfo=UTC)
    LOOKBACK_HOURS = 48

    @staticmethod
    def _entry(**overrides):
        entry = {
            "title": "Post",
            "link": "https://example.com/p/1",
            "id": "p1",
            "summary": "body",
            "author": "alice",
            "published_parsed": (2026, 6, 2, 0, 0, 0, 0, 0, 0),
        }
        entry.update(overrides)
        return _AttrDict(entry)

    def _parse(self, entries, **kwargs):
        feed = _AttrDict({"entries": entries})
        return parse_feed_entries(
            feed,
            source_type=kwargs.pop("source_type", SourceType.RSS),
            lookback_hours=kwargs.pop("lookback_hours", self.LOOKBACK_HOURS),
            reference_time=kwargs.pop("reference_time", self.REFERENCE_TIME),
            description="RSS feed 'f'",
            metadata=kwargs.pop("metadata", {"feed_url": "f"}),
            **kwargs,
        )

    def test_builds_an_item_from_an_entry(self):
        items = self._parse([self._entry()])
        assert len(items) == 1
        assert items[0].item_id == "p1"
        assert items[0].author == "alice"
        assert items[0].text == "body"
        assert items[0].metadata == {"feed_url": "f"}

    def test_entry_before_the_cutoff_is_dropped(self):
        old = self._entry(published_parsed=(2026, 5, 1, 0, 0, 0, 0, 0, 0))
        assert self._parse([old]) == []

    def test_entry_after_the_reference_time_is_dropped(self):
        # Only the parked path used to close the upper end, so a `--date` backfill of an older day
        # ingested TODAY's live items from every feed-based source alongside that day's.
        future = self._entry(published_parsed=(2026, 6, 4, 0, 0, 0, 0, 0, 0))
        assert self._parse([future]) == []

    def test_entry_without_a_date_is_kept(self):
        # A missing date is not evidence of falling outside the window; dropping it would silently
        # shrink a source over a metadata gap.
        undated = self._entry()
        del undated["published_parsed"]
        assert len(self._parse([undated])) == 1

    def test_full_content_beats_the_summary(self):
        entry = self._entry(content=[{"value": "full text"}])
        assert self._parse([entry])[0].text == "full text"

    def test_entry_without_an_id_falls_back_to_a_url_hash(self):
        item = self._parse([self._entry(id="")])[0]
        assert item.item_id and item.item_id != "p1"

    def test_a_malformed_entry_is_skipped_not_fatal(self):
        broken = self._entry(content="not a list of dicts")
        items = self._parse([broken, self._entry(id="p2", link="https://example.com/p/2")])
        assert [i.item_id for i in items] == ["p2"]

    def test_author_override_wins(self):
        # RSSHub attributes every item to the account, not to the entry's own author field.
        assert self._parse([self._entry()], author="karpathy")[0].author == "karpathy"

    def test_item_id_override_wins(self):
        # Reddit derives the post id from the permalink.
        items = self._parse([self._entry()], item_id_of=lambda entry, link: f"custom-{link[-1]}")
        assert items[0].item_id == "custom-1"


class _AttrDict(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class TestLoadItemsFromS3:
    def test_absent_without_bucket(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.ABSENT
        assert parked.usable is False and parked.degraded is False

    def test_reads_items_from_parent_prefix(self, monkeypatch):
        # S3_PREFIX is '<root>/digest_state'; parked items live one level up at '<root>/'.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        body = MagicMock()
        body.read.return_value = json.dumps(
            [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "https://y/v1", "text": "x"}]
        ).encode("utf-8")
        client = MagicMock()
        client.get_object.return_value = {"Body": body}
        with patch("collectors.base.boto3.client", return_value=client):
            parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.FRESH
        assert [i.item_id for i in parked.items] == ["v1"]
        assert client.get_object.call_args.kwargs["Key"] == "omnisummary/youtube_items.json"

    def test_reads_root_level_key_when_prefix_is_bare(self, monkeypatch):
        # With no configured root prefix the CDK sets S3_PREFIX='digest_state', and the sync
        # scripts write the park file at the bucket root ('<file>'). The reader used to look under
        # 'digest_state/<file>' and never found it, silently falling back to live collection.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "digest_state")
        client = _s3_client_returning(b"[]")
        with patch("collectors.base.boto3.client", return_value=client):
            load_items_from_s3("youtube_items.json")
        assert client.get_object.call_args.kwargs["Key"] == "youtube_items.json"

    def test_park_key_matches_between_writer_and_reader(self):
        # The sync scripts key off the config's aws.s3_prefix (bucket ROOT); the Lambda reader
        # derives that root from S3_PREFIX ('<root>/digest_state'). Both must land on one key.
        for root in ("omnisummary", ""):
            state_prefix = f"{root}/digest_state" if root else "digest_state"
            assert park_file_key("rsshub_items.json", root) == park_file_key(
                "rsshub_items.json", park_root_prefix(state_prefix)
            )
        assert park_file_key("rsshub_items.json", "omnisummary") == "omnisummary/rsshub_items.json"
        assert park_file_key("rsshub_items.json", "") == "rsshub_items.json"

    def test_missing_object_is_absent(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        client = MagicMock()
        client.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        with patch("collectors.base.boto3.client", return_value=client):
            parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.ABSENT
        assert parked.degraded is False  # a missing park file is routine, not a stale sync

    def test_unexpected_client_error_is_error_not_absent(self, monkeypatch):
        # An AccessDenied read used to be logged at info as "no items found" and looked identical
        # to an absent file. It must still fall through to live collection (never raise) but be
        # reported as a degraded park so the misconfiguration surfaces.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        client = MagicMock()
        client.get_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        with patch("collectors.base.boto3.client", return_value=client):
            parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.ERROR
        assert parked.usable is False and parked.degraded is True
        assert "AccessDenied" in parked.detail

    @pytest.mark.parametrize(
        "error",
        [NoCredentialsError(), ReadTimeoutError(endpoint_url="https://s3.example")],
        ids=["no-credentials", "read-timeout"],
    )
    def test_a_botocore_error_falls_back_instead_of_failing_the_source(self, monkeypatch, error):
        # Only ClientError was caught, so a missing credential chain or a connect/read timeout raised
        # straight out of the collector and failed the whole source — contradicting this function's
        # documented contract that EVERY S3 failure degrades to live collection.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        client = MagicMock()
        client.get_object.side_effect = error
        with patch("collectors.base.boto3.client", return_value=client):
            parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.ERROR
        assert parked.usable is False and parked.degraded is True

    def test_reads_envelope_shape(self, monkeypatch):
        # The newer {"generated_at", "items"} envelope must load like the legacy bare list.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        fresh = datetime.now(UTC).isoformat()
        body = json.dumps(
            {
                "generated_at": fresh,
                "items": [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "https://y/v1"}],
            }
        ).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            parked = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in parked.items] == ["v1"]
        assert parked.outcome == ParkOutcome.FRESH

    def test_stale_envelope_still_loads_but_warns(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        old = (datetime.now(UTC) - timedelta(hours=72)).isoformat()
        body = json.dumps(
            {"generated_at": old, "items": [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "u"}]}
        ).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            with patch("collectors.base.logger.warning") as warn:
                parked = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in parked.items] == ["v1"]  # stale beats empty
        assert parked.outcome == ParkOutcome.STALE
        assert parked.usable is True and parked.degraded is True  # used, but reported STALE
        assert parked.age_hours is not None and parked.age_hours > 36
        assert any("stalled" in str(c.args) for c in warn.call_args_list)

    def test_park_age_budget_is_configurable(self, monkeypatch):
        # A source whose sync runs less often can widen the window instead of alerting daily.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        old = (datetime.now(UTC) - timedelta(hours=48)).isoformat()
        body = json.dumps(
            {"generated_at": old, "items": [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "u"}]}
        ).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            parked = load_items_from_s3("youtube_items.json", max_age_hours=72)
        assert parked.outcome == ParkOutcome.FRESH  # 48h is inside a 72h budget

    def test_stale_empty_envelope_is_treated_as_absent(self, monkeypatch):
        # A park file that is BOTH empty and stale means the local sync stopped producing; falling
        # through to live collection lets a real outage report FAILED instead of silent EMPTY.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        old = (datetime.now(UTC) - timedelta(hours=72)).isoformat()
        body = json.dumps({"generated_at": old, "items": []}).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            parked = load_items_from_s3("rsshub_items.json")
        assert parked.outcome == ParkOutcome.ABSENT
        assert parked.usable is False  # -> live collection, so a real outage can report FAILED

    def test_fresh_empty_envelope_is_returned_not_absent(self, monkeypatch):
        # A legitimately quiet sync day must NOT fall through to live collection (which would
        # raise a false FAILED from a Lambda IP that the source blocks).
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        body = json.dumps({"generated_at": datetime.now(UTC).isoformat(), "items": []}).encode("utf-8")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(body)):
            parked = load_items_from_s3("rsshub_items.json")
        assert parked.outcome == ParkOutcome.FRESH
        assert parked.usable is True and parked.items == []
        assert parked.degraded is False  # a quiet sync day is not a stale sync

    def test_unstamped_empty_list_is_returned_not_absent(self, monkeypatch):
        # Legacy bare list carries no age, so it can't be proven stale → keep prior behavior.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(b"[]")):
            parked = load_items_from_s3("rsshub_items.json")
        assert parked.outcome == ParkOutcome.FRESH
        assert parked.items == [] and parked.age_hours is None

    def test_malformed_json_is_error(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(b"{not json")):
            parked = load_items_from_s3("youtube_items.json")
        assert parked.outcome == ParkOutcome.ERROR
        assert parked.usable is False and parked.degraded is True


class TestDumpItemsEnvelope:
    def test_roundtrips_through_loader(self, monkeypatch):
        items = [_item("v1"), _item("v2")]
        payload = dump_items_envelope(items).encode("utf-8")
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(payload)):
            loaded = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in loaded.items] == ["v1", "v2"]
        assert loaded.outcome == ParkOutcome.FRESH

    def test_meta_block_roundtrips_and_is_optional(self, monkeypatch):
        # The writer/reader contract for the OPTIONAL meta block: what a sync recorded about how it
        # went must survive the round trip, and a payload without it must still load.
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        with_meta = dump_items_envelope([_item("v1")], meta={"accounts_total": 40, "accounts_failed": 30}).encode()
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(with_meta)):
            loaded = load_items_from_s3("rsshub_items.json")
        assert loaded.meta == {"accounts_total": 40, "accounts_failed": 30}

        without_meta = dump_items_envelope([_item("v1")]).encode()
        assert "meta" not in json.loads(without_meta)
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(without_meta)):
            assert load_items_from_s3("rsshub_items.json").meta == {}

    def test_legacy_shapes_still_load_with_an_empty_meta(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "b")
        monkeypatch.setenv("S3_PREFIX", "omnisummary/digest_state")
        legacy_list = json.dumps(
            [{"item_id": "v1", "source_type": "youtube", "title": "T", "url": "https://y/v1"}]
        ).encode()
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(legacy_list)):
            parked = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in parked.items] == ["v1"]
        assert parked.meta == {}

        stamped_no_meta = json.dumps(
            {
                "generated_at": datetime.now(UTC).isoformat(),
                "items": [{"item_id": "v2", "source_type": "youtube", "title": "T", "url": "https://y/v2"}],
            }
        ).encode()
        with patch("collectors.base.boto3.client", return_value=_s3_client_returning(stamped_no_meta)):
            parked = load_items_from_s3("youtube_items.json")
        assert [i.item_id for i in parked.items] == ["v2"]
        assert parked.meta == {}
