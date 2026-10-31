from unittest.mock import AsyncMock

import pytest

from application import research as dlv
from application.research import DeliveryContext
from domain.formatting import normalize_citation_url
from domain.models import ImageAsset, ThreadsDelivery
from tests.fakes import FakeThreads


def _img(content_type="image/jpeg"):
    return ImageAsset(
        data=b"abc",
        source_url="https://src/article",
        image_url="https://cdn/x",
        content_type=content_type,
        alt="A Title",
    )


def _threads_config(d, *, max_posts: int = 8) -> None:
    d.config.agent.research_max_threads_posts = max_posts


class TestMarkdownReport:
    @pytest.mark.asyncio
    async def test_the_report_is_recorded_and_publishes_nothing(self):
        d = DeliveryContext()
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock()
        pt = d.ports.threads.post_thread
        ok = await dlv.deliver_research_report("# 제목\n\n본문이다.", channel="markdown", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        assert d.report == "# 제목\n\n본문이다."
        assert d.delivered_channels == {"markdown"}
        pt.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_revision_replaces_the_recorded_report(self):
        # Nothing external happened, so a corrected report is simply the better answer.
        d = DeliveryContext()
        await dlv.deliver_research_report("첫 보고서다.", channel="markdown", delivery=d)
        again = await dlv.deliver_research_report("고친 보고서다.", channel="markdown", delivery=d)
        assert again is dlv.DeliveryOutcome.POSTED
        assert d.report == "고친 보고서다."

    def test_threads_is_allowed_only_when_the_caller_asked(self):
        assert DeliveryContext().allowed_channels == {"markdown"}
        assert DeliveryContext(publish_threads=True).allowed_channels == {"markdown", "threads"}


class TestDeliverThreads:
    @pytest.mark.asyncio
    async def test_passes_first_image_with_content_type_and_key(self):
        d = DeliveryContext(publish_threads=True, staged_images=[_img("image/jpeg"), _img("image/png")])
        d.seen_urls.add(normalize_citation_url("https://src/article"))
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(1, 1))
        pt = d.ports.threads.post_thread
        _threads_config(d)
        ok = await dlv.deliver_research_report("리드 문장이다.\nhttps://src/article", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        kw = pt.await_args.kwargs
        assert kw["image_content_type"] == "image/jpeg"  # the FIRST staged image
        assert kw["image_bytes"] == b"abc"

    @pytest.mark.asyncio
    async def test_empty_report_skips_threads_api(self):
        # An empty report must not call post_to_threads (an empty root 400s the Threads API).
        d = DeliveryContext(publish_threads=True)
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock()
        pt = d.ports.threads.post_thread
        _threads_config(d, max_posts=6)
        ok = await dlv.deliver_research_report("   \n\n ", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.FAILED
        pt.assert_not_awaited()
        assert "threads" not in d.delivered_channels

    @pytest.mark.asyncio
    async def test_no_images_text_only(self):
        d = DeliveryContext(publish_threads=True)
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(1, 1))
        pt = d.ports.threads.post_thread
        await dlv.deliver_research_report("리드.", channel="threads", delivery=d)
        assert pt.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_a_repeat_threads_call_posts_nothing_and_says_so(self):
        # The likely reason for a second call is a REVISED report; announcing it as delivered told
        # the requester a correction had gone out that never did.
        d = DeliveryContext(publish_threads=True)
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(1, 1))
        pt = d.ports.threads.post_thread
        first = await dlv.deliver_research_report("첫 보고서다.", channel="threads", delivery=d)
        second = await dlv.deliver_research_report("고친 보고서다.", channel="threads", delivery=d)
        assert first is dlv.DeliveryOutcome.POSTED
        assert second is dlv.DeliveryOutcome.NOT_POSTED
        assert pt.await_count == 1

    @pytest.mark.asyncio
    async def test_an_image_from_a_page_the_thread_does_not_cite_is_not_posted(self):
        # The image is shown as the thread's own; one from an uncited page named no source for it.
        uncited = _img().model_copy(update={"source_url": "https://elsewhere/page"})
        d = DeliveryContext(publish_threads=True, staged_images=[uncited, _img("image/png")])
        d.seen_urls |= {normalize_citation_url("https://src/article"), normalize_citation_url("https://elsewhere/page")}
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(1, 1))
        pt = d.ports.threads.post_thread
        _threads_config(d)
        await dlv.deliver_research_report("리드.\nhttps://src/article", channel="threads", delivery=d)
        assert pt.await_args.kwargs["image_content_type"] == "image/png"  # the cited one, not the first

    @pytest.mark.asyncio
    async def test_a_citation_with_a_korean_particle_still_carries_its_image(self):
        # The citation guard strips a glued particle ("...article에서"); the image match must too.
        d = DeliveryContext(publish_threads=True, staged_images=[_img()])
        d.seen_urls.add(normalize_citation_url("https://src/article"))
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(1, 1))
        pt = d.ports.threads.post_thread
        _threads_config(d)
        await dlv.deliver_research_report("https://src/article에서 밝혔다.", channel="threads", delivery=d)
        assert pt.await_args.kwargs["image_bytes"] == b"abc"

    @pytest.mark.asyncio
    async def test_the_per_channel_outcome_is_kept_for_the_caller(self):
        d = DeliveryContext(publish_threads=True)
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(4, 6))
        _threads_config(d)
        await dlv.deliver_research_report("리드다.", channel="threads", delivery=d)
        await dlv.deliver_research_report("# 보고서", channel="markdown", delivery=d)
        stats = d.threads_stats
        assert (stats.delivered, stats.rendered) == (4, 6)


class TestARootThatLandedAloneIsNotAPlainFailure:
    """ThreadsDelivery.published is `posted > 1` when more than one post was intended, so a run that
    posted the ROOT and nothing else came back as a plain failure with nothing recorded — a retry
    would have posted a second root over the first."""

    @pytest.mark.asyncio
    async def test_a_lone_root_spends_the_channel(self):
        d = DeliveryContext(publish_threads=True)
        report = "\n---\n".join(f"{i}/6 소제목\n\n본문 {i}이다." for i in range(6))
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(1, 8))
        pt = d.ports.threads.post_thread
        _threads_config(d)
        ok = await dlv.deliver_research_report(report, channel="threads", delivery=d)
        again = await dlv.deliver_research_report(report, channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        assert again is dlv.DeliveryOutcome.NOT_POSTED
        assert pt.await_count == 1
        assert d.partial_channels == {"threads"}
        assert d.delivered_channels == set()

    @pytest.mark.asyncio
    async def test_a_root_that_may_be_live_spends_the_channel(self):
        # A publish step that failed ambiguously may have posted: a second call would be a second root.
        d = DeliveryContext(publish_threads=True)
        report = "\n---\n".join(f"{i}/6 소제목\n\n본문 {i}이다." for i in range(6))
        outcome = ThreadsDelivery(0, 8, uncertain=True)
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=outcome)
        pt = d.ports.threads.post_thread
        _threads_config(d)
        await dlv.deliver_research_report(report, channel="threads", delivery=d)
        again = await dlv.deliver_research_report(report, channel="threads", delivery=d)
        assert again is dlv.DeliveryOutcome.NOT_POSTED
        assert pt.await_count == 1
        assert d.partial_channels == {"threads"}

    @pytest.mark.asyncio
    async def test_a_delivery_where_nothing_landed_is_still_a_failure(self):
        d = DeliveryContext(publish_threads=True)
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(0, 8))
        _threads_config(d)
        ok = await dlv.deliver_research_report("리드다.", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.FAILED
        assert d.partial_channels == set() and d.delivered_channels == set()


class TestATooLongThreadNeverPosts:
    """A rewrite that still overflowed used to post cut, renumbered "1/10 … 10/10" without its dropped
    conclusion, so the reader could not tell. Every over-long text is sent back; none is published."""

    @pytest.mark.asyncio
    async def test_every_over_long_text_is_sent_back(self):
        d = DeliveryContext(publish_threads=True)
        report = "\n---\n".join(f"{i}/20 소제목\n\n본문 {i}이다." for i in range(20))
        d.ports.threads = FakeThreads()
        d.ports.threads.post_thread = AsyncMock(return_value=ThreadsDelivery(4, 4))
        pt = d.ports.threads.post_thread
        _threads_config(d, max_posts=4)
        first = await dlv.deliver_research_report(report, channel="threads", delivery=d)
        rewrite = await dlv.deliver_research_report(report + "\n---\n덧붙임", channel="threads", delivery=d)
        fits = await dlv.deliver_research_report("\n---\n".join(["a", "b", "c"]), channel="threads", delivery=d)
        assert first is rewrite is dlv.DeliveryOutcome.TOO_LONG
        assert fits is dlv.DeliveryOutcome.POSTED and pt.await_count == 1

    def test_a_trimmed_post_alone_is_too_long(self):
        # One post over the cap and no dropped post: the trim alone sends it back.
        assert dlv.sends_back_as_too_long("소제목\n\n" + "문장이다. " * 120, DeliveryContext())
        assert not dlv.sends_back_as_too_long("소제목\n\n짧은 본문이다.", DeliveryContext())
