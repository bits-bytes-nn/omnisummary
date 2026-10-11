from unittest.mock import AsyncMock, patch

import pytest

from agent.research_tools import DeliveryContext
from output import delivery as dlv
from output.threads_handler import ThreadsDelivery
from shared import ImageAsset


def _img(content_type="image/jpeg"):
    return ImageAsset(
        data=b"abc",
        source_url="https://src/article",
        image_url="https://cdn/x",
        content_type=content_type,
        alt="A Title",
    )


def _threads_config(cfg, *, bucket: str = "", prefix: str = "", max_posts: int = 8) -> None:
    cfg.return_value.aws.state_bucket_name = bucket
    cfg.return_value.aws.s3_prefix = prefix
    cfg.return_value.agent.research_max_threads_posts = max_posts


class TestMarkdownReport:
    @pytest.mark.asyncio
    async def test_the_report_is_recorded_and_publishes_nothing(self):
        d = DeliveryContext()
        with patch("output.threads_handler.post_to_threads", new=AsyncMock()) as pt:
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
        d = DeliveryContext(publish_threads=True, staged_images=[_img("image/webp"), _img("image/png")])
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 1))) as pt:
            with patch.object(dlv, "get_config") as cfg:
                _threads_config(cfg, bucket="bkt", prefix="omni")
                ok = await dlv.deliver_research_report("리드 문장이다.", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        kw = pt.await_args.kwargs
        assert kw["image_content_type"] == "image/webp"  # the FIRST staged image
        assert kw["image_bucket"] == "bkt"
        assert kw["image_key"].startswith("omni/threads/research_")
        assert kw["image_key"].endswith(".webp")  # extension derived from content_type

    @pytest.mark.asyncio
    async def test_empty_report_skips_threads_api(self):
        # An empty report must not call post_to_threads (an empty root 400s the Threads API).
        d = DeliveryContext(publish_threads=True)
        with patch("output.threads_handler.post_to_threads", new=AsyncMock()) as pt:
            with patch.object(dlv, "get_config") as cfg:
                _threads_config(cfg, max_posts=6)
                ok = await dlv.deliver_research_report("   \n\n ", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.FAILED
        pt.assert_not_awaited()
        assert "threads" not in d.delivered_channels

    @pytest.mark.asyncio
    async def test_no_images_text_only(self):
        d = DeliveryContext(publish_threads=True)
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 1))) as pt:
            await dlv.deliver_research_report("리드.", channel="threads", delivery=d)
        assert pt.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_a_repeat_threads_call_posts_nothing_and_says_so(self):
        # The likely reason for a second call is a REVISED report; announcing it as delivered told
        # the requester a correction had gone out that never did.
        d = DeliveryContext(publish_threads=True)
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 1))) as pt:
            first = await dlv.deliver_research_report("첫 보고서다.", channel="threads", delivery=d)
            second = await dlv.deliver_research_report("고친 보고서다.", channel="threads", delivery=d)
        assert first is dlv.DeliveryOutcome.POSTED
        assert second is dlv.DeliveryOutcome.NOT_POSTED
        assert pt.await_count == 1

    @pytest.mark.asyncio
    async def test_no_bucket_posts_text_only(self, monkeypatch):
        # Staged image but no state bucket → text-only post, no image bytes passed.
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        d = DeliveryContext(publish_threads=True, staged_images=[_img()])
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 1))) as pt:
            with patch.object(dlv, "get_config") as cfg:
                _threads_config(cfg)
                ok = await dlv.deliver_research_report("리드.", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        assert pt.await_args.kwargs["image_bytes"] is None
        assert pt.await_args.kwargs["image_bucket"] == ""

    @pytest.mark.asyncio
    async def test_the_per_channel_outcome_is_kept_for_the_caller(self):
        d = DeliveryContext(publish_threads=True)
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(4, 6))):
            with patch.object(dlv, "get_config") as cfg:
                _threads_config(cfg)
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
        report = "\n---\n".join(f"{i}/10 소제목\n\n본문 {i}이다." for i in range(10))
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 8))) as pt:
            with patch.object(dlv, "get_config") as cfg:
                _threads_config(cfg)
                ok = await dlv.deliver_research_report(report, channel="threads", delivery=d)
                again = await dlv.deliver_research_report(report, channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        assert again is dlv.DeliveryOutcome.NOT_POSTED
        assert pt.await_count == 1
        assert d.partial_channels == {"threads"}
        assert d.delivered_channels == set()

    @pytest.mark.asyncio
    async def test_a_delivery_where_nothing_landed_is_still_a_failure(self):
        d = DeliveryContext(publish_threads=True)
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(0, 8))):
            with patch.object(dlv, "get_config") as cfg:
                _threads_config(cfg)
                ok = await dlv.deliver_research_report("리드다.", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.FAILED
        assert d.partial_channels == set() and d.delivered_channels == set()


class TestIncompleteDeliveryIsReported:
    """A truncated thread used to be indistinguishable from a complete one. There is deliberately NO
    re-delivery path (the recorded channel makes a second deliver_report a no-op); the counts say
    what landed instead."""

    @pytest.mark.asyncio
    async def test_dropped_posts_count_as_incomplete_even_when_all_land(self):
        d = DeliveryContext(publish_threads=True)
        report = "\n---\n".join(f"{i}/20 소제목\n\n본문 {i}이다." for i in range(20))
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(4, 4))):
            with patch.object(dlv, "get_config") as cfg:
                _threads_config(cfg, max_posts=4)
                ok = await dlv.deliver_research_report(report, channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        assert d.threads_stats.dropped == 16
        assert not d.threads_stats.complete
