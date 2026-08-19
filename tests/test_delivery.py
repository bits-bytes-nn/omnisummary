from unittest.mock import AsyncMock, MagicMock, patch

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


class TestDeliverSlackImages:
    @pytest.mark.asyncio
    async def test_image_uploaded_without_caption(self):
        # The OG image is posted as a bare file upload — no "이미지: <url>" caption text.
        d = DeliveryContext(channel_id="C9", staged_images=[_img()])
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch("output.slack_handler.send_image_to_slack", new=AsyncMock()) as si:
            with patch.object(dlv, "AsyncWebClient", return_value=client):
                with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                    await dlv.deliver_research_report("본문이다.", channel="slack", delivery=d)
        assert "comment" not in si.await_args.kwargs

    @pytest.mark.asyncio
    async def test_image_not_reuploaded_on_retry(self):
        # If a prior attempt failed AFTER uploading the image (channel not recorded), a retried
        # _deliver_slack must not re-upload the same image bytes.
        d = DeliveryContext(channel_id="C9", staged_images=[_img()])
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch("output.slack_handler.send_image_to_slack", new=AsyncMock()) as si:
            with patch.object(dlv, "AsyncWebClient", return_value=client):
                with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                    await dlv._deliver_slack("본문이다.", d)
                    await dlv._deliver_slack("본문이다.", d)  # retry
        assert si.await_count == 1  # uploaded once across both attempts


class TestDryRun:
    @pytest.mark.asyncio
    async def test_slack_dry_run_posts_nothing(self):
        d = DeliveryContext(channel_id="C", dry_run=True, staged_images=[_img()])
        with patch("output.slack_handler.send_image_to_slack", new=AsyncMock()) as si:
            with patch.object(dlv, "AsyncWebClient") as web:
                ok = await dlv.deliver_research_report("*보고서* 본문이다.", channel="slack", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        si.assert_not_awaited()
        web.assert_not_called()

    @pytest.mark.asyncio
    async def test_threads_dry_run_renders_without_posting(self):
        d = DeliveryContext(channel_id="C", dry_run=True)
        with patch("output.threads_handler.post_to_threads", new=AsyncMock()) as pt:
            ok = await dlv.deliver_research_report("문단 하나다.\n\n문단 둘이다.", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        pt.assert_not_awaited()


class TestDeliverSlack:
    @pytest.mark.asyncio
    async def test_posts_images_then_blocks(self):
        d = DeliveryContext(channel_id="C9", thread_ts="t1", staged_images=[_img("image/jpeg")])
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch("output.slack_handler.send_image_to_slack", new=AsyncMock()) as si:
            with patch.object(dlv, "AsyncWebClient", return_value=client):
                with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                    ok = await dlv.deliver_research_report("본문 보고서다.", channel="slack", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        # image uploaded with the jpeg extension derived from content_type
        assert si.await_args.kwargs["file_ext"] == "jpg"
        assert client.chat_postMessage.await_count >= 1
        assert client.chat_postMessage.await_args.kwargs["thread_ts"] == "t1"

    @pytest.mark.asyncio
    async def test_returns_false_without_token(self):
        d = DeliveryContext(channel_id="C")
        with patch.object(dlv, "resolve_secret", return_value=""):
            ok = await dlv.deliver_research_report("body", channel="slack", delivery=d)
        assert ok is dlv.DeliveryOutcome.FAILED
        assert "slack" not in d.delivered_channels

    @pytest.mark.asyncio
    async def test_api_failure_returns_false_without_raising(self):
        d = DeliveryContext(channel_id="C")
        client = MagicMock()
        client.chat_postMessage = AsyncMock(side_effect=RuntimeError("slack down"))
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                ok = await dlv.deliver_research_report("body", channel="slack", delivery=d)
        assert ok is dlv.DeliveryOutcome.FAILED
        assert "slack" not in d.delivered_channels

    @pytest.mark.asyncio
    async def test_success_records_channel(self):
        d = DeliveryContext(channel_id="C")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                ok = await dlv.deliver_research_report("body", channel="slack", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        assert "slack" in d.delivered_channels

    @pytest.mark.asyncio
    async def test_first_block_has_header(self):
        d = DeliveryContext(channel_id="C")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                await dlv.deliver_research_report("본문이다.", channel="slack", delivery=d)
        first_blocks = client.chat_postMessage.await_args_list[0].kwargs["blocks"]
        assert first_blocks[0]["type"] == "header"

    @pytest.mark.asyncio
    async def test_sanitizes_before_posting(self):
        d = DeliveryContext(channel_id="C")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                with patch.object(dlv, "sanitize_slack_mrkdwn", return_value="cleaned") as san:
                    await dlv.deliver_research_report("**raw** ## heading", channel="slack", delivery=d)
        san.assert_called_once()


class TestDeliverThreads:
    @pytest.mark.asyncio
    async def test_passes_first_image_with_content_type_and_key(self):
        d = DeliveryContext(channel_id="C", staged_images=[_img("image/webp"), _img("image/png")])
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 1))) as pt:
            with patch.object(dlv, "get_config") as cfg:
                cfg.return_value.aws.state_bucket_name = "bkt"
                cfg.return_value.aws.s3_prefix = "omni"
                cfg.return_value.agent.research_max_threads_posts = 8
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
        d = DeliveryContext(channel_id="C")
        with patch("output.threads_handler.post_to_threads", new=AsyncMock()) as pt:
            with patch.object(dlv, "get_config") as cfg:
                cfg.return_value.agent.research_max_threads_posts = 6
                ok = await dlv.deliver_research_report("   \n\n ", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.FAILED
        pt.assert_not_awaited()
        assert "threads" not in d.delivered_channels

    @pytest.mark.asyncio
    async def test_no_images_text_only(self):
        d = DeliveryContext(channel_id="C")
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 1))) as pt:
            await dlv.deliver_research_report("리드.", channel="threads", delivery=d)
        assert pt.await_args.kwargs["image_bytes"] is None

    @pytest.mark.asyncio
    async def test_both_channels_recorded_and_idempotent(self):
        # Delivering to both channels records both; a repeat call to a delivered channel is a no-op.
        d = DeliveryContext(channel_id="C")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                with patch(
                    "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 1))
                ) as pt:
                    await dlv.deliver_research_report("body", channel="slack", delivery=d)
                    await dlv.deliver_research_report("body", channel="threads", delivery=d)
                    # repeat slack call must not double-post
                    await dlv.deliver_research_report("body", channel="slack", delivery=d)
        assert d.delivered_channels == {"slack", "threads"}
        assert pt.await_count == 1
        # slack posted once (the repeat was skipped) — at least one call, not two rounds
        first_round_calls = client.chat_postMessage.await_count
        assert first_round_calls >= 1

    @pytest.mark.asyncio
    async def test_no_bucket_posts_text_only(self, monkeypatch):
        # Staged image but no state bucket → text-only post, no image bytes passed.
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        d = DeliveryContext(channel_id="C", staged_images=[_img()])
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 1))) as pt:
            with patch.object(dlv, "get_config") as cfg:
                cfg.return_value.aws.state_bucket_name = ""
                cfg.return_value.aws.s3_prefix = ""
                cfg.return_value.agent.research_max_threads_posts = 8
                ok = await dlv.deliver_research_report("리드.", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        assert pt.await_args.kwargs["image_bytes"] is None
        assert pt.await_args.kwargs["image_bucket"] == ""


class TestARootThatLandedAloneIsNotAPlainFailure:
    """ThreadsDelivery.published is `posted > 1` when more than one post was intended, so a run that
    posted the ROOT and nothing else came back as a plain failure: no partial-delivery notice, and
    nothing recorded — a retry would have posted a second root over the first."""

    @staticmethod
    def _threads_config(cfg):
        cfg.return_value.aws.state_bucket_name = ""
        cfg.return_value.aws.s3_prefix = ""
        cfg.return_value.agent.research_max_threads_posts = 8

    @pytest.mark.asyncio
    async def test_a_lone_root_is_reported_as_incomplete_and_spends_the_channel(self):
        d = DeliveryContext(channel_id="C7", thread_ts="ts-1")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        report = "\n---\n".join(f"{i}/10 소제목\n\n본문 {i}이다." for i in range(10))
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                with patch(
                    "output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(1, 8))
                ) as pt:
                    with patch.object(dlv, "get_config") as cfg:
                        self._threads_config(cfg)
                        ok = await dlv.deliver_research_report(report, channel="threads", delivery=d)
                        # A retry must not post a second root over the one that landed.
                        again = await dlv.deliver_research_report(report, channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED
        assert again is dlv.DeliveryOutcome.NOT_POSTED
        assert pt.await_count == 1
        assert d.partial_channels == {"threads"}
        # Not a completed delivery: the runtime's last-resort Slack fallback must still fire.
        assert d.delivered_channels == set()
        assert "1/8" in client.chat_postMessage.await_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_a_delivery_where_nothing_landed_is_still_a_failure(self):
        d = DeliveryContext(channel_id="C7")
        with patch.object(dlv, "resolve_secret", return_value="xoxb"):
            with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(0, 8))):
                with patch.object(dlv, "get_config") as cfg:
                    self._threads_config(cfg)
                    ok = await dlv.deliver_research_report("리드다.", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.FAILED
        assert d.partial_channels == set() and d.delivered_channels == set()


class TestARevisedReportIsNotAnnouncedAsDelivered:
    """The already-delivered branch stored the new report text and returned success without posting
    it or touching last_stats, so a REVISED report was reported "Delivered" with the previous
    attempt's counts."""

    @pytest.mark.asyncio
    async def test_a_second_call_with_different_text_posts_nothing_and_says_so(self):
        d = DeliveryContext(channel_id="C")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                first = await dlv.deliver_research_report("첫 보고서다.", channel="slack", delivery=d)
                calls_after_first = client.chat_postMessage.await_count
                second = await dlv.deliver_research_report("고친 보고서다.", channel="slack", delivery=d)
        assert first is dlv.DeliveryOutcome.POSTED
        assert second is dlv.DeliveryOutcome.NOT_POSTED
        assert client.chat_postMessage.await_count == calls_after_first  # nothing new posted
        # The revised text is still what the runtime fallback would send.
        assert d.last_report == "고친 보고서다."


class TestPartialDeliveryNotice:
    """A truncated report used to be indistinguishable from a complete one. There is deliberately NO
    re-delivery path (the recorded channel makes a second deliver_report a no-op) — the requester is
    told instead, in their own thread."""

    @pytest.mark.asyncio
    async def test_incomplete_threads_delivery_posts_a_one_line_notice(self):
        d = DeliveryContext(channel_id="C7", thread_ts="ts-9")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(4, 6))):
                    with patch.object(dlv, "get_config") as cfg:
                        cfg.return_value.aws.state_bucket_name = ""
                        cfg.return_value.aws.s3_prefix = ""
                        cfg.return_value.agent.research_max_threads_posts = 8
                        ok = await dlv.deliver_research_report("리드다.", channel="threads", delivery=d)
        assert ok is dlv.DeliveryOutcome.POSTED  # what landed stays landed
        assert d.last_stats.delivered == 4 and d.last_stats.rendered == 6
        kwargs = client.chat_postMessage.await_args.kwargs
        assert kwargs["thread_ts"] == "ts-9"  # posted into the requester's thread
        assert "4/6" in kwargs["text"]

    @pytest.mark.asyncio
    async def test_complete_delivery_posts_no_notice(self):
        d = DeliveryContext(channel_id="C7", thread_ts="ts-9")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(2, 2))):
                    with patch.object(dlv, "get_config") as cfg:
                        cfg.return_value.aws.state_bucket_name = ""
                        cfg.return_value.aws.s3_prefix = ""
                        cfg.return_value.agent.research_max_threads_posts = 8
                        await dlv.deliver_research_report("리드다.", channel="threads", delivery=d)
        client.chat_postMessage.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_dropped_posts_count_as_incomplete_even_when_all_land(self):
        # Everything that was RENDERED landed, but the renderer had to drop posts over the cap —
        # the reader still has half a report, so the notice fires.
        d = DeliveryContext(channel_id="C7")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        report = "\n---\n".join(f"{i}/20 소제목\n\n본문 {i}이다." for i in range(20))
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=ThreadsDelivery(4, 4))):
                    with patch.object(dlv, "get_config") as cfg:
                        cfg.return_value.aws.state_bucket_name = ""
                        cfg.return_value.aws.s3_prefix = ""
                        cfg.return_value.agent.research_max_threads_posts = 4
                        await dlv.deliver_research_report(report, channel="threads", delivery=d)
        assert d.last_stats.dropped == 16
        assert "DROPPED" in client.chat_postMessage.await_args.kwargs["text"]
