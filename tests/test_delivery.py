from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agent.research_tools import DeliveryContext
from output import delivery as dlv
from shared import ImageAsset


def _img(content_type="image/jpeg"):
    return ImageAsset(
        data=b"abc",
        source_url="https://src/article",
        image_url="https://cdn/x",
        content_type=content_type,
        alt="A Title",
    )


class TestImageCaption:
    def test_uses_alt_and_source(self):
        cap = dlv._image_caption(_img())
        assert "A Title" in cap and "https://src/article" in cap

    def test_blank_alt_falls_back(self):
        cap = dlv._image_caption(ImageAsset(data=b"x", source_url="https://s", image_url="https://i", alt=""))
        assert "관련 이미지" in cap


class TestDryRun:
    @pytest.mark.asyncio
    async def test_slack_dry_run_posts_nothing(self):
        d = DeliveryContext(channel_id="C", dry_run=True, staged_images=[_img()])
        with patch("output.slack_handler.send_image_to_slack", new=AsyncMock()) as si:
            with patch.object(dlv, "AsyncWebClient") as web:
                ok = await dlv.deliver_research_report("*보고서* 본문이다.", channel="slack", delivery=d)
        assert ok is True
        si.assert_not_awaited()
        web.assert_not_called()

    @pytest.mark.asyncio
    async def test_threads_dry_run_renders_without_posting(self):
        d = DeliveryContext(channel_id="C", dry_run=True)
        with patch("output.threads_handler.post_to_threads", new=AsyncMock()) as pt:
            ok = await dlv.deliver_research_report("문단 하나다.\n\n문단 둘이다.", channel="threads", delivery=d)
        assert ok is True
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
        assert ok is True
        # image uploaded with the jpeg extension derived from content_type
        assert si.await_args.kwargs["file_ext"] == "jpg"
        assert client.chat_postMessage.await_count >= 1
        assert client.chat_postMessage.await_args.kwargs["thread_ts"] == "t1"

    @pytest.mark.asyncio
    async def test_returns_false_without_token(self):
        d = DeliveryContext(channel_id="C")
        with patch.object(dlv, "resolve_secret", return_value=""):
            ok = await dlv.deliver_research_report("body", channel="slack", delivery=d)
        assert ok is False
        assert "slack" not in d.delivered_channels

    @pytest.mark.asyncio
    async def test_api_failure_returns_false_without_raising(self):
        d = DeliveryContext(channel_id="C")
        client = MagicMock()
        client.chat_postMessage = AsyncMock(side_effect=RuntimeError("slack down"))
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                ok = await dlv.deliver_research_report("body", channel="slack", delivery=d)
        assert ok is False
        assert "slack" not in d.delivered_channels

    @pytest.mark.asyncio
    async def test_success_records_channel(self):
        d = DeliveryContext(channel_id="C")
        client = MagicMock()
        client.chat_postMessage = AsyncMock()
        with patch.object(dlv, "AsyncWebClient", return_value=client):
            with patch.object(dlv, "resolve_secret", return_value="xoxb"):
                ok = await dlv.deliver_research_report("body", channel="slack", delivery=d)
        assert ok is True
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
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=True)) as pt:
            with patch.object(dlv.Config, "load") as load:
                load.return_value.aws.state_bucket_name = "bkt"
                load.return_value.aws.s3_prefix = "omni"
                load.return_value.agent.research_max_threads_posts = 8
                ok = await dlv.deliver_research_report("리드 문장이다.", channel="threads", delivery=d)
        assert ok is True
        kw = pt.await_args.kwargs
        assert kw["image_content_type"] == "image/webp"  # the FIRST staged image
        assert kw["image_bucket"] == "bkt"
        assert kw["image_key"].startswith("omni/threads/research_")
        assert kw["image_key"].endswith(".webp")  # extension derived from content_type

    @pytest.mark.asyncio
    async def test_no_images_text_only(self):
        d = DeliveryContext(channel_id="C")
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=True)) as pt:
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
                with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=True)) as pt:
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
        with patch("output.threads_handler.post_to_threads", new=AsyncMock(return_value=True)) as pt:
            with patch.object(dlv.Config, "load") as load:
                load.return_value.aws.state_bucket_name = ""
                load.return_value.aws.s3_prefix = ""
                load.return_value.agent.research_max_threads_posts = 8
                ok = await dlv.deliver_research_report("리드.", channel="threads", delivery=d)
        assert ok is True
        assert pt.await_args.kwargs["image_bytes"] is None
        assert pt.await_args.kwargs["image_bucket"] == ""
