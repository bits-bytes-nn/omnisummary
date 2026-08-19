import re
from datetime import date, datetime
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

import pytest
from slack_sdk.errors import SlackApiError

from output.slack_handler import _split_message, send_digest_to_slack, send_image_to_slack
from shared import get_config
from shared.config import SlackConfig
from shared.constants import SourceType
from shared.models import CollectedItem, DigestContent, DigestItem, DigestResult, RankedItem


class TestSplitMessage:
    def test_short_message_single_chunk(self):
        text = "Short message"
        assert _split_message(text) == ["Short message"]

    def test_long_message_with_paragraphs_split(self):
        paragraphs = ["x" * 2000, "y" * 2000, "z" * 2000]
        text = "\n\n".join(paragraphs)
        chunks = _split_message(text, max_len=3900)
        assert all(len(c) <= 3900 for c in chunks)
        assert len(chunks) >= 2

    def test_split_on_paragraph_boundary(self):
        paragraphs = ["A" * 2000, "B" * 2000, "C" * 2000]
        text = "\n\n".join(paragraphs)
        chunks = _split_message(text, max_len=3900)
        assert len(chunks) >= 2
        assert "A" * 100 in chunks[0]

    def test_exact_limit(self):
        text = "x" * 3900
        assert _split_message(text, max_len=3900) == [text]

    def test_preserves_all_content(self):
        paragraphs = [f"Paragraph {i} content here" for i in range(20)]
        text = "\n\n".join(paragraphs)
        chunks = _split_message(text, max_len=200)
        reconstructed = "\n\n".join(chunks)
        for p in paragraphs:
            assert p in reconstructed

    def test_empty_string(self):
        assert _split_message("") == [""]

    def test_single_huge_paragraph_is_split(self):
        text = "x" * 8000
        chunks = _split_message(text, max_len=3900)
        assert len(chunks) >= 2
        assert all(len(c) <= 3900 for c in chunks)


def _make_digest(text: str = "Test digest content") -> DigestResult:
    return DigestResult(digest_text=text, ranked_items=[])


def _make_digest_with_content(n_items: int = 2) -> DigestResult:
    content = DigestContent(
        lead="오늘의 리드다.",
        headline_index=1,
        items=[
            DigestItem(title=f"제목 {i}", url=f"https://e.com/{i}", body="본문이다.", implication="시사점이다.")
            for i in range(1, n_items + 1)
        ],
    )
    return DigestResult(digest_text="fallback text", ranked_items=[], content=content)


def _make_config(bot_token: str = "xoxb-test", channel_id: str = "C123") -> SlackConfig:
    return SlackConfig(bot_token=bot_token, channel_id=channel_id)


class TestSendDigestToSlack:
    @pytest.mark.asyncio
    async def test_sends_single_message(self):
        digest = _make_digest("Short digest")
        config = _make_config()
        mock_client = AsyncMock()

        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            result = await send_digest_to_slack(digest, config)

        assert result is True
        mock_client.chat_postMessage.assert_called_once()
        call_kwargs = mock_client.chat_postMessage.call_args
        assert call_kwargs.kwargs["channel"] == "C123"
        assert "Short digest" in call_kwargs.kwargs["text"]
        assert call_kwargs.kwargs["mrkdwn"] is True

    @pytest.mark.asyncio
    async def test_includes_header_with_date(self):
        digest = _make_digest("Content")
        config = _make_config()
        mock_client = AsyncMock()

        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            await send_digest_to_slack(digest, config)

        text_sent = mock_client.chat_postMessage.call_args.kwargs["text"]
        assert ":satellite: *OmniSummary*" in text_sent

    @pytest.mark.asyncio
    async def test_header_uses_the_runs_digest_date(self):
        # The pipeline computes digest_date in the configured timezone; the header must use it
        # rather than the process clock (UTC in Lambda), which stamped the PREVIOUS day at 19:00 KST.
        digest = _make_digest("Content")
        mock_client = AsyncMock()
        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            await send_digest_to_slack(digest, _make_config(), date(2026, 6, 10))
        assert "2026-06-10" in mock_client.chat_postMessage.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_header_defaults_to_configured_timezone(self):
        # No date threaded in (e.g. an ad-hoc call): fall back to the configured timezone's today,
        # never the UTC clock.
        digest = _make_digest("Content")
        mock_client = AsyncMock()
        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            await send_digest_to_slack(digest, _make_config())
        expected = datetime.now(ZoneInfo(get_config().aws.timezone)).date().isoformat()
        assert expected in mock_client.chat_postMessage.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_default_header_date_follows_the_configured_timezone(self):
        # Two zones 26h apart: the same instant can never share a local date, so this pins that the
        # fallback reads the CONFIGURED timezone rather than the process clock.
        digest = _make_digest("Content")
        dates = {}
        for tz in ("Etc/GMT-14", "Etc/GMT+12"):
            cfg = MagicMock()
            cfg.aws.timezone = tz
            mock_client = AsyncMock()
            with patch("output.slack_handler.get_config", return_value=cfg):
                with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
                    await send_digest_to_slack(digest, _make_config())
            text = mock_client.chat_postMessage.call_args.kwargs["text"]
            dates[tz] = re.search(r"\d{4}-\d{2}-\d{2}", text).group()
        assert dates["Etc/GMT-14"] != dates["Etc/GMT+12"]

    @pytest.mark.asyncio
    async def test_splits_long_message(self):
        paragraphs = [f"Section {i}\n" + "x" * 2000 for i in range(5)]
        long_text = "\n\n".join(paragraphs)
        digest = _make_digest(long_text)
        config = _make_config()
        mock_client = AsyncMock()

        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            result = await send_digest_to_slack(digest, config)

        assert result is True
        assert mock_client.chat_postMessage.call_count > 1

    @pytest.mark.asyncio
    async def test_returns_false_when_no_token(self):
        # No token in config and none in the environment (cleared by the hermetic_env fixture),
        # so resolve_secret finds nothing — and its SSM fallback is stubbed out, which is what
        # used to make this test spend ~2s on a real AWS round trip.
        digest = _make_digest()
        config = _make_config(bot_token="", channel_id="C123")

        result = await send_digest_to_slack(digest, config)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_channel(self):
        digest = _make_digest()
        config = _make_config(bot_token="xoxb-test", channel_id="")

        result = await send_digest_to_slack(digest, config)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_slack_api_error(self):
        digest = _make_digest()
        config = _make_config()
        mock_client = AsyncMock()
        mock_client.chat_postMessage.side_effect = SlackApiError(
            message="error", response={"error": "channel_not_found"}
        )

        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            result = await send_digest_to_slack(digest, config)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_on_unexpected_error(self):
        digest = _make_digest()
        config = _make_config()
        mock_client = AsyncMock()
        mock_client.chat_postMessage.side_effect = RuntimeError("network error")

        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            result = await send_digest_to_slack(digest, config)

        assert result is False

    @pytest.mark.asyncio
    async def test_structured_content_goes_out_as_block_kit(self):
        # The daily path: structured content renders Block Kit with a text fallback, not mrkdwn.
        digest = _make_digest_with_content(2)
        mock_client = AsyncMock()
        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            assert await send_digest_to_slack(digest, _make_config(), date(2026, 6, 10)) is True
        kwargs = mock_client.chat_postMessage.call_args.kwargs
        assert kwargs["blocks"][0]["type"] == "header"
        assert "2026-06-10 · 2 stories" in kwargs["text"]
        assert "mrkdwn" not in kwargs

    @pytest.mark.asyncio
    async def test_header_counts_curated_items_not_ranked_items(self):
        # The editor may MERGE ranked items into fewer stories; the header must not overstate.
        digest = _make_digest_with_content(2)
        digest.ranked_items = [
            RankedItem(
                item=CollectedItem(
                    source_type=SourceType.RSS, title=f"ranked {i}", url=f"https://e.com/r{i}", text="body"
                ),
                score=0.9,
            )
            for i in range(7)
        ]
        mock_client = AsyncMock()
        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            await send_digest_to_slack(digest, _make_config(), date(2026, 6, 10))
        assert "2 stories" in mock_client.chat_postMessage.call_args.kwargs["text"]

    @pytest.mark.asyncio
    async def test_block_overflow_posts_continuation_messages(self):
        # A long digest exceeds Slack's per-message block limit; every chunk must be posted and the
        # continuation fallbacks numbered.
        digest = _make_digest_with_content(30)
        mock_client = AsyncMock()
        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            assert await send_digest_to_slack(digest, _make_config(), date(2026, 6, 10)) is True
        assert mock_client.chat_postMessage.call_count > 1
        fallbacks = [c.kwargs["text"] for c in mock_client.chat_postMessage.call_args_list]
        assert "(cont. 2)" in fallbacks[1]

    @pytest.mark.asyncio
    async def test_block_kit_failure_returns_false(self):
        digest = _make_digest_with_content(1)
        mock_client = AsyncMock()
        mock_client.chat_postMessage.side_effect = SlackApiError(message="e", response={"error": "invalid_blocks"})
        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            assert await send_digest_to_slack(digest, _make_config(), date(2026, 6, 10)) is False

    @pytest.mark.asyncio
    async def test_falls_back_to_env_vars(self, monkeypatch):
        digest = _make_digest("Content")
        config = _make_config(bot_token="", channel_id="")
        mock_client = AsyncMock()

        monkeypatch.setenv("SLACK_BOT_TOKEN", "xoxb-env")
        monkeypatch.setenv("SLACK_CHANNEL_ID", "C_ENV")
        with patch("output.slack_handler.AsyncWebClient", return_value=mock_client):
            result = await send_digest_to_slack(digest, config)

        assert result is True
        mock_client.chat_postMessage.assert_called_once()


class TestSendImageToSlack:
    @pytest.mark.asyncio
    async def test_uploads_with_comment_and_thread(self):
        # The daily-visual path: an image upload carrying the caption as the initial comment.
        client = AsyncMock()
        with patch("output.slack_handler.AsyncWebClient", return_value=client):
            ok = await send_image_to_slack(
                b"PNG", channel_id="C1", title="T", comment="🎨 *T*\nc", thread_ts="1.2", bot_token="xoxb-test"
            )
        assert ok is True
        kwargs = client.files_upload_v2.call_args.kwargs
        assert kwargs["channel"] == "C1"
        assert kwargs["filename"] == "T.png"
        assert kwargs["initial_comment"] == "🎨 *T*\nc"
        assert kwargs["thread_ts"] == "1.2"

    @pytest.mark.asyncio
    async def test_omits_optional_kwargs_when_absent(self):
        client = AsyncMock()
        with patch("output.slack_handler.AsyncWebClient", return_value=client):
            assert await send_image_to_slack(b"PNG", channel_id="C1", title="T", bot_token="xoxb-test") is True
        kwargs = client.files_upload_v2.call_args.kwargs
        assert "initial_comment" not in kwargs and "thread_ts" not in kwargs

    @pytest.mark.asyncio
    async def test_no_token_skips_upload(self):
        # Env is cleared by the hermetic_env fixture, so resolve_secret finds nothing.
        client = AsyncMock()
        with patch("output.slack_handler.AsyncWebClient", return_value=client):
            assert await send_image_to_slack(b"PNG", channel_id="C1", title="T") is False
        client.files_upload_v2.assert_not_called()

    @pytest.mark.asyncio
    async def test_slack_error_returns_false(self):
        client = AsyncMock()
        client.files_upload_v2.side_effect = SlackApiError(message="e", response={"error": "file_too_large"})
        with patch("output.slack_handler.AsyncWebClient", return_value=client):
            assert await send_image_to_slack(b"PNG", channel_id="C1", title="T", bot_token="xoxb-test") is False

    @pytest.mark.asyncio
    async def test_unexpected_error_returns_false(self):
        client = AsyncMock()
        client.files_upload_v2.side_effect = RuntimeError("socket closed")
        with patch("output.slack_handler.AsyncWebClient", return_value=client):
            assert await send_image_to_slack(b"PNG", channel_id="C1", title="T", bot_token="xoxb-test") is False
