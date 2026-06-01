from unittest.mock import AsyncMock, patch

import pytest
from slack_sdk.errors import SlackApiError

from output.slack_handler import _split_message, send_digest_to_slack
from shared.config import SlackConfig
from shared.models import DigestResult


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
    return DigestResult(digest_text=text, ranked_items=[], total_collected=0, total_ranked=0)


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
        digest = _make_digest()
        config = _make_config(bot_token="", channel_id="C123")

        with patch.dict("os.environ", {}, clear=True):
            result = await send_digest_to_slack(digest, config)

        assert result is False

    @pytest.mark.asyncio
    async def test_returns_false_when_no_channel(self):
        digest = _make_digest()
        config = _make_config(bot_token="xoxb-test", channel_id="")

        with patch.dict("os.environ", {}, clear=True):
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
    async def test_falls_back_to_env_vars(self):
        digest = _make_digest("Content")
        config = _make_config(bot_token="", channel_id="")
        mock_client = AsyncMock()

        env = {"SLACK_BOT_TOKEN": "xoxb-env", "SLACK_CHANNEL_ID": "C_ENV"}
        with (
            patch.dict("os.environ", env, clear=False),
            patch("output.slack_handler.AsyncWebClient", return_value=mock_client),
        ):
            result = await send_digest_to_slack(digest, config)

        assert result is True
        mock_client.chat_postMessage.assert_called_once()
