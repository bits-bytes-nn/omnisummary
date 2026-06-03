from __future__ import annotations

from datetime import datetime
from typing import Any

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from shared import DigestResult, logger, resolve_secret
from shared.config import SlackConfig

SLACK_MAX_TEXT_LENGTH = 3900


def _split_message(text: str, max_len: int = SLACK_MAX_TEXT_LENGTH) -> list[str]:
    if len(text) <= max_len:
        return [text]
    chunks: list[str] = []
    current = ""
    for paragraph in text.split("\n\n"):
        if len(paragraph) > max_len:
            if current:
                chunks.append(current.strip())
                current = ""
            for line in paragraph.split("\n"):
                if len(line) > max_len:
                    for i in range(0, len(line), max_len):
                        chunks.append(line[i : i + max_len])
                elif len(current) + len(line) + 1 > max_len:
                    if current:
                        chunks.append(current.strip())
                    current = line
                else:
                    current = f"{current}\n{line}" if current else line
        elif len(current) + len(paragraph) + 2 > max_len:
            if current:
                chunks.append(current.strip())
            current = paragraph
        else:
            current = f"{current}\n\n{paragraph}" if current else paragraph
    if current:
        chunks.append(current.strip())
    return chunks


async def send_image_to_slack(
    image_bytes: bytes,
    *,
    channel_id: str,
    title: str,
    comment: str = "",
    thread_ts: str = "",
    bot_token: str = "",
) -> bool:
    token = bot_token or resolve_secret("SLACK_BOT_TOKEN", "slack-bot-token")
    channel_id = channel_id or resolve_secret("SLACK_CHANNEL_ID", "slack-channel-id")
    if not token or not channel_id:
        logger.warning("Slack bot_token or channel_id not configured. Skipping image upload.")
        return False

    client = AsyncWebClient(token=token)
    try:
        kwargs: dict[str, Any] = {
            "channel": channel_id,
            "content": image_bytes,
            "title": title,
            "filename": f"{title}.png",
        }
        if comment:
            kwargs["initial_comment"] = comment
        if thread_ts:
            kwargs["thread_ts"] = thread_ts
        await client.files_upload_v2(**kwargs)
        logger.info("Uploaded image to Slack channel '%s'", channel_id)
        return True
    except SlackApiError as e:
        logger.warning("Failed to upload image to Slack: %s", e.response["error"])
        return False
    except Exception as e:
        logger.warning("Unexpected error uploading image to Slack: %s", e)
        return False


async def send_digest_to_slack(digest: DigestResult, config: SlackConfig) -> bool:
    bot_token = config.bot_token or resolve_secret("SLACK_BOT_TOKEN", "slack-bot-token")
    channel_id = config.channel_id or resolve_secret("SLACK_CHANNEL_ID", "slack-channel-id")

    if not bot_token or not channel_id:
        logger.warning("Slack bot_token or channel_id not configured. Skipping Slack delivery.")
        return False

    today = datetime.now().strftime("%Y-%m-%d")
    n_stories = len(digest.ranked_items)
    header = f":satellite: *OmniSummary* — {today} · {n_stories} stories\n"
    message_text = header + "\n" + digest.digest_text

    client = AsyncWebClient(token=bot_token)
    chunks = _split_message(message_text)

    try:
        for i, chunk in enumerate(chunks):
            await client.chat_postMessage(channel=channel_id, text=chunk, mrkdwn=True)
            if len(chunks) > 1:
                logger.debug("Sent Slack message chunk %d/%d", i + 1, len(chunks))
        logger.info("Successfully sent digest to Slack channel '%s' (%d message(s))", channel_id, len(chunks))
        return True
    except SlackApiError as e:
        logger.warning("Failed to send digest to Slack: %s", e.response["error"])
        return False
    except Exception as e:
        logger.warning("Unexpected error sending digest to Slack: %s", e)
        return False
