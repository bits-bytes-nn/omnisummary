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
    # Count the stories actually shown (the LLM may merge ranked items into fewer), not the
    # raw ranked count — otherwise the header overstates how many stories are in the digest.
    n_stories = len(digest.content.items) if digest.content and digest.content.items else len(digest.ranked_items)
    header = f":satellite: OmniSummary — {today} · {n_stories} stories"
    client = AsyncWebClient(token=bot_token)

    try:
        if digest.content and digest.content.items:
            from output.renderers import render_slack_blocks

            block_chunks = render_slack_blocks(digest.content, header=header)
            for i, blocks in enumerate(block_chunks):
                fallback = header if i == 0 else f"{header} (cont. {i + 1})"
                await client.chat_postMessage(channel=channel_id, blocks=blocks, text=fallback)
            logger.info(
                "Successfully sent digest to Slack channel '%s' (%d Block Kit message(s))",
                channel_id,
                len(block_chunks),
            )
            return True

        # Fallback: no structured content (e.g. empty-digest day) — send plain mrkdwn text.
        text_chunks = _split_message(
            f":satellite: *OmniSummary* — {today} · {n_stories} stories\n\n" + digest.digest_text
        )
        for chunk in text_chunks:
            await client.chat_postMessage(channel=channel_id, text=chunk, mrkdwn=True)
        logger.info("Successfully sent digest to Slack channel '%s' (%d text message(s))", channel_id, len(text_chunks))
        return True
    except SlackApiError as e:
        logger.warning("Failed to send digest to Slack: %s", e.response["error"])
        return False
    except Exception as e:
        logger.warning("Unexpected error sending digest to Slack: %s", e)
        return False
