import os
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from shared import (
    ContentType,
    EnvVars,
    ParseResult,
    SummaryResult,
    format_slack_message,
    logger,
)


async def send_slack_message(
    parsed_result: ParseResult,
    summary_result: SummaryResult,
    enable_business_channels: bool = False,
) -> bool:
    message_text = format_slack_message(parsed_result, summary_result)
    logger.debug("Sending Slack message: '%s'", message_text)
    bot_configs = _get_bot_configs(enable_business_channels)

    async def action(client: AsyncWebClient, channel_id: str):
        await client.chat_postMessage(channel=channel_id, text=message_text, mrkdwn=True)

        if parsed_result.content.content_type == ContentType.PDF and summary_result.thumbnails:
            await _upload_images(client, channel_id, summary_result.thumbnails)

    return await _broadcast_to_bots(bot_configs, action)


def _get_bot_configs(enable_business_channels: bool) -> list[dict]:
    return [
        {
            "token": os.getenv(EnvVars.SLACK_BUSINESS_TOKEN.value),
            "channel_ids_str": os.getenv(EnvVars.SLACK_BUSINESS_CHANNEL_IDS.value, ""),
            "name": "Business Slack",
            "enabled": enable_business_channels,
        },
        {
            "token": os.getenv(EnvVars.SLACK_PERSONAL_TOKEN.value),
            "channel_ids_str": os.getenv(EnvVars.SLACK_PERSONAL_CHANNEL_IDS.value, ""),
            "name": "Personal Slack",
            "enabled": True,
        },
    ]


async def _upload_images(client: AsyncWebClient, channel_id: str, image_paths: list[str | Path]) -> None:
    for img_path_obj in image_paths:
        img_path = str(img_path_obj)
        try:
            await client.files_upload_v2(
                channel=channel_id,
                file=img_path,
                filename=Path(img_path).name,
            )
        except SlackApiError as e:
            logger.warning("Failed to upload image '%s' to channel '%s': %s", img_path, channel_id, e.response["error"])
        except Exception as e:
            logger.warning("An unexpected error occurred while uploading '%s': %s", img_path, e)


async def _broadcast_to_bots(
    bot_configs: list[dict[str, Any]],
    action_per_channel: Callable[[AsyncWebClient, str], Awaitable[None]],
) -> bool:
    successful_sends = []
    for config in filter(lambda c: c.get("enabled"), bot_configs):
        bot_token = config.get("token")
        name = config.get("name")
        channel_ids_str = config.get("channel_ids_str", "")

        if not bot_token or not channel_ids_str:
            continue

        channel_ids = [cid.strip() for cid in channel_ids_str.split(",") if cid.strip()]
        if not channel_ids:
            continue

        logger.info("Processing '%s' for channels: '%s'", name, channel_ids)
        client = AsyncWebClient(token=bot_token)

        for channel_id in channel_ids:
            try:
                await action_per_channel(client, channel_id)
                successful_sends.append(True)
                logger.info("Successfully performed action for channel '%s'", channel_id)
            except SlackApiError as e:
                logger.error("Failed action for '%s' via '%s': %s", channel_id, name, e.response["error"])
            except Exception as e:
                logger.error("An unexpected error occurred for '%s' via '%s': %s", channel_id, name, e)

    if not any(successful_sends):
        logger.warning("No Slack actions succeeded. Check Slack tokens and channel IDs.")

    return any(successful_sends)
