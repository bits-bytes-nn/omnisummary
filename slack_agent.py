from __future__ import annotations

import argparse
import logging
import os
import re
import threading
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from dotenv import load_dotenv
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

load_dotenv()

logging.basicConfig(level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")))

from agent import create_digest_agent
from agent.agent_tools import DeliveryContext, request_context, state_manager
from agent.tool_state import DigestStateManager
from output.slack_handler import _split_message
from shared import LOGGING_TRUNCATION_CHARS, Config, LocalPaths, logger, sanitize_slack_mrkdwn


def _local_timezone() -> ZoneInfo:
    return ZoneInfo(Config.load().aws.timezone)


def _find_state_file(digest_date: date) -> Path | None:
    state_dir = Path(LocalPaths.DIGEST_STATE_DIR.value)
    if not state_dir.exists():
        return None

    target = state_dir / f"digest_{digest_date.isoformat()}.json"
    if target.exists():
        return target

    files = sorted(state_dir.glob("digest_*.json"), reverse=True)
    if files:
        logger.info("State for '%s' not found, falling back to '%s'", digest_date, files[0].name)
        return files[0]
    return None


_current_state_file: Path | None = None


def _load_state(digest_date: date | None = None) -> bool:
    global _current_state_file
    state_file = _find_state_file(digest_date or datetime.now(_local_timezone()).date())
    if not state_file:
        logger.warning("No digest state file found")
        return False

    if state_file == _current_state_file:
        return True

    loaded = DigestStateManager.load_from_file(state_file)
    state_manager.load_from(loaded)
    _current_state_file = state_file
    logger.info("Loaded %d items from '%s'", state_manager.get_item_count(), state_file)
    return True


def _reload_if_newer() -> None:
    global _current_state_file
    latest = _find_state_file(datetime.now(_local_timezone()).date())
    if latest and latest != _current_state_file:
        loaded = DigestStateManager.load_from_file(latest)
        state_manager.load_from(loaded)
        _current_state_file = latest
        logger.info("Auto-reloaded state: %d items from '%s'", state_manager.get_item_count(), latest)


def _strip_mention(text: str) -> str:
    return re.sub(r"<@[A-Z0-9]+>\s*", "", text).strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSummary Slack Agent")
    parser.add_argument("--date", type=str, help="Load digest for specific date (YYYY-MM-DD)")
    args = parser.parse_args()

    tz = _local_timezone()
    digest_date = date.fromisoformat(args.date) if args.date else datetime.now(tz).date()

    if not _load_state(digest_date):
        logger.error("Cannot start agent: no digest state available")
        return

    bot_token = os.getenv("SLACK_BOT_TOKEN", "")
    app_token = os.getenv("SLACK_APP_TOKEN", "")

    if not bot_token or not app_token:
        logger.error("SLACK_BOT_TOKEN and SLACK_APP_TOKEN must be set")
        return

    app = App(token=bot_token)
    agent = create_digest_agent()

    def _handle_user_query(event: dict, say) -> None:
        user_text = _strip_mention(event.get("text", ""))
        thread_ts = event.get("thread_ts") or event.get("ts")
        channel_id = event.get("channel", "")

        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        if not user_text:
            say(text="Please enter a question. e.g. '1번 자세히', '관련 논문 찾아줘'", thread_ts=thread_ts)
            return

        logger.info(
            "Agent query from user '%s': '%s'",
            event.get("user"),
            user_text[: LOGGING_TRUNCATION_CHARS["user_query"]],
        )
        _reload_if_newer()

        def _respond():
            try:
                delivery = DeliveryContext(channel_id=channel_id, thread_ts=thread_ts or "")
                with request_context(state_manager, delivery):
                    response = sanitize_slack_mrkdwn(str(agent(user_text)))
                chunks = _split_message(response)
                for chunk in chunks:
                    say(text=chunk, thread_ts=thread_ts)
            except Exception as e:
                logger.error("Agent error: %s", e)
                say(text=f"Error processing request: {e}", thread_ts=thread_ts)

        threading.Thread(target=_respond, daemon=True).start()

    @app.event("app_mention")
    def handle_mention(event, say):
        _handle_user_query(event, say)

    @app.event("message")
    def handle_message(event, say):
        if event.get("channel_type") == "im":
            _handle_user_query(event, say)

    logger.info("Starting Slack Socket Mode agent (%d items loaded)", state_manager.get_item_count())
    handler = SocketModeHandler(app, app_token)
    handler.start()


if __name__ == "__main__":
    main()
