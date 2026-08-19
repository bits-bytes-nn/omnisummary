from __future__ import annotations

import argparse
import logging
import os

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.getLevelName(os.getenv("LOG_LEVEL", "INFO")))

from agent import create_research_agent, research_run_limits
from agent.research_tools import DeliveryContext, request_context
from shared import logger


def main() -> None:
    parser = argparse.ArgumentParser(description="OmniSummary Deep Research Agent (local)")
    parser.add_argument("topic", help="The AI/ML topic to research")
    parser.add_argument(
        "--channel",
        choices=["slack", "threads", "both"],
        default="slack",
        help="Channel(s) this run may publish to (enforced by deliver_report, not inferred from prose)",
    )
    parser.add_argument("--channel-id", default="", help="Slack channel id to post to (omit for dry-run)")
    parser.add_argument("--thread-ts", default="", help="Slack thread ts to reply within")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the rendered report (and resolved image URLs) instead of posting",
    )
    args = parser.parse_args()

    # The typed flag travels as DATA. It used to be turned back into a Korean sentence for the model
    # to re-parse ("(쓰레드에 올려줘)"), two lines above the context that already threads
    # per-invocation state into every tool — so a typed choice became a phrase-matching problem.
    channels = {"slack", "threads"} if args.channel == "both" else {args.channel}
    delivery = DeliveryContext(
        channel_id=args.channel_id,
        thread_ts=args.thread_ts,
        dry_run=args.dry_run or not args.channel_id,
        requested_channels=channels,
    )
    # The agent still needs to KNOW where to publish; it is told, not asked to infer it.
    prompt = f"{args.topic}\n\n[DELIVERY] Publish the finished report to: {', '.join(sorted(channels))}"
    agent = create_research_agent()

    with request_context(delivery):
        try:
            response = str(agent(prompt, limits=research_run_limits()))
        except Exception as e:
            logger.error("Agent error: %s", e)
            print(f"\nError: {e}\n")
            return

    if not delivery.delivered:
        print(f"\n=== Agent final response (not delivered via tool) ===\n{response}\n")


if __name__ == "__main__":
    main()
