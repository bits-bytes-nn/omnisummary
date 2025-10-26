import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import state_manager, summarization_agent
from shared import logger


async def run_agent(message: str) -> None:
    logger.info("Running agent with message: '%s'", message)
    try:
        state_manager.clear()
        response = await asyncio.to_thread(summarization_agent, message)
        logger.info("Agent response: '%s'", response)
        logger.info("✓ Successfully completed")
    except Exception as e:
        logger.error("✗ Agent execution failed: %s", e, exc_info=True)
        raise


async def main(message: str) -> None:
    logger.info("Starting Agent Execution\n")
    logger.info("=" * 60)
    await run_agent(message)
    logger.info("=" * 60)
    logger.info("Agent execution completed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the summarization agent with a given message")
    parser.add_argument(
        "--message",
        type=str,
        required=True,
        help="Message to send to the summarization agent (e.g., URL with instructions)",
    )
    args = parser.parse_args()

    logger.info("Processing message: '%s'", args.message)
    asyncio.run(main(args.message))
