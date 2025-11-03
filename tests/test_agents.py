import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import create_summarization_agent, state_manager
from shared import logger


async def run_agent(message: str) -> None:
    logger.info("Running agent with message: '%s'", message)
    try:
        state_manager.clear()
        agent = create_summarization_agent()
        response = await asyncio.to_thread(agent, message)
        logger.info("Agent response: '%s'", response)

        if state_manager._summary_results:
            logger.info("\n" + "=" * 60)
            logger.info("SUMMARY RESULT DETAILS")
            logger.info("=" * 60)
            for hash_key, summary_result in state_manager._summary_results.items():
                logger.info("Summary Hash: %s", hash_key)
                logger.info("Number of thumbnails: %d", len(summary_result.thumbnails))
                for i, thumbnail in enumerate(summary_result.thumbnails):
                    logger.info("  Thumbnail %d: %s", i + 1, thumbnail)
                    if Path(thumbnail).exists():
                        logger.info("    ✓ File exists")
                    else:
                        logger.info("    ✗ File does NOT exist")
            logger.info("=" * 60 + "\n")

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
