from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

import boto3

from pipeline.comment_replies import CommentResponder
from shared import BedrockLanguageModelFactory, Config, create_state_store, logger, set_correlation_id


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Scheduled: answer readers' new comments on the account's recent Threads posts.

    A comment that could not be answered is RE-RAISED as a failure once the rest are done, so the
    Errors alarm fires; a transient failure stays unrecorded and the next run tries it again
    (retry_attempts=0)."""
    set_correlation_id(getattr(context, "aws_request_id", "") or None)
    config = Config.load()
    if not config.pipeline.enable_comment_replies:
        logger.info("Comment replies are disabled (pipeline.enable_comment_replies)")
        return {"statusCode": 200, "body": "disabled"}
    try:
        run = asyncio.run(_run(config))
    except Exception as e:
        logger.error("Comment-reply Lambda failed: %s", e, exc_info=True)
        raise
    if run.failed:
        raise RuntimeError(f"Could not answer {len(run.failed)} comment(s): {', '.join(run.failed)}")
    return {"statusCode": 200, "body": f"{len(run.replied)} replied, {len(run.declined)} declined"}


async def _run(config: Config):
    session = boto3.Session(region_name=config.aws.bedrock_region)
    factory = BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)
    return await CommentResponder(config, factory, create_state_store(config)).run(datetime.now(UTC))
