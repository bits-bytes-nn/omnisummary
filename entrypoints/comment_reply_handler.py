from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from application.comment_replies import CommentResponder
from entrypoints import bootstrap
from shared import Config, logger, set_correlation_id


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
    responder = CommentResponder(
        config, bootstrap.language_models(config), bootstrap.state_store(config), bootstrap.threads(config)
    )
    return await responder.run(datetime.now(UTC))
