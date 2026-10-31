#!/usr/bin/env python3
"""Draft the replies the comment-reply Lambda would post right now, without posting or recording
anything. Reads the live Threads comments and the deployed reply ledger (config.aws.state_bucket_name).

Usage: uv run python scripts/sample_comment_replies.py
"""

import asyncio
import json
import os
import sys
from dataclasses import asdict
from datetime import UTC, datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from application.comment_replies import CommentResponder
from entrypoints import bootstrap
from shared import Config, create_state_store


async def main() -> None:
    config = Config.load()
    responder = CommentResponder(
        config, bootstrap.language_models(config), create_state_store(config), bootstrap.threads(config)
    )
    run = await responder.run(datetime.now(UTC), dry_run=True)
    print(json.dumps(asdict(run), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
