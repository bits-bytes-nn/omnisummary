from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

import httpx

from output.threads_handler import (
    error_detail,
    fetch_conversation,
    fetch_own_posts,
    is_media_not_found,
    is_transient_reply_error,
    publish_reply,
    threads_client,
)
from shared import (
    KOREAN_PROSE_RULES,
    BedrockLanguageModelFactory,
    CommentReply,
    CommentReplyLedger,
    Config,
    StateStore,
    logger,
    resolve_secret,
)
from shared.prompts import CommentReplyPrompt

# Meta's moderation states a reader still sees; a HIDDEN, COVERED or BLOCKED comment is left alone.
VISIBLE_HIDE_STATUSES = frozenset({"", "NOT_HUSHED", "UNHUSHED"})
REQUEST_TIMEOUT_SEC = 30


@dataclass
class PendingComment:
    comment: dict[str, Any]
    # The posts above the comment, oldest first: the account's post, then each reply on the way down.
    thread: list[dict[str, Any]]
    # The account's own posts in this conversation, so the model can tell its words from the reader's.
    own_ids: set[str]


@dataclass
class CommentReplyRun:
    replied: list[dict[str, str]] = field(default_factory=list)
    declined: list[str] = field(default_factory=list)
    failed: list[str] = field(default_factory=list)


def _parse_time(raw: str) -> datetime | None:
    try:
        return datetime.strptime(raw, "%Y-%m-%dT%H:%M:%S%z")
    except (TypeError, ValueError):
        return None


def pending_comments(
    post: dict[str, Any],
    conversation: list[dict[str, Any]],
    *,
    handled: set[str],
    since: datetime,
    max_per_commenter: int,
) -> list[PendingComment]:
    """The comments under `post` the account still owes an answer: written by someone else, visible,
    newer than `since`, not yet dealt with, and addressed to one of the account's own posts. A reply
    between two readers is their conversation, not the account's."""
    by_id = {post["id"]: post, **{c["id"]: c for c in conversation}}
    own = {post["id"]} | {c["id"] for c in conversation if c.get("is_reply_owned_by_me")}

    def parent_id(item: dict[str, Any]) -> str:
        return str((item.get("replied_to") or {}).get("id", ""))

    answered = {parent_id(c) for c in conversation if c["id"] in own}
    replies_to: dict[str, int] = {}
    for parent in answered:
        user = by_id.get(parent, {}).get("username", "")
        replies_to[user] = replies_to.get(user, 0) + 1

    pending = []
    for c in conversation:
        created = _parse_time(c.get("timestamp", ""))
        if (
            c["id"] in own
            or c["id"] in handled
            or c["id"] in answered
            or parent_id(c) not in own
            or not (c.get("text") or "").strip()
            or c.get("hide_status", "") not in VISIBLE_HIDE_STATUSES
            or created is None
            or created < since
        ):
            continue
        if replies_to.get(c.get("username", ""), 0) >= max_per_commenter:
            logger.info("Not answering @%s again under post %s: already answered", c.get("username"), post["id"])
            continue
        if parent_id(c) == post["id"]:
            # A comment on the root may be about any story: the root only lists their titles.
            thread = [post] + [s for s in conversation if s["id"] in own and parent_id(s) == post["id"]]
        else:
            thread = []
            cursor = by_id.get(parent_id(c))
            while cursor is not None and cursor not in thread:
                thread.insert(0, cursor)
                cursor = None if cursor is post else by_id.get(parent_id(cursor))
        pending.append(PendingComment(comment=c, thread=thread, own_ids=own))
        user = c.get("username", "")
        replies_to[user] = replies_to.get(user, 0) + 1
    return pending


def _thread_text(pending: PendingComment) -> str:
    def author(item: dict[str, Any]) -> str:
        return "YOU" if item["id"] in pending.own_ids else f"@{item.get('username', 'reader')}"

    return "\n\n".join(f"[{author(item)}]\n{(item.get('text') or '').strip()}" for item in pending.thread)


class CommentResponder:
    """Answers readers' comments on the account's recent posts: one model call per comment, which
    may decline it, then one reply under the comment."""

    def __init__(self, config: Config, llm_factory: BedrockLanguageModelFactory, store: StateStore) -> None:
        self.config = config.pipeline
        # json_schema is Bedrock's constrained output, so every answer is a verdict. Sonnet 5.5 takes the
        # schema tool only under "auto", and then answered 4 of 5 English comments in plain prose.
        self.llm = llm_factory.get_model(self.config.digest_model, stage="comment-reply").with_structured_output(
            CommentReply, method="json_schema"
        )
        self.ledger = CommentReplyLedger(store, retention=timedelta(days=2 * self.config.comment_reply_lookback_days))

    async def draft(self, pending: PendingComment) -> CommentReply:
        chain = CommentReplyPrompt.get_prompt() | self.llm
        result = await chain.ainvoke(
            {
                "audience": self.config.digest_audience_description,
                "language_rules": KOREAN_PROSE_RULES,
                "thread": _thread_text(pending),
                "username": pending.comment.get("username", "reader"),
                "comment": pending.comment.get("text", "").strip(),
            }
        )
        return CommentReply.model_validate(result)

    async def run(self, now: datetime, *, dry_run: bool = False) -> CommentReplyRun:
        """Answer what is pending. `dry_run` drafts the replies and posts and records nothing.
        A comment whose draft fails, or whose publish may have failed transiently, is not recorded, so
        the next run tries it again; one Threads rejects for good is recorded as failed and left alone."""
        token = resolve_secret("THREADS_ACCESS_TOKEN", "threads-access-token", strict=True)
        user_id = resolve_secret("THREADS_USER_ID", "threads-user-id", strict=True)
        if not token or not user_id:
            raise RuntimeError("Threads access token / user id not configured; cannot answer comments")
        since = now - timedelta(days=self.config.comment_reply_lookback_days)
        handled = self.ledger.handled()
        run = CommentReplyRun()
        async with threads_client(token, REQUEST_TIMEOUT_SEC) as client:
            pending: list[PendingComment] = []
            for post in await fetch_own_posts(client, since):
                pending += pending_comments(
                    post,
                    await fetch_conversation(client, post["id"]),
                    handled=handled,
                    since=since,
                    max_per_commenter=self.config.comment_reply_max_per_commenter,
                )
            pending.sort(key=lambda p: p.comment.get("timestamp", ""))
            cap = self.config.comment_reply_max_per_run
            if len(pending) > cap:
                logger.info("%d comments pending; answering the oldest %d this run", len(pending), cap)
            for item in pending[:cap]:
                comment_id = item.comment["id"]
                try:
                    decision = await self.draft(item)
                    text = decision.text.strip()
                    if not decision.reply or not text:
                        logger.info("Declined comment %s by @%s", comment_id, item.comment.get("username"))
                        run.declined.append(comment_id)
                        if not dry_run:
                            self.ledger.record(comment_id, "declined", now)
                        continue
                    if not dry_run:
                        reply_id = await publish_reply(client, user_id, token, text, comment_id)
                        self.ledger.record(comment_id, "replied", now)
                        logger.info(
                            "Replied %s to comment %s by @%s", reply_id, comment_id, item.comment.get("username")
                        )
                    run.replied.append(
                        {"comment_id": comment_id, "comment": item.comment.get("text", ""), "reply": text}
                    )
                except httpx.HTTPStatusError as e:
                    logger.warning(
                        "Could not answer comment %s (%s): %s", comment_id, e.response.status_code, error_detail(e)
                    )
                    run.failed.append(comment_id)
                    # Only a verdict retrying cannot change (the comment is gone, auth, a permanent
                    # rejection) is recorded; a 400 that may be transient is tried again next run.
                    if (is_media_not_found(e) or not is_transient_reply_error(e)) and not dry_run:
                        try:
                            self.ledger.record(comment_id, "failed", now)
                        except Exception:
                            logger.warning("Could not record comment %s as failed", comment_id, exc_info=True)
                except Exception as e:
                    logger.warning("Could not answer comment %s: %s", comment_id, e)
                    run.failed.append(comment_id)
        logger.info(
            "Comment replies: %d replied, %d declined, %d failed", len(run.replied), len(run.declined), len(run.failed)
        )
        return run
