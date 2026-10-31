from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from application.history import CommentReplyLedger
from application.ports import LanguageModels, PublishError, StateStore, ThreadsPublisher
from shared import KOREAN_PROSE_RULES, CommentReply, Config, logger
from shared.formatting import URL_RE
from shared.prompts import CommentReplyPrompt

# Meta's moderation states a reader still sees; a HIDDEN, COVERED or BLOCKED comment is left alone.
VISIBLE_HIDE_STATUSES = frozenset({"", "NOT_HUSHED", "UNHUSHED"})


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


# A reply the account publishes on its own, with no human in the loop, must not carry anything a
# commenter could have planted: a link or a domain, someone else's handle, a hashtag. The prompt asks for
# none of these, so a reply that has one followed instructions written into the comment.
# Any name.tld, including a shortener (lnkd.in, youtu.be) and the spelled-out or bracketed dot ("evil 닷
# com", "evil[.]com", "evil . com"; a space only AFTER a dot is a sentence end). The text is NFKC-normalized
# first, so a full-width or ideographic dot is a dot, and the TLD may be followed by a Korean particle
# ("evil.com으로"): Hangul is a word character, so \b found no boundary there.
# Its cost is declining a reply that names a file or library ("node.js"): the account then stays silent.
_BARE_DOMAIN = re.compile(r"(?i)[\w-]+(?:\.|\s+\.\s*|\s*(?:\[\.\]|\(dot\)|닷|\bdot\b)\s*)[a-z]{2,}(?![a-z0-9-])")
_ZERO_WIDTH = re.compile("[\u200b-\u200f\u2060\ufeff]")
_HANDLE = re.compile(r"(?<![\w@])@([A-Za-z0-9._]+)")
_HASHTAG = re.compile(r"(?<![\w#&])#[^\W\d_]")


def unsafe_reply_reason(text: str, commenter: str) -> str:
    """Why `text` must not be published as the account's reply to `commenter`; "" when it may."""
    text = _ZERO_WIDTH.sub("", unicodedata.normalize("NFKC", text)).replace("。", ".")
    if URL_RE.search(text) or _BARE_DOMAIN.search(text):
        return "it carries a link or a domain"
    if any(handle.rstrip(".").lower() != commenter.lower() for handle in _HANDLE.findall(text)):
        return "it names another account"
    if _HASHTAG.search(text):
        return "it carries a hashtag"
    return ""


def _reader_text(text: str) -> str:
    """Reader-written text with the prompt's own markers defused, so a comment cannot close the
    <comment> block or pass itself off as the account's words."""
    return text.replace("<", "‹").replace(">", "›").replace("[YOU]", "[you]")


def _thread_text(pending: PendingComment) -> str:
    def author(item: dict[str, Any]) -> str:
        return "YOU" if item["id"] in pending.own_ids else f"@{item.get('username', 'reader')}"

    def text(item: dict[str, Any]) -> str:
        raw = (item.get("text") or "").strip()
        return raw if item["id"] in pending.own_ids else _reader_text(raw)

    return "\n\n".join(f"[{author(item)}]\n{text(item)}" for item in pending.thread)


class CommentResponder:
    """Answers readers' comments on the account's recent posts: one model call per comment, which
    may decline it, then one reply under the comment."""

    def __init__(self, config: Config, models: LanguageModels, store: StateStore, threads: ThreadsPublisher) -> None:
        self.config = config.pipeline
        self.threads = threads
        # json_schema is Bedrock's constrained output, so every answer is a verdict. Sonnet 5.5 takes the
        # schema tool only under "auto", and then answered 4 of 5 English comments in plain prose.
        self.llm = models.get_model(self.config.digest_model, stage="comment-reply").with_structured_output(
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
                "username": _reader_text(pending.comment.get("username", "reader")),
                "comment": _reader_text(pending.comment.get("text", "").strip()),
            }
        )
        return CommentReply.model_validate(result)

    def _record_failure(self, comment_id: str, now: datetime, *, final: bool) -> None:
        try:
            if final:
                self.ledger.record(comment_id, "failed", now)
            else:
                self.ledger.record_failed_attempt(comment_id, now)
        except Exception:
            logger.warning("Could not record the failure on comment %s", comment_id, exc_info=True)

    async def run(self, now: datetime, *, dry_run: bool = False) -> CommentReplyRun:
        """Answer what is pending. `dry_run` drafts the replies and posts and records nothing.
        A comment whose draft fails is tried again next run; one whose publish may have failed transiently
        is tried again up to the ledger's MAX_ATTEMPTS; one Threads rejects for good is recorded as failed."""
        since = now - timedelta(days=self.config.comment_reply_lookback_days)
        handled = self.ledger.handled()
        run = CommentReplyRun()
        async with self.threads.comments() as threads:
            pending: list[PendingComment] = []
            for post in await threads.fetch_own_posts(since):
                pending += pending_comments(
                    post,
                    await threads.fetch_conversation(post["id"]),
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
                    unsafe = unsafe_reply_reason(text, item.comment.get("username", "")) if decision.reply else ""
                    if unsafe:
                        logger.warning("Not posting the reply to comment %s: %s", comment_id, unsafe)
                    if not decision.reply or not text or unsafe:
                        logger.info("Declined comment %s by @%s", comment_id, item.comment.get("username"))
                        run.declined.append(comment_id)
                        if not dry_run:
                            self.ledger.record(comment_id, "declined", now)
                        continue
                    if not dry_run:
                        reply_id = await threads.publish_reply(text, comment_id)
                        try:
                            self.ledger.record(comment_id, "replied", now)
                        except Exception:
                            # The reply is live: counting it as failed would raise the alarm for a
                            # comment that was answered.
                            logger.error("Replied to comment %s but could not record it", comment_id, exc_info=True)
                        logger.info(
                            "Replied %s to comment %s by @%s", reply_id, comment_id, item.comment.get("username")
                        )
                    run.replied.append(
                        {"comment_id": comment_id, "comment": item.comment.get("text", ""), "reply": text}
                    )
                except PublishError as e:
                    logger.warning("Could not answer comment %s (%s): %s", comment_id, e.status, e.detail)
                    run.failed.append(comment_id)
                    # A verdict retrying cannot change (auth, a permanent rejection) is final, and so is a
                    # failure of the publish step itself: that reply may be live, and the next run would post
                    # a second one whenever the conversation lags. A 400 that may be transient is tried again.
                    if not dry_run:
                        self._record_failure(comment_id, now, final=e.may_be_live or not e.transient)
                except Exception as e:
                    logger.warning("Could not answer comment %s: %s", comment_id, e)
                    run.failed.append(comment_id)
                    # Only a failure that repeats on this comment counts (the answer failed validation or
                    # parsing, both ValueErrors): uncounted, such a comment held a slot and raised the alarm
                    # for the whole lookback. A throttle, timeout or outage fails every comment alike, and
                    # counting it retired comments that were never answered.
                    if isinstance(e, ValueError) and not dry_run:
                        self._record_failure(comment_id, now, final=False)
        logger.info(
            "Comment replies: %d replied, %d declined, %d failed", len(run.replied), len(run.declined), len(run.failed)
        )
        return run
