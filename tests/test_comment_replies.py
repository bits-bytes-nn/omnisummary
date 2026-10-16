from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from langchain_core.runnables import RunnableLambda

from lambda_handlers import comment_reply_handler
from output import threads_handler
from pipeline import comment_replies
from pipeline.comment_replies import CommentReplyRun, CommentResponder, pending_comments
from shared import CommentReply, CommentReplyLedger, Config, LocalStateStore, StateReadError

NOW = datetime(2026, 10, 16, 12, 0, tzinfo=UTC)
SINCE = NOW - timedelta(days=7)
POST = {"id": "root", "text": "AGI 등장 808일 전이다. 오늘의 리드."}


def _c(cid, parent, *, user="kim", own=False, text="질문이 있어요", at="2026-10-16T10:00:00+0000", **extra):
    return {
        "id": cid,
        "text": text,
        "username": "me" if own else user,
        "timestamp": at,
        "replied_to": {"id": parent},
        "is_reply_owned_by_me": own,
        **extra,
    }


def _pending(conversation, handled=frozenset(), max_per_commenter=3):
    return pending_comments(POST, conversation, handled=set(handled), since=SINCE, max_per_commenter=max_per_commenter)


class TestPendingComments:
    def test_a_comment_on_a_story_reply_carries_the_thread_above_it(self):
        story = _c("story", "root", own=True, text="스토리 1")
        found = _pending([story, _c("c1", "story")])
        assert [p.comment["id"] for p in found] == ["c1"]
        assert [t["id"] for t in found[0].thread] == ["root", "story"]
        assert found[0].own_ids == {"root", "story"}

    def test_already_answered_handled_or_own_comments_are_skipped(self):
        conversation = [
            _c("story", "root", own=True),
            _c("answered", "story"),
            _c("mine", "answered", own=True),
            _c("handled", "story"),
            _c("new", "story"),
        ]
        assert [p.comment["id"] for p in _pending(conversation, handled={"handled"})] == ["new"]

    def test_a_reply_between_two_readers_is_not_the_accounts_to_answer(self):
        conversation = [_c("c1", "root", user="kim"), _c("c2", "c1", user="lee")]
        assert [p.comment["id"] for p in _pending(conversation)] == ["c1"]

    def test_hidden_empty_and_old_comments_are_skipped(self):
        conversation = [
            _c("hidden", "root", hide_status="HIDDEN"),
            _c("blank", "root", text="  "),
            _c("old", "root", at="2026-10-01T10:00:00+0000"),
            _c("ok", "root", hide_status="NOT_HUSHED"),
        ]
        assert [p.comment["id"] for p in _pending(conversation)] == ["ok"]

    def test_a_commenter_is_answered_at_most_max_per_commenter_times_under_one_post(self):
        conversation = [
            _c("c1", "root", user="bot"),
            _c("r1", "c1", own=True),
            _c("c2", "r1", user="bot"),
            _c("r2", "c2", own=True),
            _c("c3", "r2", user="bot"),
            _c("other", "root", user="kim"),
        ]
        assert [p.comment["id"] for p in _pending(conversation, max_per_commenter=2)] == ["other"]
        assert [p.comment["id"] for p in _pending(conversation, max_per_commenter=3)] == ["c3", "other"]

    def test_the_cap_also_counts_the_answers_this_run_will_give(self):
        conversation = [_c(f"c{i}", "root", user="bot") for i in range(6)]
        assert len(_pending(conversation, max_per_commenter=3)) == 3

    def test_a_comment_on_the_root_sees_the_story_replies(self):
        conversation = [
            _c("story1", "root", own=True, text="스토리 1 본문"),
            _c("story2", "root", own=True, text="스토리 2 본문"),
            _c("c1", "root"),
        ]
        assert [t["id"] for t in _pending(conversation)[0].thread] == ["root", "story1", "story2"]


class TestCommentReplyLedger:
    def test_records_and_prunes_past_the_retention(self, tmp_path):
        ledger = CommentReplyLedger(LocalStateStore(tmp_path), retention=timedelta(days=14))
        ledger.record("old", "replied", NOW - timedelta(days=20))
        ledger.record("new", "declined", NOW)
        assert ledger.handled() == {"new"}

    def test_an_unreadable_store_raises_instead_of_reading_as_empty(self):
        store = MagicMock()
        store.read_json.side_effect = StateReadError("s3 down")
        with pytest.raises(StateReadError):
            CommentReplyLedger(store, retention=timedelta(days=14)).handled()


def _responder(tmp_path, decisions, **pipeline):
    config = Config()
    config.pipeline.comment_reply_max_per_run = pipeline.get("max_per_run", 10)
    factory = MagicMock()
    responder = CommentResponder(config, factory, LocalStateStore(tmp_path))
    responder.draft = AsyncMock(side_effect=decisions)
    return responder


def _patch_threads(conversation, publish=None):
    return (
        patch.object(comment_replies, "resolve_secret", side_effect=["tok", "uid"]),
        patch.object(comment_replies, "fetch_own_posts", new=AsyncMock(return_value=[POST])),
        patch.object(comment_replies, "fetch_conversation", new=AsyncMock(return_value=conversation)),
        patch.object(comment_replies, "publish_reply", new=publish or AsyncMock(return_value="reply-id")),
    )


async def _run(responder, patches, **kwargs):
    with patches[0], patches[1], patches[2], patches[3] as publish:
        return await responder.run(NOW, **kwargs), publish


class TestCommentResponder:
    async def test_replies_declines_and_records_both(self, tmp_path):
        responder = _responder(tmp_path, [CommentReply(reply=True, text="답글이에요"), CommentReply(reply=False)])
        conversation = [_c("c1", "root", at="2026-10-16T09:00:00+0000"), _c("c2", "root")]
        run, publish = await _run(responder, _patch_threads(conversation))
        publish.assert_awaited_once_with(ANY_CLIENT, "uid", "tok", "답글이에요", "c1")
        assert [r["comment_id"] for r in run.replied] == ["c1"] and run.declined == ["c2"]
        assert responder.ledger.handled() == {"c1", "c2"}

    async def test_a_dry_run_posts_and_records_nothing(self, tmp_path):
        responder = _responder(tmp_path, [CommentReply(reply=True, text="답글")])
        run, publish = await _run(responder, _patch_threads([_c("c1", "root")]), dry_run=True)
        publish.assert_not_awaited()
        assert run.replied[0]["reply"] == "답글" and responder.ledger.handled() == set()

    async def test_a_failed_publish_stays_unrecorded_for_the_next_run(self, tmp_path):
        responder = _responder(tmp_path, [CommentReply(reply=True, text="답글")])
        failing = AsyncMock(side_effect=RuntimeError("400"))
        run, _ = await _run(responder, _patch_threads([_c("c1", "root")], publish=failing))
        assert run.failed == ["c1"] and responder.ledger.handled() == set()

    async def test_a_reply_threads_rejects_is_recorded_and_a_server_error_is_retried(self, tmp_path):
        def status_error(code):
            request = httpx.Request("POST", "https://graph.threads.net/v1.0/uid/threads")
            return httpx.HTTPStatusError("x", request=request, response=httpx.Response(code, request=request))

        responder = _responder(tmp_path, [CommentReply(reply=True, text="a"), CommentReply(reply=True, text="b")])
        publish = AsyncMock(side_effect=[status_error(400), status_error(503)])
        conversation = [_c("rejected", "root", at="2026-10-16T09:00:00+0000"), _c("flaky", "root")]
        run, _ = await _run(responder, _patch_threads(conversation, publish=publish))
        assert run.failed == ["rejected", "flaky"]
        assert responder.ledger.handled() == {"rejected"}

    async def test_the_run_cap_answers_the_oldest_first(self, tmp_path):
        responder = _responder(tmp_path, [CommentReply(reply=True, text="a")], max_per_run=1)
        conversation = [_c("newer", "root", at="2026-10-16T11:00:00+0000"), _c("older", "root")]
        run, _ = await _run(responder, _patch_threads(conversation))
        assert [r["comment_id"] for r in run.replied] == ["older"]

    async def test_draft_reads_the_verdict_from_the_model(self, tmp_path):
        responder = _responder(tmp_path, [])
        del responder.draft
        responder.llm = RunnableLambda(lambda _: CommentReply(reply=True, text="답글"))
        found = _pending([_c("c1", "root")])
        assert (await responder.draft(found[0])).text == "답글"

    async def test_a_reply_verdict_without_text_is_a_decline(self, tmp_path):
        responder = _responder(tmp_path, [CommentReply(reply=True, text="  ")])
        run, publish = await _run(responder, _patch_threads([_c("c1", "root")]))
        publish.assert_not_awaited()
        assert run.declined == ["c1"]

    async def test_a_failed_draft_stays_unrecorded(self, tmp_path):
        responder = _responder(tmp_path, [ValueError("unparseable")])
        run, publish = await _run(responder, _patch_threads([_c("c1", "root")]))
        publish.assert_not_awaited()
        assert run.failed == ["c1"] and responder.ledger.handled() == set()

    async def test_missing_credentials_raise(self, tmp_path):
        responder = _responder(tmp_path, [])
        with patch.object(comment_replies, "resolve_secret", return_value=""):
            with pytest.raises(RuntimeError, match="not configured"):
                await responder.run(NOW)


class _AnyClient:
    def __eq__(self, other):
        return isinstance(other, httpx.AsyncClient)


ANY_CLIENT = _AnyClient()


class TestHandler:
    def test_disabled_does_nothing(self):
        config = Config()
        with patch.object(comment_reply_handler.Config, "load", return_value=config):
            with patch.object(comment_reply_handler, "_run", new=AsyncMock()) as run:
                assert comment_reply_handler.handler({}, None)["body"] == "disabled"
        run.assert_not_awaited()

    def test_a_failed_comment_raises_so_the_errors_alarm_fires(self):
        config = Config()
        config.pipeline.enable_comment_replies = True
        with patch.object(comment_reply_handler.Config, "load", return_value=config):
            with patch.object(
                comment_reply_handler, "_run", new=AsyncMock(return_value=CommentReplyRun(failed=["c1"]))
            ):
                with pytest.raises(RuntimeError, match="c1"):
                    comment_reply_handler.handler({}, None)


class TestThreadsReads:
    async def test_follows_paging_and_drops_replies_with_the_token_out_of_the_url(self):
        pages = [
            {
                "data": [{"id": "a"}, {"id": "b", "is_reply": True}],
                "paging": {"next": "https://graph.threads.net/next"},
            },
            {"data": [{"id": "c"}]},
        ]
        seen: list[httpx.Request] = []

        def respond(request):
            seen.append(request)
            return httpx.Response(200, json=pages[len(seen) - 1])

        client = threads_handler.threads_client("secret-token", 5)
        client._transport = httpx.MockTransport(respond)
        async with client:
            posts = await threads_handler.fetch_own_posts(client, SINCE)
        assert [p["id"] for p in posts] == ["a", "c"]
        assert all("secret-token" not in str(r.url) for r in seen)
        assert seen[0].headers["Authorization"] == "Bearer secret-token"
