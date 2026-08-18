import time
from unittest.mock import AsyncMock, MagicMock, patch
from urllib.parse import parse_qs

import httpx
import pytest

from output import threads_handler
from output.threads_handler import _is_media_not_found, post_to_threads


def _ctx(obj):
    """Wrap an object as an async context manager (stand-in for httpx.AsyncClient(...))."""
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=obj)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


class TestPostToThreads:
    @pytest.mark.asyncio
    async def test_skips_without_credentials(self):
        with patch.object(threads_handler, "resolve_secret", return_value=""):
            assert (await post_to_threads(root_text="hi")).published is False

    @pytest.mark.asyncio
    async def test_posts_flat_replies_under_root(self):
        # Root + one reply PER pre-rendered item, every reply hanging off the ROOT (a flat
        # thread, NOT nested reply-of-reply); an over-long reply is hard-capped, still one post.
        published: list[dict] = []

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            pid = f"id{len(published)}"
            published.append({"text": text, "image_url": image_url, "reply_to_id": reply_to_id, "id": pid})
            return pid

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                ok = await post_to_threads(root_text="ROOT", replies=["reply one", "a" * 1100])

        assert ok.published is True
        assert len(published) == 3  # root + exactly 2 replies (one per input reply)
        assert published[0]["reply_to_id"] == "" and published[0]["text"] == "ROOT"
        # both replies point at the ROOT (id0), not at each other
        assert published[1]["reply_to_id"] == "id0"
        assert published[2]["reply_to_id"] == "id0"
        assert len(published[2]["text"]) <= 500  # over-long reply hard-capped, not re-split

    @pytest.mark.asyncio
    async def test_hosts_image_and_posts_with_url(self):
        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_upload_image_for_hosting", return_value="https://s3/img.png") as up:
                with patch.object(threads_handler, "_publish_post", new=AsyncMock(return_value="rid")) as pub:
                    ok = await post_to_threads(
                        root_text="R", replies=[], image_bytes=b"PNG", image_bucket="b", image_key="k.png"
                    )
        assert ok.published is True
        up.assert_called_once()
        # the root publish call received the hosted image url
        assert pub.await_args_list[0].kwargs["image_url"] == "https://s3/img.png"

    @pytest.mark.asyncio
    async def test_api_failure_returns_false(self):
        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        resp = httpx.Response(400, request=req, text="bad")
        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(
                threads_handler,
                "_publish_post",
                new=AsyncMock(side_effect=httpx.HTTPStatusError("err", request=req, response=resp)),
            ):
                assert (await post_to_threads(root_text="R")).published is False

    @pytest.mark.asyncio
    async def test_reply_retries_on_media_not_found(self, monkeypatch):
        # The just-published root isn't instantly addressable as a reply target; the first
        # reply attempt 400s with code 24 and must be retried, not dropped.
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_BACKOFF_SEC", 0)
        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        resp = httpx.Response(400, request=req, json={"error": {"code": 24, "error_subcode": 4279009}})
        not_found = httpx.HTTPStatusError("media not found", request=req, response=resp)

        calls = {"n": 0}

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            if reply_to_id:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise not_found  # first reply attempt: target not indexed yet
            return "id"

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                ok = await post_to_threads(root_text="R", replies=["only reply"])
        assert ok.published is True
        assert calls["n"] == 2  # failed once, retried once

    @pytest.mark.asyncio
    async def test_one_failing_reply_does_not_abandon_the_rest(self, monkeypatch):
        # A single reply that exhausts the shared indexing budget must not drop the remaining
        # replies — otherwise the thread posts a half-finished comment chain ("댓글이 달리다 말았다").
        # A code-24 reply now rides the shared indexing deadline, so mock the clock and set a small
        # budget: the failing reply drains it, then the rest still post (they see the budget spent).
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_BACKOFF_SEC", 10)
        monkeypatch.setattr(threads_handler, "THREADS_INDEXING_BUDGET_SEC", 30)
        clock = {"t": 0.0}
        monkeypatch.setattr(threads_handler.time, "monotonic", lambda: clock["t"])

        async def fake_sleep(sec):
            clock["t"] += sec

        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        resp = httpx.Response(400, request=req, json={"error": {"code": 24, "error_subcode": 4279009}})
        not_found = httpx.HTTPStatusError("media not found", request=req, response=resp)

        seen: list[str] = []

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            if reply_to_id:
                seen.append(text)
                if text == "second":  # second reply never indexes
                    raise not_found
            return "rid"

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                with patch.object(threads_handler.asyncio, "sleep", side_effect=fake_sleep):
                    ok = await post_to_threads(root_text="R", replies=["first", "second", "third"])
        assert ok.published is True
        # first and third land; second is attempted (and retried to the budget) but never blocks the others
        assert "first" in seen and "third" in seen

    @pytest.mark.asyncio
    async def test_polls_root_readiness_before_replies(self, monkeypatch):
        # An image root isn't instantly addressable. The handler must POLL it (cheap GET) until
        # ready and only THEN post replies — so a reply doesn't 400 on an un-indexed root.
        monkeypatch.setattr(threads_handler, "THREADS_READINESS_POLL_INTERVAL_SEC", 0)
        monkeypatch.setattr(threads_handler.asyncio, "sleep", AsyncMock())

        get_calls = {"n": 0}

        class FakeClient:
            async def get(self, url, params=None):
                get_calls["n"] += 1
                # not ready on the first probe, ready on the second
                return httpx.Response(200 if get_calls["n"] >= 2 else 400, request=httpx.Request("GET", url))

        published: list[str] = []

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            published.append(reply_to_id or "root")
            return "rid"

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_upload_image_for_hosting", return_value="https://s3/i.png"):
                with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                    with patch.object(threads_handler.httpx, "AsyncClient", return_value=_ctx(FakeClient())):
                        ok = await post_to_threads(
                            root_text="R", replies=["only"], image_bytes=b"P", image_bucket="b", image_key="k"
                        )
        assert ok.published is True
        assert get_calls["n"] >= 2  # polled until ready
        assert published == ["root", "rid"]  # reply posted after the root indexed

    @pytest.mark.asyncio
    async def test_root_never_indexes_reports_failure(self, monkeypatch):
        # If the image root never becomes addressable within the budget, no replies land → a
        # lone-image, story-less digest. Report failure so the ledger rollback keeps it retryable.
        monkeypatch.setattr(threads_handler, "THREADS_READINESS_POLL_INTERVAL_SEC", 10)
        monkeypatch.setattr(threads_handler, "THREADS_INDEXING_BUDGET_SEC", 100)

        clock = {"t": 0.0}
        monkeypatch.setattr(threads_handler.time, "monotonic", lambda: clock["t"])

        async def fake_sleep(sec):
            clock["t"] += sec

        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        resp = httpx.Response(400, request=req, json={"error": {"code": 24, "error_subcode": 4279009}})
        not_found = httpx.HTTPStatusError("media not found", request=req, response=resp)

        class FakeClient:
            async def get(self, url, params=None):
                return httpx.Response(400, request=httpx.Request("GET", url))  # never ready

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            if reply_to_id:  # the un-indexed root can't accept replies
                raise not_found
            return "rid"

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_upload_image_for_hosting", return_value="https://s3/i.png"):
                with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                    with patch.object(threads_handler.asyncio, "sleep", side_effect=fake_sleep):
                        with patch.object(threads_handler.httpx, "AsyncClient", return_value=_ctx(FakeClient())):
                            ok = await post_to_threads(
                                root_text="R", replies=["a"], image_bytes=b"P", image_bucket="b", image_key="k"
                            )
        # poll never succeeds and the reply can't land → overall failure (retryable)
        assert ok.published is False

    @pytest.mark.asyncio
    async def test_media_not_found_retries_past_fixed_cap_until_budget(self, monkeypatch):
        # Regression (2026-07-25): the up-front GET poll reported the image root addressable, but
        # it wasn't yet a valid REPLY target, so the first replies 400'd code-24 and each burned only
        # its 3 short attempts — dropping the Opus-5 headline + one more. A code-24 reply must ride
        # the SHARED indexing deadline (many attempts), not the fixed 3-attempt cap.
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_BACKOFF_SEC", 10)
        monkeypatch.setattr(threads_handler, "THREADS_INDEXING_BUDGET_SEC", 200)
        clock = {"t": 0.0}
        monkeypatch.setattr(threads_handler.time, "monotonic", lambda: clock["t"])

        async def fake_sleep(sec):
            clock["t"] += sec

        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        resp = httpx.Response(400, request=req, json={"error": {"code": 24, "error_subcode": 4279009}})
        not_found = httpx.HTTPStatusError("media not found", request=req, response=resp)

        calls = {"n": 0}

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            if reply_to_id:
                calls["n"] += 1
                if calls["n"] <= 6:  # indexing lags well past the fixed 3-attempt cap
                    raise not_found
            return "id"

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                with patch.object(threads_handler.asyncio, "sleep", side_effect=fake_sleep):
                    ok = await post_to_threads(root_text="R", replies=["headline reply"])
        assert ok.published is True
        assert calls["n"] == 7  # retried 6 code-24 failures (past the 3-cap) then landed

    @pytest.mark.asyncio
    async def test_transient_400_still_capped_at_fixed_attempts(self, monkeypatch):
        # A non-code-24 transient 400 (container processing) is NOT indexing lag, so it must stay
        # capped at the fixed attempt count — not ride the long indexing deadline.
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_ATTEMPTS", 3)
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_BACKOFF_SEC", 10)
        monkeypatch.setattr(threads_handler, "THREADS_INDEXING_BUDGET_SEC", 200)
        clock = {"t": 0.0}
        monkeypatch.setattr(threads_handler.time, "monotonic", lambda: clock["t"])

        async def fake_sleep(sec):
            clock["t"] += sec

        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        resp = httpx.Response(400, request=req, json={"error": {"code": 100, "message": "processing"}})
        transient = httpx.HTTPStatusError("bad", request=req, response=resp)

        calls = {"n": 0}

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            if reply_to_id:
                calls["n"] += 1
                raise transient  # never recovers
            return "id"

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                with patch.object(threads_handler.asyncio, "sleep", side_effect=fake_sleep):
                    ok = await post_to_threads(root_text="R", replies=["only reply"])
        assert ok.published is False
        assert calls["n"] == 3  # capped at the fixed attempt count, not the indexing budget

    @pytest.mark.asyncio
    async def test_reply_retries_on_transient_400(self, monkeypatch):
        # A container-processing 400 that is NOT "media not found" (the 2026-07-17 failure that
        # silently dropped one story) must be retried, not dropped on the first attempt.
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_BACKOFF_SEC", 0)
        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        resp = httpx.Response(400, request=req, json={"error": {"code": 100, "message": "processing"}})
        transient = httpx.HTTPStatusError("bad", request=req, response=resp)

        calls = {"n": 0}

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            if reply_to_id:
                calls["n"] += 1
                if calls["n"] == 1:
                    raise transient  # first attempt: transient processing failure
            return "id"

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                ok = await post_to_threads(root_text="R", replies=["only reply"])
        assert ok.published is True
        assert calls["n"] == 2  # failed once, retried once, then succeeded

    @pytest.mark.asyncio
    async def test_reply_does_not_retry_on_auth_error(self, monkeypatch):
        # A non-transient error (401 auth) must raise immediately — retrying wastes the budget and
        # a bad token won't heal. The reply is dropped (best-effort) without burning 3 attempts.
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_BACKOFF_SEC", 0)
        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        resp = httpx.Response(401, request=req, json={"error": {"code": 190, "message": "bad token"}})
        auth_err = httpx.HTTPStatusError("unauthorized", request=req, response=resp)

        calls = {"n": 0}

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            if reply_to_id:
                calls["n"] += 1
                raise auth_err
            return "id"

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                ok = await post_to_threads(root_text="R", replies=["only reply"])
        # root posted, sole reply failed → all replies failed → overall failure (retryable)
        assert ok.published is False
        assert calls["n"] == 1  # raised on the first attempt, no retry

    @pytest.mark.asyncio
    async def test_counts_delivered_posts_and_flags_a_partial_chain(self, monkeypatch):
        # Regression: a digest whose reply chain lost one story reported plain success, so nothing
        # downstream could tell 4-of-6 from 6-of-6. The outcome now carries both counts.
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_BACKOFF_SEC", 0)
        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        resp = httpx.Response(401, request=req, json={"error": {"code": 190}})
        auth_err = httpx.HTTPStatusError("unauthorized", request=req, response=resp)

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            if text == "second":
                raise auth_err
            return "id"

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                ok = await post_to_threads(root_text="R", replies=["first", "second", "third"])
        assert (ok.posted, ok.expected) == (3, 4)  # root + 2 of 3 replies
        assert ok.published is True
        assert ok.partial is True
        assert ok.summary() == "3/4 posts"

    @pytest.mark.asyncio
    async def test_missing_credentials_report_every_intended_post_as_undelivered(self):
        # Empty credentials used to be an INFO "skipping" line: the day's only delivery path
        # silently did nothing. The counts now say the whole digest was undelivered.
        with patch.object(threads_handler, "resolve_secret", return_value=""):
            ok = await post_to_threads(root_text="R", replies=["a", "b"])
        assert (ok.posted, ok.expected) == (0, 3)
        assert ok.published is False
        assert ok.partial is False

    @pytest.mark.asyncio
    async def test_a_complete_post_is_not_partial(self):
        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", new=AsyncMock(return_value="id")):
                ok = await post_to_threads(root_text="R", replies=["a", "b"])
        assert (ok.posted, ok.expected) == (3, 3)
        assert ok.partial is False

    def test_is_media_not_found_detects_code_24(self):
        req = httpx.Request("POST", "https://x")
        resp = httpx.Response(400, request=req, json={"error": {"code": 24}})
        assert _is_media_not_found(httpx.HTTPStatusError("e", request=req, response=resp))
        other = httpx.Response(400, request=req, json={"error": {"code": 100}})
        assert not _is_media_not_found(httpx.HTTPStatusError("e", request=req, response=other))

    def test_is_transient_reply_error_classification(self):
        req = httpx.Request("POST", "https://x")

        def err(status, body=None):
            resp = httpx.Response(status, request=req, json=body or {})
            return httpx.HTTPStatusError("e", request=req, response=resp)

        assert threads_handler._is_transient_reply_error(err(400, {"error": {"code": 24}}))  # media-not-found
        assert threads_handler._is_transient_reply_error(err(400, {"error": {"code": 100}}))  # processing
        assert threads_handler._is_transient_reply_error(err(429))
        assert threads_handler._is_transient_reply_error(err(503))
        assert not threads_handler._is_transient_reply_error(err(401))  # auth: don't retry
        assert not threads_handler._is_transient_reply_error(err(403))


class TestPublishPost:
    @pytest.mark.asyncio
    async def test_text_post_sends_text_media_type_and_no_wait(self):
        client = MagicMock()
        with patch.object(threads_handler, "_create_container", new=AsyncMock(return_value="c1")) as create:
            with patch.object(threads_handler, "_publish_container", new=AsyncMock(return_value="p1")) as publish:
                with patch.object(threads_handler.asyncio, "sleep", new=AsyncMock()) as sleep:
                    post_id = await threads_handler._publish_post(client, "u", "tok", text="hello")
        assert post_id == "p1"
        assert create.await_args.kwargs["media_type"] == "TEXT"
        assert create.await_args.kwargs["text"] == "hello"
        assert "image_url" not in create.await_args.kwargs
        publish.assert_awaited_once()
        sleep.assert_not_awaited()  # only an image container needs the processing wait

    @pytest.mark.asyncio
    async def test_image_post_waits_for_media_processing(self):
        client = MagicMock()
        with patch.object(threads_handler, "_create_container", new=AsyncMock(return_value="c1")):
            with patch.object(threads_handler, "_publish_container", new=AsyncMock(return_value="p1")):
                with patch.object(threads_handler.asyncio, "sleep", new=AsyncMock()) as sleep:
                    await threads_handler._publish_post(client, "u", "tok", text="t", image_url="https://s3/i.png")
        sleep.assert_awaited_once_with(threads_handler.THREADS_MEDIA_PROCESS_WAIT_SEC)

    @pytest.mark.asyncio
    async def test_text_is_hard_capped_at_the_api_limit(self):
        client = MagicMock()
        with patch.object(threads_handler, "_create_container", new=AsyncMock(return_value="c1")) as create:
            with patch.object(threads_handler, "_publish_container", new=AsyncMock(return_value="p1")):
                await threads_handler._publish_post(client, "u", "tok", text="x" * 900, reply_to_id="rid")
        assert len(create.await_args.kwargs["text"]) == threads_handler.THREADS_MAX_TEXT_LENGTH
        assert create.await_args.kwargs["reply_to_id"] == "rid"

    @pytest.mark.asyncio
    async def test_create_container_posts_token_and_returns_id(self):
        req = httpx.Request("POST", "https://graph.threads.net/v1.0/u/threads")
        client = MagicMock()
        client.post = AsyncMock(return_value=httpx.Response(200, request=req, json={"id": "c9"}))
        assert await threads_handler._create_container(client, "u", "tok", media_type="TEXT") == "c9"
        assert client.post.await_args.kwargs["data"]["access_token"] == "tok"


class TestWireProtocol:
    """Every other test in this file patches `_publish_post` or `_create_container`, so nothing ever
    checked what actually goes ON THE WIRE — and that is the layer both 2026-08 incidents lived in.
    These drive the real request-building code through an httpx MockTransport: no new dependency,
    nothing patched but the transport and the credentials."""

    @staticmethod
    def _client_factory(recorded: list[httpx.Request]):
        def handle(request: httpx.Request) -> httpx.Response:
            recorded.append(request)
            if request.method == "GET":  # readiness probe
                return httpx.Response(200, json={"id": "root"})
            if request.url.path.endswith("/threads_publish"):
                return httpx.Response(200, json={"id": f"published-{len(recorded)}"})
            return httpx.Response(200, json={"id": f"container-{len(recorded)}"})

        transport = httpx.MockTransport(handle)
        # Bind the REAL class now: threads_handler.httpx is the global module, so patching
        # httpx.AsyncClient would make this factory call itself and recurse forever.
        real_client = httpx.AsyncClient
        return lambda *a, **kw: real_client(transport=transport)

    @staticmethod
    def _form(request: httpx.Request) -> dict[str, str]:
        return {k: v[0] for k, v in parse_qs(request.content.decode()).items()}

    @pytest.mark.asyncio
    async def test_image_root_then_reply_hit_the_documented_two_step_endpoints(self):
        recorded: list[httpx.Request] = []
        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_upload_image_for_hosting", return_value="https://s3/i.png"):
                with patch.object(threads_handler.asyncio, "sleep", new=AsyncMock()):
                    with patch.object(threads_handler.httpx, "AsyncClient", new=self._client_factory(recorded)):
                        ok = await post_to_threads(
                            root_text="LEAD",
                            replies=["STORY"],
                            image_bytes=b"PNG",
                            image_bucket="b",
                            image_key="k.png",
                        )
        assert ok.published is True
        posts = [r for r in recorded if r.method == "POST"]
        assert [r.url.path for r in posts] == [
            "/v1.0/user1/threads",  # create the image container
            "/v1.0/user1/threads_publish",  # publish it
            "/v1.0/user1/threads",  # create the reply container
            "/v1.0/user1/threads_publish",  # publish the reply
        ]
        # every call is authenticated
        assert all(self._form(r)["access_token"] == "tok" for r in posts)

        root_create, root_publish, reply_create, _ = (self._form(r) for r in posts)
        assert root_create["media_type"] == "IMAGE"
        assert root_create["image_url"] == "https://s3/i.png"
        assert root_create["text"] == "LEAD"
        # the publish step must carry the id the CREATE step returned, not a re-derived one
        assert root_publish["creation_id"] == "container-1"
        # the reply is a TEXT post pointing at the PUBLISHED root id
        assert reply_create["media_type"] == "TEXT"
        assert reply_create["reply_to_id"] == "published-2"
        assert "image_url" not in reply_create

    @pytest.mark.asyncio
    async def test_overlong_reply_is_capped_before_it_reaches_the_api(self):
        # Threads rejects >500 chars outright; the cap has to be applied on the request, not just in
        # the renderer, or one long story 400s the whole reply.
        recorded: list[httpx.Request] = []
        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler.asyncio, "sleep", new=AsyncMock()):
                with patch.object(threads_handler.httpx, "AsyncClient", new=self._client_factory(recorded)):
                    ok = await post_to_threads(root_text="R", replies=["가" * 900])
        assert ok.published is True
        creates = [self._form(r) for r in recorded if r.url.path.endswith("/threads")]
        assert all(len(c["text"]) <= threads_handler.THREADS_MAX_TEXT_LENGTH for c in creates)

    @pytest.mark.asyncio
    async def test_api_rejection_on_the_publish_step_is_reported_not_swallowed(self):
        # A 400 on threads_publish (as opposed to the create step) must surface as failure.
        def handle(request: httpx.Request) -> httpx.Response:
            if request.url.path.endswith("/threads_publish"):
                return httpx.Response(400, json={"error": {"code": 100, "message": "nope"}})
            return httpx.Response(200, json={"id": "c1"})

        transport = httpx.MockTransport(handle)
        real_client = httpx.AsyncClient
        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler.asyncio, "sleep", new=AsyncMock()):
                with patch.object(
                    threads_handler.httpx, "AsyncClient", new=lambda *a, **kw: real_client(transport=transport)
                ):
                    assert (await post_to_threads(root_text="R", replies=["one"])).published is False


class TestImageHosting:
    def test_uploads_with_content_type_and_presigns(self):
        s3 = MagicMock()
        s3.generate_presigned_url.return_value = "https://s3/presigned"
        with patch.object(threads_handler.boto3, "client", return_value=s3):
            url = threads_handler._upload_image_for_hosting(b"PNG", "bucket", "k/p.jpg", content_type="image/jpeg")
        assert url == "https://s3/presigned"
        assert s3.put_object.call_args.kwargs["ContentType"] == "image/jpeg"
        assert s3.put_object.call_args.kwargs["Key"] == "k/p.jpg"
        assert s3.generate_presigned_url.call_args.kwargs["ExpiresIn"] == threads_handler.THREADS_IMAGE_URL_TTL_SEC

    @pytest.mark.asyncio
    async def test_hosting_failure_falls_back_to_text_only(self):
        # A dead S3 upload must not sink the digest: the root still posts, just without the image.
        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_upload_image_for_hosting", side_effect=RuntimeError("denied")):
                with patch.object(threads_handler, "_publish_post", new=AsyncMock(return_value="rid")) as pub:
                    ok = await post_to_threads(
                        root_text="R", replies=[], image_bytes=b"PNG", image_bucket="b", image_key="k.png"
                    )
        assert ok.published is True
        assert pub.await_args_list[0].kwargs["image_url"] == ""

    @pytest.mark.asyncio
    async def test_no_bucket_configured_posts_text_only(self):
        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_upload_image_for_hosting") as up:
                with patch.object(threads_handler, "_publish_post", new=AsyncMock(return_value="rid")) as pub:
                    ok = await post_to_threads(root_text="R", image_bytes=b"PNG", image_bucket="", image_key="")
        assert ok.published is True
        up.assert_not_called()
        assert pub.await_args_list[0].kwargs["image_url"] == ""


class TestAddressability:
    @pytest.mark.asyncio
    async def test_addressable_on_200_and_not_on_400(self):
        req = httpx.Request("GET", "https://graph.threads.net/v1.0/rid")
        client = MagicMock()
        client.get = AsyncMock(return_value=httpx.Response(200, request=req, json={"id": "rid"}))
        assert await threads_handler._is_addressable(client, "rid", "tok") is True
        client.get = AsyncMock(return_value=httpx.Response(400, request=req, text="media not found"))
        assert await threads_handler._is_addressable(client, "rid", "tok") is False

    @pytest.mark.asyncio
    async def test_transport_error_is_not_addressable(self):
        client = MagicMock()
        client.get = AsyncMock(side_effect=httpx.ConnectError("no route"))
        assert await threads_handler._is_addressable(client, "rid", "tok") is False

    def test_error_detail_survives_an_unreadable_body(self):
        exc = MagicMock()
        type(exc).response = property(lambda self: (_ for _ in ()).throw(RuntimeError("gone")))
        assert threads_handler._error_detail(exc) == "<no response body>"


class TestIndexingBudget:
    """The indexing wait may be bounded by the CALLER's remaining time, but never shortened while
    the full budget still fits — too little indexing patience is what dropped stories before."""

    def test_no_deadline_keeps_the_full_budget(self):
        assert threads_handler._indexing_budget_sec(None) == float(threads_handler.THREADS_INDEXING_BUDGET_SEC)

    def test_generous_deadline_keeps_the_full_budget(self):
        deadline = time.monotonic() + threads_handler.THREADS_INDEXING_BUDGET_SEC * 3
        assert threads_handler._indexing_budget_sec(deadline) == float(threads_handler.THREADS_INDEXING_BUDGET_SEC)

    def test_tight_deadline_reserves_room_for_the_reply_chain(self):
        deadline = time.monotonic() + threads_handler.THREADS_PUBLISH_RESERVE_SEC + 40
        budget = threads_handler._indexing_budget_sec(deadline)
        assert 30 < budget <= 40

    def test_expired_deadline_waits_not_at_all(self):
        assert threads_handler._indexing_budget_sec(time.monotonic() - 5) == 0.0
