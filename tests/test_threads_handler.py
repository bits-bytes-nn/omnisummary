from unittest.mock import AsyncMock, MagicMock, patch

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
            assert await post_to_threads(root_text="hi") is False

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

        assert ok is True
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
        assert ok is True
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
                assert await post_to_threads(root_text="R") is False

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
        assert ok is True
        assert calls["n"] == 2  # failed once, retried once

    @pytest.mark.asyncio
    async def test_one_failing_reply_does_not_abandon_the_rest(self, monkeypatch):
        # A single reply that exhausts its indexing retries must not drop the remaining replies
        # — otherwise the thread posts a half-finished comment chain ("댓글이 달리다 말았다").
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_BACKOFF_SEC", 0)
        monkeypatch.setattr(threads_handler, "THREADS_REPLY_RETRY_ATTEMPTS", 2)
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
                ok = await post_to_threads(root_text="R", replies=["first", "second", "third"])
        assert ok is True
        # first and third land; second is attempted (and retried) but never blocks the others
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
        assert ok is True
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
        assert ok is False

    def test_is_media_not_found_detects_code_24(self):
        req = httpx.Request("POST", "https://x")
        resp = httpx.Response(400, request=req, json={"error": {"code": 24}})
        assert _is_media_not_found(httpx.HTTPStatusError("e", request=req, response=resp))
        other = httpx.Response(400, request=req, json={"error": {"code": 100}})
        assert not _is_media_not_found(httpx.HTTPStatusError("e", request=req, response=other))
