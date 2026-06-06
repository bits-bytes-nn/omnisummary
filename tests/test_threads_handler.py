from unittest.mock import AsyncMock, patch

import httpx
import pytest

from output import threads_handler
from output.threads_handler import _is_media_not_found, _split_text, post_to_threads


class TestSplitText:
    def test_short_text_single_chunk(self):
        assert _split_text("hello") == ["hello"]

    def test_empty_text_no_chunks(self):
        assert _split_text("") == []

    def test_long_text_respects_max_len(self):
        chunks = _split_text("x" * 1200, max_len=500)
        assert all(len(c) <= 500 for c in chunks)
        assert "".join(chunks) == "x" * 1200

    def test_splits_on_paragraph_boundaries(self):
        text = "para one\n\n" + ("y" * 480) + "\n\npara three"
        chunks = _split_text(text, max_len=500)
        assert len(chunks) >= 2
        assert all(len(c) <= 500 for c in chunks)


class TestPostToThreads:
    @pytest.mark.asyncio
    async def test_skips_without_credentials(self):
        with patch.object(threads_handler, "resolve_secret", return_value=""):
            assert await post_to_threads(root_text="hi") is False

    @pytest.mark.asyncio
    async def test_posts_root_and_reply_chain(self):
        # token + user id resolve; verify the root post + one reply per body chunk,
        # each reply threaded onto the previous post id.
        published: list[dict] = []

        async def fake_publish(client, user_id, token, *, text="", image_url="", reply_to_id=""):
            pid = f"id{len(published)}"
            published.append({"text": text, "image_url": image_url, "reply_to_id": reply_to_id, "id": pid})
            return pid

        with patch.object(threads_handler, "resolve_secret", side_effect=["tok", "user1"]):
            with patch.object(threads_handler, "_publish_post", side_effect=fake_publish):
                ok = await post_to_threads(root_text="ROOT", replies=["a" * 1100])

        assert ok is True
        # root has no reply_to_id; subsequent posts chain onto the prior id
        assert published[0]["reply_to_id"] == ""
        assert published[0]["text"] == "ROOT"
        assert published[1]["reply_to_id"] == "id0"
        assert all(p["reply_to_id"] == f"id{i - 1}" for i, p in enumerate(published) if i >= 1)

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

    def test_is_media_not_found_detects_code_24(self):
        req = httpx.Request("POST", "https://x")
        resp = httpx.Response(400, request=req, json={"error": {"code": 24}})
        assert _is_media_not_found(httpx.HTTPStatusError("e", request=req, response=resp))
        other = httpx.Response(400, request=req, json={"error": {"code": 100}})
        assert not _is_media_not_found(httpx.HTTPStatusError("e", request=req, response=other))
