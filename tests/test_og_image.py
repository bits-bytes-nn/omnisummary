from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from shared.media import og_image

_PAGE_WITH_OG = """
<html><head>
<meta property="og:image" content="https://cdn.example.com/pic.jpg">
<meta property="og:title" content="A Headline">
</head><body>hi</body></html>
"""

_PAGE_RELATIVE = """
<html><head><meta property="og:image" content="/images/pic.png"></head></html>
"""

_PAGE_TWITTER = """
<html><head><meta name="twitter:image" content="https://cdn.example.com/tw.png"></head></html>
"""

_PAGE_NONE = "<html><head><title>nothing</title></head></html>"


def _stream_response(content_type="image/jpeg", body=b"\xff\xd8\xff\xff", content_length=None, chunk=8):
    """A streaming image response: an async context manager exposing headers + aiter_bytes()."""
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    headers = {"content-type": content_type}
    if content_length is not None:
        headers["content-length"] = str(content_length)
    resp.headers = headers

    async def _aiter():
        for i in range(0, len(body), chunk):
            yield body[i : i + chunk]

    resp.aiter_bytes = _aiter
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _page_response(html, url="https://example.com/post"):
    resp = MagicMock()
    resp.raise_for_status = MagicMock()
    resp.text = html
    resp.url = url
    return resp


def _client_with(page_resp, stream_ctx):
    client = MagicMock()
    client.get = AsyncMock(return_value=page_resp)
    client.stream = MagicMock(return_value=stream_ctx)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx, client


class TestExtractImageUrl:
    def test_og_image_and_alt(self):
        url, alt = og_image._extract_image_url(_PAGE_WITH_OG, "https://example.com/post")
        assert url == "https://cdn.example.com/pic.jpg"
        assert alt == "A Headline"

    def test_relative_url_resolved(self):
        url, _ = og_image._extract_image_url(_PAGE_RELATIVE, "https://example.com/post")
        assert url == "https://example.com/images/pic.png"

    def test_twitter_fallback(self):
        url, _ = og_image._extract_image_url(_PAGE_TWITTER, "https://example.com/post")
        assert url == "https://cdn.example.com/tw.png"

    def test_none_present(self):
        url, alt = og_image._extract_image_url(_PAGE_NONE, "https://example.com/post")
        assert url == "" and alt == ""


class TestFetchOgImage:
    @pytest.mark.asyncio
    async def test_happy_path_returns_asset(self):
        ctx, _ = _client_with(_page_response(_PAGE_WITH_OG), _stream_response())
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            asset = await og_image.fetch_og_image("https://example.com/post")
        assert asset is not None
        assert asset.image_url == "https://cdn.example.com/pic.jpg"
        assert asset.source_url == "https://example.com/post"
        assert asset.alt == "A Headline"
        assert asset.content_type == "image/jpeg"
        assert asset.data == b"\xff\xd8\xff\xff"

    @pytest.mark.asyncio
    async def test_no_og_tag_returns_none(self):
        ctx, _ = _client_with(_page_response(_PAGE_NONE), _stream_response())
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            asset = await og_image.fetch_og_image("https://example.com/post")
        assert asset is None

    @pytest.mark.asyncio
    async def test_non_renderable_content_type_rejected(self):
        # SVG is an image/* type but not in the renderable allowlist — must be rejected.
        ctx, _ = _client_with(_page_response(_PAGE_WITH_OG), _stream_response(content_type="image/svg+xml"))
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            asset = await og_image.fetch_og_image("https://example.com/post")
        assert asset is None

    @pytest.mark.asyncio
    async def test_oversize_image_rejected_mid_stream(self):
        ctx, _ = _client_with(_page_response(_PAGE_WITH_OG), _stream_response(body=b"x" * 100))
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            asset = await og_image.fetch_og_image("https://example.com/post", max_bytes=10)
        assert asset is None

    @pytest.mark.asyncio
    async def test_oversize_rejected_by_content_length(self):
        # The declared Content-Length rejects before any body is read.
        ctx, _ = _client_with(_page_response(_PAGE_WITH_OG), _stream_response(body=b"x", content_length=999999))
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            asset = await og_image.fetch_og_image("https://example.com/post", max_bytes=10)
        assert asset is None

    @pytest.mark.asyncio
    async def test_empty_body_returns_none(self):
        ctx, _ = _client_with(_page_response(_PAGE_WITH_OG), _stream_response(body=b""))
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            asset = await og_image.fetch_og_image("https://example.com/post")
        assert asset is None

    @pytest.mark.asyncio
    async def test_network_error_returns_none(self):
        client = MagicMock()
        client.get = AsyncMock(side_effect=RuntimeError("boom"))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            asset = await og_image.fetch_og_image("https://example.com/post")
        assert asset is None


class TestExtensionFor:
    def test_known_types(self):
        assert og_image.extension_for("image/jpeg") == "jpg"
        assert og_image.extension_for("image/webp") == "webp"
        assert og_image.extension_for("image/gif") == "gif"

    def test_unknown_defaults_png(self):
        assert og_image.extension_for("application/octet-stream") == "png"
        assert og_image.extension_for("image/png; charset=binary") == "png"
