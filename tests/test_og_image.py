from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from adapters import og_image

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


@pytest.fixture(autouse=True)
def _stub_dns():
    """Resolution is stubbed (never real DNS): the guard resolves every hop's host, and tests must
    not depend on the network. Public address by default."""
    with patch.object(og_image, "_resolve_addresses", return_value=["93.184.216.34"]):
        yield


def _stream_response(
    content_type="image/jpeg", body=b"\xff\xd8\xff\xff", content_length=None, chunk=8, location="", url=""
):
    """A streaming response: an async context manager exposing headers + aiter_bytes()."""
    resp = MagicMock()
    resp.url = url
    resp.encoding = "utf-8"
    resp.raise_for_status = MagicMock()
    headers = {"content-type": content_type}
    if content_length is not None:
        headers["content-length"] = str(content_length)
    if location:
        headers["location"] = location
    resp.headers = headers
    resp.is_redirect = bool(location)

    async def _aiter():
        for i in range(0, len(body), chunk):
            yield body[i : i + chunk]

    resp.aiter_bytes = _aiter
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=resp)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


def _page_response(html, url="https://example.com/post", location=""):
    return _stream_response(content_type="text/html", body=html.encode(), location=location, url=url)


def _client_with(page_resp, stream_ctx):
    client = MagicMock()
    client.stream = MagicMock(side_effect=[page_resp, stream_ctx])
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
        client.stream = MagicMock(side_effect=RuntimeError("boom"))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            asset = await og_image.fetch_og_image("https://example.com/post")
        assert asset is None


class TestTheWholeFetchIsBounded:
    @pytest.mark.asyncio
    async def test_an_oversize_page_is_abandoned(self):
        ctx, client = _client_with(_page_response(_PAGE_WITH_OG + "x" * 50_000), _stream_response())
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            assert await og_image.fetch_og_image("https://example.com/post", max_bytes=10_000) is None
        assert client.stream.call_count == 1  # the image was never requested

    @pytest.mark.asyncio
    async def test_a_trickling_server_cannot_hold_the_call(self):
        # httpx's timeout is per read, so a byte every few seconds never tripped it.
        import asyncio

        slow = _page_response(_PAGE_WITH_OG)
        resp = await slow.__aenter__()

        async def _trickle():
            while True:
                await asyncio.sleep(0.05)
                yield b"x"

        resp.aiter_bytes = _trickle
        ctx, _ = _client_with(slow, _stream_response())
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            assert await asyncio.wait_for(og_image.fetch_og_image("https://example.com/post", timeout=0.3), 5) is None


class TestExtensionFor:
    def test_known_types(self):
        assert og_image.extension_for("image/jpeg") == "jpg"
        # Meta documents only JPEG and PNG for Threads image posts.
        assert "image/webp" not in og_image._RENDERABLE_IMAGE_TYPES

    def test_unknown_defaults_png(self):
        assert og_image.extension_for("application/octet-stream") == "png"
        assert og_image.extension_for("image/png; charset=binary") == "png"


class TestFetchGuards:
    @pytest.mark.asyncio
    async def test_private_host_is_not_fetched(self):
        ctx, client = _client_with(_page_response(_PAGE_WITH_OG), _stream_response())
        with patch.object(og_image, "_resolve_addresses", return_value=["127.0.0.1"]):
            with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
                assert await og_image.fetch_og_image("http://localhost/post") is None
        client.stream.assert_not_called()  # rejected before any request left the process

    @pytest.mark.asyncio
    async def test_link_local_metadata_address_is_not_fetched(self):
        with patch.object(og_image, "_resolve_addresses", return_value=["169.254.169.254"]):
            assert await og_image._is_fetchable("http://metadata.example/latest") is False

    @pytest.mark.asyncio
    async def test_non_http_scheme_is_not_fetched(self):
        assert await og_image._is_fetchable("file:///etc/passwd") is False
        assert await og_image._is_fetchable("gopher://example.com/") is False

    @pytest.mark.asyncio
    async def test_unresolvable_host_is_not_fetched(self):
        with patch.object(og_image, "_resolve_addresses", side_effect=OSError("no such host")):
            assert await og_image._is_fetchable("https://nope.example/") is False

    @pytest.mark.asyncio
    async def test_mixed_resolution_is_rejected(self):
        # One public + one private answer must NOT pass: the connect could pick either.
        with patch.object(og_image, "_resolve_addresses", return_value=["93.184.216.34", "10.0.0.5"]):
            assert await og_image._is_fetchable("https://dual.example/") is False

    @pytest.mark.asyncio
    async def test_ipv4_mapped_loopback_is_rejected(self):
        with patch.object(og_image, "_resolve_addresses", return_value=["::ffff:127.0.0.1"]):
            assert await og_image._is_fetchable("https://mapped.example/") is False

    @pytest.mark.asyncio
    async def test_redirect_into_a_private_host_is_refused(self):
        # The first hop is public and 302s to a private address — the classic bypass httpx's own
        # follow_redirects would have chased.
        ctx, client = _client_with(
            _page_response("", location="http://169.254.169.254/latest/meta-data/"), _stream_response()
        )

        def _resolve(host):
            return ["93.184.216.34"] if host == "example.com" else ["169.254.169.254"]

        with patch.object(og_image, "_resolve_addresses", side_effect=_resolve):
            with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
                assert await og_image.fetch_og_image("https://example.com/post") is None
        assert client.stream.call_count == 1  # the private hop was never requested

    @pytest.mark.asyncio
    async def test_redirect_loop_is_bounded(self):
        client = MagicMock()
        client.stream = MagicMock(return_value=_page_response("", location="https://example.com/post"))
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            assert await og_image.fetch_og_image("https://example.com/post") is None
        assert client.stream.call_count == og_image._MAX_REDIRECT_HOPS + 1

    @pytest.mark.asyncio
    async def test_public_redirect_is_followed(self):
        pages = [
            _page_response("", location="https://example.com/final"),
            _page_response(_PAGE_WITH_OG, url="https://example.com/final"),
        ]
        client = MagicMock()
        client.stream = MagicMock(side_effect=[*pages, _stream_response()])
        ctx = MagicMock()
        ctx.__aenter__ = AsyncMock(return_value=client)
        ctx.__aexit__ = AsyncMock(return_value=False)
        with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
            asset = await og_image.fetch_og_image("https://example.com/post")
        assert asset is not None and asset.image_url == "https://cdn.example.com/pic.jpg"

    @pytest.mark.asyncio
    async def test_image_redirect_into_a_private_host_is_refused(self):
        ctx, _ = _client_with(_page_response(_PAGE_WITH_OG), _stream_response(location="http://10.0.0.9/pic.jpg"))

        def _resolve(host):
            return ["10.0.0.9"] if host == "10.0.0.9" else ["93.184.216.34"]

        with patch.object(og_image, "_resolve_addresses", side_effect=_resolve):
            with patch.object(og_image.httpx, "AsyncClient", return_value=ctx):
                assert await og_image.fetch_og_image("https://example.com/post") is None

    @pytest.mark.asyncio
    async def test_guard_failure_never_raises(self):
        # Best-effort contract: even an exploding resolver yields None.
        with patch.object(og_image, "_resolve_addresses", side_effect=RuntimeError("resolver blew up")):
            assert await og_image.fetch_og_image("https://example.com/post") is None


class TestPublicAddress:
    @pytest.mark.parametrize(
        "addr",
        [
            "10.0.0.1",
            "127.0.0.1",
            "169.254.169.254",
            "100.64.1.2",
            "100.100.100.100",
            "::ffff:127.0.0.1",
            "fd00::1",
            # IPv6 forms that carry a private IPv4 address: deprecated compatible, NAT64, 6to4.
            "::169.254.169.254",
            "64:ff9b::a9fe:a9fe",
            "2002:a9fe:a9fe::1",
        ],
    )
    def test_non_public_addresses_are_refused(self, addr):
        # 100.64.0.0/10 (carrier-grade NAT, Tailscale's MagicDNS) slipped past the old flag list.
        assert og_image._is_public_address(addr) is False

    def test_a_public_address_is_allowed(self):
        assert og_image._is_public_address("8.8.8.8") is True
