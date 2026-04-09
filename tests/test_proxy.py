import os

from shared.proxy import get_proxied_url, is_proxy_configured


class TestProxy:
    def test_no_proxy_returns_original(self):
        os.environ.pop("CLOUDFLARE_PROXY_URL", None)
        os.environ.pop("CLOUDFLARE_PROXY_TOKEN", None)
        assert get_proxied_url("http://example.com") == "http://example.com"

    def test_not_configured_without_env(self):
        os.environ.pop("CLOUDFLARE_PROXY_URL", None)
        os.environ.pop("CLOUDFLARE_PROXY_TOKEN", None)
        assert not is_proxy_configured()

    def test_configured_with_env(self):
        os.environ["CLOUDFLARE_PROXY_URL"] = "https://proxy.example.com"
        os.environ["CLOUDFLARE_PROXY_TOKEN"] = "token123"
        try:
            assert is_proxy_configured()
            result = get_proxied_url("http://example.com")
            assert "proxy.example.com" in result
            assert "token123" in result
            assert "example.com" in result
        finally:
            os.environ.pop("CLOUDFLARE_PROXY_URL", None)
            os.environ.pop("CLOUDFLARE_PROXY_TOKEN", None)
