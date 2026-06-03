import os
from unittest.mock import patch

from shared.proxy import get_proxied_url, is_proxy_configured, parse_feed_with_fallback


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


class _Feed(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class TestParseFeedWithFallback:
    def test_direct_success_no_proxy_attempt(self):
        os.environ.pop("CLOUDFLARE_PROXY_URL", None)
        os.environ.pop("CLOUDFLARE_PROXY_TOKEN", None)
        good = _Feed(status=200, bozo=False, entries=[{"x": 1}])
        with patch("shared.proxy.feedparser.parse", return_value=good) as mp:
            feed = parse_feed_with_fallback("https://example.com/feed")
        assert feed.entries
        assert mp.call_count == 1  # no proxy configured -> single direct attempt

    def test_falls_back_to_proxy_when_direct_blocked(self):
        os.environ["CLOUDFLARE_PROXY_URL"] = "https://proxy.example.com"
        os.environ["CLOUDFLARE_PROXY_TOKEN"] = "tok"
        try:
            blocked = _Feed(status=403, bozo=False, entries=[])
            good = _Feed(status=200, bozo=False, entries=[{"x": 1}])
            with patch("shared.proxy.feedparser.parse", side_effect=[blocked, good]) as mp:
                feed = parse_feed_with_fallback("https://example.com/feed")
            assert feed.entries  # second (proxy) attempt succeeded
            assert mp.call_count == 2
        finally:
            os.environ.pop("CLOUDFLARE_PROXY_URL", None)
            os.environ.pop("CLOUDFLARE_PROXY_TOKEN", None)

    def test_returns_last_when_all_fail(self):
        os.environ.pop("CLOUDFLARE_PROXY_URL", None)
        os.environ.pop("CLOUDFLARE_PROXY_TOKEN", None)
        bad = _Feed(status=503, bozo=True, bozo_exception=Exception("x"), entries=[])
        with patch("shared.proxy.feedparser.parse", return_value=bad):
            feed = parse_feed_with_fallback("https://example.com/feed")
        assert feed.get("status") == 503  # caller can inspect failure
