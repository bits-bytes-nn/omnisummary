import re
from pathlib import Path
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from shared.proxy import get_proxied_url, is_proxy_configured, parse_feed_with_fallback

PROXY_DIR = Path(__file__).resolve().parent.parent / "cloudflare-proxy"

# The proxy env vars are cleared for every test by the hermetic_env fixture (tests/conftest.py),
# and set here with monkeypatch — never by mutating os.environ, which leaked across tests when a
# test failed before its cleanup ran.


class TestProxy:
    def test_no_proxy_returns_original(self):
        assert get_proxied_url("http://example.com") == "http://example.com"

    def test_not_configured_without_env(self):
        assert not is_proxy_configured()

    def test_configured_with_env(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_PROXY_URL", "https://proxy.example.com")
        monkeypatch.setenv("CLOUDFLARE_PROXY_TOKEN", "token123")
        assert is_proxy_configured()
        result = get_proxied_url("http://example.com")
        assert "proxy.example.com" in result
        assert "token123" in result
        assert "example.com" in result

    def test_token_is_percent_encoded(self, monkeypatch):
        # An unencoded '&'/'#' in the token would truncate the query string, so every proxied
        # fetch would come back 401.
        monkeypatch.setenv("CLOUDFLARE_PROXY_URL", "https://proxy.example.com")
        monkeypatch.setenv("CLOUDFLARE_PROXY_TOKEN", "a&b#c")
        query = parse_qs(urlparse(get_proxied_url("https://www.reddit.com/r/x/.rss")).query)
        assert query["token"] == ["a&b#c"]
        assert query["url"] == ["https://www.reddit.com/r/x/.rss"]


class TestWorkerHardening:
    """The worker is deployed from this repo, so its security posture is asserted here rather than
    trusted to a manual `wrangler deploy` review."""

    def test_token_is_not_committed_to_wrangler_vars(self):
        toml = (PROXY_DIR / "wrangler.toml").read_text()
        # A [vars] PROXY_TOKEN sits in version control in plaintext; it must be a wrangler secret.
        assert not re.search(r"^\s*PROXY_TOKEN\s*=", toml, re.MULTILINE)
        assert "wrangler secret put PROXY_TOKEN" in toml

    def test_allowed_hosts_cover_the_live_callers(self):
        toml = (PROXY_DIR / "wrangler.toml").read_text()
        match = re.search(r'^\s*ALLOWED_HOSTS\s*=\s*"([^"]*)"', toml, re.MULTILINE)
        assert match, "the worker needs an ALLOWED_HOSTS var to enforce its allowlist"
        hosts = [h.strip() for h in match.group(1).split(",") if h.strip()]
        # Suffix matching means the bare domains must cover the hosts the collectors actually
        # proxy: www.reddit.com (reddit collector) and www.youtube.com (youtube RSS fallback).
        for live_host in ("www.reddit.com", "www.youtube.com"):
            assert any(live_host == h or live_host.endswith(f".{h}") for h in hosts), live_host

    def test_the_executable_worker_suite_is_wired_into_ci(self):
        # The worker's actual BRANCHES (token, URL parse, scheme, suffix host match, fixed outbound
        # headers) are asserted by cloudflare-proxy/test/worker.test.js with node:test. Substring
        # checks on worker.js used to stand in for that, and they stayed green even if isAllowedHost
        # returned true unconditionally or the 401/403 branches were inverted — so all that is
        # checked here is that the executable suite exists and that CI runs it.
        assert (PROXY_DIR / "test" / "worker.test.js").exists()
        ci = (PROXY_DIR.parent / ".github" / "workflows" / "ci.yml").read_text()
        assert "node --test cloudflare-proxy/test/*.test.js" in ci


class _Feed(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as e:
            raise AttributeError(name) from e


class TestParseFeedWithFallback:
    def test_direct_success_no_proxy_attempt(self):
        good = _Feed(status=200, bozo=False, entries=[{"x": 1}])
        with patch("shared.proxy.feedparser.parse", return_value=good) as mp:
            feed = parse_feed_with_fallback("https://example.com/feed")
        assert feed.entries
        assert mp.call_count == 1  # no proxy configured -> single direct attempt

    def test_falls_back_to_proxy_when_direct_blocked(self, monkeypatch):
        monkeypatch.setenv("CLOUDFLARE_PROXY_URL", "https://proxy.example.com")
        monkeypatch.setenv("CLOUDFLARE_PROXY_TOKEN", "tok")
        blocked = _Feed(status=403, bozo=False, entries=[])
        good = _Feed(status=200, bozo=False, entries=[{"x": 1}])
        with patch("shared.proxy.feedparser.parse", side_effect=[blocked, good]) as mp:
            feed = parse_feed_with_fallback("https://example.com/feed")
        assert feed.entries  # second (proxy) attempt succeeded
        assert mp.call_count == 2

    def test_returns_last_when_all_fail(self):
        bad = _Feed(status=503, bozo=True, bozo_exception=Exception("x"), entries=[])
        with patch("shared.proxy.feedparser.parse", return_value=bad):
            feed = parse_feed_with_fallback("https://example.com/feed")
        assert feed.get("status") == 503  # caller can inspect failure
