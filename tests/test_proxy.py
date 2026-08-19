import re
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

from shared.proxy import fetch_with_proxy_fallback, get_proxied_url, is_proxy_configured

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


def _fetcher(outcomes: dict[str, object]):
    """A stand-in fetch keyed by URL; an exception value is raised, anything else returned. Records
    which candidates were attempted, in order."""
    seen: list[str] = []

    async def _fetch(url: str):
        seen.append(url)
        outcome = outcomes[url]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    return _fetch, seen


def _has_entries(feed) -> bool:
    return bool(feed.entries)


@pytest.fixture
def proxy_env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_PROXY_URL", "https://proxy.example.com")
    monkeypatch.setenv("CLOUDFLARE_PROXY_TOKEN", "tok")


class TestFetchWithProxyFallback:
    URL = "https://example.com/feed"

    @pytest.mark.asyncio
    async def test_direct_success_no_proxy_attempt(self):
        good = _Feed(entries=[{"x": 1}])
        fetch, seen = _fetcher({self.URL: good})
        feed = await fetch_with_proxy_fallback(self.URL, fetch, has_entries=_has_entries)
        assert feed is good
        assert seen == [self.URL]  # no proxy configured -> single direct attempt

    @pytest.mark.asyncio
    async def test_falls_back_to_proxy_when_direct_blocked(self, proxy_env):
        good = _Feed(entries=[{"x": 1}])
        proxied = get_proxied_url(self.URL)
        fetch, seen = _fetcher({self.URL: RuntimeError("returned HTTP 403"), proxied: good})
        feed = await fetch_with_proxy_fallback(self.URL, fetch, has_entries=_has_entries)
        assert feed is good
        assert seen == [self.URL, proxied]

    @pytest.mark.asyncio
    async def test_a_quiet_direct_feed_wins_over_a_failing_proxy(self, proxy_env):
        # THE REGRESSION: a direct 200 with zero entries (a quiet subreddit) used to be overwritten
        # by the proxy's 429, so the caller saw a transient failure, burned every retry, and with two
        # configured subreddits reported the whole source FAILED on a clean empty day.
        quiet = _Feed(entries=[])
        proxied = get_proxied_url(self.URL)
        fetch, seen = _fetcher({self.URL: quiet, proxied: RuntimeError("returned HTTP 429")})
        feed = await fetch_with_proxy_fallback(self.URL, fetch, has_entries=_has_entries)
        assert feed is quiet
        assert seen == [self.URL, proxied]  # the proxy is still tried; its failure just doesn't win

    @pytest.mark.asyncio
    async def test_entries_beat_an_earlier_empty_candidate(self, proxy_env):
        quiet = _Feed(entries=[])
        good = _Feed(entries=[{"x": 1}])
        proxied = get_proxied_url(self.URL)
        fetch, _ = _fetcher({self.URL: quiet, proxied: good})
        assert await fetch_with_proxy_fallback(self.URL, fetch, has_entries=_has_entries) is good

    @pytest.mark.asyncio
    async def test_raises_the_last_error_when_every_candidate_failed(self, proxy_env):
        proxied = get_proxied_url(self.URL)
        fetch, _ = _fetcher({self.URL: RuntimeError("returned HTTP 403"), proxied: RuntimeError("returned HTTP 429")})
        # The LAST error is the proxy's, which for a datacenter-blocked host is the informative one
        # (and transient, so the caller's retry chain still applies).
        with pytest.raises(RuntimeError, match="429"):
            await fetch_with_proxy_fallback(self.URL, fetch, has_entries=_has_entries)
