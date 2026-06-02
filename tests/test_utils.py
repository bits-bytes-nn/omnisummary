import hashlib
from types import SimpleNamespace
from unittest.mock import patch

from shared.utils import (
    generate_item_id,
    parse_feed_published_date,
    resolve_secret,
    sanitize_slack_mrkdwn,
    truncate_text_by_tokens,
)


class TestResolveSecret:
    def test_prefers_env(self, monkeypatch):
        monkeypatch.setenv("MY_SECRET", "from-env")
        assert resolve_secret("MY_SECRET", "my-secret") == "from-env"

    def test_falls_back_to_ssm(self, monkeypatch):
        monkeypatch.delenv("MY_SECRET", raising=False)
        monkeypatch.setenv("PROJECT_NAME", "proj")
        monkeypatch.setenv("STAGE", "dev")
        ssm = patch("shared.utils.boto3.client").start()
        ssm.return_value.get_parameter.return_value = {"Parameter": {"Value": "from-ssm"}}
        try:
            assert resolve_secret("MY_SECRET", "my-secret") == "from-ssm"
            assert ssm.return_value.get_parameter.call_args.kwargs["Name"] == "/proj/dev/my-secret"
        finally:
            patch.stopall()

    def test_returns_empty_on_failure(self, monkeypatch):
        monkeypatch.delenv("MY_SECRET", raising=False)
        with patch("shared.utils.boto3.client", side_effect=Exception("no ssm")):
            assert resolve_secret("MY_SECRET", "my-secret") == ""


class TestGenerateItemId:
    def test_deterministic(self):
        url = "http://example.com/article"
        assert generate_item_id(url) == generate_item_id(url)

    def test_length(self):
        assert len(generate_item_id("http://example.com")) == 16

    def test_matches_sha256_prefix(self):
        url = "http://test.com"
        expected = hashlib.sha256(url.encode()).hexdigest()[:16]
        assert generate_item_id(url) == expected

    def test_different_urls_different_ids(self):
        assert generate_item_id("http://a.com") != generate_item_id("http://b.com")


class TestParseFeedPublishedDate:
    def test_published_parsed(self):
        entry = SimpleNamespace(published_parsed=(2024, 6, 15, 12, 0, 0, 5, 167, 0))
        entry.get = lambda k, d=None: None
        result = parse_feed_published_date(entry)
        assert result is not None
        assert result.year == 2024
        assert result.month == 6

    def test_published_string_rfc2822(self):
        entry = SimpleNamespace()
        entry.published_parsed = None
        entry.updated_parsed = None
        entry.get = lambda k, d=None: "Sat, 15 Jun 2024 12:00:00 +0000" if k == "published" else d
        result = parse_feed_published_date(entry)
        assert result is not None
        assert result.year == 2024

    def test_updated_parsed_fallback(self):
        entry = SimpleNamespace(published_parsed=None, updated_parsed=(2024, 3, 1, 0, 0, 0, 4, 61, 0))
        entry.get = lambda k, d=None: None
        result = parse_feed_published_date(entry)
        assert result is not None
        assert result.month == 3

    def test_none_when_no_date(self):
        entry = SimpleNamespace(published_parsed=None, updated_parsed=None)
        entry.get = lambda k, d=None: None
        result = parse_feed_published_date(entry)
        assert result is None


class TestTruncateTextByTokens:
    def test_short_text_unchanged(self):
        text = "Hello world"
        assert truncate_text_by_tokens(text, max_tokens=100) == text

    def test_long_text_truncated(self):
        text = "word " * 1000
        result = truncate_text_by_tokens(text, max_tokens=10)
        assert len(result) < len(text)


class TestSanitizeSlackMrkdwn:
    def test_bold_conversion(self):
        assert sanitize_slack_mrkdwn("**bold**") == "*bold*"

    def test_header_removal(self):
        assert sanitize_slack_mrkdwn("## Header") == "Header"

    def test_horizontal_rule_removal(self):
        result = sanitize_slack_mrkdwn("above\n---\nbelow")
        assert "---" not in result
