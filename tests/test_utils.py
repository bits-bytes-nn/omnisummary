import hashlib
import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from shared.utils import (
    extract_json_from_llm_output,
    generate_item_id,
    parse_feed_published_date,
    parse_json_from_llm_output,
    resolve_secret,
    retry_async,
    sanitize_slack_mrkdwn,
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


class TestExtractJsonFromLlmOutput:
    def test_bare_object(self):
        assert json.loads(extract_json_from_llm_output('{"a": 1}')) == {"a": 1}

    def test_object_with_prose(self):
        raw = 'Here it is:\n{"a": 1, "b": 2}\nThanks!'
        assert json.loads(extract_json_from_llm_output(raw)) == {"a": 1, "b": 2}

    def test_fenced_json_block(self):
        raw = 'note\n```json\n{"x": [1, 2]}\n```\nend'
        assert json.loads(extract_json_from_llm_output(raw)) == {"x": [1, 2]}

    def test_bare_array(self):
        raw = 'queries:\n["a", "b"]\n'
        assert json.loads(extract_json_from_llm_output(raw)) == ["a", "b"]

    def test_picks_outermost_value(self):
        raw = '{"rankings": [{"item_id": "1", "score": 0.5}]}'
        assert json.loads(extract_json_from_llm_output(raw)) == {"rankings": [{"item_id": "1", "score": 0.5}]}


class TestParseJsonFromLlmOutput:
    def test_bare_object(self):
        assert parse_json_from_llm_output('{"a": 1}') == {"a": 1}

    def test_fenced_and_prose(self):
        raw = 'sure:\n```json\n{"x": [1, 2]}\n```\ndone'
        assert parse_json_from_llm_output(raw) == {"x": [1, 2]}

    def test_raw_newline_in_string_value(self):
        # Sonnet 5 emits an unescaped newline inside a string literal; strict json.loads
        # would raise 'Invalid control character', strict=False must accept and preserve it.
        raw = '{"body": "line one\nline two"}'
        assert parse_json_from_llm_output(raw) == {"body": "line one\nline two"}

    def test_raw_tab_in_string_value(self):
        raw = '{"body": "col1\tcol2"}'
        assert parse_json_from_llm_output(raw) == {"body": "col1\tcol2"}


class TestRetryAsync:
    @pytest.mark.asyncio
    async def test_returns_on_first_success(self):
        calls = {"n": 0}

        async def ok():
            calls["n"] += 1
            return "done"

        result = await retry_async(ok, max_retries=3, backoff_sec=0)
        assert result == "done"
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self):
        calls = {"n": 0}

        async def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ValueError("transient")
            return "ok"

        result = await retry_async(flaky, max_retries=3, backoff_sec=0)
        assert result == "ok"
        assert calls["n"] == 3

    @pytest.mark.asyncio
    async def test_reraises_after_exhausting_attempts(self):
        async def always_fail():
            raise RuntimeError("nope")

        with pytest.raises(RuntimeError):
            await retry_async(always_fail, max_retries=2, backoff_sec=0)

    @pytest.mark.asyncio
    async def test_does_not_retry_unlisted_exception(self):
        calls = {"n": 0}

        async def boom():
            calls["n"] += 1
            raise KeyError("unexpected")

        with pytest.raises(KeyError):
            await retry_async(boom, max_retries=3, backoff_sec=0, retry_on=(ValueError,))
        assert calls["n"] == 1

    @pytest.mark.asyncio
    async def test_backoff_is_linear(self):
        # Sleep grows linearly (backoff_sec * attempt), matching the documented contract.
        sleeps: list[float] = []

        async def always_fail():
            raise ValueError("transient")

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        with patch("shared.utils.asyncio.sleep", side_effect=fake_sleep):
            with pytest.raises(ValueError):
                await retry_async(always_fail, max_retries=4, backoff_sec=2.0)
        # 4 attempts -> sleeps after attempts 1, 2, 3 (none after the final attempt)
        assert sleeps == [2.0, 4.0, 6.0]


class TestSanitizeSlackMrkdwn:
    def test_bold_conversion(self):
        assert sanitize_slack_mrkdwn("**bold**") == "*bold*"

    def test_header_removal(self):
        assert sanitize_slack_mrkdwn("## Header") == "Header"

    def test_horizontal_rule_removal(self):
        result = sanitize_slack_mrkdwn("above\n---\nbelow")
        assert "---" not in result

    def test_korean_bold_not_broken_by_space_padding(self):
        # Korean particles attach directly to bold (*규모*가); the space-padding rule
        # must NOT insert a space inside the markers (which breaks Slack rendering).
        result = sanitize_slack_mrkdwn("*규모*가 아니라 *설계*가 이기고 있다")
        assert "*규모*" in result and "*규모 *" not in result
        assert "*설계*" in result and "*설계 *" not in result

    def test_english_bold_still_padded(self):
        # English words touching a bold marker should still get a separating space.
        assert sanitize_slack_mrkdwn("word*bold*word") == "word *bold* word"

    def test_bold_before_paren_not_broken(self):
        # Real regression: *Name* (note) must not become *Name * (note) or merge spans.
        out = sanitize_slack_mrkdwn("추론 특화 *MAI-Thinking-1* (35B)과 코드 특화 *MAI-Code-1-Flash* (5B)")
        assert "*MAI-Thinking-1*" in out and "*MAI-Thinking-1 *" not in out
        assert "*MAI-Code-1-Flash*" in out and "특화*MAI-Code" not in out

    def test_english_italic_padded(self):
        # Multi-word english italic touching neighbours gets boundary spaces.
        assert sanitize_slack_mrkdwn("a_italic phrase_b") == "a _italic phrase_ b"

    def test_snake_case_not_treated_as_italic(self):
        # A single ASCII token with underscores is an identifier, not emphasis.
        assert sanitize_slack_mrkdwn("see config_value_here today") == "see config_value_here today"

    def test_italic_no_space_inside_markers(self):
        out = sanitize_slack_mrkdwn("text _ padded phrase _ end")
        assert "_ padded" not in out and "phrase _" not in out

    def test_markdown_link_with_parens_in_url_preserved(self):
        # A citation URL containing balanced parens (Wikipedia, arXiv, DOIs) must survive the
        # [text](url) → <url|text> conversion intact, not truncate at the first ')'.
        out = sanitize_slack_mrkdwn("see [Foo](https://en.wikipedia.org/wiki/Foo_(bar))")
        assert "<https://en.wikipedia.org/wiki/Foo_(bar)|Foo>" in out

    def test_markdown_link_simple_url(self):
        out = sanitize_slack_mrkdwn("[label](https://example.com/x)")
        assert out == "<https://example.com/x|label>"
