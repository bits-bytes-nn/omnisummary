import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from shared.utils import (
    aws_region,
    backoff_delay,
    coerce_bool,
    extract_json_from_llm_output,
    generate_item_id,
    parse_feed_published_date,
    parse_json_from_llm_output,
    resolve_secret,
    retry_async,
    sanitize_slack_mrkdwn,
)


class TestAwsRegion:
    """Four modules carried the literal os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION",
    "ap-northeast-2")) — one developer's region baked into the code, free to diverge from
    config.aws.region and to send an SSM/AgentCore call to the wrong region anywhere else."""

    def test_prefers_the_runtime_environment(self, monkeypatch):
        monkeypatch.setenv("AWS_REGION", "eu-west-1")
        assert aws_region() == "eu-west-1"

    def test_falls_back_to_aws_default_region(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-2")
        assert aws_region() == "us-east-2"

    def test_falls_back_to_the_configured_region_not_a_literal(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        from shared.config import Config, get_config

        configured = Config()
        configured.aws.region = "ap-southeast-2"
        get_config.cache_clear()
        with patch("shared.config.Config.load", return_value=configured):
            assert aws_region() == "ap-southeast-2"

    def test_none_when_nothing_is_configured_so_boto3_resolves_it(self, monkeypatch):
        monkeypatch.delenv("AWS_REGION", raising=False)
        monkeypatch.delenv("AWS_DEFAULT_REGION", raising=False)
        from shared.config import Config, get_config

        configured = Config()
        configured.aws.region = ""
        get_config.cache_clear()
        with patch("shared.config.Config.load", return_value=configured):
            assert aws_region() is None


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

    def test_strict_distinguishes_an_ssm_outage_from_an_unset_secret(self, monkeypatch):
        # Default behaviour (every existing caller) is unchanged: "" degrades gracefully. The
        # opt-in strict mode raises, so a caller whose whole job IS the secret cannot report
        # "nothing to do" while SSM is simply unreachable.
        from botocore.exceptions import ClientError

        from shared.utils import SecretUnavailableError

        monkeypatch.delenv("MY_SECRET", raising=False)
        with patch("shared.utils.boto3.client", side_effect=Exception("no ssm")):
            with pytest.raises(SecretUnavailableError):
                resolve_secret("MY_SECRET", "my-secret", strict=True)

        # A parameter that genuinely does not exist is still just "unset", even in strict mode.
        client = MagicMock()
        client.get_parameter.side_effect = ClientError({"Error": {"Code": "ParameterNotFound"}}, "GetParameter")
        with patch("shared.utils.boto3.client", return_value=client):
            assert resolve_secret("MY_SECRET", "my-secret", strict=True) == ""


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

    @pytest.mark.asyncio
    async def test_a_jitter_seed_spreads_concurrent_retries(self):
        # Plain linear backoff resynchronises dozens of concurrent callers into exactly the burst
        # the upstream rate-limited (40 RSSHub account feeds all retrying at the same instant).
        sleeps: list[float] = []

        async def always_fail():
            raise ValueError("transient")

        async def fake_sleep(seconds):
            sleeps.append(seconds)

        with patch("shared.utils.asyncio.sleep", side_effect=fake_sleep):
            for seed in ("feed-a", "feed-b"):
                with pytest.raises(ValueError):
                    await retry_async(always_fail, max_retries=2, backoff_sec=2.0, jitter_seed=seed)
        assert len(sleeps) == 2
        assert sleeps[0] != sleeps[1]
        assert all(2.0 <= s <= 4.0 for s in sleeps)  # never below the linear delay, never past 2x


class TestBackoffDelay:
    def test_linear_without_a_seed(self):
        assert backoff_delay(3.0, 2) == 6.0

    def test_jitter_is_deterministic_per_seed_and_attempt(self):
        first = backoff_delay(5.0, 1, "r/LocalLLaMA")
        assert first == backoff_delay(5.0, 1, "r/LocalLLaMA")  # no RNG: reproducible
        assert first != backoff_delay(5.0, 1, "r/MachineLearning")
        assert first != backoff_delay(5.0, 2, "r/LocalLLaMA")

    def test_jitter_never_shortens_the_linear_delay(self):
        for attempt in (1, 2, 3):
            delay = backoff_delay(4.0, attempt, "seed")
            assert 4.0 * attempt <= delay < 4.0 * (attempt + 1)


class TestCoerceBool:
    """LLM plan flags arrive as JSON the model wrote by hand, so a boolean may be a STRING. Bare
    truthiness read "false" as True: the visual editor's `skip` silently killed the day's visual and
    `use_character` injected the mascot against the editor's judgment."""

    def test_false_spellings_are_false(self):
        for value in ("false", "False", " FALSE ", "0", "no", "off", "none", "null", ""):
            assert coerce_bool(value) is False, value

    def test_true_spellings_and_real_booleans(self):
        for value in ("true", "True", "yes", "1", True, 1):
            assert coerce_bool(value) is True, value
        assert coerce_bool(False) is False

    def test_missing_falls_back_to_the_default(self):
        assert coerce_bool(None) is False
        assert coerce_bool(None, default=True) is True


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


class TestAvailableBotoProfile:
    """Credential selection by platform sniff was wrong in both directions, so the question the code
    asks is now the only one that matters: does this profile actually resolve here?"""

    def test_no_configured_profile_means_ambient(self):
        from shared.utils import available_boto_profile

        assert available_boto_profile("") is None

    def test_a_configured_profile_that_exists_is_returned(self):
        from unittest.mock import MagicMock, patch

        from shared.utils import available_boto_profile

        with patch("boto3.session.Session", return_value=MagicMock(available_profiles=["research"])):
            assert available_boto_profile("research") == "research"

    def test_a_configured_profile_that_does_not_exist_means_ambient(self):
        from unittest.mock import MagicMock, patch

        from shared.utils import available_boto_profile

        with patch("boto3.session.Session", return_value=MagicMock(available_profiles=["default"])):
            assert available_boto_profile("research") is None

    def test_an_unenumerable_profile_list_means_ambient(self):
        # A broken/unreadable ~/.aws/config must never be able to stop a run.
        from unittest.mock import patch

        from shared.utils import available_boto_profile

        with patch("boto3.session.Session", side_effect=RuntimeError("bad config")):
            assert available_boto_profile("research") is None
