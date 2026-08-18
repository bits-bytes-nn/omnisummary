from unittest.mock import MagicMock, patch

from shared.constants import LanguageModelId
from shared.utils import _LANGUAGE_MODEL_INFO, TOKEN_COUNT_MODEL, BedrockLanguageModelFactory


def _factory(client=None):
    with patch("shared.utils.boto3.Session") as session:
        session.return_value.client.return_value = client or MagicMock()
        session.return_value.region_name = "us-west-2"
        session.return_value.profile_name = None
        return BedrockLanguageModelFactory(region_name="us-west-2")


class TestCountTokens:
    def test_always_counts_with_supported_base_model_id(self):
        # CountTokens only works on some base models; we always count with the supported Sonnet
        # base id (shared tokenizer), never the caller's model, and strip any cross-region prefix.
        client = MagicMock()
        client.count_tokens.return_value = {"inputTokens": 42}
        f = _factory(client)
        assert f.count_tokens("some text") == 42
        assert client.count_tokens.call_args.kwargs["modelId"] == TOKEN_COUNT_MODEL.value

    def test_falls_back_to_char_estimate_on_api_error(self):
        client = MagicMock()
        client.count_tokens.side_effect = RuntimeError("throttled")
        f = _factory(client)
        assert f.count_tokens("x" * 40) == 10  # len//4 fallback

    def test_truncate_to_tokens_binary_searches_to_fit(self):
        client = MagicMock()
        # token count ≈ chars/5 for this fake, so a 1000-char text over a 20-token budget truncates
        client.count_tokens.side_effect = lambda modelId, input: {
            "inputTokens": len(input["converse"]["messages"][0]["content"][0]["text"]) // 5
        }
        f = _factory(client)
        out = f.truncate_to_tokens("가 " * 500, 20)
        assert len(out) < 1000
        assert f.count_tokens(out) <= 20

    def test_truncate_returns_text_when_within_budget(self):
        client = MagicMock()
        client.count_tokens.return_value = {"inputTokens": 5}
        f = _factory(client)
        assert f.truncate_to_tokens("short", 100) == "short"

    def test_count_tokens_memoizes_identical_text(self):
        # The same text is counted many times (prompt building across stages, truncate's binary
        # search); repeat calls must hit the instance cache, not re-bill the Bedrock API.
        client = MagicMock()
        client.count_tokens.return_value = {"inputTokens": 7}
        f = _factory(client)
        assert f.count_tokens("repeated") == 7
        assert f.count_tokens("repeated") == 7
        assert client.count_tokens.call_count == 1  # second call served from cache

    def test_count_tokens_degrades_after_first_failure(self):
        # After CountTokens fails once, the rest of the run must use the char estimate WITHOUT
        # hammering the failing API — otherwise truncate's binary search amplifies one blip into
        # a storm of failing round-trips.
        client = MagicMock()
        client.count_tokens.side_effect = RuntimeError("throttled")
        f = _factory(client)
        assert f.count_tokens("a" * 40) == 10  # first: tries API, fails, estimates
        assert f.count_tokens("b" * 80) == 20  # second: short-circuits to estimate
        assert f.count_tokens("c" * 12) == 3
        assert client.count_tokens.call_count == 1  # API called only once, then degraded


class TestTemperatureGating:
    def test_opus_48_omits_temperature(self):
        # Opus 4.7/4.8 reject the temperature param -> must not be sent.
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_8_OPUS]
        cfg = f._build_model_config(info, "global.anthropic.claude-opus-4-8", True)
        assert "temperature" not in cfg
        assert cfg["max_tokens"] > 0

    def test_sonnet_46_includes_temperature(self):
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-4-6", True)
        assert "temperature" in cfg

    def test_non_cross_region_opus_48_omits_temperature_and_top_k(self):
        # Models that reject sampling params must omit BOTH temperature and top_k on the
        # non-converse (ChatBedrock) path, or Bedrock 400s.
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_8_OPUS]
        cfg = f._build_model_config(info, "anthropic.claude-opus-4-8", False)
        assert "temperature" not in cfg["model_kwargs"]
        assert "top_k" not in cfg["model_kwargs"]

    def test_non_cross_region_sonnet_46_includes_top_k(self):
        # A sampling-param-accepting model DOES get top_k on the non-converse path.
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        cfg = f._build_model_config(info, "anthropic.claude-sonnet-4-6", False)
        assert cfg["model_kwargs"]["top_k"] == BedrockLanguageModelFactory.DEFAULT_TOP_K

    def test_sonnet_5_omits_temperature(self):
        # Sonnet 5 rejects non-default sampling params (400), same as Opus 4.7/4.8. It is now
        # the default digest/agent/trend model, so lock the gating in explicitly.
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True)
        assert "temperature" not in cfg
        assert cfg["max_tokens"] > 0

    def test_sonnet_5_omits_top_k_on_non_converse(self):
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "anthropic.claude-sonnet-5", False)
        assert "temperature" not in cfg["model_kwargs"]
        assert "top_k" not in cfg["model_kwargs"]


class TestThinkingFormat:
    """Newer models (Sonnet 5, Opus 4.7/4.8) reject the legacy
    thinking.type='enabled' + budget_tokens form and require adaptive thinking.
    Guards against re-emitting the deprecated shape when thinking is enabled."""

    def test_adaptive_model_emits_adaptive_thinking(self):
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        assert info.uses_adaptive_thinking is True
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True, enable_thinking=True)
        amrf = cfg["additional_model_request_fields"]
        assert amrf["thinking"] == {"type": "adaptive"}
        assert "budget_tokens" not in amrf["thinking"]
        assert amrf["output_config"]["effort"] == f.DEFAULT_THINKING_EFFORT

    def test_adaptive_model_honors_thinking_effort(self):
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_8_OPUS]
        cfg = f._build_model_config(
            info,
            "global.anthropic.claude-opus-4-8",
            True,
            enable_thinking=True,
            thinking_effort="high",
        )
        assert cfg["additional_model_request_fields"]["output_config"]["effort"] == "high"

    def test_legacy_model_keeps_budget_form(self):
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_6_SONNET]
        assert info.uses_adaptive_thinking is False
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-4-6", True, enable_thinking=True)
        thinking = cfg["additional_model_request_fields"]["thinking"]
        assert thinking["type"] == "enabled"
        assert thinking["budget_tokens"] == f.DEFAULT_THINKING_BUDGET_TOKENS


class TestOpus5Registry:
    """Opus 5 is selectable from config. Its capability flags were VERIFIED against Converse on
    global.anthropic.claude-opus-5, not inferred from the version number: a `temperature` param and
    the legacy thinking.type="enabled"/budget_tokens form both return ValidationException, while
    thinking.type="adaptive" + output_config.effort succeeds. Getting these wrong 400s every call."""

    def test_opus_5_is_registered(self):
        assert LanguageModelId.CLAUDE_V5_OPUS in _LANGUAGE_MODEL_INFO

    def test_opus_5_rejects_sampling_params_and_uses_adaptive_thinking(self):
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_OPUS]
        assert info.supports_temperature is False
        assert info.uses_adaptive_thinking is True
        assert info.supports_thinking is True
        assert info.supports_prompt_caching is True

    def test_opus_5_matches_the_other_claude_5_family_gates(self):
        # Same shape as Sonnet 5 / Opus 4.8 — a new family member that silently differs on these two
        # flags is the failure mode this pins.
        opus5 = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_OPUS]
        for sibling in (LanguageModelId.CLAUDE_V5_SONNET, LanguageModelId.CLAUDE_V4_8_OPUS):
            other = _LANGUAGE_MODEL_INFO[sibling]
            assert opus5.supports_temperature == other.supports_temperature
            assert opus5.uses_adaptive_thinking == other.uses_adaptive_thinking


class TestTokenUsageAttribution:
    """Cost Explorer bills per MODEL, and several pipeline stages share one model — so a token total
    could not be traced to the stage that spent it. Every model carries a usage logger tagged with
    its stage; telemetry must never be able to fail a generation."""

    def test_stage_tagged_usage_logger_is_attached(self):
        from shared.utils import _TokenUsageLogger

        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True, stage="ranking")
        loggers = [c for c in cfg["callbacks"] if isinstance(c, _TokenUsageLogger)]
        assert [h.stage for h in loggers] == ["ranking"]
        assert loggers[0].model_id == "global.anthropic.claude-sonnet-5"

    def test_untagged_call_is_labelled_rather_than_dropped(self):
        from shared.utils import _TokenUsageLogger

        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True)
        assert [c.stage for c in cfg["callbacks"] if isinstance(c, _TokenUsageLogger)] == ["unattributed"]

    def test_a_caller_supplied_callback_is_kept(self):
        from shared.utils import _TokenUsageLogger

        mine = MagicMock()
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V5_SONNET]
        cfg = f._build_model_config(info, "global.anthropic.claude-sonnet-5", True, callbacks=[mine], stage="digest")
        assert mine in cfg["callbacks"]
        assert any(isinstance(c, _TokenUsageLogger) for c in cfg["callbacks"])

    def test_a_response_without_usage_metadata_is_survivable(self):
        from shared.utils import _TokenUsageLogger

        handler = _TokenUsageLogger("digest", "global.anthropic.claude-sonnet-5")
        handler.on_llm_end(MagicMock(generations=[], llm_output=None))  # must not raise
        handler.on_llm_end(object())  # not even the expected shape
