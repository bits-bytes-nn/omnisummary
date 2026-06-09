from unittest.mock import MagicMock, patch

from shared.constants import LanguageModelId
from shared.utils import _LANGUAGE_MODEL_INFO, BedrockLanguageModelFactory


def _factory(client=None):
    with patch("shared.utils.boto3.Session") as session:
        session.return_value.client.return_value = client or MagicMock()
        session.return_value.region_name = "us-west-2"
        session.return_value.profile_name = None
        return BedrockLanguageModelFactory(region_name="us-west-2")


class TestCountTokens:
    def test_uses_bedrock_count_tokens_with_base_model_id(self):
        client = MagicMock()
        client.count_tokens.return_value = {"inputTokens": 42}
        f = _factory(client)
        n = f.count_tokens("some text", LanguageModelId.CLAUDE_V4_6_SONNET)
        assert n == 42
        # CountTokens needs the BASE model id (no cross-region global./us. prefix)
        assert client.count_tokens.call_args.kwargs["modelId"] == "anthropic.claude-sonnet-4-6"

    def test_pins_supported_model_even_for_unsupported_caller(self):
        # Opus 4.8 doesn't expose CountTokens; counting must still hit the supported Sonnet
        # base id (shared tokenizer), not the caller's model.
        client = MagicMock()
        client.count_tokens.return_value = {"inputTokens": 7}
        f = _factory(client)
        f.count_tokens("hi", LanguageModelId.CLAUDE_V4_8_OPUS)
        assert client.count_tokens.call_args.kwargs["modelId"] == "anthropic.claude-sonnet-4-6"

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
        out = f.truncate_to_tokens("가 " * 500, 20, LanguageModelId.CLAUDE_V4_6_SONNET)
        assert len(out) < 1000
        assert f.count_tokens(out, LanguageModelId.CLAUDE_V4_6_SONNET) <= 20

    def test_truncate_returns_text_when_within_budget(self):
        client = MagicMock()
        client.count_tokens.return_value = {"inputTokens": 5}
        f = _factory(client)
        assert f.truncate_to_tokens("short", 100) == "short"


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

    def test_non_cross_region_opus_48_omits_temperature(self):
        f = _factory()
        info = _LANGUAGE_MODEL_INFO[LanguageModelId.CLAUDE_V4_8_OPUS]
        cfg = f._build_model_config(info, "anthropic.claude-opus-4-8", False)
        assert "temperature" not in cfg["model_kwargs"]
