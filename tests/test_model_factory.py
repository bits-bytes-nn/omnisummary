from unittest.mock import MagicMock, patch

from shared.constants import LanguageModelId
from shared.utils import _LANGUAGE_MODEL_INFO, BedrockLanguageModelFactory


def _factory():
    with patch("shared.utils.boto3.Session") as session:
        session.return_value.client.return_value = MagicMock()
        session.return_value.region_name = "us-west-2"
        session.return_value.profile_name = None
        return BedrockLanguageModelFactory(region_name="us-west-2")


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
