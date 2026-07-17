import pytest
from pydantic import ValidationError

from shared.config import Config, PipelineConfig, YouTubeCollectorConfig
from shared.utils import _LANGUAGE_MODEL_INFO


class TestStrictConfig:
    def test_unknown_key_rejected(self):
        # A typo'd config key must fail loudly (extra="forbid"), not be silently dropped and fall
        # back to a code default — critical for the delivery toggles.
        with pytest.raises(ValidationError):
            PipelineConfig(enable_thread_post=True)  # typo of enable_threads_post
        with pytest.raises(ValidationError):
            Config(pipeline={"min_scor": 0.5})  # typo of min_score

    def test_real_config_yaml_still_loads(self):
        # The shipped config.yaml must contain only known keys (guards against a strict-mode
        # regression where a real key isn't modeled).
        assert Config.load().pipeline.top_n >= 1

    def test_top_n_lower_bound(self):
        with pytest.raises(ValidationError):
            PipelineConfig(top_n=0)
        with pytest.raises(ValidationError):
            PipelineConfig(top_n=-3)


class TestConfiguredModelsAreRegistered:
    def test_every_configured_model_has_registry_info(self):
        # A model set in config that lacks a _LANGUAGE_MODEL_INFO entry passes Pydantic load
        # (valid enum) but hits the runtime max_tokens/gating fallback with only a warning.
        # This locks the two in sync so a Sonnet-5-style bump can't half-land.
        cfg = Config.load()
        configured = {
            cfg.pipeline.ranking_model,
            cfg.pipeline.digest_model,
            cfg.pipeline.trend_model,
            cfg.collectors.web_search.refine_model,
            cfg.agent.model_id,
        }
        missing = [m.value for m in configured if m not in _LANGUAGE_MODEL_INFO]
        assert not missing, f"configured models missing from _LANGUAGE_MODEL_INFO: {missing}"


class TestImageSizes:
    def test_default_orientation_map(self):
        # orientation -> gpt-image size; the brief picks the orientation per visual.
        sizes = PipelineConfig().image_sizes
        assert sizes["square"] == "1024x1024"
        assert sizes["landscape"] == "1536x1024"
        assert sizes["portrait"] == "1024x1536"

    def test_orientation_map_is_overridable(self):
        cfg = PipelineConfig(image_sizes={"square": "512x512", "landscape": "768x512", "portrait": "512x768"})
        assert cfg.image_sizes["portrait"] == "512x768"


class TestTranscriptLanguage:
    def test_default_is_en(self):
        assert YouTubeCollectorConfig().transcript_language == "en"

    def test_is_configurable(self):
        assert YouTubeCollectorConfig(transcript_language="ko").transcript_language == "ko"
