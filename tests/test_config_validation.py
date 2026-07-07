from shared.config import Config, PipelineConfig, YouTubeCollectorConfig
from shared.utils import _LANGUAGE_MODEL_INFO


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
