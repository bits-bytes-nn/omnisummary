from shared.config import PipelineConfig, YouTubeCollectorConfig


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
