import pytest
from pydantic import ValidationError

from shared.config import PipelineConfig, YouTubeCollectorConfig


class TestImageSizeValidation:
    def test_default_is_valid(self):
        # portrait by default so multi-panel comics aren't cropped
        assert PipelineConfig().image_size == "1024x1536"

    def test_accepts_well_formed_size(self):
        assert PipelineConfig(image_size="512x768").image_size == "512x768"

    def test_rejects_malformed_size(self):
        with pytest.raises(ValidationError):
            PipelineConfig(image_size="1024")
        with pytest.raises(ValidationError):
            PipelineConfig(image_size="large")


class TestTranscriptLanguage:
    def test_default_is_en(self):
        assert YouTubeCollectorConfig().transcript_language == "en"

    def test_is_configurable(self):
        assert YouTubeCollectorConfig(transcript_language="ko").transcript_language == "ko"
