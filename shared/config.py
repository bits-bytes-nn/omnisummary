from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from shared.constants import LanguageModelId, PdfParserType


class Resources(BaseModel):
    project_name: str = Field(min_length=1)
    stage: Literal["dev", "prod"] = Field(default="dev")
    profile_name: str | None = Field(default=None)
    default_region_name: str = Field(default="ap-northeast-2")
    bedrock_region_name: str = Field(default="us-west-2")
    vpc_id: str | None = Field(default=None)
    subnet_ids: list[str] | None = Field(default=None)
    enable_business_slack_channels: bool = Field(default=False)


class ContentParser(BaseModel):
    figure_analysis_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V3_HAIKU)
    metadata_extraction_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V4_5_SONNET)
    pdf_parser_type: PdfParserType = Field(default=PdfParserType.UNSTRUCTURED)


class Summarization(BaseModel):
    summarization_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V4_5_SONNET)
    n_thumbnails: int = Field(default=1)


class Agent(BaseModel):
    agent_model_id: LanguageModelId = Field(default=LanguageModelId.CLAUDE_V4_5_SONNET)


class Infrastructure(BaseModel):
    lambda_timeout_seconds: int = Field(default=60)
    lambda_memory_mb: int = Field(default=128)
    agentcore_image_tag: str = Field(default="latest")
    event_deduplication_ttl_sec: int = Field(default=300)
    slack_signature_expiration_sec: int = Field(default=300)


class Config(BaseModel):
    resources: Resources = Field(default_factory=lambda: Resources(project_name="omnisummary"))
    content_parser: ContentParser = Field(default_factory=lambda: ContentParser())
    summarization: Summarization = Field(default_factory=lambda: Summarization())
    agent: Agent = Field(default_factory=lambda: Agent())
    infrastructure: Infrastructure = Field(default_factory=lambda: Infrastructure())

    @classmethod
    def from_yaml(cls, file_path: str) -> "Config":
        with open(file_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)
        return cls(**config_data if config_data else {})

    @classmethod
    def load(cls) -> "Config":
        load_dotenv()
        config_path = Path(__file__).parent.parent / "config" / "config.yaml"
        if not config_path.exists():
            return cls()
        return cls.from_yaml(str(config_path))
