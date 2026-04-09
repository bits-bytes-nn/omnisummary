from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from .constants import LanguageModelId


class BaseCollectorConfig(BaseModel):
    enabled: bool = True
    lookback_hours: int = 24
    reference_time: datetime | None = None


class YouTubeCollectorConfig(BaseCollectorConfig):
    channels: list[str] = Field(default_factory=list)
    max_videos_per_channel: int = 3


class RedditCollectorConfig(BaseCollectorConfig):
    subreddits: list[str] = Field(default_factory=lambda: ["MachineLearning", "artificial", "LocalLLaMA"])
    sort: Literal["hot", "top", "new"] = "hot"
    limit: int = 20


class RSSCollectorConfig(BaseCollectorConfig):
    feeds: list[str] = Field(default_factory=list)


class TrendSearch(BaseModel):
    name: str
    queries: list[str]
    domains: list[str] = Field(default_factory=list)
    topic: Literal["news", "general"] = "news"


class WebSearchCollectorConfig(BaseCollectorConfig):
    trend_searches: list[TrendSearch] = Field(default_factory=list)
    max_results_per_query: int = 10
    lookback_hours: int = 72
    refine_model: LanguageModelId = LanguageModelId.CLAUDE_V4_6_SONNET
    max_refine_queries: int = 3


class RSSHubAccount(BaseModel):
    username: str
    platform: str


class RSSHubCollectorConfig(BaseCollectorConfig):
    base_url: str = "http://localhost:1200"
    accounts: list[RSSHubAccount] = Field(default_factory=list)
    lookback_hours: int = 72
    error_rate_threshold: float = Field(default=50.0, ge=0.0, le=100.0)


class CollectorsConfig(BaseModel):
    youtube: YouTubeCollectorConfig = Field(default_factory=YouTubeCollectorConfig)
    reddit: RedditCollectorConfig = Field(default_factory=RedditCollectorConfig)
    rss: RSSCollectorConfig = Field(default_factory=RSSCollectorConfig)
    web_search: WebSearchCollectorConfig = Field(default_factory=WebSearchCollectorConfig)
    rsshub: RSSHubCollectorConfig = Field(default_factory=RSSHubCollectorConfig)

    def set_reference_time(self, reference_time: datetime) -> None:
        for cfg in (self.youtube, self.reddit, self.rss, self.web_search, self.rsshub):
            cfg.reference_time = reference_time


class PipelineConfig(BaseModel):
    top_n: int = 7
    min_score: float = Field(default=0.7, ge=0.0, le=1.0)
    ranking_model: LanguageModelId = LanguageModelId.CLAUDE_V4_6_SONNET
    digest_model: LanguageModelId = LanguageModelId.CLAUDE_V4_6_SONNET
    item_text_max_tokens: int = 8000
    source_slots: dict[str, int] = Field(
        default_factory=lambda: {
            "web": 2,
            "x": 2,
            "rss": 1,
            "reddit": 1,
            "youtube": 1,
        }
    )
    source_cap_multiplier: int = Field(default=2, ge=1)
    origin_weights: dict[str, float] = Field(default_factory=dict)
    origin_weight_default: float = Field(default=1.0, ge=0.0)
    trend_model: LanguageModelId = LanguageModelId.CLAUDE_V4_6_SONNET
    trend_retention_days: int = Field(default=30, ge=1)


class AgentConfig(BaseModel):
    model_id: LanguageModelId = LanguageModelId.CLAUDE_V4_6_SONNET
    enable_interactive: bool = True


class SlackConfig(BaseModel):
    bot_token: str = ""
    channel_id: str = ""


class AWSConfig(BaseModel):
    region: str = "us-east-1"
    bedrock_region: str = "us-west-2"
    profile: str = ""
    project_name: str = "omnisummary"
    stage: str = "dev"
    vpc_id: str = ""
    subnet_ids: list[str] = Field(default_factory=list)
    state_bucket_name: str = ""
    s3_prefix: str = ""


class Config(BaseModel):
    collectors: CollectorsConfig = Field(default_factory=CollectorsConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    slack: SlackConfig = Field(default_factory=SlackConfig)
    aws: AWSConfig = Field(default_factory=AWSConfig)

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
