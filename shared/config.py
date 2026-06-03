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
    request_timeout: int = Field(default=30, ge=1)
    max_retries: int = Field(default=3, ge=1)
    retry_backoff_sec: int = Field(default=5, ge=0)


class YouTubeCollectorConfig(BaseCollectorConfig):
    channels: list[str] = Field(default_factory=list)
    max_videos_per_channel: int = 3
    resolve_timeout: int = Field(default=15, ge=1)
    transcript_timeout: int = Field(default=15, ge=1)
    transcript_language: str = Field(default="en")


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
    min_search_score: float = Field(default=0.3, ge=0.0, le=1.0)


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
    # Language rules injected into the digest prompt's *Language* block. Defaults to the
    # Korean editorial rules + translation glossary; other deployments can override to
    # write the digest in another language without forking the prompt.
    digest_language_rules: str = (
        "- Write in Korean (95%+). English ONLY for proper nouns and untranslatable technical terms.\n"
        "- Translate terms that have established Korean equivalents: architecture → 아키텍처, "
        "benchmark → 벤치마크, inference → 추론, training → 학습, deployment → 배포, "
        "weight → 가중치, parameter → 파라미터, token → 토큰, open-source → 오픈소스, "
        "pipeline → 파이프라인, optimization → 최적화, compression → 압축, memory → 메모리.\n"
        "- General words MUST be Korean: practitioner → 실무자, implication → 시사점, "
        "release → 출시/공개, breakthrough → 돌파구, approach → 접근법, ecosystem → 생태계.\n"
        "- If the original item title is in English, translate it to Korean for the display text."
    )
    item_text_max_tokens: int = 8000
    ranking_batch_size: int = Field(default=40, ge=1)
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
    max_per_origin: int = Field(default=1, ge=1)
    origin_weights: dict[str, float] = Field(default_factory=dict)
    origin_weight_default: float = Field(default=1.0, ge=0.0)
    origin_weight_nudge: float = Field(default=0.1, ge=0.0, le=1.0)
    # Engagement bonus tiers (views threshold -> score bonus) the ranking prompt applies
    # to items carrying view counts. Tunable instead of baked into the prompt text.
    engagement_tiers: list[tuple[int, float]] = Field(
        default_factory=lambda: [(10000, 0.05), (100000, 0.1), (500000, 0.15)]
    )
    # Taxonomy the ranking prompt assigns to each item. Configurable so non-AI
    # deployments can supply their own categories.
    ranking_categories: list[str] = Field(
        default_factory=lambda: [
            "research",
            "tools",
            "news",
            "release",
            "industry",
            "paper",
            "interview",
            "infrastructure",
            "community",
        ]
    )
    # Score the ranking prompt assigns to duplicate items within a same-topic cluster.
    ranking_duplicate_score_penalty: float = Field(default=0.3, ge=0.0, le=1.0)
    # Score-calibration buckets the ranking prompt applies, injected as template text so
    # ops can retune the distribution without editing the prompt.
    ranking_scoring_rubric: str = (
        "0.9+: field-defining. 0.8-0.89: very important. 0.7-0.79: notable. " "0.6-0.69: worth noting. <0.6: low value."
    )
    # Target count of items the ranking prompt should aim to score above the bar per batch.
    ranking_target_count: str = "~10-20 items scoring 0.6+"
    trend_model: LanguageModelId = LanguageModelId.CLAUDE_V4_6_SONNET
    trend_retention_days: int = Field(default=30, ge=1)
    trend_cooling_days: int = Field(default=7, ge=1)
    trend_max_evidence: int = Field(default=5, ge=1)
    trend_max_active_trends: int = Field(default=10, ge=1)
    trend_max_chars: int = Field(default=15000, ge=1)
    enable_daily_visual: bool = True
    image_model: str = "gpt-image-2"
    # Portrait by default so multi-panel comics aren't cropped in a square frame.
    image_size: str = Field(default="1024x1536", pattern=r"^\d+x\d+$")
    visual_synopsis_source_max_tokens: int = Field(default=2000, ge=1)
    visual_synopsis_context_max_tokens: int = Field(default=1500, ge=1)
    visual_context_max_results: int = Field(default=5, ge=1)
    visual_context_preview_chars: int = Field(default=300, ge=1)
    # Audience/domain the visual prompts target. Configurable so the visual pipeline can
    # be reused across domains without forking the prompts.
    visual_audience_description: str = "a daily AI/ML digest aimed at practicing ML engineers"
    # Language rules for visual output: which language the title/caption use and which
    # language must appear inside the rendered image (image models garble non-Latin glyphs).
    visual_caption_language: str = "Korean"
    visual_on_image_language: str = "SHORT ENGLISH (the image model garbles Korean and other non-Latin glyphs)"


class AgentConfig(BaseModel):
    model_id: LanguageModelId = LanguageModelId.CLAUDE_V4_6_SONNET
    enable_interactive: bool = True
    community_search_domains: list[str] = Field(
        default_factory=lambda: ["twitter.com", "x.com", "reddit.com", "news.ycombinator.com", "substack.com"]
    )
    search_result_limit: int = Field(default=5, ge=1)
    detail_max_tokens: int = Field(default=2000, ge=1)
    search_content_preview_chars: int = Field(default=300, ge=1)
    search_request_timeout: int = Field(default=30, ge=1)
    search_max_retries: int = Field(default=3, ge=1)
    search_retry_backoff_sec: int = Field(default=2, ge=0)
    search_paper_max_authors: int = Field(default=3, ge=1)
    search_paper_abstract_max_chars: int = Field(default=200, ge=1)
    recall_memory_top_k: int = Field(default=5, ge=1)


class SlackConfig(BaseModel):
    bot_token: str = ""
    channel_id: str = ""


class AWSConfig(BaseModel):
    region: str = "us-east-1"
    bedrock_region: str = "us-west-2"
    profile: str = ""
    project_name: str = "omnisummary"
    stage: str = "dev"
    timezone: str = "Asia/Seoul"
    vpc_id: str = ""
    subnet_ids: list[str] = Field(default_factory=list)
    state_bucket_name: str = ""
    s3_prefix: str = ""
    digest_cron_hour: str = "13"
    digest_cron_minute: str = "0"
    api_throttle_rate_limit: int = Field(default=20, ge=1)
    api_throttle_burst_limit: int = Field(default=10, ge=1)
    waf_rate_limit: int = Field(default=2000, ge=100)


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
