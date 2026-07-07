from enum import Enum


class SourceType(str, Enum):
    REDDIT = "reddit"
    RSS = "rss"
    WEB = "web"
    X = "x"
    YOUTUBE = "youtube"


class LanguageModelId(str, Enum):
    CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
    CLAUDE_V3_SONNET = "anthropic.claude-3-sonnet-20240229-v1:0"
    CLAUDE_V3_OPUS = "anthropic.claude-3-opus-20240229-v1:0"
    CLAUDE_V3_5_HAIKU = "anthropic.claude-3-5-haiku-20241022-v1:0"
    CLAUDE_V4_5_HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
    CLAUDE_V3_5_SONNET = "anthropic.claude-3-5-sonnet-20240620-v1:0"
    CLAUDE_V3_5_SONNET_V2 = "anthropic.claude-3-5-sonnet-20241022-v2:0"
    CLAUDE_V3_7_SONNET = "anthropic.claude-3-7-sonnet-20250219-v1:0"
    CLAUDE_V4_SONNET = "anthropic.claude-sonnet-4-20250514-v1:0"
    CLAUDE_V4_5_SONNET = "anthropic.claude-sonnet-4-5-20250929-v1:0"
    CLAUDE_V4_OPUS = "anthropic.claude-opus-4-20250514-v1:0"
    CLAUDE_V4_1_OPUS = "anthropic.claude-opus-4-1-20250805-v1:0"
    CLAUDE_V4_5_OPUS = "anthropic.claude-opus-4-5-20251101-v1:0"
    CLAUDE_V4_6_SONNET = "anthropic.claude-sonnet-4-6"
    CLAUDE_V5_SONNET = "anthropic.claude-sonnet-5"
    CLAUDE_V4_6_OPUS = "anthropic.claude-opus-4-6-v1"
    CLAUDE_V4_7_OPUS = "anthropic.claude-opus-4-7"
    CLAUDE_V4_8_OPUS = "anthropic.claude-opus-4-8"
    # NOTE: add new models here


class EnvVars(str, Enum):
    AWS_BEDROCK_REGION = "AWS_BEDROCK_REGION"
    AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
    AWS_PROFILE_NAME = "AWS_PROFILE_NAME"
    LOG_LEVEL = "LOG_LEVEL"
    SLACK_BOT_TOKEN = "SLACK_BOT_TOKEN"
    SLACK_CHANNEL_ID = "SLACK_CHANNEL_ID"
    TAVILY_API_KEY = "TAVILY_API_KEY"
    CLOUDFLARE_PROXY_URL = "CLOUDFLARE_PROXY_URL"
    CLOUDFLARE_PROXY_TOKEN = "CLOUDFLARE_PROXY_TOKEN"
    YOUTUBE_API_KEY = "YOUTUBE_API_KEY"
    STATE_BUCKET = "STATE_BUCKET"
    RSSHUB_BASE_URL = "RSSHUB_BASE_URL"
    AGENTCORE_RUNTIME_ARN = "AGENTCORE_RUNTIME_ARN"
    DDB_TABLE_NAME = "DDB_TABLE_NAME"
    PROJECT_NAME = "PROJECT_NAME"
    STAGE = "STAGE"


class LocalPaths(str, Enum):
    DIGEST_STATE_DIR = "digest_state"
    LOGS_DIR = "logs"
    LOGS_FILE = "logs.txt"


DOMAIN_TO_SOURCE: dict[str, SourceType] = {
    "x.com": SourceType.X,
    "twitter.com": SourceType.X,
}

# Platform aliases that RSSHub routes through its `twitter` namespace.
TWITTER_PLATFORMS: tuple[str, ...] = ("x", "twitter")

# Port the RSSHub service listens on (collector base URL + Fargate container/DNS).
# Single source of truth so config and infrastructure can't drift.
RSSHUB_PORT: int = 1200

# State-store key for the cross-day trends artifact, read by both the digest pipeline
# (TrendTracker) and the research agent (recall_trends). A shared-core artifact, so its key
# lives in the core rather than in either consuming workload.
TRENDS_KEY: str = "trends.json"

# Character limits applied to titles/queries when written to log lines. Centralized so
# log verbosity can be tuned in one place instead of scattered slice literals.
LOGGING_TRUNCATION_CHARS: dict[str, int] = {
    "title": 70,
    "title_short": 50,
    "brief_title": 60,
    "user_query": 100,
}
