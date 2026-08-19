from enum import Enum
from typing import Literal, get_args


class SourceType(str, Enum):
    REDDIT = "reddit"
    RSS = "rss"
    WEB = "web"
    X = "x"
    YOUTUBE = "youtube"


class LanguageModelId(str, Enum):
    CLAUDE_V3_HAIKU = "anthropic.claude-3-haiku-20240307-v1:0"
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
    CLAUDE_V5_OPUS = "anthropic.claude-opus-5"
    # NOTE: add new models here


class EnvVars(str, Enum):
    AWS_BEDROCK_REGION = "AWS_BEDROCK_REGION"
    AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
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

# User-Agent for every server-side fetch of a third-party page or feed. ONE definition: several
# hosts (Reddit's public .rss above all) answer a bare library UA with 403/429, and the feed fetch
# and the OG-image fetch used to carry their own literal — which drifted.
BROWSER_USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)

# Platform aliases that RSSHub routes through its `twitter` namespace.
TWITTER_PLATFORMS: tuple[str, ...] = ("x", "twitter")

# Port the RSSHub service listens on (collector base URL + Fargate container/DNS).
# Single source of truth so config and infrastructure can't drift.
RSSHUB_PORT: int = 1200

# State-store key for the cross-day trends artifact, read by both the digest pipeline
# (TrendTracker) and the research agent (recall_trends). A shared-core artifact, so its key
# lives in the core rather than in either consuming workload.
TRENDS_KEY: str = "trends.json"

# SSM parameter name (under /{PROJECT_NAME}/{STAGE}/) -> the environment variable holding its
# value locally. The CDK stack creates each parameter holding SSM_PLACEHOLDER so no secret ever
# enters a CloudFormation template; scripts/put_secrets.py writes the real values as SecureStrings.
# One mapping, used by both, so a renamed parameter can't drift between them.
SSM_SECRET_ENV_VARS: dict[str, str] = {
    "slack-signing-secret": "SLACK_SIGNING_SECRET",
    "slack-bot-token": "SLACK_BOT_TOKEN",
    "slack-channel-id": "SLACK_CHANNEL_ID",
    "tavily-api-key": "TAVILY_API_KEY",
    "openai-api-key": "OPENAI_API_KEY",
    "youtube-api-key": "YOUTUBE_API_KEY",
    "threads-access-token": "THREADS_ACCESS_TOKEN",
    "threads-user-id": "THREADS_USER_ID",
}

# X/Twitter session cookies for the RSSHub container. They authenticate as the account, so they are
# strictly more sensitive than an API key — and they used to sit in plaintext in the Fargate task
# definition's `environment` block, i.e. in the CloudFormation template. The task now reads them
# through ECS `secrets`, which puts only the parameter ARN there. Owned by the FOUNDATION stack
# (which defines the task) rather than the application stack, so the parameters exist before the
# service that consumes them starts; hence a separate mapping.
SSM_RSSHUB_SECRET_ENV_VARS: dict[str, str] = {
    "twitter-auth-token": "TWITTER_AUTH_TOKEN",
    "twitter-ct0": "TWITTER_CT0",
}

# Every parameter scripts/put_secrets.py is responsible for.
ALL_SSM_SECRET_ENV_VARS: dict[str, str] = {**SSM_SECRET_ENV_VARS, **SSM_RSSHUB_SECRET_ENV_VARS}

# Value the stack writes instead of a real secret. resolve_secret() treats it as "unset" so a
# deploy whose put_secrets step was skipped degrades to the normal missing-credential path
# (logged, feature skipped) rather than sending the literal placeholder to an API as a token.
SSM_PLACEHOLDER: str = "PLACEHOLDER-run-scripts/put_secrets.py"

# The collector names the pipeline knows: the keys of pipeline/runner.py's collector registry, the
# per-source fields of CollectorsConfig, and the vocabulary `collectors.alert_on_empty` is validated
# against. ONE definition, because a typo'd alert_on_empty entry never matches a source name — the
# dark source it was meant to watch simply stays dark, with no signal anywhere.
COLLECTOR_NAMES: tuple[str, ...] = ("reddit", "rss", "rsshub", "web_search", "youtube")

# CloudWatch namespace every EMF record and every metric-backed alarm uses. ONE definition: the
# emitting code and the CDK alarm must agree, and a drifting literal makes an alarm watch a
# namespace nothing writes to (which reads as "no data", i.e. permanently green).
METRIC_NAMESPACE: str = "OmniSummary"

# Project name assumed when PROJECT_NAME is unset (local runs, un-wired stages). Alerts and metrics
# are both attributed per project/stage, so both need the same fallback.
DEFAULT_PROJECT_NAME: str = "omnisummary"

# Character limits applied to titles/queries when written to log lines. Centralized so
# log verbosity can be tuned in one place instead of scattered slice literals.
LOGGING_TRUNCATION_CHARS: dict[str, int] = {
    "title": 70,
    "title_short": 50,
    "brief_title": 60,
    "user_query": 100,
}

# Threads caps each post at 500 characters, and one item's post is assembled as
# title / source line / prose / URL joined by blank lines. Both live HERE because three layers
# depend on them and used to carry their own literal: the renderer that fits a post to the cap, the
# publisher that truncates to it, and the prose budget the digest editor is told about.
THREADS_MAX_POST_CHARS: int = 500
THREADS_POST_SEPARATOR: str = "\n\n"


# Aspect ratios a visual brief may choose. ONE definition: VisualBrief.orientation is typed with it
# and pipeline.image_sizes is validated to map exactly these keys, so renaming an image_sizes key
# now fails config load instead of silently making every brief coerce to the default orientation.
VisualOrientation = Literal["square", "landscape", "portrait"]
VISUAL_ORIENTATIONS: tuple[str, ...] = get_args(VisualOrientation)
