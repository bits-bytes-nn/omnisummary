from enum import Enum
from typing import Literal, get_args


class SourceType(str, Enum):
    REDDIT = "reddit"
    RSS = "rss"
    WEB = "web"
    X = "x"
    YOUTUBE = "youtube"


class LanguageModelId(str, Enum):
    CLAUDE_V4_6_SONNET = "anthropic.claude-sonnet-4-6"
    CLAUDE_V5_SONNET = "anthropic.claude-sonnet-5"
    CLAUDE_V4_8_OPUS = "anthropic.claude-opus-4-8"
    CLAUDE_V5_OPUS = "anthropic.claude-opus-5"
    CLAUDE_V5_5_HAIKU = "anthropic.claude-haiku-5-5"
    CLAUDE_V5_5_SONNET = "anthropic.claude-sonnet-5-5"
    CLAUDE_V5_5_OPUS = "anthropic.claude-opus-5-5"
    GPT_V6_1_TERRA = "openai.gpt-6.1-terra"
    GPT_V6_1_SOL = "openai.gpt-6.1-sol"


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

# Port the local RSSHub container listens on (the collector's base URL).
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
    "tavily-api-key": "TAVILY_API_KEY",
    "openai-api-key": "OPENAI_API_KEY",
    "youtube-api-key": "YOUTUBE_API_KEY",
    "threads-access-token": "THREADS_ACCESS_TOKEN",
    "threads-user-id": "THREADS_USER_ID",
}

# Value the stack writes instead of a real secret. resolve_secret() treats it as "unset" so a
# deploy whose put_secrets step was skipped degrades to the normal missing-credential path
# (logged, feature skipped) rather than sending the literal placeholder to an API as a token.
SSM_PLACEHOLDER: str = "PLACEHOLDER-run-scripts/put_secrets.py"

# The collector names the pipeline knows: the keys of pipeline/runner.py's collector registry, the
# per-source fields of CollectorsConfig, and the vocabulary `collectors.alert_on_empty` is validated
# against. ONE definition, because a typo'd alert_on_empty entry never matches a source name — the
# dark source it was meant to watch simply stays dark, with no signal anywhere.
COLLECTOR_NAMES: tuple[str, ...] = ("reddit", "rss", "rsshub", "web_search", "youtube")

# empty_rate_threshold value that DISABLES the all-inputs-empty tripwire. degradation_reason tests
# `empty_rate > empty_threshold`, so 100 can never be exceeded — the check is unreachable at this
# value. Named here because the config default, the collector helpers' defaults and the config test
# that asserts the knob stays armed must all mean the same sentinel.
EMPTY_RATE_CHECK_DISABLED: float = 100.0

# CloudWatch namespace every EMF record and every metric-backed alarm uses. ONE definition: the
# emitting code and the CDK alarm must agree, and a drifting literal makes an alarm watch a
# namespace nothing writes to (which reads as "no data", i.e. permanently green).
METRIC_NAMESPACE: str = "OmniSummary"
# The digest Lambda's daily story-count metric; the stack's zero-stories alarm reads the same name.
DIGEST_ITEMS_METRIC = "DigestItemsPublished"

# Project name assumed when PROJECT_NAME is unset (local runs, un-wired stages). Alerts and metrics
# are both attributed per project/stage, so both need the same fallback.
DEFAULT_PROJECT_NAME: str = "omnisummary"

# Threads caps each post at 500 characters, and one item's post is assembled as
# title / source line / prose / URL joined by blank lines. Both live HERE because three layers
# depend on them and used to carry their own literal: the renderer that fits a post to the cap, the
# publisher that truncates to it, and the prose budget the digest editor is told about.
THREADS_MAX_POST_CHARS: int = 500
THREADS_POST_SEPARATOR: str = "\n\n"


# Aspect ratios a visual brief may choose. ONE definition: VisualBrief.orientation is typed with it
# and pipeline.image_sizes is validated to map exactly these keys, so renaming an image_sizes key
# now fails config load instead of silently making every brief coerce to the default orientation.
VisualOrientation = Literal["tall", "portrait", "square", "landscape"]
VISUAL_ORIENTATIONS: tuple[str, ...] = get_args(VisualOrientation)

# The digest runs once a day (the EventBridge cron). One skipped local park sync is one period behind.
DIGEST_PERIOD_HOURS = 24
