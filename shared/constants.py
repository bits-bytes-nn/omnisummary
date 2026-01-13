from enum import Enum


class AppConstants:
    class External(str, Enum):
        UPSTAGE_DOCUMENT_PARSE = "https://api.upstage.ai/v1/document-ai/document-parse"


class ContentType(str, Enum):
    HTML = "html"
    PDF = "pdf"
    YOUTUBE = "youtube"


class EnvVars(str, Enum):
    AGENTCORE_RUNTIME_ARN = "AGENTCORE_RUNTIME_ARN"
    AWS_BEDROCK_REGION = "AWS_BEDROCK_REGION"
    AWS_DEFAULT_REGION = "AWS_DEFAULT_REGION"
    AWS_PROFILE_NAME = "AWS_PROFILE_NAME"
    DDB_TABLE_NAME = "DDB_TABLE_NAME"
    EVENT_DEDUPLICATION_TTL_SEC = "EVENT_DEDUPLICATION_TTL_SEC"
    LANGCHAIN_API_KEY = "LANGCHAIN_API_KEY"
    LANGCHAIN_TRACING_V2 = "LANGCHAIN_TRACING_V2"
    LANGCHAIN_ENDPOINT = "LANGCHAIN_ENDPOINT"
    LANGCHAIN_PROJECT = "LANGCHAIN_PROJECT"
    LOG_LEVEL = "LOG_LEVEL"
    PROJECT_NAME = "PROJECT_NAME"
    SLACK_BUSINESS_TOKEN = "SLACK_BUSINESS_TOKEN"
    SLACK_BUSINESS_CHANNEL_IDS = "SLACK_BUSINESS_CHANNEL_IDS"
    SLACK_PERSONAL_TOKEN = "SLACK_PERSONAL_TOKEN"
    SLACK_PERSONAL_CHANNEL_IDS = "SLACK_PERSONAL_CHANNEL_IDS"
    SLACK_SIGNATURE_EXPIRATION_SEC = "SLACK_SIGNATURE_EXPIRATION_SEC"
    SLACK_SIGNING_SECRET = "SLACK_SIGNING_SECRET"
    STAGE = "STAGE"
    UPSTAGE_API_KEY = "UPSTAGE_API_KEY"


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
    # NOTE: add new models here


class LocalPaths(str, Enum):
    ASSETS_DIR = "assets"
    FIGURES_DIR = "figures"
    DOCS_DIR = "docs"
    LOGS_DIR = "logs"
    POSTS_DIR = "_posts"
    LOGS_FILE = "logs.txt"
    PARSED_FILE = "parsed.json"


class PdfParserType(str, Enum):
    UNSTRUCTURED = "unstructured"
    UPSTAGE = "upstage"


class SSMParams(str, Enum):
    LANGCHAIN_API_KEY = "langchain-api-key"
    SLACK_BUSINESS_TOKEN = "slack-business-token"
    SLACK_BUSINESS_CHANNEL_IDS = "slack-business-channel-ids"
    SLACK_PERSONAL_TOKEN = "slack-personal-token"
    SLACK_PERSONAL_CHANNEL_IDS = "slack-personal-channel-ids"
    SLACK_SIGNING_SECRET = "slack-signing-secret"
    UPSTAGE_API_KEY = "upstage-api-key"
