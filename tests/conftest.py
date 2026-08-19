import pytest

from shared.config import get_config
from shared.utils import BedrockCrossRegionModelHelper

# Env vars a developer's shell or `.env` supplies in real runs. Any of them leaking into a test
# changes behavior: a real SLACK_BOT_TOKEN makes a "no token configured" path take the token
# branch, a STATE_BUCKET sends a collector to the S3 park file instead of live collection. Cleared
# for every test so the suite behaves the same on a laptop with a full .env and in CI without one.
LEAKY_ENV_VARS = (
    "SLACK_BOT_TOKEN",
    "SLACK_CHANNEL_ID",
    "SLACK_SIGNING_SECRET",
    "TAVILY_API_KEY",
    "OPENAI_API_KEY",
    "YOUTUBE_API_KEY",
    "THREADS_ACCESS_TOKEN",
    "THREADS_USER_ID",
    "CLOUDFLARE_PROXY_URL",
    "CLOUDFLARE_PROXY_TOKEN",
    "STATE_BUCKET",
    "S3_PREFIX",
    "MEMORY_ID",
    "ALERT_SNS_TOPIC_ARN",
    "VISUAL_FUNCTION_NAME",
    "RSSHUB_BASE_URL",
    "AGENTCORE_RUNTIME_ARN",
    "DDB_TABLE_NAME",
)


@pytest.fixture(autouse=True)
def clear_config_cache():
    """get_config() caches the parsed Config process-wide, so a Config built (or patched) by one
    test would otherwise leak into the next. Cleared on both sides of every test."""
    get_config.cache_clear()
    yield
    get_config.cache_clear()


@pytest.fixture(autouse=True)
def clear_model_resolution_cache():
    """BedrockCrossRegionModelHelper memoizes (model_id, region) -> resolved model id on the CLASS.
    Nothing clears it, so the first test to exercise resolution would decide which id every later
    test resolves to — including tests that mean to assert the ladder's other branches."""
    BedrockCrossRegionModelHelper._resolution_cache.clear()
    yield
    BedrockCrossRegionModelHelper._resolution_cache.clear()


@pytest.fixture(autouse=True)
def hermetic_env(monkeypatch):
    """Isolate every test from the ambient environment and from live AWS.

    Uses monkeypatch only (never a global os.environ mutation), so state is restored per test even
    when a test fails mid-way.
    """
    for name in LEAKY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)

    # Config.load() calls load_dotenv(), which would put the cleared secrets straight back.
    monkeypatch.setattr("shared.config.load_dotenv", lambda *args, **kwargs: False)

    # resolve_secret falls back to SSM whenever an env var is absent, which meant a real
    # credential-resolution + SSM round trip per call (seconds each, and dependent on whatever
    # AWS profile the developer had). Stub the client FACTORY rather than resolve_secret itself:
    # the env-first/SSM-fallback logic still executes, and tests that legitimately exercise the
    # SSM path (tests/test_utils.py) patch this same target themselves and win.
    def _aws_disabled(*args, **kwargs):
        raise RuntimeError("AWS access is disabled in tests; patch boto3.client to opt in")

    monkeypatch.setattr("shared.utils.boto3.client", _aws_disabled)
