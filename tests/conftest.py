import logging
import socket

import pytest

from adapters.bedrock import BedrockCrossRegionModelHelper
from shared.config import get_config

# Env vars a developer's shell or `.env` supplies in real runs. Any of them leaking into a test
# changes behavior: a real OPENAI_API_KEY makes a "no key configured" path take the key branch,
# a STATE_BUCKET sends a collector to the S3 park file instead of live collection. Cleared
# for every test so the suite behaves the same on a laptop with a full .env and in CI without one.
LEAKY_ENV_VARS = (
    "OMNISUMMARY_EDITION",
    "TAVILY_API_KEY",
    "OPENAI_API_KEY",
    "YOUTUBE_API_KEY",
    "THREADS_ACCESS_TOKEN",
    "THREADS_USER_ID",
    "STATE_BUCKET",
    "S3_PREFIX",
    "MEMORY_ID",
    "ALERT_SNS_TOPIC_ARN",
    "VISUAL_FUNCTION_NAME",
    # research_cli / recall_cli call load_dotenv() at import, i.e. during collection.
    "PROJECT_NAME",
    "STAGE",
    "LOG_LEVEL",
    "S3_SYNC_ACCESS_KEY_ID",
    "S3_SYNC_SECRET_ACCESS_KEY",
    # Ambient AWS credentials sent ChatBedrockConverse down its own session path (12 failures), and
    # tracing variables from .env would ship prompts to LangSmith.
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "LANGCHAIN_TRACING_V2",
    "LANGSMITH_TRACING",
    "LANGCHAIN_API_KEY",
    "LANGSMITH_API_KEY",
)


@pytest.fixture(autouse=True, scope="session")
def log_file_in_tmp(tmp_path_factory):
    """The suite's log lines go to a temp file, not logs/: the cost report and a reader of a real
    run's log read that directory, and test lines carry fake spend."""
    log = logging.getLogger("omnisummary")
    real = [h for h in log.handlers if isinstance(h, logging.FileHandler)]
    for handler in real:
        log.removeHandler(handler)
        handler.close()
    if not real:
        yield
        return
    temp = logging.FileHandler(tmp_path_factory.mktemp("logs") / real[0].baseFilename.rsplit("/", 1)[-1])
    temp.setFormatter(real[0].formatter)
    log.addHandler(temp)
    yield
    log.removeHandler(temp)
    temp.close()


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


class _DisabledSession:
    """A class, not a function: libraries evaluate `boto3.Session | None` in signatures at import."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("AWS access is disabled in tests; patch boto3.Session to opt in")


async def _never_detached(*args, **kwargs) -> bool:
    return False


def _network_disabled(*args, **kwargs):
    raise RuntimeError("Network access is disabled in tests; stub the client")


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

    # Stub the client FACTORIES rather than resolve_secret or the park reader themselves: the
    # env-first/SSM-fallback and park-file logic still executes, and tests that exercise those paths
    # patch the same targets and win. boto3.client covers the module-level clients; boto3.Session
    # covers every client built from a session (the SSM client, the park reader, the state store and
    # AgentCore memory).
    def _aws_disabled(*args, **kwargs):
        raise RuntimeError("AWS access is disabled in tests; patch boto3.client to opt in")

    monkeypatch.setattr("boto3.client", _aws_disabled)
    monkeypatch.setattr("adapters.aws._ssm_client", _aws_disabled)
    monkeypatch.setattr("boto3.Session", _DisabledSession)
    # The reply read-back GETs graph.threads.net on whatever client the test handed in: a real one
    # would go to the network. Tests of the read-back itself call `real_landed_detached`.
    from adapters import threads

    monkeypatch.setattr(threads, "_landed_detached", _never_detached)
    # Everything above is a deny-list, and both incidents (a real post on 2026-10-27, a real S3 read
    # on 10-28) were misses of one. Below it, no test may open a connection or resolve a name at all:
    # a missed stub fails loudly here instead of reaching Threads, Bedrock, OpenAI or the web.
    monkeypatch.setattr(socket.socket, "connect", _network_disabled)
    monkeypatch.setattr(socket.socket, "connect_ex", _network_disabled)
    monkeypatch.setattr(socket, "getaddrinfo", _network_disabled)
    # Name lookups outside getaddrinfo and UDP sends reach the network too.
    for name in ("gethostbyname", "gethostbyname_ex", "gethostbyaddr"):
        monkeypatch.setattr(socket, name, _network_disabled)
    monkeypatch.setattr(socket.socket, "sendto", _network_disabled)
