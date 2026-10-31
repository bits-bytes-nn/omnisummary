"""The composition root: the one place that builds adapters and wires them into the application.

Every entrypoint (the Lambda handlers, main.py, research_cli.py, recall_cli.py, the scripts) gets its
collaborators from here, so the application core never constructs an AWS, Threads or OpenAI client
itself, and a test hands it fakes instead."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import boto3

from adapters.aws import available_boto_profile
from adapters.bedrock import BedrockLanguageModelFactory
from adapters.memory import create_memory_store
from adapters.openai_images import OpenAIImageRenderer
from adapters.research_backends import WebResearchSources
from adapters.state_store import create_state_store
from adapters.threads import ThreadsGateway
from application.daily_visual import DailyVisualMaker
from application.ports import ContentSource, LanguageModels, MemoryStore, StateStore
from application.research import ResearchPorts
from collectors import RedditCollector, RSSCollector, RSSHubCollector, WebSearchCollector, YouTubeCollector
from shared import Config, logger


def language_models(config: Config) -> BedrockLanguageModelFactory:
    """Bedrock in config.aws.bedrock_region, on the configured profile where it resolves (a developer
    laptop) and on ambient credentials everywhere else (the Lambdas)."""
    session = boto3.Session(
        profile_name=available_boto_profile(config.aws.profile), region_name=config.aws.bedrock_region
    )
    return BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)


def state_store(config: Config) -> StateStore:
    """S3 when a state bucket is configured (STATE_BUCKET, else config.aws.state_bucket_name), else the
    local digest_state directory."""
    return create_state_store(config)


def memory_store(base_dir: Path | None = None) -> MemoryStore:
    """The stack's AgentCore Memory (MEMORY_ID, or found by name for a local run), else the local
    store under `base_dir`. Raises MemoryReadError when the lookup itself fails."""
    return create_memory_store(base_dir)


def collector_registry(
    config: Config, models: LanguageModels
) -> dict[str, tuple[Callable[..., ContentSource], Any, dict]]:
    """collector name -> (class, its config, extra constructor kwargs).

    The keys are COLLECTOR_NAMES, which is also what `collectors.alert_on_empty` is validated
    against — a source this registry names but that vocabulary doesn't could never be watched for
    going dark."""
    return {
        "reddit": (RedditCollector, config.collectors.reddit, {}),
        "rsshub": (RSSHubCollector, config.collectors.rsshub, {}),
        "rss": (RSSCollector, config.collectors.rss, {}),
        "web_search": (WebSearchCollector, config.collectors.web_search, {"llm_factory": models}),
        "youtube": (YouTubeCollector, config.collectors.youtube, {}),
    }


def collectors(
    config: Config, models: LanguageModels, sources: list[str] | None = None
) -> list[tuple[str, ContentSource]]:
    """The (name, collector) pairs to run: the enabled ones, or the named ones that are enabled, in
    the order named."""
    registry = collector_registry(config, models)
    names = [name for name, (_, cfg, _kw) in registry.items() if cfg.enabled] if sources is None else sources
    built: list[tuple[str, ContentSource]] = []
    for name in names:
        if name not in registry:
            logger.warning("Unknown source: '%s'", name)
            continue
        collector_cls, collector_cfg, kwargs = registry[name]
        if not collector_cfg.enabled:
            logger.info("Skipping disabled source: '%s'", name)
            continue
        built.append((name, collector_cls(collector_cfg, **kwargs)))
    return built


def threads(config: Config) -> ThreadsGateway:
    return ThreadsGateway(config)


def image_renderer(config: Config) -> OpenAIImageRenderer:
    return OpenAIImageRenderer(
        image_model=config.pipeline.image_model,
        image_quality=config.pipeline.visual_image_quality,
        timeout_sec=config.pipeline.visual_image_timeout_sec,
        max_retries=config.pipeline.visual_image_max_retries,
    )


def optional_state_store(config: Config) -> StateStore | None:
    """The state store, or None when it cannot be built: the digest still publishes without it."""
    try:
        return create_state_store(config)
    except Exception:
        logger.error("State store unavailable (init failed); publishing without the Threads ledger", exc_info=True)
        return None


def daily_visual_maker(config: Config, models: BedrockLanguageModelFactory | None = None) -> DailyVisualMaker:
    return DailyVisualMaker(
        config,
        models or language_models(config),
        optional_state_store(config),
        threads(config),
        image_renderer(config),
    )


def research_ports(config: Config) -> ResearchPorts:
    """A research run's reading, its Threads account, the fact check's model and the read-only stores
    its recall tools use. The models are built here, the stores on first use."""
    return ResearchPorts(
        sources=WebResearchSources(),
        threads=threads(config),
        models=language_models(config),
        state_store=lambda: state_store(config),
        memory_store=memory_store,
    )
