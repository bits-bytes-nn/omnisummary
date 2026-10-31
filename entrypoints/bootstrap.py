"""The composition root: the one place that builds adapters and wires them into the application.

Every entrypoint (the Lambda handlers, main.py, research_cli.py, recall_cli.py, the scripts) gets its
collaborators from here, so the application core never constructs an AWS, Threads or OpenAI client
itself, and a test hands it fakes instead."""

from __future__ import annotations

import boto3

from adapters.openai_images import OpenAIImageRenderer
from adapters.threads import ThreadsGateway
from application.daily_visual import DailyVisualMaker
from shared import BedrockLanguageModelFactory, Config, available_boto_profile, create_state_store, logger
from shared.state_store import StateStore


def language_models(config: Config) -> BedrockLanguageModelFactory:
    """Bedrock in config.aws.bedrock_region, on the configured profile where it resolves (a developer
    laptop) and on ambient credentials everywhere else (the Lambdas)."""
    session = boto3.Session(
        profile_name=available_boto_profile(config.aws.profile), region_name=config.aws.bedrock_region
    )
    return BedrockLanguageModelFactory(boto_session=session, region_name=config.aws.bedrock_region)


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
