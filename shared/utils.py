import hashlib
import re
from abc import ABC, abstractmethod
from calendar import timegm
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Any, ClassVar, Generic, TypeVar

import boto3
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrock, ChatBedrockConverse
from pydantic import BaseModel
from tiktoken import get_encoding

from .constants import LanguageModelId
from .logger import logger


class LanguageModelInfo(BaseModel):
    context_window_size: int
    max_output_tokens: int
    supports_performance_optimization: bool = False
    supports_prompt_caching: bool = False
    supports_thinking: bool = False
    supports_1m_context_window: bool = False


_LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
    LanguageModelId.CLAUDE_V3_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=4096,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V3_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=8192,
        supports_performance_optimization=True,
        supports_prompt_caching=True,
    ),
    LanguageModelId.CLAUDE_V4_5_HAIKU: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V3_5_SONNET: LanguageModelInfo(context_window_size=200000, max_output_tokens=8192),
    LanguageModelId.CLAUDE_V3_5_SONNET_V2: LanguageModelInfo(
        context_window_size=200000, max_output_tokens=8192, supports_prompt_caching=True
    ),
    LanguageModelId.CLAUDE_V3_7_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_5_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_6_SONNET: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_1_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_5_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=64000,
        supports_prompt_caching=True,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_6_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_7_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    LanguageModelId.CLAUDE_V4_8_OPUS: LanguageModelInfo(
        context_window_size=1000000,
        max_output_tokens=64000,
        supports_thinking=True,
        supports_1m_context_window=True,
    ),
    # NOTE: add new models here
}


ModelIdT = TypeVar("ModelIdT")
ModelInfoT = TypeVar("ModelInfoT")
WrapperT = TypeVar("WrapperT")

MAX_TOKENS: int = 150000


class BaseBedrockModelFactory(Generic[ModelIdT, ModelInfoT, WrapperT], ABC):
    BOTO_READ_TIMEOUT: ClassVar[int] = 300
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 3
    MAX_POOL_CONNECTIONS: ClassVar[int] = 50

    def __init__(
        self,
        boto_session: boto3.Session | None = None,
        region_name: str | None = None,
        profile_name: str | None = None,
    ) -> None:
        self.boto_session = boto_session or boto3.Session(profile_name=profile_name)
        self.region_name = region_name or self.boto_session.region_name
        boto_config = BotoConfig(
            read_timeout=self.BOTO_READ_TIMEOUT,
            connect_timeout=60,
            retries={"max_attempts": self.BOTO_MAX_ATTEMPTS, "mode": "adaptive"},
            max_pool_connections=self.MAX_POOL_CONNECTIONS,
        )
        self._client = self.boto_session.client(
            self._get_boto_service_name(),
            region_name=self.region_name,
            config=boto_config,
        )
        logger.debug("Initialized %s for region: '%s'", self.__class__.__name__, self.region_name)

    @abstractmethod
    def _get_boto_service_name(self) -> str: ...

    @abstractmethod
    def _get_model_info_dict(self) -> dict[ModelIdT, ModelInfoT]: ...

    @abstractmethod
    def get_model(self, model_id: ModelIdT, **kwargs: Any) -> WrapperT: ...

    def get_model_info(self, model_id: ModelIdT) -> ModelInfoT | None:
        return self._get_model_info_dict().get(model_id)

    def get_supported_models(self) -> list[ModelIdT]:
        return list(self._get_model_info_dict().keys())


class BedrockCrossRegionModelHelper:
    @staticmethod
    def get_cross_region_model_id(
        boto_session: boto3.Session,
        model_id: LanguageModelId,
        region_name: str,
    ) -> str:
        try:
            bedrock_client = boto_session.client("bedrock", region_name=region_name)
            global_model_id = BedrockCrossRegionModelHelper._build_cross_region_model_id(
                model_id, region_name, is_global=True
            )
            if BedrockCrossRegionModelHelper._is_cross_region_model_available(bedrock_client, global_model_id):
                logger.debug("Using global cross-region model: '%s'", global_model_id)
                return global_model_id
            regional_model_id = BedrockCrossRegionModelHelper._build_cross_region_model_id(
                model_id, region_name, is_global=False
            )
            if BedrockCrossRegionModelHelper._is_cross_region_model_available(bedrock_client, regional_model_id):
                logger.debug("Using regional cross-region model: '%s'", regional_model_id)
                return regional_model_id
            logger.debug(
                "Cross-region models not available, using standard model: '%s'",
                model_id.value,
            )
            return model_id.value
        except Exception as e:
            logger.warning(
                "Failed to resolve cross-region model for '%s': %s. Falling back to standard model.",
                model_id.value,
                e,
            )
            return model_id.value

    @staticmethod
    def _build_cross_region_model_id(model_id: LanguageModelId, region_name: str, is_global: bool = False) -> str:
        if is_global:
            return f"global.{model_id.value}"
        prefix = "apac" if region_name.startswith("ap-") else region_name[:2]
        return f"{prefix}.{model_id.value}"

    @staticmethod
    def _is_cross_region_model_available(bedrock_client: Any, cross_region_id: str) -> bool:
        try:
            response = bedrock_client.list_inference_profiles(maxResults=1000, typeEquals="SYSTEM_DEFINED")
            available_profiles = {
                profile["inferenceProfileId"] for profile in response.get("inferenceProfileSummaries", [])
            }
            return cross_region_id in available_profiles
        except Exception as e:
            raise RuntimeError(f"Failed to check cross-region model availability: {e}") from e


class BedrockLanguageModelFactory(
    BaseBedrockModelFactory[LanguageModelId, LanguageModelInfo, ChatBedrock | ChatBedrockConverse]
):
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0
    DEFAULT_TOP_K: ClassVar[int] = 50
    DEFAULT_THINKING_BUDGET_TOKENS: ClassVar[int] = 2048
    DEFAULT_LATENCY_MODE: ClassVar[str] = "normal"

    def _get_boto_service_name(self) -> str:
        return "bedrock-runtime"

    def _get_model_info_dict(self) -> dict[LanguageModelId, LanguageModelInfo]:
        return _LANGUAGE_MODEL_INFO

    def get_model(self, model_id: LanguageModelId, **kwargs: Any) -> ChatBedrock | ChatBedrockConverse:
        model_info = self.get_model_info(model_id)
        if not model_info:
            raise ValueError(f"Unsupported language model ID: '{model_id.value}'")
        resolved_model_id = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            self.boto_session, model_id, self.region_name or ""
        )
        is_cross_region = resolved_model_id != model_id.value
        enable_thinking = kwargs.get("enable_thinking", False)
        use_converse = is_cross_region or (enable_thinking and model_info.supports_thinking)
        model_config = self._build_model_config(model_info, resolved_model_id, use_converse, **kwargs)
        model_class = ChatBedrockConverse if use_converse else ChatBedrock
        model = model_class(**model_config)
        logger.debug(
            "Created language model: '%s' with class %s",
            resolved_model_id,
            model_class.__name__,
        )
        return model

    def _build_model_config(
        self,
        model_info: LanguageModelInfo,
        resolved_model_id: str,
        is_cross_region: bool,
        **kwargs: Any,
    ) -> dict[str, Any]:
        enable_thinking = kwargs.get("enable_thinking", False)
        supports_1m_context_window = kwargs.get("supports_1m_context_window", False)
        temperature = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        final_temperature = 1.0 if self._should_enable_thinking(enable_thinking, model_info) else temperature
        if final_temperature != temperature:
            logger.debug("Adjusting temperature to 1.0 for thinking mode")
        final_max_tokens = self._validate_max_tokens(kwargs.get("max_tokens"), model_info)
        config = self._build_base_config(resolved_model_id, is_cross_region, **kwargs)
        if is_cross_region:
            config.update({"max_tokens": final_max_tokens, "temperature": final_temperature})
        else:
            config["model_kwargs"].update({"max_tokens": final_max_tokens, "temperature": final_temperature})
        if supports_1m_context_window and model_info.supports_1m_context_window:
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(
                    {"anthropic_beta": ["context-1m-2025-08-07"]}
                )
            else:
                config["model_kwargs"].setdefault("additionalModelRequestFields", {}).update(
                    {"anthropic_beta": ["context-1m-2025-08-07"]}
                )
            logger.debug("Applied 1M context window support")
        self._apply_model_features(config, model_info, is_cross_region, **kwargs)
        return config

    def _build_base_config(self, resolved_model_id: str, is_cross_region: bool, **kwargs: Any) -> dict[str, Any]:
        config = {
            "model_id": resolved_model_id,
            "region_name": self.region_name,
            "client": self._client,
            "callbacks": kwargs.get("callbacks"),
        }
        if self.boto_session.profile_name and self.boto_session.profile_name != "default":
            config["credentials_profile_name"] = self.boto_session.profile_name
        common_params = {
            "stop_sequences": ["\n\nHuman:"],
        }
        if is_cross_region:
            config.update(common_params)
        else:
            config["model_kwargs"] = {
                "top_k": kwargs.get("top_k", self.DEFAULT_TOP_K),
                **common_params,
            }
        return config

    def _apply_model_features(
        self,
        config: dict[str, Any],
        model_info: LanguageModelInfo,
        is_cross_region: bool,
        **kwargs: Any,
    ) -> None:
        enable_perf = kwargs.get("enable_performance_optimization", False)
        enable_think = kwargs.get("enable_thinking", False)
        if self._should_enable_performance_optimization(enable_perf, model_info, is_cross_region):
            latency = kwargs.get("latency_mode", self.DEFAULT_LATENCY_MODE)
            config.setdefault("performanceConfig", {}).update({"latency": latency})
            logger.debug("Applied performance optimization (latency_mode='%s')", latency)
        if self._should_enable_thinking(enable_think, model_info):
            budget = kwargs.get("thinking_budget_tokens", self.DEFAULT_THINKING_BUDGET_TOKENS)
            think_config = {"thinking": {"type": "enabled", "budget_tokens": budget}}
            if is_cross_region:
                config.setdefault("additional_model_request_fields", {}).update(think_config)
            else:
                config.setdefault("model_kwargs", {}).update(think_config)
            logger.debug("Applied thinking mode (budget_tokens=%d)", budget)

    @staticmethod
    def _validate_max_tokens(max_tokens: int | None, model_info: LanguageModelInfo) -> int:
        final_max_tokens = max_tokens or model_info.max_output_tokens
        if final_max_tokens > model_info.max_output_tokens:
            logger.warning(
                "Requested max_tokens (%d) exceeds model's maximum (%d). Adjusting.",
                final_max_tokens,
                model_info.max_output_tokens,
            )
            return model_info.max_output_tokens
        return final_max_tokens

    @staticmethod
    def _should_enable_performance_optimization(
        enable: bool, model_info: LanguageModelInfo, is_cross_region: bool
    ) -> bool:
        return enable and model_info.supports_performance_optimization and not is_cross_region

    @staticmethod
    def _should_enable_thinking(enable: bool, model_info: LanguageModelInfo) -> bool:
        return enable and model_info.supports_thinking


def generate_item_id(url: str) -> str:
    return hashlib.sha256(url.encode()).hexdigest()[:16]


def parse_feed_published_date(entry) -> datetime | None:
    if hasattr(entry, "published_parsed") and entry.published_parsed:
        return datetime.fromtimestamp(timegm(entry.published_parsed), tz=UTC)

    published_str = entry.get("published")
    if published_str:
        try:
            return parsedate_to_datetime(published_str).astimezone(UTC)
        except (TypeError, ValueError):
            pass

    if hasattr(entry, "updated_parsed") and entry.updated_parsed:
        return datetime.fromtimestamp(timegm(entry.updated_parsed), tz=UTC)

    return None


def truncate_text_by_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    encoding = get_encoding("cl100k_base")
    tokens = encoding.encode(text)
    if len(tokens) <= max_tokens:
        return text
    truncated_tokens = tokens[:max_tokens]
    truncated = encoding.decode(truncated_tokens)
    logger.warning("Text truncated from %d to %d tokens", len(tokens), len(truncated_tokens))
    return truncated


def sanitize_slack_mrkdwn(text: str) -> str:
    text = re.sub(r"\n---+\n", "\n\n", text)
    text = re.sub(r"\n\*\*\*+\n", "\n\n", text)
    text = re.sub(r"\n___+\n", "\n\n", text)

    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    text = re.sub(r"\*\*([^*\n]+?)\*\*", r"*\1*", text)

    text = re.sub(r"\* ([^*\n]+?)\*", r"*\1*", text)
    text = re.sub(r"\*([^*\n]+?) \*", r"*\1*", text)

    text = re.sub(r'"\*([^*\n]+?)\*"', r"*\1*", text)

    text = re.sub(r"(\S)\*([^*\n]+?)\*", r"\1 *\2*", text)
    text = re.sub(r"\*([^*\n]+?)\*(\S)", r"*\1* \2", text)

    text = re.sub(r"!\[([^\]]*)\]\(([^)]+)\)", r"<\2|\1>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"<\2|\1>", text)

    text = re.sub(
        r"[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF"
        r"\U0001F900-\U0001F9FF\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF"
        r"\U00002702-\U000027B0\U0000FE00-\U0000FE0F\U0000200D"
        r"\U00002194-\U00002199\U000021A9-\U000021AA\U0000231A-\U0000231B"
        r"\U000023E9-\U000023F3\U000025AA-\U000025B7\U000025B9-\U000025FE\U00002600-\U000026FF"
        r"\U00002934-\U00002935\U00003030\U0000303D\U00003297\U00003299]+",
        "",
        text,
    )

    text = re.sub(r"\n{3,}", "\n\n", text)

    text = re.sub(r"(?<!\n)\n(\*<)", r"\n\n\1", text)
    return text.strip()
