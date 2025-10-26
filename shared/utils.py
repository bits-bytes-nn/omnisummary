import asyncio
import re
from abc import ABC, abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, Generic, TypeVar

import boto3
import tenacity
from boto3.s3.transfer import TransferConfig
from botocore.config import Config as BotoConfig
from botocore.exceptions import ClientError
from bs4 import BeautifulSoup
from langchain_aws import ChatBedrock, ChatBedrockConverse
from langchain_core.output_parsers import BaseOutputParser
from pydantic import BaseModel, Field, HttpUrl, PrivateAttr, TypeAdapter
from tiktoken import Encoding, get_encoding

from .constants import LanguageModelId
from .logger import logger

MAX_RETRIES: int = 5
MAX_TOKENS: int = 150000
RETRY_MAX_WAIT: int = 120
RETRY_MULTIPLIER: int = 30


class AWSHandlerError(Exception):
    pass


class S3OperationError(AWSHandlerError):
    pass


class LanguageModelInfo(BaseModel):
    context_window_size: int = Field(description="Maximum context window size in tokens that the model can handle.")
    max_output_tokens: int = Field(description="Maximum number of tokens the model can generate in a single response.")
    supports_performance_optimization: bool = Field(
        default=False,
        description="Whether the model supports performance optimization features.",
    )
    supports_prompt_caching: bool = Field(
        default=False,
        description="Whether the model supports prompt caching to improve performance.",
    )
    supports_thinking: bool = Field(
        default=False,
        description="Whether the model supports thinking/reasoning capabilities.",
    )
    supports_1m_context_window: bool = Field(
        default=False,
        description="Whether the model supports 1M context window.",
    )


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
    LanguageModelId.CLAUDE_V4_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=32000,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    LanguageModelId.CLAUDE_V4_1_OPUS: LanguageModelInfo(
        context_window_size=200000,
        max_output_tokens=32000,
        supports_prompt_caching=True,
        supports_thinking=True,
    ),
    # NOTE: add new models here
}

ModelIdT = TypeVar("ModelIdT")
ModelInfoT = TypeVar("ModelInfoT")
WrapperT = TypeVar("WrapperT")


class BaseBedrockWrapper:
    buffer_tokens: int = Field(default=128, ge=0)
    _tokenizer: Encoding = PrivateAttr()

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._tokenizer = get_encoding("cl100k_base")

    def _truncate_text(self, text: str, max_chars: int | None, max_tokens: int | None, text_type: str) -> str:
        if not max_chars and not max_tokens:
            return text

        token_ids = self._tokenizer.encode(text, allowed_special="all")
        final_text = text
        truncated = False

        if max_tokens and len(token_ids) > max_tokens:
            effective_tokens = max_tokens - self.buffer_tokens
            truncated_token_ids = token_ids[:effective_tokens]
            final_text = self._tokenizer.decode(truncated_token_ids)
            logger.warning(
                f"{text_type.capitalize()} token count ({len(token_ids)}) exceeds maximum ({max_tokens}). Truncating."
            )
            truncated = True

        if max_chars and len(text) > max_chars:
            if not truncated or len(text[:max_chars]) < len(final_text):
                final_text = text[:max_chars]
                logger.warning(
                    f"{text_type.capitalize()} character count ({len(text)}) exceeds maximum ({max_chars}). Truncating."
                )

        return final_text


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
        model_config = self._build_model_config(model_info, resolved_model_id, is_cross_region, **kwargs)
        model_class = ChatBedrockConverse if is_cross_region else ChatBedrock
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


class HTMLTagOutputParser(BaseOutputParser):
    tag_names: str | list[str]

    def parse(self, text: str) -> str | dict[str, str]:
        if not text:
            return {} if isinstance(self.tag_names, list) else ""
        soup = BeautifulSoup(text, "html.parser")
        parsed: dict[str, str] = {}
        tag_list = self.tag_names if isinstance(self.tag_names, list) else [self.tag_names]
        for tag_name in tag_list:
            if tag := soup.find(tag_name):
                if hasattr(tag, "decode_contents"):
                    parsed[tag_name] = str(tag.decode_contents()).strip()
                else:
                    parsed[tag_name] = str(tag).strip()
        if isinstance(self.tag_names, list):
            return parsed
        return next(iter(parsed.values()), "")

    @property
    def _type(self) -> str:
        return "html_tag_output_parser"


class RetryableBase:
    @staticmethod
    def _retry(operation_name: str) -> Callable:
        return tenacity.retry(
            wait=tenacity.wait_exponential(multiplier=RETRY_MULTIPLIER, max=RETRY_MAX_WAIT),
            stop=tenacity.stop_after_attempt(MAX_RETRIES),
            before_sleep=lambda retry_state: logger.warning(
                "Retrying '%s' (attempt %d failed). Waiting %.1fs",
                operation_name,
                retry_state.attempt_number,
                retry_state.next_action.sleep if retry_state.next_action else 0,
            ),
            reraise=True,
        )


class S3Handler:
    def __init__(self, boto_session: boto3.Session, bucket_name: str):
        if not bucket_name:
            raise ValueError("S3 bucket name is required.")
        self.boto_session = boto_session
        self.bucket_name = bucket_name
        self._s3_client = self.boto_session.client("s3")

    def download_file(self, s3_key: str, local_path: Path):
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            self._s3_client.download_file(self.bucket_name, s3_key, str(local_path))
            logger.info(
                "Successfully downloaded 's3://%s/%s' to '%s'",
                self.bucket_name,
                s3_key,
                local_path,
            )
        except ClientError as e:
            raise S3OperationError(f"Failed to download '{s3_key}'") from e

    async def download_file_async(self, s3_key: str, local_path: Path):
        try:
            await asyncio.to_thread(self.download_file, s3_key, local_path)
        except S3OperationError as e:
            logger.error("Async download failed: %s", e)
            raise

    def exists(self, s3_key: str) -> bool:
        try:
            self._s3_client.head_object(Bucket=self.bucket_name, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise S3OperationError(f"Failed to check existence of '{s3_key}'") from e

    def upload_directory(
        self,
        local_dir: Path,
        s3_prefix: str,
        file_extensions: list[str] | None = None,
        public_readable: bool = False,
    ) -> int:
        if not local_dir.is_dir():
            raise NotADirectoryError(f"Local path is not a directory: {local_dir}")

        config = TransferConfig(use_threads=True, max_concurrency=10)
        extra_args = {"ACL": "public-read"} if public_readable else {}
        upload_count = 0

        files_to_upload = [
            p for p in local_dir.rglob("*") if p.is_file() and (not file_extensions or p.suffix in file_extensions)
        ]

        for local_path in files_to_upload:
            relative_path = local_path.relative_to(local_dir)
            s3_key = (Path(s3_prefix) / relative_path).as_posix()
            try:
                self._s3_client.upload_file(
                    str(local_path),
                    self.bucket_name,
                    s3_key,
                    Config=config,
                    ExtraArgs=extra_args,
                )
                upload_count += 1
            except ClientError as e:
                logger.error(
                    "Failed to upload '%s' to 's3://%s/%s': %s",
                    local_path,
                    self.bucket_name,
                    s3_key,
                    e,
                )
        logger.info(
            "Completed directory upload. Uploaded %d of %d files.",
            upload_count,
            len(files_to_upload),
        )
        return upload_count

    def upload_file(self, local_path: Path, s3_prefix: str | None = None):
        if not local_path.exists():
            raise FileNotFoundError(f"Local file not found: {local_path}")

        s3_key = self._construct_s3_key(local_path.name, s3_prefix)
        try:
            self._s3_client.upload_file(str(local_path), self.bucket_name, s3_key)
            logger.info(
                "Successfully uploaded '%s' to 's3://%s/%s'",
                local_path.name,
                self.bucket_name,
                s3_key,
            )
        except ClientError as e:
            raise S3OperationError(f"Failed to upload '{local_path}'") from e

    @staticmethod
    def _construct_s3_key(file_name: str, s3_prefix: str | None = None) -> str:
        if not s3_prefix:
            return file_name
        return f"{s3_prefix.strip('/')}/{file_name}"

    async def upload_file_async(self, local_path: Path, s3_prefix: str | None = None):
        try:
            await asyncio.to_thread(self.upload_file, local_path, s3_prefix)
        except (S3OperationError, FileNotFoundError) as e:
            logger.error("Async upload failed: %s", e)
            raise


def extract_video_id(url: str) -> str:
    patterns = [
        r"(?:v=|\/v\/|youtu\.be\/|embed\/|\/v=|^)([a-zA-Z0-9_-]{11})",
    ]
    for pattern in patterns:
        if match := re.search(pattern, url):
            return match.group(1)
    raise ValueError(f"Invalid or unsupported YouTube URL: '{url}'")


def get_account_id(boto_session: boto3.Session) -> str | None:
    sts_client = boto_session.client("sts")
    account_id = sts_client.get_caller_identity().get("Account")
    if account_id is None:
        return None
    return str(account_id)


def get_ssm_param_value(boto_session: boto3.Session, param_name: str) -> str:
    try:
        ssm_client = boto_session.client("ssm")
        response = ssm_client.get_parameter(Name=param_name, WithDecryption=True)
        value = response["Parameter"]["Value"]
        if not isinstance(value, str):
            raise RuntimeError(f"SSM parameter '{param_name}' value is not a string")
        return value
    except ClientError as e:
        raise RuntimeError(f"Failed to get SSM parameter '{param_name}'") from e


def sanitize_name(name: str, max_length: int = 50) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_-]", "-", name)
    return sanitized[:max_length]


def truncate_text_by_tokens(text: str, max_tokens: int = MAX_TOKENS) -> str:
    encoding = get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    if len(tokens) <= max_tokens:
        return text

    truncated_tokens = tokens[:max_tokens]
    truncated = encoding.decode(truncated_tokens)
    logger.warning(
        "Text truncated from %d to %d tokens",
        len(tokens),
        len(truncated_tokens),
    )
    return truncated


def validate_path(path: str) -> str | None:
    stripped_path = path.strip() if path else ""
    if not stripped_path:
        return None

    try:
        http_url_adapter = TypeAdapter(HttpUrl)
        http_url_adapter.validate_python(stripped_path)
        return stripped_path
    except Exception:
        pass

    try:
        Path(stripped_path)
        return stripped_path
    except Exception:
        pass

    return None
