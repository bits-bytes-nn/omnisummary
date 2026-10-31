"""Amazon Bedrock as the LanguageModels port: the model registry the factory acts on, cross-region
and application inference-profile resolution, and CountTokens."""

import hashlib
import os
import re
from typing import Any, ClassVar, Literal

import boto3
from botocore.config import Config as BotoConfig
from langchain_aws import ChatBedrockConverse
from langchain_core.callbacks import BaseCallbackHandler
from pydantic import BaseModel

from shared.constants import LanguageModelId
from shared.logger import logger

# Model used for the Bedrock CountTokens API. Only some base models expose CountTokens
# (Sonnet does; Opus 4.8 does not), but the Claude family shares a tokenizer, so counting
# with a supported model is accurate for all of them and avoids per-model failures. GPT models expose
# no CountTokens either; for them the count is a close estimate, which is all a truncation budget needs.
TOKEN_COUNT_MODEL = LanguageModelId.CLAUDE_V4_6_SONNET


class LanguageModelInfo(BaseModel):
    """What the factory acts on, hand-verified against Converse (re-establishing it costs real
    invocations, so it is written down here):

    - Every registered model rejects a non-default `temperature` (ValidationException), except
      Sonnet 4.6, which is only used for CountTokens.
    - Opus 5.5 and Sonnet 5.5 reject tool_choice "any"/"tool"; Haiku 5.5 and the older models accept
      it. langchain-aws infers ("auto", "any", "tool") for any id containing "claude" and () for an
      application-inference-profile ARN, so the factory states the flag instead of inheriting either.
    - Opus 5.5 and Haiku 5.5 default to effort "medium" where every other model defaults to "high";
      `effort` pins it so swapping a stage's model does not silently lower its reasoning depth.
    - Fable 5.1 is not registered: Bedrock rejects it under the account's default data-retention mode
      ("data retention mode 'default' is not available for this model").
    - The GPT-6.1 models run on the same Converse API through their global profiles. They accept a
      forced tool_choice, but langchain-aws infers () for any non-Claude id, and with_structured_output
      then got plain prose back and parsed it as None. They take effort as `reasoning.effort` (they
      reject `output_config`) and default to a lower effort than "high". They reject a Converse
      `cachePoint`; their prompt caching is implicit."""

    provider: Literal["anthropic", "openai"] = "anthropic"
    max_output_tokens: int
    supports_temperature: bool = False
    supports_forced_tool_choice: bool = True
    effort: str | None = None

    def request_fields(self) -> dict[str, Any]:
        """The provider-specific Converse `additionalModelRequestFields` that pin the effort."""
        if not self.effort:
            return {}
        if self.provider == "openai":
            return {"reasoning": {"effort": self.effort}}
        return {"output_config": {"effort": self.effort}}


# 64000, not the 128K ceiling: every pipeline call is a non-streaming ainvoke, and a higher cap would
# only raise the HTTP-timeout risk with no caller needing that much output.
LANGUAGE_MODEL_INFO: dict[LanguageModelId, LanguageModelInfo] = {
    LanguageModelId.CLAUDE_V4_6_SONNET: LanguageModelInfo(max_output_tokens=64000, supports_temperature=True),
    LanguageModelId.CLAUDE_V5_SONNET: LanguageModelInfo(max_output_tokens=64000),
    LanguageModelId.CLAUDE_V4_8_OPUS: LanguageModelInfo(max_output_tokens=64000),
    LanguageModelId.CLAUDE_V5_OPUS: LanguageModelInfo(max_output_tokens=64000),
    LanguageModelId.CLAUDE_V5_5_HAIKU: LanguageModelInfo(max_output_tokens=64000, effort="high"),
    LanguageModelId.CLAUDE_V5_5_SONNET: LanguageModelInfo(max_output_tokens=64000, supports_forced_tool_choice=False),
    LanguageModelId.CLAUDE_V5_5_OPUS: LanguageModelInfo(
        max_output_tokens=64000, supports_forced_tool_choice=False, effort="high"
    ),
    LanguageModelId.GPT_V6_1_TERRA: LanguageModelInfo(provider="openai", max_output_tokens=64000, effort="high"),
    LanguageModelId.GPT_V6_1_SOL: LanguageModelInfo(provider="openai", max_output_tokens=64000, effort="high"),
}


class _TokenUsageLogger(BaseCallbackHandler):
    """Log the token usage of every LLM call, tagged with the pipeline stage that made it.

    Cost Explorer bills per MODEL, not per stage, so a bill of "Sonnet 5: 4.7M input tokens / 30d"
    could not be attributed: the digest, the grounding pass, trend classification, the visual editor,
    the visual synopsis, query refinement and the ad-hoc research agent all share that one model. Any
    optimisation without this is a guess. Best-effort by construction — a callback must never be able
    to fail a generation, so every read is defensive."""

    def __init__(self, stage: str, model_id: str) -> None:
        self.stage = stage
        self.model_id = model_id

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            usage: dict[str, Any] = {}
            for generations in getattr(response, "generations", []) or []:
                for generation in generations or []:
                    message = getattr(generation, "message", None)
                    meta = getattr(message, "usage_metadata", None)
                    if meta:
                        usage = dict(meta)
            if not usage:
                usage = (getattr(response, "llm_output", None) or {}).get("usage", {}) or {}
            if not usage:
                return
            details = usage.get("input_token_details") or {}
            logger.info(
                "LLM usage stage=%s model=%s input=%s output=%s cache_read=%s cache_write=%s",
                self.stage,
                self.model_id,
                usage.get("input_tokens"),
                usage.get("output_tokens"),
                details.get("cache_read"),
                details.get("cache_creation"),
            )
        except Exception:  # pragma: no cover - telemetry must never break a generation
            logger.debug("Could not read LLM usage metadata", exc_info=True)


class BedrockCrossRegionModelHelper:
    # Resolution is identical for a given (model_id, region) within a process, but each
    # call hits list_inference_profiles. Cache it so ranker/digest/trend/refine model
    # builds don't each pay the round-trip (and don't each risk the AccessDenied path).
    _resolution_cache: ClassVar[dict[tuple[str, str], str]] = {}

    @staticmethod
    def get_cross_region_model_id(
        boto_session: boto3.Session,
        model_id: LanguageModelId,
        region_name: str,
    ) -> str:
        cache_key = (model_id.value, region_name)
        cached = BedrockCrossRegionModelHelper._resolution_cache.get(cache_key)
        if cached is not None:
            return cached
        resolved = BedrockCrossRegionModelHelper._resolve(boto_session, model_id, region_name)
        # Prefer this project's APPLICATION inference profile when one exists: it is the only way
        # on-demand Bedrock token spend gets attributed to a cost-allocation tag (there is no
        # taggable resource behind InvokeModel otherwise), and this account bills several workloads
        # against the same models. Resolution goes through here, so BOTH the LangChain factory and
        # the Strands research agent pick it up. Absent → today's system-defined id, unchanged.
        resolved = BedrockCrossRegionModelHelper._application_profile_arn(boto_session, model_id, region_name, resolved)
        BedrockCrossRegionModelHelper._resolution_cache[cache_key] = resolved
        return resolved

    @staticmethod
    def application_profile_name(model_id: LanguageModelId, project: str, stage: str) -> str:
        """Deterministic name for a project/stage's application inference profile for a model.
        Shared by the resolver (project/stage from the env CDK sets on the Lambdas) and
        scripts/put_inference_profiles.py (from the config it tags with), so neither can drift."""
        slug = re.sub(r"[^A-Za-z0-9]+", "-", model_id.value).strip("-")
        return f"{project}-{stage}-{slug}"

    @staticmethod
    def _application_profile_arn(
        boto_session: boto3.Session,
        model_id: LanguageModelId,
        region_name: str,
        fallback: str,
    ) -> str:
        """The ARN of this project's application inference profile for the model, or `fallback`.

        Best-effort by design: cost attribution must never be able to stop a generation, so a
        missing profile, a denied ListInferenceProfiles or any unexpected error keeps the
        system-defined id the caller would have used anyway."""
        wanted = BedrockCrossRegionModelHelper.application_profile_name(
            model_id, os.getenv("PROJECT_NAME", "omnisummary"), os.getenv("STAGE", "dev")
        )
        try:
            client = boto_session.client("bedrock", region_name=region_name)
            paginator = client.get_paginator("list_inference_profiles")
            for page in paginator.paginate(typeEquals="APPLICATION"):
                for summary in page.get("inferenceProfileSummaries", []):
                    if summary.get("inferenceProfileName") == wanted:
                        arn = summary["inferenceProfileArn"]
                        logger.debug("Using application inference profile '%s' (%s)", wanted, arn)
                        return arn
        except Exception as e:
            logger.debug("Could not look up application inference profile '%s': %s", wanted, e)
            return fallback
        logger.debug("No application inference profile named '%s'; using '%s'", wanted, fallback)
        return fallback

    @staticmethod
    def _resolve(
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


class BedrockLanguageModelFactory:
    DEFAULT_TEMPERATURE: ClassVar[float] = 0.0
    BOTO_READ_TIMEOUT: ClassVar[int] = 300
    BOTO_MAX_ATTEMPTS: ClassVar[int] = 3
    MAX_POOL_CONNECTIONS: ClassVar[int] = 50

    def __init__(
        self,
        boto_session: boto3.Session | None = None,
        region_name: str | None = None,
    ) -> None:
        self.boto_session = boto_session or boto3.Session()
        self.region_name = region_name or self.boto_session.region_name
        boto_config = BotoConfig(
            read_timeout=self.BOTO_READ_TIMEOUT,
            connect_timeout=60,
            retries={"max_attempts": self.BOTO_MAX_ATTEMPTS, "mode": "adaptive"},
            max_pool_connections=self.MAX_POOL_CONNECTIONS,
        )
        self._client = self.boto_session.client("bedrock-runtime", region_name=self.region_name, config=boto_config)
        logger.debug("Initialized %s for region: '%s'", self.__class__.__name__, self.region_name)

    @staticmethod
    def get_model_info(model_id: LanguageModelId) -> LanguageModelInfo:
        return LANGUAGE_MODEL_INFO[model_id]

    def count_tokens(self, text: str) -> int:
        """Authoritative token count via the Bedrock CountTokens API (not a local heuristic).

        Only some base models expose CountTokens (e.g. Sonnet does; Opus 4.8 returns
        'doesn't support counting tokens'). The Claude family shares a tokenizer, so we always
        count with TOKEN_COUNT_MODEL regardless of the caller's model — accurate for all of them
        and avoids per-model 'unsupported' failures. Falls back to a char estimate on error.
        Results are memoized on the factory instance keyed by a text hash: prompt-building counts
        the same item text across ranker/digest/grounding stages, and truncate_to_tokens binary-
        searches many overlapping prefixes — each an identical repeat call otherwise billed anew."""
        # Once CountTokens has failed this run, stop calling it: the estimate is deterministic, so
        # a transient throttle/region error otherwise turns truncate_to_tokens' ~log2(n)-call binary
        # search — repeated per oversized item across ranker/digest/grounding — into a self-inflicted
        # storm of failing round-trips + warnings. Degrade to the char estimate for the rest of the run.
        if self.__dict__.get("_count_tokens_degraded"):
            return len(text) // 4
        cache = self.__dict__.setdefault("_token_count_cache", {})
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cached = cache.get(key)
        if cached is not None:
            return cached
        try:
            resp = self._client.count_tokens(
                modelId=TOKEN_COUNT_MODEL.value,
                input={"converse": {"messages": [{"role": "user", "content": [{"text": text}]}]}},
            )
            result = int(resp["inputTokens"])
        except Exception as e:
            logger.warning("Bedrock count_tokens failed (%s); using char/4 estimate for the rest of the run", e)
            self.__dict__["_count_tokens_degraded"] = True
            return len(text) // 4
        cache[key] = result
        return result

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to <= max_tokens, measured by the Bedrock CountTokens API. Binary-searches
        the character cut point (O(log n) API calls) since CountTokens counts but can't decode token
        boundaries. Cuts on a whitespace boundary near the found point so words aren't split."""
        if not text or self.count_tokens(text) <= max_tokens:
            return text
        lo, hi, best = 0, len(text), 0
        while lo <= hi:
            mid = (lo + hi) // 2
            if self.count_tokens(text[:mid]) <= max_tokens:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        cut = text.rfind(" ", 0, best)
        truncated = text[: cut if cut > best // 2 else best].rstrip()
        logger.warning("Text truncated to <=%d tokens (%d chars)", max_tokens, len(truncated))
        return truncated

    def get_model(self, model_id: LanguageModelId, *, stage: str = "", **kwargs: Any) -> ChatBedrockConverse:
        """Build a chat model. `stage` names the caller (ranking / digest / grounding / ...) purely so
        token usage can be attributed in the logs — the bill is per MODEL, and several stages share
        one model, so without it a token total cannot be traced to the call that spent it."""
        model_info = self.get_model_info(model_id)
        resolved_model_id = BedrockCrossRegionModelHelper.get_cross_region_model_id(
            self.boto_session, model_id, self.region_name or ""
        )
        model_config = self._build_model_config(model_info, resolved_model_id, stage=stage, **kwargs)
        model = ChatBedrockConverse(**model_config)
        logger.debug("Created language model: '%s'", resolved_model_id)
        return model

    def _build_model_config(
        self, model_info: LanguageModelInfo, resolved_model_id: str, **kwargs: Any
    ) -> dict[str, Any]:
        final_max_tokens = self._validate_max_tokens(kwargs.get("max_tokens"), model_info)
        config = self._build_base_config(resolved_model_id, model_info.provider, **kwargs)
        # Telemetry sits with the rest of the config, not bolted on afterwards. The base config
        # takes `callbacks` from kwargs, which is None when the caller passed none, so build the list.
        config["callbacks"] = [
            *(config.get("callbacks") or []),
            _TokenUsageLogger(str(kwargs.get("stage") or "unattributed"), resolved_model_id),
        ]
        config["max_tokens"] = final_max_tokens
        if model_info.supports_temperature:
            config["temperature"] = kwargs.get("temperature", self.DEFAULT_TEMPERATURE)
        # Stated, never inferred: langchain-aws guesses from the id, and its guess is wrong for an
        # opaque application-profile ARN and for every GPT id. "auto" only makes with_structured_output
        # bind the schema tool without forcing it.
        config["supports_tool_choice_values"] = (
            ("auto", "any", "tool") if model_info.supports_forced_tool_choice else ("auto",)
        )
        if model_info.effort:
            config["additional_model_request_fields"] = model_info.request_fields()
        if model_info.provider != "anthropic":
            # Every call here is a non-streaming ainvoke; stated, langchain-aws stops warning on every
            # GPT model build that it has not verified the provider's streaming.
            config["disable_streaming"] = True
        return config

    def _build_base_config(self, resolved_model_id: str, provider: str, **kwargs: Any) -> dict[str, Any]:
        config: dict[str, Any] = {
            "model_id": resolved_model_id,
            "region_name": self.region_name,
            "client": self._client,
            "callbacks": kwargs.get("callbacks"),
        }
        # An application-inference-profile ARN carries no model family in its id, so
        # ChatBedrockConverse refuses it ("Model provider should be supplied when passing a model ARN
        # as model_id"). The registry knows the provider, so name it.
        if resolved_model_id.startswith("arn:"):
            config["provider"] = provider
        if self.boto_session.profile_name and self.boto_session.profile_name != "default":
            config["credentials_profile_name"] = self.boto_session.profile_name
        return config

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
