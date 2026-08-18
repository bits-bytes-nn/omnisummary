from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

from shared import (
    LOGGING_TRUNCATION_CHARS,
    BedrockLanguageModelFactory,
    VisualBrief,
    VisualSynopsisPrompt,
    logger,
    resolve_secret,
)
from shared.config import LanguageModelId


def _usage_summary(usage: Any) -> str:
    """Compact token counts from an images.generate response, or "unknown".

    The image is billed on output image tokens, so this is the only number that turns spend from an
    estimate (published per-image price x a log count) into a measurement. Tolerant of the field
    being absent or renamed: a usage-shape change in the SDK must never break a render."""
    if usage is None:
        return "unknown"
    parts = []
    for field in ("input_tokens", "output_tokens", "total_tokens"):
        value = getattr(usage, field, None)
        if value is not None:
            parts.append(f"{field.split('_')[0]}={value}")
    return " ".join(parts) or "unknown"


class VisualGenerator:
    """Free-form synopsis -> image. The caller (agent) supplies a natural-language
    instruction describing the desired visual (a 1-page presentation slide, an N-panel
    comic, a concept diagram, an infographic, ...) plus the source material and any
    research context it gathered. No fixed modes or panel counts.

    brief (Claude via Bedrock) -> single image prompt -> gpt-image (OpenAI) -> PNG bytes.
    """

    # Every knob is a REQUIRED keyword: duplicating the defaults here let them drift from
    # PipelineConfig (the copy of `style_aesthetic` had already decayed to "clean modern style"
    # while config shipped the real art-direction paragraph), and the drifted copy is what any
    # partial-kwargs caller silently got. PipelineConfig is now the single source of truth.
    def __init__(
        self,
        llm_factory: BedrockLanguageModelFactory,
        brief_model: LanguageModelId,
        *,
        image_model: str,
        image_sizes: dict[str, str],
        source_max_tokens: int,
        context_max_tokens: int,
        caption_language: str,
        on_image_language: str,
        moderation_softening_instruction: str,
        style_guidance: str,
        humor_guidance: str,
        style_aesthetic: str,
        image_timeout_sec: int,
        image_max_retries: int,
        image_quality: str = "",
    ) -> None:
        self.llm_factory = llm_factory
        self.brief_model = brief_model
        # Bind the VisualBrief schema as a tool so Bedrock returns a validated object instead
        # of free text: the brief's `prompt` is up to 4000 chars of free-form copy that often
        # contains unescaped quotes/newlines, which broke hand-parsing the model's JSON.
        self.llm = llm_factory.get_model(brief_model, stage="visual-synopsis").with_structured_output(VisualBrief)
        self.image_model = image_model
        # orientation -> gpt-image size; the brief picks the orientation that fits the visual.
        self.image_sizes = image_sizes
        self.image_timeout_sec = image_timeout_sec
        self.image_quality = image_quality
        self.image_max_retries = image_max_retries
        self.source_max_tokens = source_max_tokens
        self.context_max_tokens = context_max_tokens
        self.caption_language = caption_language
        self.on_image_language = on_image_language
        self.moderation_softening_instruction = moderation_softening_instruction
        self.style_guidance = style_guidance
        self.humor_guidance = humor_guidance
        self.style_aesthetic = style_aesthetic

    async def brief(self, instruction: str, source: str, context: str = "") -> VisualBrief:
        chain = VisualSynopsisPrompt.get_prompt() | self.llm
        result = await chain.ainvoke(
            {
                "instruction": instruction,
                "source": self.llm_factory.truncate_to_tokens(source, self.source_max_tokens),
                "context": self.llm_factory.truncate_to_tokens(context, self.context_max_tokens),
                "orientations": ", ".join(self.image_sizes),
                "caption_language": self.caption_language,
                "on_image_language": self.on_image_language,
                "style_guidance": self.style_guidance,
                "humor_guidance": self.humor_guidance,
                "style_aesthetic": self.style_aesthetic,
            }
        )
        # with_structured_output is typed as returning dict | BaseModel; re-validate so the
        # type is concretely VisualBrief (and a dict-shaped return is coerced consistently).
        brief = VisualBrief.model_validate(result)
        logger.info("Generated visual brief '%s'", brief.title[: LOGGING_TRUNCATION_CHARS["brief_title"]])
        return brief

    def render(self, brief: VisualBrief) -> bytes:
        api_key = resolve_secret("OPENAI_API_KEY", "openai-api-key")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured — visualization disabled")
        from openai import OpenAI

        if not brief.prompt:
            raise ValueError("Visual brief has no image prompt")
        size = self.image_sizes.get(brief.orientation) or next(iter(self.image_sizes.values()))
        # Bound the render: the SDK defaults (10-min timeout, 2 retries) can outlive the visual
        # Lambda's 15-min budget, which shows up as a timeout with no image instead of a clean
        # failure. Both bounds are config-driven (PipelineConfig.visual_image_*).
        client = OpenAI(api_key=api_key, timeout=self.image_timeout_sec, max_retries=self.image_max_retries)
        # `quality` is only sent when configured: omitting it leaves OpenAI's "auto", which picks
        # between quality tiers whose per-image prices differ ~4x, so the bill is unpredictable and
        # the code cannot say which tier it bought. Pin it to make cost deterministic.
        params: dict[str, Any] = {"model": self.image_model, "prompt": brief.prompt, "size": size}
        if self.image_quality:
            params["quality"] = self.image_quality
        response = client.images.generate(**params)
        b64 = response.data[0].b64_json if response.data else None
        if not b64:
            raise RuntimeError("gpt-image returned no image data")
        # The response carries the token counts the image is billed on, and they were being
        # discarded — leaving spend as an estimate from published per-image prices multiplied by a
        # log count. Logged so the real per-render usage is in CloudWatch.
        logger.info(
            "Rendered visual image (%s, %s, quality=%s, tokens=%s)",
            brief.orientation,
            size,
            self.image_quality or "auto",
            _usage_summary(getattr(response, "usage", None)),
        )
        return base64.b64decode(b64)

    @staticmethod
    def _is_moderation_error(exc: Exception) -> bool:
        # Prefer the typed OpenAI exception / structured error code, which survives API
        # version changes; fall back to substring matching only as a documented last resort.
        try:
            from openai import BadRequestError

            if isinstance(exc, BadRequestError):
                body = getattr(exc, "body", None)
                code = body.get("code") if isinstance(body, dict) else None
                error_type = body.get("type") if isinstance(body, dict) else None
                if code == "moderation_blocked" or error_type == "image_generation_user_error":
                    return True
        except ImportError:
            # openai SDK lacks BadRequestError (older/partial install): fall through to the
            # documented last-resort substring detection below.
            pass
        msg = str(exc).lower()
        return "moderation_blocked" in msg or "safety system" in msg

    async def generate(
        self, instruction: str, source: str, context: str = "", *, deadline: float | None = None
    ) -> tuple[bytes, VisualBrief]:
        """`deadline` is an optional monotonic timestamp bounding the caller (the visual Lambda's
        remaining time). It only gates the SECOND, moderation-softened render — a whole extra
        image_timeout_sec — so a retry can't push the run past its caller's timeout and lose the
        text digest too. None (local runs, the research agent) behaves exactly as before."""
        brief = await self.brief(instruction, source, context)
        try:
            # render() makes a blocking 30-120s OpenAI HTTP call; run it off the event loop so
            # concurrent coroutines (Slack/Threads I/O) aren't frozen for the whole gpt-image render.
            return await asyncio.to_thread(self.render, brief), brief
        except Exception as e:
            if not self._is_moderation_error(e):
                raise
            if deadline is not None and deadline - time.monotonic() < self.image_timeout_sec:
                logger.warning("Image moderation blocked the prompt, but there is no time left to re-render")
                raise
            # gpt-image moderation is intermittent and sensitive to real-person likenesses /
            # edgy parody. Regenerate the brief once with a softened, safe-for-work instruction
            # rather than losing the visual entirely.
            logger.warning("Image moderation blocked the prompt; retrying with a softened brief")
            safe_instruction = f"{instruction}\n\n{self.moderation_softening_instruction}"
            brief = await self.brief(safe_instruction, source, context)
            return await asyncio.to_thread(self.render, brief), brief
