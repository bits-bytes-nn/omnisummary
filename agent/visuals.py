from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Sequence
from typing import Any

from shared import (
    BedrockLanguageModelFactory,
    VisualBrief,
    VisualSynopsisPrompt,
    logger,
    resolve_secret,
)
from shared.config import LanguageModelId

# Prepended to the image prompt when the narrator's reference sheet rides along as an input image.
REFERENCE_NOTE = (
    "The attached image is the character reference sheet for the recurring narrator. Wherever he "
    "appears, draw him exactly as the sheet shows him (face, hair, glasses, two-tone cardigan), in this "
    "comic's style. Do not reproduce the sheet itself.\n\n"
)


def _image_file(index: int, data: bytes) -> tuple[str, bytes, str]:
    """An upload tuple for images.edit, typed by the bytes' own signature (JPEG or PNG)."""
    if data[:2] == b"\xff\xd8":
        return f"input{index}.jpg", data, "image/jpeg"
    return f"input{index}.png", data, "image/png"


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
    """Free-form synopsis -> image. The caller supplies a natural-language instruction
    describing the desired visual (normally a four-panel comic) plus the source material.

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
        on_image_language: str,
        moderation_softening_instruction: str,
        style_guidance: str,
        style_aesthetic: str,
        image_timeout_sec: int,
        image_max_retries: int,
        image_quality: str = "",
    ) -> None:
        self.llm_factory = llm_factory
        self.brief_model = brief_model
        # Bind the VisualBrief schema as a tool so Bedrock returns a validated object instead
        # of free text: the brief's `prompt` is thousands of chars of free-form copy that often
        # contains unescaped quotes/newlines, which broke hand-parsing the model's JSON.
        self.llm = llm_factory.get_model(brief_model, stage="visual-synopsis").with_structured_output(VisualBrief)
        self.image_model = image_model
        # orientation -> gpt-image size; the brief picks the orientation that fits the visual.
        self.image_sizes = image_sizes
        self.image_timeout_sec = image_timeout_sec
        self.image_quality = image_quality
        self.image_max_retries = image_max_retries
        self.source_max_tokens = source_max_tokens
        self.on_image_language = on_image_language
        self.moderation_softening_instruction = moderation_softening_instruction
        self.style_guidance = style_guidance
        self.style_aesthetic = style_aesthetic

    async def brief(self, instruction: str, source: str) -> VisualBrief:
        chain = VisualSynopsisPrompt.get_prompt() | self.llm
        result = await chain.ainvoke(
            {
                "instruction": instruction,
                "source": self.llm_factory.truncate_to_tokens(source, self.source_max_tokens),
                "orientations": ", ".join(self.image_sizes),
                "on_image_language": self.on_image_language,
                "style_guidance": self.style_guidance,
                "style_aesthetic": self.style_aesthetic,
            }
        )
        # Models that reject forced tool_choice (Opus 5.5 / Sonnet 5.5) get the schema
        # tool under "auto", so a reply without the tool call parses to None; name that failure.
        if result is None:
            raise ValueError(f"Visual brief model returned no {VisualBrief.__name__} tool call")
        # with_structured_output is typed as returning dict | BaseModel; re-validate so the
        # type is concretely VisualBrief (and a dict-shaped return is coerced consistently).
        brief = VisualBrief.model_validate(result)
        logger.info("Generated visual brief '%s'", brief.title[:60])
        return brief

    def _resolve_size(self, brief: VisualBrief) -> str:
        """The gpt-image size for the brief's orientation. A miss cannot happen through the shipped
        path (the orientation is a Literal and config validates image_sizes' keys), so it raises."""
        size = self.image_sizes.get(brief.orientation, "")
        if not size:
            raise RuntimeError(
                f"No image size configured for orientation '{brief.orientation}' "
                f"(configured: {', '.join(self.image_sizes) or 'none'}); cannot render a visual"
            )
        return size

    def _images_client(self) -> Any:
        api_key = resolve_secret("OPENAI_API_KEY", "openai-api-key")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured — visualization disabled")
        from openai import OpenAI

        # Bound the call: the SDK defaults (10-min timeout, 2 retries) can outlive the visual
        # Lambda's 15-min budget, which shows up as a timeout with no image instead of a clean failure.
        return OpenAI(api_key=api_key, timeout=self.image_timeout_sec, max_retries=self.image_max_retries)

    def _call_images(self, prompt: str, size: str, inputs: Sequence[bytes], label: str) -> bytes:
        """images.generate, or images.edit when input images ride along (the narrator's reference
        sheet). `quality` is only sent when configured: OpenAI's "auto"
        picks between tiers whose prices differ several-fold."""
        params: dict[str, Any] = {"model": self.image_model, "prompt": prompt, "size": size}
        if self.image_quality:
            params["quality"] = self.image_quality
        client = self._images_client()
        if inputs:
            files = [_image_file(i, data) for i, data in enumerate(inputs)]
            response = client.images.edit(image=files, **params)
        else:
            response = client.images.generate(**params)
        b64 = response.data[0].b64_json if response.data else None
        if not b64:
            raise RuntimeError("gpt-image returned no image data")
        # The token counts the image is billed on, and the quality the response reports (not just
        # the one requested), so spend is a measurement rather than an estimate.
        logger.info(
            "%s visual image (%s, quality=%s->%s, inputs=%d, tokens=%s)",
            label,
            size,
            self.image_quality or "auto",
            getattr(response, "quality", None) or "unreported",
            len(inputs),
            _usage_summary(getattr(response, "usage", None)),
        )
        return base64.b64decode(b64)

    def render(self, brief: VisualBrief, references: Sequence[bytes] = ()) -> bytes:
        if not brief.prompt:
            raise ValueError("Visual brief has no image prompt")
        prompt = REFERENCE_NOTE + brief.prompt if references else brief.prompt
        return self._call_images(prompt, self._resolve_size(brief), references, f"Rendered {brief.orientation}")

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
        self, instruction: str, source: str, *, references: Sequence[bytes] = (), deadline: float | None = None
    ) -> tuple[bytes, VisualBrief]:
        """`references` are input images every render carries (the narrator's reference sheet).
        `deadline` is an optional monotonic timestamp bounding the caller (the visual Lambda's
        remaining time); it gates the moderation-softened re-render (a whole extra image_timeout_sec).
        None (local runs) is unbounded."""
        brief = await self.brief(instruction, source)
        try:
            # The OpenAI call blocks for 30-120s; run it off the event loop so concurrent coroutines
            # (Threads I/O) aren't frozen for the whole render.
            image = await asyncio.to_thread(self.render, brief, references)
        except Exception as e:
            if not self._is_moderation_error(e):
                raise
            if deadline is not None and deadline - time.monotonic() < self.image_timeout_sec:
                logger.warning("Image moderation blocked the prompt, but there is no time left to re-render")
                raise
            # gpt-image moderation is intermittent and sensitive to real-person likenesses / edgy
            # parody. Regenerate the brief once with a softened instruction rather than lose the visual.
            logger.warning("Image moderation blocked the prompt; retrying with a softened brief")
            brief = await self.brief(f"{instruction}\n\n{self.moderation_softening_instruction}", source)
            image = await asyncio.to_thread(self.render, brief, references)
        return image, brief
