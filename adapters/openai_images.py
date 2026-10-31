from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

from adapters.aws import SecretUnavailableError, resolve_secret
from shared import logger


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


class OpenAIImageRenderer:
    """gpt-image through the OpenAI Images API: images.generate, or images.edit when reference images
    ride along (the narrator's sheet)."""

    def __init__(self, *, image_model: str, image_quality: str, timeout_sec: int, max_retries: int) -> None:
        self.image_model = image_model
        self.image_quality = image_quality
        self.timeout_sec = timeout_sec
        self.max_retries = max_retries

    def ready(self) -> bool:
        # strict=True so an SSM read FAILURE (throttled, denied, wrong region) is distinguishable
        # from "no key configured": the lenient read returned "" for both, so a broken parameter
        # store silently produced text-only digests that looked like a deliberate config.
        try:
            api_key = resolve_secret("OPENAI_API_KEY", "openai-api-key", strict=True)
        except SecretUnavailableError as e:
            logger.error("Could not read OPENAI_API_KEY (%s) — publishing the digest text-only", e)
            return False
        if not api_key:
            logger.info("OPENAI_API_KEY not set — publishing the digest text-only, without a visual")
            return False
        return True

    def _client(self) -> Any:
        api_key = resolve_secret("OPENAI_API_KEY", "openai-api-key")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured — visualization disabled")
        from openai import OpenAI

        # Bound the call: the SDK defaults (10-min timeout, 2 retries) can outlive the visual
        # Lambda's 15-min budget, which shows up as a timeout with no image instead of a clean failure.
        return OpenAI(api_key=api_key, timeout=self.timeout_sec, max_retries=self.max_retries)

    def render(self, prompt: str, size: str, inputs: Sequence[bytes], label: str) -> bytes:
        """`quality` is only sent when configured: OpenAI's "auto" picks between tiers whose prices
        differ several-fold."""
        params: dict[str, Any] = {"model": self.image_model, "prompt": prompt, "size": size}
        if self.image_quality:
            params["quality"] = self.image_quality
        client = self._client()
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

    @staticmethod
    def is_moderation_error(exc: Exception) -> bool:
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
