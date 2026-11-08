from __future__ import annotations

import base64
from collections.abc import Sequence
from typing import Any

from openai import BadRequestError

from adapters.aws import SecretUnavailableError, resolve_secret
from adapters.metrics import emit_spend
from shared import logger


def _image_file(index: int, data: bytes) -> tuple[str, bytes, str]:
    """An upload tuple for images.edit, typed by the bytes' own signature (JPEG or PNG)."""
    if data[:2] == b"\xff\xd8":
        return f"input{index}.jpg", data, "image/jpeg"
    return f"input{index}.png", data, "image/png"


# USD per million tokens (text input, image input, image output), from OpenAI's model page, checked
# 2026-11-03. gpt-image bills by token, not per image: the same render costs several times more at
# xhigh than at medium, and an edit pays for its input images too.
IMAGE_PRICES: dict[str, tuple[float, float, float]] = {
    "gpt-image-2.5-sunburst": (5.0, 8.0, 30.0),
    "gpt-image-2.5-flare": (5.0, 8.0, 30.0),
    "gpt-image-2": (5.0, 8.0, 30.0),
}


def image_usage(usage: Any) -> dict[str, int]:
    """The token counts an images response is billed on. Tolerant of fields being absent or renamed: a
    usage-shape change in the SDK must never break a render."""
    if usage is None:
        return {}
    details = getattr(usage, "input_tokens_details", None)
    counts = {
        "input_tokens": getattr(usage, "input_tokens", None),
        "text_input_tokens": getattr(details, "text_tokens", None),
        "image_input_tokens": getattr(details, "image_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
    }
    return {name: int(value) for name, value in counts.items() if isinstance(value, int)}


def image_usd(model: str, counts: dict[str, int]) -> float | None:
    """The render's price, or None for an unpriced model or a response without output tokens. Input
    without a text/image split is priced as text (the cheaper rate; an edit's reference image is the
    only image input)."""
    price = IMAGE_PRICES.get(model)
    if price is None or "output_tokens" not in counts:
        return None
    text_rate, image_rate, output_rate = price
    image_in = counts.get("image_input_tokens", 0)
    text_in = counts.get("text_input_tokens", counts.get("input_tokens", 0) - image_in)
    return (text_in * text_rate + image_in * image_rate + counts["output_tokens"] * output_rate) / 1_000_000


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
        counts = image_usage(getattr(response, "usage", None))
        usd = image_usd(self.image_model, counts)
        logger.info(
            "Image usage stage=%s model=%s size=%s quality=%s->%s inputs=%d %s usd=%s",
            label,
            self.image_model,
            size,
            self.image_quality or "auto",
            getattr(response, "quality", None) or "unreported",
            len(inputs),
            " ".join(f"{k}={v}" for k, v in counts.items()) or "tokens=unknown",
            f"{usd:.4f}" if usd is not None else "unpriced",
        )
        emit_spend("image", label, self.image_model, usd, counts)
        return base64.b64decode(b64)

    @staticmethod
    def is_moderation_error(exc: Exception) -> bool:
        """The OpenAI SDK's typed answer for a prompt its moderation blocked."""
        if not isinstance(exc, BadRequestError):
            return False
        body = exc.body if isinstance(exc.body, dict) else {}
        return body.get("code") == "moderation_blocked" or body.get("type") == "image_generation_user_error"
