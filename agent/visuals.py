from __future__ import annotations

import base64

from shared import (
    LOGGING_TRUNCATION_CHARS,
    BedrockLanguageModelFactory,
    VisualBrief,
    VisualSynopsisPrompt,
    logger,
    resolve_secret,
)
from shared.config import LanguageModelId


class VisualGenerator:
    """Free-form synopsis -> image. The caller (agent) supplies a natural-language
    instruction describing the desired visual (a 1-page presentation slide, an N-panel
    comic, a concept diagram, an infographic, ...) plus the source material and any
    research context it gathered. No fixed modes or panel counts.

    brief (Claude via Bedrock) -> single image prompt -> gpt-image (OpenAI) -> PNG bytes.
    """

    def __init__(
        self,
        llm_factory: BedrockLanguageModelFactory,
        brief_model: LanguageModelId,
        *,
        image_model: str = "gpt-image-2",
        image_sizes: dict[str, str] | None = None,
        source_max_tokens: int = 2000,
        context_max_tokens: int = 1500,
        caption_language: str = "Korean",
        on_image_language: str = "SHORT ENGLISH (the image model garbles Korean and other non-Latin glyphs)",
        moderation_softening_instruction: str = (
            "IMPORTANT: keep it clearly safe-for-work and good-natured. "
            "Use brand mascots/logos and generic stylized characters rather than realistic "
            "depictions of real named individuals; avoid anything that could read as defamatory."
        ),
        style_guidance: str = (
            "Multi-panel: same characters and a single consistent, polished art style across panels; "
            "each panel follows from the previous so the sequence reads in order without explanation."
        ),
        humor_guidance: str = (
            "For comics/cartoons, aim for genuinely funny and shareable — internet-humor sensibility, "
            "a clear setup-and-payoff, expressive characters — in a clean, modern, appealing illustration style."
        ),
        style_aesthetic: str = "clean modern style",
    ) -> None:
        self.llm_factory = llm_factory
        self.brief_model = brief_model
        # Bind the VisualBrief schema as a tool so Bedrock returns a validated object instead
        # of free text: the brief's `prompt` is up to 4000 chars of free-form copy that often
        # contains unescaped quotes/newlines, which broke hand-parsing the model's JSON.
        self.llm = llm_factory.get_model(brief_model).with_structured_output(VisualBrief)
        self.image_model = image_model
        # orientation -> gpt-image size; the brief picks the orientation that fits the visual.
        self.image_sizes = image_sizes or {
            "square": "1024x1024",
            "landscape": "1536x1024",
            "portrait": "1024x1536",
        }
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
        brief = await chain.ainvoke(
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
        client = OpenAI(api_key=api_key)
        response = client.images.generate(model=self.image_model, prompt=brief.prompt, size=size)
        b64 = response.data[0].b64_json if response.data else None
        if not b64:
            raise RuntimeError("gpt-image returned no image data")
        logger.info("Rendered visual image (%s, %s)", brief.orientation, size)
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

    async def generate(self, instruction: str, source: str, context: str = "") -> tuple[bytes, VisualBrief]:
        brief = await self.brief(instruction, source, context)
        try:
            return self.render(brief), brief
        except Exception as e:
            if not self._is_moderation_error(e):
                raise
            # gpt-image moderation is intermittent and sensitive to real-person likenesses /
            # edgy parody. Regenerate the brief once with a softened, safe-for-work instruction
            # rather than losing the visual entirely.
            logger.warning("Image moderation blocked the prompt; retrying with a softened brief")
            safe_instruction = f"{instruction}\n\n{self.moderation_softening_instruction}"
            brief = await self.brief(safe_instruction, source, context)
            return self.render(brief), brief
