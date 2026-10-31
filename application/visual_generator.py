from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence

from application.ports import ImageRenderer, LanguageModels
from application.prompts import VisualSynopsisPrompt
from domain.models import VisualBrief
from shared import logger
from shared.config import LanguageModelId

# Prepended to the image prompt when the narrator's reference sheet rides along as an input image.
REFERENCE_NOTE = (
    "The attached image is the character reference sheet for the recurring narrator. Wherever he "
    "appears, draw him exactly as the sheet shows him (face, hair, glasses, two-tone cardigan), in this "
    "comic's style. Do not reproduce the sheet itself.\n\n"
)


class VisualGenerator:
    """Free-form synopsis -> image. The caller supplies a natural-language instruction
    describing the desired visual (normally a four-panel comic) plus the source material.

    brief (the art director model) -> single image prompt -> the image renderer -> PNG bytes.
    """

    # Every knob is a REQUIRED keyword: duplicating the defaults here let them drift from
    # PipelineConfig (the copy of `style_aesthetic` had already decayed to "clean modern style"
    # while config shipped the real art-direction paragraph), and the drifted copy is what any
    # partial-kwargs caller silently got. PipelineConfig is now the single source of truth.
    def __init__(
        self,
        models: LanguageModels,
        brief_model: LanguageModelId,
        renderer: ImageRenderer,
        *,
        image_sizes: dict[str, str],
        source_max_tokens: int,
        on_image_language: str,
        moderation_softening_instruction: str,
        style_guidance: str,
        style_aesthetic: str,
        image_timeout_sec: int,
    ) -> None:
        self.models = models
        self.renderer = renderer
        self.brief_model = brief_model
        # Bind the VisualBrief schema as a tool so Bedrock returns a validated object instead
        # of free text: the brief's `prompt` is thousands of chars of free-form copy that often
        # contains unescaped quotes/newlines, which broke hand-parsing the model's JSON.
        self.llm = models.get_model(brief_model, stage="visual-synopsis").with_structured_output(VisualBrief)
        # orientation -> gpt-image size; the brief picks the orientation that fits the visual.
        self.image_sizes = image_sizes
        self.image_timeout_sec = image_timeout_sec
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
                "source": self.models.truncate_to_tokens(source, self.source_max_tokens),
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

    def render(self, brief: VisualBrief, references: Sequence[bytes] = ()) -> bytes:
        if not brief.prompt:
            raise ValueError("Visual brief has no image prompt")
        prompt = REFERENCE_NOTE + brief.prompt if references else brief.prompt
        return self.renderer.render(prompt, self._resolve_size(brief), references, f"Rendered {brief.orientation}")

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
            if not self.renderer.is_moderation_error(e):
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
