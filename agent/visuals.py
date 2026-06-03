from __future__ import annotations

import base64
import json

from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from shared import (
    BedrockLanguageModelFactory,
    VisualBrief,
    VisualSynopsisPrompt,
    extract_json_from_llm_output,
    logger,
    resolve_secret,
    truncate_text_by_tokens,
)
from shared.config import LanguageModelId


def _parse_brief(raw: str) -> VisualBrief:
    try:
        data = json.loads(extract_json_from_llm_output(raw))
    except json.JSONDecodeError as e:
        raise ValueError(f"No valid JSON object in visual brief output: {e}") from e
    try:
        return VisualBrief.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Visual brief is missing required fields: {e}") from e


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
        image_model: str = "gpt-image-1",
        image_size: str = "1024x1024",
        source_max_tokens: int = 2000,
        context_max_tokens: int = 1500,
    ) -> None:
        self.llm = llm_factory.get_model(brief_model)
        self.image_model = image_model
        self.image_size = image_size
        self.source_max_tokens = source_max_tokens
        self.context_max_tokens = context_max_tokens

    async def brief(self, instruction: str, source: str, context: str = "") -> VisualBrief:
        chain = VisualSynopsisPrompt.get_prompt() | self.llm | StrOutputParser()
        raw = await chain.ainvoke(
            {
                "instruction": instruction,
                "source": truncate_text_by_tokens(source, self.source_max_tokens),
                "context": truncate_text_by_tokens(context, self.context_max_tokens),
                "image_size": self.image_size,
            }
        )
        brief = _parse_brief(raw)
        logger.info("Generated visual brief '%s'", brief.title[:60])
        return brief

    def render(self, brief: VisualBrief) -> bytes:
        api_key = resolve_secret("OPENAI_API_KEY", "openai-api-key")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured — visualization disabled")
        from openai import OpenAI

        if not brief.prompt:
            raise ValueError("Visual brief has no image prompt")
        client = OpenAI(api_key=api_key)
        response = client.images.generate(model=self.image_model, prompt=brief.prompt, size=self.image_size)
        b64 = response.data[0].b64_json if response.data else None
        if not b64:
            raise RuntimeError("gpt-image returned no image data")
        logger.info("Rendered visual image")
        return base64.b64decode(b64)

    async def generate(self, instruction: str, source: str, context: str = "") -> tuple[bytes, VisualBrief]:
        brief = await self.brief(instruction, source, context)
        return self.render(brief), brief
