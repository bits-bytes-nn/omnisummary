from __future__ import annotations

import base64
import json
import re

from langchain_core.output_parsers import StrOutputParser

from shared import BedrockLanguageModelFactory, VisualSynopsisPrompt, logger, resolve_secret
from shared.config import LanguageModelId

IMAGE_MODEL = "gpt-image-1"
IMAGE_SIZE = "1024x1024"


def _parse_json_object(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


class VisualGenerator:
    """Free-form synopsis -> image. The caller (agent) supplies a natural-language
    instruction describing the desired visual (a 1-page presentation slide, an N-panel
    comic, a concept diagram, an infographic, ...) plus the source material and any
    research context it gathered. No fixed modes or panel counts.

    brief (Claude via Bedrock) -> single image prompt -> gpt-image (OpenAI) -> PNG bytes.
    """

    def __init__(self, llm_factory: BedrockLanguageModelFactory, brief_model: LanguageModelId) -> None:
        self.llm = llm_factory.get_model(brief_model)

    async def brief(self, instruction: str, source: str, context: str = "") -> dict:
        chain = VisualSynopsisPrompt.get_prompt() | self.llm | StrOutputParser()
        raw = await chain.ainvoke({"instruction": instruction, "source": source[:8000], "context": context[:6000]})
        brief = _parse_json_object(raw)
        logger.info("Generated visual brief '%s'", brief.get("title", "")[:60])
        return brief

    @staticmethod
    def render(brief: dict) -> bytes:
        api_key = resolve_secret("OPENAI_API_KEY", "openai-api-key")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured — visualization disabled")
        from openai import OpenAI

        prompt = brief.get("prompt", "")
        if not prompt:
            raise ValueError("Visual brief has no image prompt")
        client = OpenAI(api_key=api_key)
        response = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size=IMAGE_SIZE)
        b64 = response.data[0].b64_json if response.data else None
        if not b64:
            raise RuntimeError("gpt-image returned no image data")
        logger.info("Rendered visual image")
        return base64.b64decode(b64)

    async def generate(self, instruction: str, source: str, context: str = "") -> tuple[bytes, dict]:
        brief = await self.brief(instruction, source, context)
        return self.render(brief), brief
