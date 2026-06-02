from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser

from shared import (
    BedrockLanguageModelFactory,
    ComicSynopsisPrompt,
    VisualizationBriefPrompt,
    logger,
    resolve_secret,
)
from shared.config import LanguageModelId
from shared.prompts.prompts import BasePrompt

IMAGE_MODEL = "gpt-image-1"
IMAGE_SIZE = "1024x1024"
MIN_PANELS = 1
MAX_PANELS = 6

_PANEL_LAYOUTS = {
    1: "a single panel",
    2: "two panels side by side",
    3: "three panels in a row (read left-to-right)",
    4: "a 2x2 grid of four panels (read left-to-right, top-to-bottom)",
    5: "five panels in a clear reading order",
    6: "a 2x3 grid of six panels (read left-to-right, top-to-bottom)",
}


def _parse_json_object(raw: str) -> dict:
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("No JSON object found in model output")
    return json.loads(match.group(0))


@dataclass(frozen=True)
class VisualMode:
    """A synopsis→image mode: how to brief the LLM and how to render the brief."""

    name: str
    prompt: type[BasePrompt]

    def brief_inputs(self, title: str, content: str, panels: int) -> dict[str, str]:
        inputs = {"title": title, "content": content[:6000]}
        if "panels" in self.prompt.input_variables:
            inputs["panels"] = str(panels)
        return inputs

    def build_image_prompt(self, brief: dict, panels: int) -> str:
        if self.name == "comic":
            return _build_comic_prompt(brief, panels)
        return _build_diagram_prompt(brief)


def _build_comic_prompt(brief: dict, panels: int) -> str:
    style = brief.get("style", "hand-drawn webcomic, warm flat colors, clean linework")
    layout = _PANEL_LAYOUTS.get(panels, f"{panels} panels in a clear reading order")
    lines = [
        f"A {panels}-panel comic illustration as {layout}.",
        f"Visual style: {style}. Friendly, clever, easy to understand at a glance.",
        "Keep any text minimal and legible.",
        "",
        "Panels:",
    ]
    for i, panel in enumerate(brief.get("panels", []), start=1):
        lines.append(f"{i}. {panel.get('visual', '')} (caption idea: {panel.get('caption', '')})")
    return "\n".join(lines)


def _build_diagram_prompt(brief: dict) -> str:
    return (
        "A single explanatory infographic/diagram, clean modern style, minimal clutter, "
        "clear labels and arrows.\n\n" + brief.get("visual", "")
    )


COMIC_MODE = VisualMode(name="comic", prompt=ComicSynopsisPrompt)
DIAGRAM_MODE = VisualMode(name="diagram", prompt=VisualizationBriefPrompt)
MODES = {"comic": COMIC_MODE, "diagram": DIAGRAM_MODE}


class VisualGenerator:
    """Synopsis -> visualization. Comic (N panels) and diagram are two modes.

    brief (Claude via Bedrock) -> image prompt -> gpt-image (OpenAI) -> PNG bytes.
    """

    def __init__(self, llm_factory: BedrockLanguageModelFactory, brief_model: LanguageModelId) -> None:
        self.llm = llm_factory.get_model(brief_model)

    async def brief(self, mode: VisualMode, title: str, content: str, panels: int) -> dict:
        chain = mode.prompt.get_prompt() | self.llm | StrOutputParser()
        raw = await chain.ainvoke(mode.brief_inputs(title, content, panels))
        brief = _parse_json_object(raw)
        logger.info("Generated %s brief '%s'", mode.name, brief.get("title", title))
        return brief

    @staticmethod
    def render(mode: VisualMode, brief: dict, panels: int) -> bytes:
        api_key = resolve_secret("OPENAI_API_KEY", "openai-api-key")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not configured — visualization disabled")
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        prompt = mode.build_image_prompt(brief, panels)
        response = client.images.generate(model=IMAGE_MODEL, prompt=prompt, size=IMAGE_SIZE)
        b64 = response.data[0].b64_json if response.data else None
        if not b64:
            raise RuntimeError("gpt-image returned no image data")
        logger.info("Rendered %s image", mode.name)
        return base64.b64decode(b64)

    async def generate(self, title: str, content: str, *, mode: str = "comic", panels: int = 4) -> tuple[bytes, dict]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {sorted(MODES)}, got '{mode}'")
        visual_mode = MODES[mode]
        if visual_mode.name == "comic" and not MIN_PANELS <= panels <= MAX_PANELS:
            raise ValueError(f"panels must be between {MIN_PANELS} and {MAX_PANELS}, got {panels}")
        brief = await self.brief(visual_mode, title, content, panels)
        image_bytes = self.render(visual_mode, brief, panels)
        return image_bytes, brief
