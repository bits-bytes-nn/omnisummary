from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import httpx
from langchain_core.runnables import Runnable
from PIL import Image
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

from .constants import ContentType
from .logger import logger
from .utils import RetryableBase, truncate_text_by_tokens, validate_path

DEFAULT_RESIZE_QUALITY: int = 85
DEFAULT_TIMEOUT: int = 60
MAX_IMAGE_SIZE_BYTES: int = 5242880

type ImageData = str


class ParserError(Exception):
    pass


class ContentParseError(ParserError):
    pass


class FigureParseError(ParserError):
    pass


class Figure(BaseModel, RetryableBase):
    figure_id: str
    path: str | Path | None = None
    caption: str | None = None
    analysis: str | None = None

    @classmethod
    @RetryableBase._retry("figure_analysing")
    async def from_llm(
        cls,
        figure_analyser: Runnable,
        figure_id: str,
        path: str,
        caption: str | None = None,
    ) -> Figure:
        analysis = None
        try:
            image_data = await cls._get_image_data(path)
            analysis = await figure_analyser.ainvoke({"caption": caption, "image_data": image_data})
        except FigureParseError as e:
            logger.warning("Failed to get image data for figure '%s': %s", figure_id, e)
        except Exception as e:
            raise RuntimeError(f"LLM failed to analyze figure '{figure_id}': {e}") from e

        return cls(figure_id=figure_id, path=path, caption=caption, analysis=analysis)

    @staticmethod
    async def _get_image_data(path: str) -> str:
        image_bytes: bytes

        if path.startswith(("http://", "https://")):
            try:
                async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
                    response = await client.get(path)
                    response.raise_for_status()
                    image_bytes = response.content
            except httpx.HTTPError as e:
                raise FigureParseError(f"HTTP error for image '{path}': {e}") from e
        else:
            try:
                with open(path, "rb") as f:
                    image_bytes = f.read()
            except OSError as e:
                raise FigureParseError(f"Failed to read file '{path}': {e}") from e

        processed_bytes = Figure._resize_if_needed(image_bytes)
        return base64.b64encode(processed_bytes).decode("utf-8")

    @staticmethod
    def _resize_if_needed(image_bytes: bytes, max_iterations: int = 20) -> bytes:
        if len(image_bytes) <= MAX_IMAGE_SIZE_BYTES:
            return image_bytes

        logger.info(
            "Image size (%.2f MB) exceeds limit. Resizing...",
            len(image_bytes) / (1024 * 1024),
        )
        try:
            img: Image.Image = Image.open(io.BytesIO(image_bytes))

            if img.mode in ("RGBA", "P", "CMYK"):
                img = img.convert("RGB")

            quality = DEFAULT_RESIZE_QUALITY
            iteration = 0

            while iteration < max_iterations:
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=quality)

                if len(buffer.getvalue()) <= MAX_IMAGE_SIZE_BYTES:
                    resized_bytes = buffer.getvalue()
                    logger.info(
                        "Image successfully resized to %.2f MB after %d iterations.",
                        len(resized_bytes) / (1024 * 1024),
                        iteration + 1,
                    )
                    return resized_bytes

                if quality <= 10:
                    new_width = int(img.width * 0.8)
                    new_height = int(img.height * 0.8)
                    if new_width < 100 or new_height < 100:
                        raise FigureParseError(
                            "Cannot resize image below 5MB limit even at minimum size"
                        )
                    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                    quality = DEFAULT_RESIZE_QUALITY
                else:
                    quality -= 10

                iteration += 1

            raise FigureParseError(
                f"Failed to resize image below 5MB limit after {max_iterations} iterations"
            )

        except FigureParseError:
            raise
        except Exception as e:
            logger.error(
                "Failed to resize image due to unexpected error: %s",
                e,
                exc_info=True,
            )
            raise FigureParseError(f"Image resizing failed: {e}") from e


class Content(BaseModel, RetryableBase):
    model_config = ConfigDict(frozen=False)

    text: str = Field(default="")
    title: str
    authors: list[str] | None = None
    affiliations: list[str] | None = None
    published_date: str | None = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    categories: list[str] | None = None
    keywords: list[str] = Field(default_factory=list)
    source_url: str
    content_type: ContentType
    metadata: dict[str, Any] | None = None

    @classmethod
    @RetryableBase._retry("content_parsing")
    async def from_llm(
        cls,
        metadata_extractor: Runnable,
        text: str,
        source_url: str,
        content_type: ContentType,
        raw_title: str | None = None,
        raw_authors: str | None = None,
        raw_published_date: str | None = None,
    ) -> Content:
        truncated_text = truncate_text_by_tokens(text)

        metadata = await metadata_extractor.ainvoke(
            {
                "text": truncated_text,
                "source_url": source_url,
                "raw_title": raw_title or "",
                "raw_authors": raw_authors or "",
                "raw_published_date": raw_published_date or "",
            }
        )

        list_fields = ["authors", "affiliations", "categories", "keywords"]
        for field in list_fields:
            if field in metadata and isinstance(metadata[field], str):
                metadata[field] = [
                    item.strip() for item in metadata[field].split(",") if item.strip()
                ]

        none_fields = ["authors", "affiliations", "published_date"]
        for field in none_fields:
            if metadata.get(field) in (["None"], "None"):
                metadata[field] = None

        return cls(
            text=truncated_text,
            source_url=source_url,
            content_type=content_type,
            **metadata,
        )

    @model_validator(mode="before")
    @classmethod
    def validate_source(cls, data: dict[str, Any]) -> dict[str, Any]:
        source_url = data.get("source_url")
        if not source_url:
            raise ValueError("'source_url' is required.")

        validated_path = validate_path(source_url)
        if not validated_path:
            raise ValueError(f"Invalid source URL: '{source_url}'")

        data["source_url"] = validated_path
        return data


class ParseResult(BaseModel):
    content: Content
    figures: list[Figure] = Field(default_factory=list)


class SummaryResult(BaseModel):
    summary: str
    thumbnails: list[str | Path] | None = None

    @field_validator("thumbnails", mode="before")
    @classmethod
    def validate_thumbnails(cls, paths: list[str | Path]) -> list[str | Path]:
        return [validated for path in paths if (validated := validate_path(str(path))) is not None]


class SlackAppMentionEvent(BaseModel):
    type: str
    user: str | None = None
    text: str
    channel: str
    ts: str | None = None
    event_ts: str | None = None


class SlackEventCallback(BaseModel):
    type: str
    event_id: str | None = None
    event: SlackAppMentionEvent | None = None
    challenge: str | None = None

    @property
    def is_url_verification(self) -> bool:
        return self.type == "url_verification"

    @property
    def is_app_mention(self) -> bool:
        return self.event is not None and self.event.type == "app_mention"
