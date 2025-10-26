import asyncio
import hashlib
import json
import os
import re
from pathlib import Path
from pprint import pformat
from typing import Any

import fitz
from bs4 import BeautifulSoup
from PIL import Image
from unstructured.documents.elements import Image as UnstructuredImage
from unstructured.partition.pdf import partition_pdf

from shared import (
    AppConstants,
    Content,
    ContentType,
    EnvVars,
    Figure,
    LocalPaths,
    ParseResult,
    PdfParserType,
    is_running_in_aws,
    logger,
)

from .base_parser import RichParser

MIN_FIGURE_AREA: float = 10000.0
ROOT_DIR: Path = Path("/tmp") if is_running_in_aws() else Path(__file__).parent.parent.parent


class PdfParser(RichParser):
    async def parse(
        self,
        url: str,
        parser_type: PdfParserType = PdfParserType.UNSTRUCTURED,
    ) -> ParseResult:
        logger.info("Parsing PDF from '%s' using '%s' parser", url, parser_type.value)

        url_hash = hashlib.md5(url.encode()).hexdigest()
        docs_dir = ROOT_DIR / LocalPaths.DOCS_DIR.value / url_hash
        docs_dir.mkdir(parents=True, exist_ok=True)

        pdf_path = await self._download_pdf(url, docs_dir)

        if parser_type == PdfParserType.UPSTAGE:
            text, raw_figures = await self._run_upstage_pipeline(pdf_path)
        else:
            text, raw_figures = self._run_unstructured_pipeline(pdf_path)

        figure_analysis_task = self._analyze_figures_concurrently(raw_figures)
        content_task = Content.from_llm(
            metadata_extractor=self.metadata_extractor,
            text=text,
            source_url=url,
            content_type=ContentType.PDF,
        )

        results = await asyncio.gather(figure_analysis_task, content_task, return_exceptions=True)

        figures = results[0] if isinstance(results[0], list) else []
        if isinstance(results[0], Exception):
            logger.warning("Figure analysis failed: '%s'", results[0])

        if isinstance(results[1], Exception):
            logger.error("Content extraction failed: '%s'", results[1])
            raise results[1]

        if not isinstance(results[1], Content):
            raise RuntimeError(f"Unexpected type for content: '{type(results[1])}'")

        content: Content = results[1]

        content.text = self._enrich_content_with_figures(text, figures)
        logger.info("Successfully parsed and enriched content from '%s'", url)
        logger.info("Extracted %d figures and %d characters.", len(figures), len(text))

        parsed_result = ParseResult(content=content, figures=figures)
        logger.debug("Parsed result: %s", pformat(parsed_result.model_dump()))

        return parsed_result

    async def _download_pdf(self, url: str, temp_dir: Path) -> Path:
        temp_dir.mkdir(parents=True, exist_ok=True)
        pdf_filename = url.split("/")[-1]
        if not pdf_filename.endswith(".pdf"):
            pdf_filename = "document.pdf"
        pdf_path = temp_dir / pdf_filename

        response = await self.async_client.get(url)
        response.raise_for_status()
        pdf_path.write_bytes(response.content)

        logger.info("Downloaded PDF to '%s'", pdf_path)
        return pdf_path

    async def _run_upstage_pipeline(self, pdf_path: Path) -> tuple[str, list[Figure]]:
        api_key = os.environ.get(EnvVars.UPSTAGE_API_KEY.value)
        if not api_key:
            raise ValueError(EnvVars.UPSTAGE_API_KEY.value + " environment variable is required")

        parsed_path = pdf_path.parent / LocalPaths.PARSED_FILE.value
        if parsed_path.exists():
            logger.info("Loading cached Upstage response from '%s'", parsed_path)
            try:
                with open(parsed_path, encoding="utf-8") as f:
                    elements = json.load(f).get("elements", [])
            except (OSError, json.JSONDecodeError) as e:
                logger.warning("Failed to load cached response: %s. Re-parsing.", e)
                elements = await self._request_upstage_parse(pdf_path, api_key)
                self._cache_upstage_response(parsed_path, elements)
        else:
            elements = await self._request_upstage_parse(pdf_path, api_key)
            self._cache_upstage_response(parsed_path, elements)

        figures = self._extract_figures_from_upstage_elements(elements, pdf_path)
        raw_text = self._extract_text_from_upstage_elements(elements, figures)

        return raw_text, figures

    async def _request_upstage_parse(self, pdf_path: Path, api_key: str) -> list[dict[Any, Any]]:
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {"document": (pdf_path.name, pdf_path.read_bytes(), "application/pdf")}

        response = await self.async_client.post(
            AppConstants.External.UPSTAGE_DOCUMENT_PARSE.value,
            headers=headers,
            files=files,
        )
        response.raise_for_status()
        elements = response.json().get("elements", [])
        return elements if isinstance(elements, list) else []

    @staticmethod
    def _cache_upstage_response(path: Path, elements: list[dict]) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"elements": elements}, f, indent=2, ensure_ascii=False)
            logger.info("Cached Upstage response to '%s'", path)
        except OSError as e:
            logger.warning("Failed to cache Upstage response: %s", e)

    def _run_unstructured_pipeline(self, pdf_path: Path) -> tuple[str, list[Figure]]:
        figures_dir = pdf_path.parent / LocalPaths.FIGURES_DIR.value
        figures_dir.mkdir(exist_ok=True)
        logger.info("Created figures directory at '%s' for Unstructured parser", figures_dir)

        elements = partition_pdf(
            filename=str(pdf_path),
            extract_images_in_pdf=True,
            extract_image_block_output_dir=str(figures_dir),
            strategy="hi_res",
            infer_table_structure=True,
        )

        text_parts, figures = [], []
        for element in elements:
            if isinstance(element, UnstructuredImage):
                img_path_str = getattr(element.metadata, "image_path", None)
                if not img_path_str:
                    continue

                img_path = Path(img_path_str)
                if not img_path.exists():
                    logger.warning("Image path does not exist: '%s'", img_path)
                    continue

                if self._is_image_large_enough(img_path):
                    figures.append(Figure(figure_id=img_path.stem, path=str(img_path)))
                    text_parts.append(f"[Image: alt=, src={img_path.name}]")
                    logger.debug("Added figure '%s' from Unstructured", img_path.name)
            elif hasattr(element, "text") and element.text:
                text_parts.append(element.text)

        raw_text = "\n\n".join(text_parts)
        logger.info("Unstructured parser extracted %d figures", len(figures))
        return raw_text, figures

    @staticmethod
    def _extract_text_from_upstage_elements(elements: list[dict], figures: list[Figure]) -> str:
        figure_paths = {fig.figure_id for fig in figures}
        text_parts = []

        for el in elements:
            category = el.get("category", "").lower()
            element_id = str(el.get("id", ""))

            if category in ("chart", "figure") and element_id in figure_paths:
                img_path = next((f.path for f in figures if f.figure_id == element_id), None)
                if img_path:
                    text_parts.append(f"[Image: alt=, src={Path(img_path).name}]")
            elif content := el.get("content", {}):
                html = content.get("html", "")
                if html:
                    soup = BeautifulSoup(html, "html.parser")
                    text = soup.get_text(separator="\n", strip=True)
                    if text:
                        text_parts.append(text)

        return "\n\n".join(text_parts)

    def _extract_figures_from_upstage_elements(
        self, elements: list[dict], pdf_path: Path
    ) -> list[Figure]:
        figures: list[Figure] = []
        figures_dir = pdf_path.parent / LocalPaths.FIGURES_DIR.value
        figures_dir.mkdir(exist_ok=True)
        logger.info("Created figures directory at '%s'", figures_dir)

        doc = fitz.open(pdf_path)
        try:
            for el in elements:
                if el.get("category", "").lower() not in ("chart", "figure"):
                    continue

                coords = el.get("coordinates")
                if not coords or len(coords) < 4:
                    continue

                page_num = el.get("page", 1)
                figure_id = str(el.get("id", f"fig_{len(figures)}"))

                try:
                    img_path = self._crop_figure_from_pdf(
                        doc, page_num, coords, figures_dir, figure_id
                    )

                    if not img_path.exists():
                        logger.warning("Cropped figure does not exist: '%s'", img_path)
                        continue

                    soup = BeautifulSoup(el.get("html", ""), "html.parser")
                    img_tag = soup.find("img")
                    caption = img_tag.get("alt") if img_tag else None
                    caption_str = str(caption) if caption and isinstance(caption, str) else None
                    figures.append(
                        Figure(figure_id=figure_id, path=str(img_path), caption=caption_str)
                    )
                    logger.info("Saved figure '%s' to '%s'", figure_id, img_path)
                except Exception as e:
                    logger.warning("Failed to extract figure '%s': %s", figure_id, e)
        finally:
            doc.close()

        logger.info("Extracted %d figures from PDF via Upstage", len(figures))
        return figures

    @staticmethod
    def _crop_figure_from_pdf(
        doc: fitz.Document, page_num: int, coords: list[dict], figure_dir: Path, fig_id: str
    ) -> Path:
        page = doc[page_num - 1]

        rect_coords = [
            coords[0]["x"] * page.rect.width,
            coords[0]["y"] * page.rect.height,
            coords[2]["x"] * page.rect.width,
            coords[2]["y"] * page.rect.height,
        ]

        clip_rect = fitz.Rect(*rect_coords)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), clip=clip_rect, dpi=300)
        figure_path = figure_dir / f"{fig_id}.png"
        pix.save(figure_path)
        return figure_path

    @staticmethod
    def _is_image_large_enough(img_path: Path) -> bool:
        try:
            with Image.open(img_path) as img:
                width, height = img.size
                return bool(width * height >= MIN_FIGURE_AREA)
        except Exception as e:
            logger.warning("Could not check image size for %s: %s", img_path.name, e)
            return False

    async def _analyze_figures_concurrently(self, figures: list[Figure]) -> list[Figure]:
        tasks = [
            Figure.from_llm(
                figure_analyser=self.figure_analyser,
                figure_id=fig.figure_id,
                path=str(fig.path),
                caption=fig.caption,
            )
            for fig in figures
            if fig.path is not None
        ]

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_figures = []
        for result in results:
            if isinstance(result, Exception):
                logger.warning("Figure analysis failed: %s", result)
            elif isinstance(result, Figure):
                successful_figures.append(result)

        return successful_figures

    @staticmethod
    def _enrich_content_with_figures(text: str, figures: list[Figure]) -> str:
        pattern = r"\[Image:\s*alt=(.*?),\s*src=(.*?)\]"
        if not figures:
            return re.sub(pattern, "", text)

        figure_map = {Path(fig.path).name: fig for fig in figures if fig.path is not None}

        def repl(match: re.Match) -> str:
            alt, src_name = match.group(1).strip(), match.group(2).strip()
            matched_fig = figure_map.get(src_name)
            if matched_fig and matched_fig.path is not None:
                if not Path(matched_fig.path).exists():
                    logger.warning(
                        "Figure path does not exist during enrichment: '%s'", matched_fig.path
                    )
                    return ""

                if matched_fig.analysis:
                    alt_text = (alt or matched_fig.caption or "").replace('"', '\\"')
                    analysis = matched_fig.analysis.replace('"', '\\"')
                    return (
                        f'[Image: alt="{alt_text}", src="{matched_fig.path}", caption="{analysis}"]'
                    )
            return ""

        return re.sub(pattern, repl, text)
