import asyncio
import re
from pprint import pformat
from urllib.parse import urljoin

from bs4 import BeautifulSoup, Tag
from bs4.element import NavigableString

from shared import Content, ContentType, Figure, ParseResult, logger

from .base_parser import RichParser


class HTMLParserConfig:
    INCLUDE_META_IMAGES_IN_TEXT: bool = True
    CONTENT_SELECTORS: list[str] = [
        ".content",
        ".entry-content",
        ".post-content",
        "article",
        "body",
        "div[itemprop='articleBody']",
        "main",
    ]
    IMAGE_META_SELECTORS: list[str] = [
        "meta[property='og:image']",
        "meta[name='twitter:image']",
    ]
    TAGS_TO_DECOMPOSE: list[str] = [
        "aside",
        "footer",
        "form",
        "header",
        "nav",
        "script",
        "style",
    ]
    TITLE_SELECTORS: list[str] = [
        "h1",
        "meta[name='twitter:title']",
        "meta[property='og:title']",
        "title",
    ]


class HTMLParser(RichParser):
    async def parse(self, url: str) -> ParseResult:
        logger.info("Starting HTML parsing for URL: '%s'", url)
        html_content = await self._fetch_html(url)
        soup = BeautifulSoup(html_content, "html.parser")

        raw_title, text = self._extract_title_and_text_with_placeholders(soup, url)

        figure_task = self._extract_and_analyze_figures(soup, url)
        content_task = Content.from_llm(
            metadata_extractor=self.metadata_extractor,
            text=text,
            source_url=url,
            content_type=ContentType.HTML,
            raw_title=raw_title,
        )

        results = await asyncio.gather(figure_task, content_task, return_exceptions=True)

        figures = results[0] if isinstance(results[0], list) else []
        if isinstance(results[0], Exception):
            logger.warning("Figure extraction failed: %s", results[0])

        if isinstance(results[1], Exception):
            logger.error("Content extraction failed: %s", results[1])
            raise results[1]

        if not isinstance(results[1], Content):
            raise RuntimeError(f"Unexpected type for content: {type(results[1])}")

        content: Content = results[1]

        content.text = self._enrich_content_with_figures(text, figures)
        content.metadata = {
            "raw_title": raw_title,
        }

        logger.info("Successfully parsed and enriched content from '%s'", url)
        logger.info("Extracted %d figures and %d characters.", len(figures), len(text))

        parsed_result = ParseResult(content=content, figures=figures)
        logger.debug("Parsed result: %s", pformat(parsed_result.model_dump()))

        return parsed_result

    async def _fetch_html(self, url: str) -> str:
        try:
            response = await self.async_client.get(url)
            response.raise_for_status()
            return response.text
        except Exception as e:
            logger.error("Failed to fetch HTML from '%s': %s", url, e)
            raise

    def _get_cleaned_main_content(self, soup: BeautifulSoup) -> Tag:
        text_soup = BeautifulSoup(str(soup), "html.parser")
        for tag_name in HTMLParserConfig.TAGS_TO_DECOMPOSE:
            for tag in text_soup.find_all(tag_name):
                tag.decompose()

        return next(
            (el for selector in HTMLParserConfig.CONTENT_SELECTORS if (el := text_soup.select_one(selector))),
            text_soup,
        )

    def _extract_meta_images(self, soup: BeautifulSoup, base_url: str, seen_urls: set[str]) -> list[dict[str, str]]:
        meta_images = []
        for selector in HTMLParserConfig.IMAGE_META_SELECTORS:
            for meta_tag in soup.select(selector):
                content = meta_tag.get("content")
                if not content or not isinstance(content, str):
                    continue

                absolute_url = urljoin(base_url, content)
                if not absolute_url.startswith("http") or absolute_url in seen_urls:
                    continue

                seen_urls.add(absolute_url)
                alt = ""
                if "og:image" in selector:
                    alt_tag = soup.select_one("meta[property='og:image:alt']")
                    if alt_tag:
                        alt_content = alt_tag.get("content", "")
                        alt = alt_content if isinstance(alt_content, str) else ""
                elif "twitter:image" in selector:
                    alt_tag = soup.select_one("meta[name='twitter:image:alt']")
                    if alt_tag:
                        alt_content = alt_tag.get("content", "")
                        alt = alt_content if isinstance(alt_content, str) else ""

                meta_images.append({"url": absolute_url, "alt": alt})
        return meta_images

    def _extract_title_and_text_with_placeholders(self, soup: BeautifulSoup, base_url: str) -> tuple[str, str]:
        title = "Untitled"
        for selector in HTMLParserConfig.TITLE_SELECTORS:
            if tag := soup.select_one(selector):
                text = tag.get("content") or tag.get_text()
                if text and isinstance(text, str) and text.strip():
                    title = text.strip()
                    break

        main_content_element = self._get_cleaned_main_content(soup)
        text = self._parse_element_to_text(main_content_element, base_url)

        meta_image_placeholders = []
        if HTMLParserConfig.INCLUDE_META_IMAGES_IN_TEXT:
            meta_images = self._extract_meta_images(soup, base_url, set())
            for img in meta_images:
                meta_image_placeholders.append(f"[Image: alt={img['alt']}, src={img['url']}]")

        if meta_image_placeholders:
            text = " ".join(meta_image_placeholders) + " " + text

        return title, re.sub(r"\s+", " ", text).strip()

    def _parse_element_to_text(self, element: Tag | NavigableString, base_url: str) -> str:
        if isinstance(element, NavigableString):
            return str(element)
        if not isinstance(element, Tag):
            return ""
        if element.name == "img":
            src_attr = element.get("src", "")
            src = urljoin(base_url, src_attr if isinstance(src_attr, str) else "")
            alt = element.get("alt", "")
            return f"[Image: alt={alt}, src={src}]"
        return " ".join(
            self._parse_element_to_text(child, base_url)
            for child in element.children
            if isinstance(child, (Tag, NavigableString))
        )

    async def _extract_and_analyze_figures(self, soup: BeautifulSoup, base_url: str) -> list[Figure]:
        main_content_element = self._get_cleaned_main_content(soup)

        tasks, seen_urls = [], set()
        for idx, img_tag in enumerate(main_content_element.find_all("img")):
            src = img_tag.get("src")
            if not src or not isinstance(src, str) or src.startswith("data:"):
                continue

            absolute_url = urljoin(base_url, src)
            if not absolute_url.startswith("http") or absolute_url in seen_urls:
                continue

            seen_urls.add(absolute_url)
            caption = img_tag.get("alt", "") or img_tag.get("title", "")
            tasks.append(
                Figure.from_llm(
                    figure_analyser=self.figure_analyser,
                    figure_id=f"fig-{idx}",
                    path=absolute_url,
                    caption=caption.strip() if isinstance(caption, str) else None,
                )
            )

        meta_images = self._extract_meta_images(soup, base_url, seen_urls)
        for idx, img in enumerate(meta_images):
            tasks.append(
                Figure.from_llm(
                    figure_analyser=self.figure_analyser,
                    figure_id=f"fig-meta-{idx}",
                    path=img["url"],
                    caption=img["alt"].strip() if img["alt"] else None,
                )
            )

        if not tasks:
            return []

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return [fig for fig in results if isinstance(fig, Figure) and not isinstance(fig, Exception)]

    @staticmethod
    def _enrich_content_with_figures(text: str, figures: list[Figure]) -> str:
        pattern = r"\[Image:\s*alt=(.*?),\s*src=(.*?)\]"
        if not figures:
            return re.sub(pattern, "", text)

        figure_map = {str(fig.path): fig for fig in figures}

        def repl(match: re.Match) -> str:
            alt_value, src_value = match.group(1).strip(), match.group(2).strip()
            matched_figure = figure_map.get(src_value)

            if matched_figure and matched_figure.analysis:
                alt_text = (alt_value or matched_figure.caption or "").replace('"', '\\"')
                analysis_text = matched_figure.analysis.replace('"', '\\"')
                return f'[Image: alt="{alt_text}", src="{matched_figure.path}", caption="{analysis_text}"]'

            logger.warning("No matching figure/analysis for src: '%s'. Removing placeholder.", src_value)
            return ""

        return re.sub(pattern, repl, text)
