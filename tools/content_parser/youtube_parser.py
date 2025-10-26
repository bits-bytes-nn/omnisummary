import asyncio
from typing import Any

import yt_dlp
from youtube_transcript_api import (
    NoTranscriptFound,
    TranscriptsDisabled,
    YouTubeTranscriptApi,
)

from shared import (
    Content,
    ContentParseError,
    ContentType,
    ParseResult,
    extract_video_id,
    logger,
)

from .base_parser import RichParser


class YouTubeParser(RichParser):
    async def parse(self, url: str, languages: list[str] | None = None) -> ParseResult:
        logger.info("Starting YouTube parsing for URL: '%s'", url)
        video_id = extract_video_id(url)
        effective_languages = languages or ["en", "ko"]

        (
            (raw_title, raw_authors, raw_published_date, metadata),
            transcript_text,
        ) = await asyncio.gather(
            asyncio.to_thread(self._get_metadata, url, video_id),
            asyncio.to_thread(self._get_transcript, video_id, effective_languages),
        )

        content = await Content.from_llm(
            metadata_extractor=self.metadata_extractor,
            text=transcript_text,
            source_url=url,
            content_type=ContentType.YOUTUBE,
            raw_title=raw_title,
            raw_authors=raw_authors,
            raw_published_date=raw_published_date,
        )

        metadata["raw_title"] = raw_title
        metadata["raw_authors"] = raw_authors
        metadata["raw_published_date"] = raw_published_date
        content.metadata = metadata

        logger.info("Successfully parsed and enriched content from '%s'", url)
        logger.info("Extracted %d characters.", len(transcript_text))

        return ParseResult(content=content, figures=[])

    @staticmethod
    def _get_metadata(url: str, video_id: str) -> tuple[str, str, str, dict[str, Any]]:
        title = f"YouTube Video '{video_id}'"
        authors = "Unknown Author"
        published_date = ""
        metadata = {}

        try:
            ydl_opts = {
                "quiet": True,
                "no_warnings": True,
                "extract_flat": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                if info:
                    title = info.get("title", title)
                    authors = info.get("uploader", authors)
                    published_date = info.get("upload_date", published_date)
                    metadata = {
                        "duration": info.get("duration"),
                        "like_count": info.get("like_count"),
                        "view_count": info.get("view_count"),
                    }
                    logger.info("Successfully fetched metadata for video: '%s'", title)
        except Exception as e:
            logger.warning(
                "Failed to fetch metadata with yt-dlp for video ID '%s', using fallback: %s",
                video_id,
                e,
            )

        return title, authors, published_date, metadata

    @staticmethod
    def _get_transcript(video_id: str, languages: list[str]) -> str:
        try:
            ytt_api = YouTubeTranscriptApi()
            transcript_list = ytt_api.list(video_id)
            transcript = transcript_list.find_transcript(languages)
            transcript_data = transcript.fetch()
            text = " ".join(snippet.text for snippet in transcript_data)
            logger.info("Successfully fetched transcript for video ID: '%s'", video_id)
            return text
        except (NoTranscriptFound, TranscriptsDisabled) as e:
            msg = f"No transcript available for video ID '{video_id}' with languages {languages}"
            logger.error("%s: %s", msg, e)
            raise ContentParseError(msg) from e
        except Exception as e:
            msg = f"Failed to fetch transcript for YouTube video '{video_id}'"
            logger.error("%s: %s", msg, e, exc_info=True)
            raise ContentParseError(msg) from e
