from typing import Any

import boto3
import httpx
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable

from shared import BedrockLanguageModelFactory, Config, HTMLTagOutputParser, LanguageModelId
from shared.prompts import FigureAnalysisPrompt, MetadataExtractionPrompt

DEFAULT_TIMEOUT: int = 60
DEFAULT_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
    "Accept-Language": "ko-KR,ko;q=0.9,en-US;q=0.8,en;q=0.7",
    "Referer": "https://www.google.com/",
}


class BaseParser:
    def __init__(self, timeout: int = DEFAULT_TIMEOUT):
        self.timeout = timeout
        self._async_client: httpx.AsyncClient | None = None

    @property
    def async_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
                follow_redirects=True,
            )
        return self._async_client

    async def __aenter__(self) -> "BaseParser":
        return self

    async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        if self._async_client and not self._async_client.is_closed:
            await self._async_client.aclose()


class RichParser(BaseParser):
    def __init__(
        self,
        config: Config,
        boto_session: boto3.Session | None = None,
        timeout: int = DEFAULT_TIMEOUT,
    ) -> None:
        super().__init__(timeout=timeout)
        self.config = config
        self.boto_session = boto_session or boto3.Session(
            region_name=config.resources.bedrock_region_name,
            profile_name=config.resources.profile_name,
        )

        self.figure_analyser, self.metadata_extractor = self._initialize_chain(
            self.config.content_parser.figure_analysis_model_id,
            self.config.content_parser.metadata_extraction_model_id,
            self.boto_session,
        )

    @staticmethod
    def _initialize_chain(
        figure_analysis_model_id: LanguageModelId,
        metadata_extraction_model_id: LanguageModelId,
        boto_session: boto3.Session,
    ) -> tuple[Runnable, Runnable]:
        llm_factory = BedrockLanguageModelFactory(boto_session=boto_session)
        figure_analyser_llm = llm_factory.get_model(figure_analysis_model_id, temperature=0.0)
        metadata_extractor_llm = llm_factory.get_model(
            metadata_extraction_model_id, temperature=0.0
        )

        figure_analyser = (
            FigureAnalysisPrompt.get_prompt() | figure_analyser_llm | StrOutputParser()
        )
        metadata_extractor = (
            MetadataExtractionPrompt.get_prompt()
            | metadata_extractor_llm
            | HTMLTagOutputParser(tag_names=MetadataExtractionPrompt.output_variables)
        )

        return figure_analyser, metadata_extractor

    async def parse(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("RichParser is an abstract class")
