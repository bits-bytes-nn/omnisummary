import json
import re
from pprint import pformat

import boto3
from langchain_core.output_parsers import StrOutputParser

from shared import (
    BedrockLanguageModelFactory,
    Config,
    ParseResult,
    SummaryResult,
    logger,
)
from shared.prompts import SummarizationPrompt


class Summarizer:
    def __init__(self, config: Config, boto_session: boto3.Session | None = None) -> None:
        self.config = config
        self.boto_session = boto_session or boto3.Session(
            region_name=config.resources.bedrock_region_name,
            profile_name=config.resources.profile_name,
        )

        llm_factory = BedrockLanguageModelFactory(boto_session=self.boto_session)
        self.summarization_llm = llm_factory.get_model(
            config.summarization.summarization_model_id,
            temperature=0.0,
            supports_1m_context_window=True,
        )

    async def generate_summary(
        self,
        parsed_result: ParseResult,
        seed_message: str | None = None,
    ) -> SummaryResult:
        logger.info("Generating summary with %d thumbnails", self.config.summarization.n_thumbnails)

        summarizer = SummarizationPrompt.get_prompt() | self.summarization_llm | StrOutputParser()

        result = await summarizer.ainvoke(
            {
                "text": parsed_result.content.text,
                "seed_message": seed_message or "",
                "n_thumbnails": self.config.summarization.n_thumbnails,
            }
        )

        summary_match = re.search(r"<summary>(.*?)</summary>", result, re.DOTALL)
        thumbnails_match = re.search(r"<thumbnails>(.*?)</thumbnails>", result, re.DOTALL)

        summary_text = (
            (summary_match.group(1).strip() if summary_match else "").replace("다:", "다.").replace("요:", "요.")
        )

        thumbnails_list = []
        if thumbnails_match:
            thumbnails_str = thumbnails_match.group(1).strip()
            try:
                thumbnails_list = json.loads(thumbnails_str)
                if not isinstance(thumbnails_list, list):
                    thumbnails_list = []
            except json.JSONDecodeError:
                thumbnails_list = [
                    item.strip().strip('"').strip("'")
                    for item in thumbnails_str.strip("[]").split(",")
                    if item.strip().strip('"').strip("'")
                ]

        summary_result = SummaryResult(
            summary=summary_text,
            thumbnails=thumbnails_list,
        )

        logger.debug("Summary result: %s", pformat(summary_result.model_dump()))

        return summary_result
