from __future__ import annotations

import re

from langchain_core.output_parsers import StrOutputParser

from shared import BedrockLanguageModelFactory, TrendUpdatePrompt, logger
from shared.config import PipelineConfig
from shared.state_store import StateStore

TRENDS_KEY = "trends.md"


class TrendTracker:
    def __init__(
        self,
        config: PipelineConfig,
        llm_factory: BedrockLanguageModelFactory,
        state_store: StateStore,
    ) -> None:
        self.config = config
        self.llm = llm_factory.get_model(config.trend_model)
        self.state_store = state_store
        self._cached_trends: str | None = None

    def get_trends_context(self) -> str:
        if self._cached_trends is not None:
            return self._cached_trends
        if self.state_store.exists(TRENDS_KEY):
            content = self.state_store.read(TRENDS_KEY) or ""
            self._cached_trends = content
            logger.info("Loaded trends context (%d chars)", len(content))
            return content
        logger.info("No existing trends document found, starting fresh")
        self._cached_trends = ""
        return ""

    async def update_trends(self, digest_text: str, today_date: str) -> str:
        current = self._cached_trends if self._cached_trends is not None else self.get_trends_context()
        chain = TrendUpdatePrompt.get_prompt() | self.llm | StrOutputParser()

        updated = await chain.ainvoke({
            "current_trends": current or "(No trends tracked yet. Start fresh.)",
            "todays_digest": digest_text,
            "today_date": today_date,
            "trend_retention_days": str(self.config.trend_retention_days),
        })

        updated = self._strip_code_fences(updated)
        self.state_store.write(TRENDS_KEY, updated)
        self._cached_trends = updated
        logger.info("Updated trends document (%d chars)", len(updated))
        return updated

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
        if text.endswith("```"):
            text = re.sub(r"\n?```$", "", text)
        return text.strip()
