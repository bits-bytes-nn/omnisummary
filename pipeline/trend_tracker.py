from __future__ import annotations

import re
from datetime import date, timedelta

from langchain_core.output_parsers import StrOutputParser

from shared import BedrockLanguageModelFactory, TrendUpdatePrompt, logger
from shared.config import PipelineConfig
from shared.state_store import StateStore

TRENDS_KEY = "trends.md"
MAX_EVIDENCE_PER_TREND = 5
MAX_TRENDS_CHARS = 15000


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
        trimmed = self._trim_for_llm(current, today_date)
        logger.info("Trimmed trends for LLM: %d → %d chars", len(current), len(trimmed))

        chain = TrendUpdatePrompt.get_prompt() | self.llm | StrOutputParser()

        updated = await chain.ainvoke({
            "current_trends": trimmed or "(No trends tracked yet. Start fresh.)",
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
    def _trim_for_llm(content: str, today_date: str) -> str:
        if not content or len(content) <= MAX_TRENDS_CHARS:
            return content

        archived_marker = "# Archived Trends"
        archived_idx = content.find(archived_marker)
        if archived_idx > 0:
            content = content[:archived_idx].rstrip()

        try:
            cutoff = date.fromisoformat(today_date) - timedelta(days=7)
        except ValueError:
            cutoff = None

        if cutoff:
            content = TrendTracker._trim_evidence(content, cutoff)

        if len(content) > MAX_TRENDS_CHARS:
            content = content[:MAX_TRENDS_CHARS] + "\n\n(... truncated for size ...)"

        return content

    @staticmethod
    def _trim_evidence(content: str, cutoff: date) -> str:
        lines = content.split("\n")
        result: list[str] = []
        evidence_count = 0
        in_evidence = False

        for line in lines:
            stripped = line.strip()
            if stripped == "- **Evidence**:":
                in_evidence = True
                evidence_count = 0
                result.append(line)
                continue

            if in_evidence:
                if stripped.startswith("- [") or stripped.startswith("- **"):
                    if stripped.startswith("- ["):
                        date_match = re.match(r"- \[(\d{4}-\d{2}-\d{2})\]", stripped)
                        if date_match:
                            try:
                                entry_date = date.fromisoformat(date_match.group(1))
                                if entry_date < cutoff:
                                    continue
                            except ValueError:
                                pass
                        evidence_count += 1
                        if evidence_count > MAX_EVIDENCE_PER_TREND:
                            continue
                        result.append(line)
                        continue

                    in_evidence = False
                    result.append(line)
                    continue

                if not stripped:
                    in_evidence = False
                    result.append(line)
                    continue

                result.append(line)
                continue

            result.append(line)

        return "\n".join(result)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
        if text.endswith("```"):
            text = re.sub(r"\n?```$", "", text)
        return text.strip()
