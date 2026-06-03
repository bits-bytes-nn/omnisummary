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
ARCHIVED_MARKER = "# Archived Trends"


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
        trimmed, old_archived = self._trim_for_llm(current, today_date)
        logger.info(
            "Trimmed trends for LLM: %d → %d chars (archived: %d chars)", len(current), len(trimmed), len(old_archived)
        )

        chain = TrendUpdatePrompt.get_prompt() | self.llm | StrOutputParser()

        updated = await chain.ainvoke(
            {
                "current_trends": trimmed or "(No trends tracked yet. Start fresh.)",
                "todays_digest": digest_text,
                "today_date": today_date,
                "trend_retention_days": str(self.config.trend_retention_days),
            }
        )

        updated = self._strip_code_fences(updated)
        updated = self._merge_archived(updated, old_archived)
        self.state_store.write(TRENDS_KEY, updated)
        self._cached_trends = updated
        logger.info("Updated trends document (%d chars)", len(updated))
        return updated

    @staticmethod
    def _trim_for_llm(content: str, today_date: str) -> tuple[str, str]:
        if not content:
            return content, ""

        old_archived = ""
        archived_idx = content.find(ARCHIVED_MARKER)
        if archived_idx > 0:
            old_archived = content[archived_idx:].strip()
            content = content[:archived_idx].rstrip()

        try:
            cutoff = date.fromisoformat(today_date) - timedelta(days=7)
        except ValueError:
            cutoff = None

        if cutoff:
            content = TrendTracker._trim_evidence(content, cutoff)

        if len(content) > MAX_TRENDS_CHARS:
            content = content[:MAX_TRENDS_CHARS] + "\n\n(... truncated for size ...)"

        return content, old_archived

    @staticmethod
    def _trim_evidence(content: str, cutoff: date) -> str:
        lines = content.split("\n")
        result: list[str] = []
        evidence_count = 0
        dropped_count = 0
        in_evidence = False

        for line in lines:
            stripped = line.strip()
            if stripped == "- **Evidence**:":
                if in_evidence and dropped_count > 0:
                    result.append(f"  - (+ {dropped_count} earlier entries omitted)")
                in_evidence = True
                evidence_count = 0
                dropped_count = 0
                result.append(line)
                continue

            if in_evidence:
                if stripped.startswith("- ["):
                    date_match = re.match(r"- \[(\d{4}-\d{2}-\d{2})\]", stripped)
                    if date_match:
                        try:
                            entry_date = date.fromisoformat(date_match.group(1))
                            if entry_date < cutoff:
                                dropped_count += 1
                                continue
                        except ValueError:
                            pass
                    evidence_count += 1
                    if evidence_count > MAX_EVIDENCE_PER_TREND:
                        dropped_count += 1
                        continue
                    result.append(line)
                    continue

                if stripped.startswith("- **") or not stripped:
                    if dropped_count > 0:
                        result.append(f"  - (+ {dropped_count} earlier entries omitted)")
                        dropped_count = 0
                    in_evidence = False
                    result.append(line)
                    continue

                result.append(line)
                continue

            result.append(line)

        if in_evidence and dropped_count > 0:
            result.append(f"  - (+ {dropped_count} earlier entries omitted)")

        return "\n".join(result)

    @staticmethod
    def _merge_archived(updated: str, old_archived: str) -> str:
        if not old_archived:
            return updated

        old_entries = set()
        for line in old_archived.split("\n"):
            stripped = line.strip()
            if stripped.startswith("- "):
                old_entries.add(stripped)

        new_archived_idx = updated.find(ARCHIVED_MARKER)
        if new_archived_idx >= 0:
            new_archived_section = updated[new_archived_idx:]
            active_section = updated[:new_archived_idx].rstrip()

            new_entries = set()
            for line in new_archived_section.split("\n"):
                stripped = line.strip()
                if stripped.startswith("- "):
                    new_entries.add(stripped)

            merged_entries = []
            for line in new_archived_section.split("\n"):
                merged_entries.append(line)

            for entry in sorted(old_entries - new_entries):
                merged_entries.append(entry)

            return active_section + "\n\n" + "\n".join(merged_entries)

        merged_lines = [updated.rstrip(), "", old_archived]
        return "\n".join(merged_lines)

    @staticmethod
    def _strip_code_fences(text: str) -> str:
        text = text.strip()
        if text.startswith("```"):
            text = re.sub(r"^```\w*\n?", "", text)
        if text.endswith("```"):
            text = re.sub(r"\n?```$", "", text)
        return text.strip()
