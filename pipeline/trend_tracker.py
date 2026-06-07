from __future__ import annotations

import asyncio
import json
import re
from datetime import date

from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from shared import (
    BedrockLanguageModelFactory,
    Trend,
    TrendClassifyPrompt,
    TrendEvidence,
    TrendMemory,
    TrendStatus,
    extract_json_from_llm_output,
    logger,
)
from shared.config import PipelineConfig
from shared.state_store import StateStore

TRENDS_KEY = "trends.json"

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


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
        self._memory: TrendMemory | None = None

    def _load_memory(self) -> TrendMemory:
        if self._memory is not None:
            return self._memory

        raw = self.state_store.read(TRENDS_KEY) if self.state_store.exists(TRENDS_KEY) else None
        if raw:
            try:
                self._memory = TrendMemory.model_validate_json(raw)
                logger.info("Loaded %d trends from '%s'", len(self._memory.trends), TRENDS_KEY)
                return self._memory
            except ValidationError as e:
                logger.warning("Failed to parse '%s' (%s); starting fresh", TRENDS_KEY, e)
                self._memory = TrendMemory()
                return self._memory

        logger.info("No existing trends found, starting fresh")
        self._memory = TrendMemory()
        return self._memory

    def get_trends_context(self, today: date | None = None) -> str:
        memory = self._load_memory()
        visible = [t for t in memory.trends if t.status != TrendStatus.ARCHIVED]
        if not visible:
            return ""
        # Use the run's digest date (KST) as the single clock so the recurrence facts agree with
        # the evidence dates (also KST-stamped); date.today() would be UTC and off by a day at the
        # 19:00-KST run boundary.
        today = today or date.today()
        visible.sort(key=lambda t: t.momentum(today, self.config.trend_momentum_half_life_days), reverse=True)
        return self._render_ammunition(visible, today)

    @staticmethod
    def _render_ammunition(trends: list[Trend], today: date) -> str:
        """Render visible trends with the recurrence facts the digest lead uses for sharp,
        grounded criticism: days the trend has been running, how many distinct days it has
        recurred, and its most recent note. Computed in code from evidence — never invented."""
        lines: list[str] = []
        for t in trends:
            dates = sorted({ev.date for ev in t.evidence})
            facts: list[str] = []
            try:
                first = date.fromisoformat(t.first_seen) if t.first_seen else None
            except ValueError:
                first = None
            if first is not None:
                facts.append(f"{(today - first).days + 1}일째 추적 중")
            if len(dates) > 1:
                facts.append(f"최근 {len(dates)}개 다른 날 재등장")
            month_hits = sum(1 for d in dates if d[:7] == today.isoformat()[:7])
            if month_hits > 1:
                facts.append(f"이번 달 {month_hits}회")
            gist = t.evidence[-1].summary if t.evidence else ""
            fact_str = f" [{', '.join(facts)}]" if facts else ""
            lines.append(f"- {t.title}{fact_str}: {gist}" if gist else f"- {t.title}{fact_str}")
        return "\n".join(lines)

    async def update_trends(self, digest_text: str, today_date: str) -> None:
        memory = self._load_memory()

        existing_summary = self._render_existing(memory)
        chain = TrendClassifyPrompt.get_prompt() | self.llm | StrOutputParser()
        raw = await chain.ainvoke(
            {
                "existing_trends": existing_summary or "(No trends tracked yet.)",
                "todays_digest": digest_text,
            }
        )

        # Idempotency: drop any evidence already stamped with today's date so a re-run
        # (Lambda retry, manual re-run) replaces the day's contribution instead of
        # appending reworded duplicates. Dates are code-stamped, never from the LLM.
        for trend in memory.trends:
            trend.evidence = [ev for ev in trend.evidence if ev.date != today_date]

        observations = self._parse_observations(raw)
        logger.info("Trend classifier returned %d observations", len(observations))
        for obs in observations:
            self._apply_observation(memory, obs, today_date)

        # Drop trends left with no evidence — e.g. a re-run wiped today's evidence (above)
        # but the LLM's second pass didn't re-emit that observation. Without this an
        # empty, contentless trend would linger forever in trends.json and the digest.
        memory.trends = [t for t in memory.trends if t.evidence]

        self._run_lifecycle(memory, today_date)

        self._memory = memory
        await asyncio.to_thread(self.state_store.write, TRENDS_KEY, memory.model_dump_json())
        logger.info("Persisted %d trends to '%s'", len(memory.trends), TRENDS_KEY)

    @staticmethod
    def _render_existing(memory: TrendMemory) -> str:
        lines: list[str] = []
        for t in memory.trends:
            if t.status == TrendStatus.ARCHIVED:
                continue
            gist = t.evidence[-1].summary if t.evidence else ""
            lines.append(f"- {t.id}: {t.title}" + (f" — {gist}" if gist else ""))
        return "\n".join(lines)

    @staticmethod
    def _parse_observations(raw: str) -> list[dict[str, str]]:
        try:
            data = json.loads(extract_json_from_llm_output(raw))
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning("Failed to parse trend classifier output: %s", e)
            return []
        items = data.get("observations", []) if isinstance(data, dict) else data
        if not isinstance(items, list):
            return []
        observations: list[dict[str, str]] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            summary = str(item.get("summary", "")).strip()
            if not summary:
                continue
            observations.append(
                {
                    "trend_id": str(item.get("trend_id", "")).strip(),
                    "new_title": str(item.get("new_title", "")).strip(),
                    "summary": summary,
                }
            )
        return observations

    def _apply_observation(self, memory: TrendMemory, obs: dict[str, str], today_date: str) -> None:
        trend = memory.by_id(obs["trend_id"]) if obs["trend_id"] else None
        if trend is None:
            title = obs["new_title"] or obs["summary"]
            trend_id = self._make_slug(title, {t.id for t in memory.trends})
            trend = Trend(id=trend_id, title=title, first_seen=today_date, last_seen=today_date)
            memory.trends.append(trend)
            logger.info("Created new trend '%s' (%s)", trend.title, trend.id)

        if any(ev.date == today_date and ev.summary == obs["summary"] for ev in trend.evidence):
            return

        trend.evidence.append(TrendEvidence(date=today_date, summary=obs["summary"]))
        trend.last_seen = today_date
        if not trend.first_seen:
            trend.first_seen = today_date

    def _run_lifecycle(self, memory: TrendMemory, today_date: str) -> None:
        try:
            today: date | None = date.fromisoformat(today_date)
        except ValueError:
            logger.warning("Invalid today_date '%s'; skipping date-based lifecycle", today_date)
            today = None

        for trend in memory.trends:
            if len(trend.evidence) > self.config.trend_max_evidence:
                trend.evidence = trend.evidence[-self.config.trend_max_evidence :]
            if today is not None:
                trend.status = self._compute_status(trend, today)

        active = [t for t in memory.trends if t.status == TrendStatus.ACTIVE]
        if len(active) > self.config.trend_max_active_trends:
            half_life = self.config.trend_momentum_half_life_days
            ref = today or date.today()
            active.sort(key=lambda t: t.momentum(ref, half_life))
            for trend in active[: len(active) - self.config.trend_max_active_trends]:
                trend.status = TrendStatus.ARCHIVED
                logger.info("Archived low-momentum trend '%s' over active cap", trend.id)

    def _compute_status(self, trend: Trend, today: date) -> TrendStatus:
        try:
            last_seen = date.fromisoformat(trend.last_seen)
        except ValueError:
            return trend.status
        age = (today - last_seen).days
        if age >= self.config.trend_retention_days:
            return TrendStatus.ARCHIVED
        if age >= self.config.trend_cooling_days:
            return TrendStatus.COOLING
        return TrendStatus.ACTIVE

    @staticmethod
    def _make_slug(title: str, existing: set[str]) -> str:
        base = _NON_ALNUM.sub("-", title.lower()).strip("-") or "trend"
        slug = base
        counter = 2
        while slug in existing:
            slug = f"{base}-{counter}"
            counter += 1
        return slug
