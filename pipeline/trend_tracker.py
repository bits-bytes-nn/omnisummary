from __future__ import annotations

import asyncio
import json
import re
from datetime import date

from langchain_core.output_parsers import StrOutputParser
from pydantic import ValidationError

from shared import (
    TRENDS_KEY,
    BedrockLanguageModelFactory,
    Trend,
    TrendClassifyPrompt,
    TrendEvidence,
    TrendMemory,
    TrendStatus,
    logger,
    parse_json_from_llm_output,
)
from shared.config import PipelineConfig
from shared.state_store import StateReadError, StateStore

__all__ = ["TRENDS_KEY", "TrendTracker"]

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


class TrendTracker:
    def __init__(
        self,
        config: PipelineConfig,
        llm_factory: BedrockLanguageModelFactory,
        state_store: StateStore,
    ) -> None:
        self.config = config
        self.llm = llm_factory.get_model(config.trend_model, stage="trends")
        self.state_store = state_store
        self._memory: TrendMemory | None = None
        # Set when trends.json could not be READ this run: history is unknown, so the merge must
        # not be persisted (it would replace every accumulated thread with today's observations).
        self._degraded = False

    def _load_memory(self) -> TrendMemory:
        if self._memory is not None:
            return self._memory

        try:
            raw = self.state_store.read(TRENDS_KEY)
        except StateReadError as e:
            logger.error("Trend memory unreadable (%s); running without trend history", e)
            self._degraded = True
            self._memory = TrendMemory()
            return self._memory
        if raw:
            try:
                self._memory = TrendMemory.model_validate_json(raw)
                logger.info("Loaded %d trends from '%s'", len(self._memory.trends), TRENDS_KEY)
                return self._memory
            except ValidationError as e:
                # A single bad record (schema drift, a removed enum value) must not wipe ALL
                # accumulated trend history. Recover per-trend: parse the raw JSON and validate
                # each trend independently, keeping every one that still validates.
                self._memory = self._recover_trends(raw, e)
                return self._memory

        logger.info("No existing trends found, starting fresh")
        self._memory = TrendMemory()
        return self._memory

    @staticmethod
    def _recover_trends(raw: str, err: ValidationError) -> TrendMemory:
        try:
            data = json.loads(raw)
            raw_trends = data.get("trends", []) if isinstance(data, dict) else []
        except (json.JSONDecodeError, ValueError):
            logger.warning("Trends blob is not valid JSON (%s); starting fresh", err)
            return TrendMemory()
        recovered: list[Trend] = []
        for rt in raw_trends:
            try:
                recovered.append(Trend.model_validate(rt))
            except ValidationError:
                logger.warning("Dropping one unparseable trend during recovery: %r", rt)
        logger.warning("Trends schema drift (%s); recovered %d/%d trends", err, len(recovered), len(raw_trends))
        return TrendMemory(trends=recovered)

    def get_trends_context(self, today: date | None = None) -> str:
        """The DIGEST-facing trend ammunition: the highest-momentum visible trends, with the
        recurrence facts the lead reasons from.

        Capped at trend_max_active_trends (cooling trends are visible too). The CLASSIFIER's view
        (_render_existing) is deliberately NOT capped: hiding a trend there would make the model coin
        a duplicate id for the same thread.

        Only evidence dated BEFORE today counts: on a same-day re-run the first run's stories are
        already in trends.json, and showing them as "recent days" fed today's news back in as history."""
        # The run's digest date (KST) is the single clock, so the recurrence facts agree with the
        # KST-stamped evidence; date.today() would be UTC and off by a day at the 19:00-KST boundary.
        today = today or date.today()
        cutoff = today.isoformat()
        visible = []
        for t in self._load_memory().trends:
            past = [ev for ev in t.evidence if ev.date < cutoff]
            if past:
                visible.append(t.model_copy(update={"evidence": past}))
        if not visible:
            return ""
        visible.sort(key=lambda t: t.momentum(today, self.config.trend_momentum_half_life_days), reverse=True)
        capped = visible[: self.config.trend_max_active_trends]
        if len(capped) < len(visible):
            logger.info(
                "Trend context capped to the %d highest-momentum of %d visible trends",
                len(capped),
                len(visible),
            )
        return self._render_ammunition(capped, today)

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
            gist = t.evidence[-1].summary if t.evidence else ""
            fact_str = f" [{', '.join(facts)}]" if facts else ""
            lines.append(f"- {t.title}{fact_str}: {gist}" if gist else f"- {t.title}{fact_str}")
        return "\n".join(lines)

    async def update_trends(self, digest_text: str, today_date: str) -> None:
        memory = self._load_memory()
        # Idempotency: drop any evidence already stamped with today's date so a re-run (Lambda retry,
        # manual re-run) replaces the day's contribution instead of appending reworded duplicates —
        # and BEFORE rendering, so the classifier never sees the first run's stories as existing
        # trends. Dates are code-stamped, never from the LLM.
        for trend in memory.trends:
            trend.evidence = [ev for ev in trend.evidence if ev.date != today_date]

        existing_summary = self._render_existing(memory)
        chain = TrendClassifyPrompt.get_prompt() | self.llm | StrOutputParser()
        raw = await chain.ainvoke(
            {
                "existing_trends": existing_summary or "(No trends tracked yet.)",
                "todays_digest": digest_text,
            }
        )

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
        if self._degraded:
            # trends.json was unreadable at load, so `memory` holds only today's observations.
            # Writing it would wipe every tracked thread; the next healthy run picks up again.
            logger.error("Not persisting trends: the existing trend memory could not be read this run")
            return
        await asyncio.to_thread(self.state_store.write, TRENDS_KEY, memory.model_dump_json())
        logger.info("Persisted %d trends to '%s'", len(memory.trends), TRENDS_KEY)

    @staticmethod
    def _render_existing(memory: TrendMemory) -> str:
        lines: list[str] = []
        for t in memory.trends:
            if not t.evidence:
                continue
            gist = t.evidence[-1].summary if t.evidence else ""
            lines.append(f"- {t.id}: {t.title}" + (f" — {gist}" if gist else ""))
        return "\n".join(lines)

    @staticmethod
    def _parse_observations(raw: str) -> list[dict[str, str]]:
        try:
            data = parse_json_from_llm_output(raw)
        except (json.JSONDecodeError, ValueError) as e:
            # Raise, not []: an empty list is persisted as "no trends today", and on a same-day re-run
            # (today's evidence already stripped) it deletes the day's trends. The runner logs it and
            # skips the write.
            raise ValueError(f"Unparseable trend classifier output: {e}") from e
        items = data.get("observations") if isinstance(data, dict) else data
        if not isinstance(items, list):
            raise ValueError("Trend classifier output has no observations list")
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

    def _run_lifecycle(self, memory: TrendMemory, today_date: str) -> None:
        """Cap evidence, derive last_seen FROM the evidence, then drop trends past retention and
        mark the rest active/cooling. last_seen is derived rather than stamped on append: a same-day
        re-run strips today's evidence first, and a stamped value kept a trend the second pass did
        not re-emit at today's date — ACTIVE, with its retention clock restarted. A trend past
        retention is dropped outright."""
        today = date.fromisoformat(today_date)
        for trend in memory.trends:
            # Sorted first: a --date backfill appends an older day last, and the cap kept it over newer ones.
            trend.evidence.sort(key=lambda ev: ev.date)
            if len(trend.evidence) > self.config.trend_max_evidence:
                trend.evidence = trend.evidence[-self.config.trend_max_evidence :]
            if trend.evidence:
                trend.last_seen = max(ev.date for ev in trend.evidence)

        kept: list[Trend] = []
        for trend in memory.trends:
            age = self._age_days(trend, today)
            if age is not None and age >= self.config.trend_retention_days:
                continue
            if age is not None:
                trend.status = TrendStatus.COOLING if age >= self.config.trend_cooling_days else TrendStatus.ACTIVE
            kept.append(trend)
        if len(kept) < len(memory.trends):
            logger.info(
                "Dropped %d trend(s) unseen for %d+ days",
                len(memory.trends) - len(kept),
                self.config.trend_retention_days,
            )
        memory.trends = kept

    @staticmethod
    def _age_days(trend: Trend, today: date) -> int | None:
        try:
            return (today - date.fromisoformat(trend.last_seen)).days
        except ValueError:
            return None

    @staticmethod
    def _make_slug(title: str, existing: set[str]) -> str:
        base = _NON_ALNUM.sub("-", title.lower()).strip("-") or "trend"
        slug = base
        counter = 2
        while slug in existing:
            slug = f"{base}-{counter}"
            counter += 1
        return slug
