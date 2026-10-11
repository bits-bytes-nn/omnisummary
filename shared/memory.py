from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import boto3

from .config import Config, get_config
from .logger import logger
from .utils import available_boto_profile, aws_region

DEFAULT_ACTOR_ID = "omnisummary"


class MemoryReadError(RuntimeError):
    """The digest snapshot could NOT be read — distinct from "there is no snapshot for that date".

    Both used to come back as None, so a throttled/denied/misconfigured AgentCore read was
    indistinguishable from a day that simply has no digest: the visual Lambda logged "nothing to
    publish", returned 200, and the day's digest — whose only delivery path that Lambda is — was
    never posted, with no alarm.

    A consumer that PUBLISHES must let this escape (a failed invoke fires the Errors alarm and
    lands in the DLQ; retry_attempts=0 means it cannot duplicate a post). A consumer that only
    ENRICHES — the cross-day dedup seed in main.py — keeps catching it and degrades, because a
    lost dedup hint is harmless while a lost digest is not.
    """


class MemoryStore(ABC):
    """Persistence boundary for the structured digest snapshot, one per digest date.

    The visual Lambda reloads the day's snapshot to publish it, the pipeline reads recent ones to
    seed cross-day dedup, and the research CLI's recall_digest reads one back. Trend memory lives in
    the structured trends.json owned by the TrendTracker, not here.
    """

    @abstractmethod
    def put_digest(self, digest_date: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_digest(self, digest_date: str) -> dict[str, Any] | None:
        """The snapshot stored FOR that ISO date, or None when the date has none.

        Keyed by the digest date rather than "the latest": digest_result.generated_at is UTC and
        would false-mismatch the KST digest date on every pre-09:00 KST run.

        Raises MemoryReadError when the snapshot could not be READ (as opposed to not existing),
        so a publish path never mistakes an unreadable store for an empty day.
        """

    def get_recent_digests(self, before: date, days: int) -> list[dict[str, Any]]:
        """The snapshots for the `days` dates preceding `before` (newest first), skipping dates that
        have none. Seeds cross-day dedup with the SAME window the published-URL ledger uses, and
        never includes `before` itself, so a same-day re-run keeps its own stories.

        Session ids are deterministic (`digest-YYYY-MM-DD`), so this reads exactly `days` sessions
        instead of listing every session ever stored. Best-effort per date: it only ENRICHES the
        dedup seed, so one unreadable day is logged and skipped rather than failing the read."""
        out: list[dict[str, Any]] = []
        for offset in range(1, max(0, days) + 1):
            day = (before - timedelta(days=offset)).isoformat()
            try:
                data = self.get_digest(day)
            except MemoryReadError as e:
                logger.warning("Skipping the digest snapshot for %s: %s", day, e)
                continue
            if data is not None:
                out.append(data)
        return out


class LocalMemoryStore(MemoryStore):
    """Filesystem-backed fallback for offline development."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put_digest(self, digest_date: str, state: dict[str, Any]) -> None:
        path = self.base_dir / f"digest_{digest_date}.json"
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Stored digest state locally at '%s'", path)

    def get_digest(self, digest_date: str) -> dict[str, Any] | None:
        path = self.base_dir / f"digest_{digest_date}.json"
        if not path.exists():
            logger.info("No local digest state for '%s' in '%s'", digest_date, self.base_dir)
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise MemoryReadError(f"Failed to read local digest state '{path}': {e}") from e


class AgentCoreMemoryStore(MemoryStore):
    """Bedrock AgentCore Memory-backed store (system of record in AWS).

    Digest snapshots are persisted as short-term session events under a stable actor.
    """

    DIGEST_SESSION_PREFIX = "digest"

    def __init__(
        self,
        memory_id: str,
        *,
        actor_id: str = DEFAULT_ACTOR_ID,
        region_name: str | None = None,
        session: boto3.Session | None = None,
    ) -> None:
        self.memory_id = memory_id
        self.actor_id = actor_id
        self._client = (session or boto3).client("bedrock-agentcore", region_name=region_name or aws_region())

    MAX_EVENT_TEXT = 100_000
    # Events read per digest session. A session only ever holds more than one event when the same
    # date's pipeline was re-run.
    EVENTS_PER_SESSION = 5
    # When even the ranked set overflows, cap each item's stored body. The digest snapshot
    # only needs enough text for cross-day dedup/recall, so a per-item cap loses nothing used.
    RANKED_TEXT_CAP = 12_000

    def put_digest(self, digest_date: str, state: dict[str, Any]) -> None:
        session_id = f"{self.DIGEST_SESSION_PREFIX}-{digest_date}"
        payload_text = self._fit_to_limit(state)
        self._client.create_event(
            memoryId=self.memory_id,
            actorId=self.actor_id,
            sessionId=session_id,
            eventTimestamp=datetime.now(UTC),
            payload=[{"conversational": {"role": "ASSISTANT", "content": {"text": payload_text}}}],
        )
        logger.info("Stored digest state to AgentCore Memory (session '%s')", session_id)

    def _fit_to_limit(self, state: dict[str, Any]) -> str:
        """Bound the serialized state under AgentCore's per-event char limit by progressively
        shedding bulk: truncate the ranked items' text (readers re-truncate anyway), then hard-cap.
        A single oversized digest must never abort the whole pipeline."""
        payload_text = json.dumps(state, ensure_ascii=False)
        if len(payload_text) <= self.MAX_EVENT_TEXT:
            return payload_text

        import copy

        trimmed = copy.deepcopy(state)
        for ranked in trimmed.get("ranked_items", []):
            item = ranked.get("item", {})
            if isinstance(item.get("text"), str):
                item["text"] = item["text"][: self.RANKED_TEXT_CAP]
        payload_text = json.dumps(trimmed, ensure_ascii=False)
        if len(payload_text) <= self.MAX_EVENT_TEXT:
            logger.warning("Digest state still large; truncated ranked-item text to %d chars", self.RANKED_TEXT_CAP)
            return payload_text

        logger.warning("Digest state exceeds limit after trimming; storing ranked metadata only")
        minimal: dict[str, Any] = {
            "ranked_items": [
                {"item": {k: v for k, v in r.get("item", {}).items() if k != "text"}, "score": r.get("score")}
                for r in trimmed.get("ranked_items", [])
            ],
            "digest_result": trimmed.get("digest_result"),
        }
        payload_text = json.dumps(minimal, ensure_ascii=False)
        if len(payload_text) <= self.MAX_EVENT_TEXT:
            return payload_text

        # Still over: shed whole keys (never byte-slice — a truncated string is invalid JSON
        # and would crash json.loads on read). Drop the digest_result, then the ranked list.
        logger.warning("Minimal digest state still over limit; dropping digest_result")
        minimal["digest_result"] = None
        payload_text = json.dumps(minimal, ensure_ascii=False)
        if len(payload_text) <= self.MAX_EVENT_TEXT:
            return payload_text

        logger.warning("Digest state still over limit; storing empty snapshot")
        return json.dumps({"ranked_items": [], "digest_result": None}, ensure_ascii=False)

    def _load_session(self, session_id: str) -> dict[str, Any] | None:
        """The NEWEST snapshot stored in a session, or None when it holds none.

        A session normally carries exactly one event, but a same-day re-run appends another and
        list_events does not promise an order — reading maxResults=1 could therefore serve the FIRST
        (stale) attempt of a day that was re-run. Read a small page and keep the newest instead. An
        event whose payload is unparseable is skipped; if that leaves nothing usable the error
        propagates, so get_digest reports an unreadable store rather than an empty day."""
        events = self._client.list_events(
            memoryId=self.memory_id,
            actorId=self.actor_id,
            sessionId=session_id,
            maxResults=self.EVENTS_PER_SESSION,
            includePayloads=True,
        )
        best_key: tuple[float, str] | None = None
        best: dict[str, Any] | None = None
        error: Exception | None = None
        for record in events.get("events", []):
            text = self._extract_text(record)
            if not text:
                continue
            try:
                data = json.loads(text)
            except ValueError as e:
                logger.warning("Skipping an unparseable event in session '%s': %s", session_id, e)
                error = e
                continue
            key = (self._event_epoch(record), self._payload_stamp(data))
            if best_key is None or key > best_key:
                best_key, best = key, data
        if best is None and error is not None:
            raise error
        return best

    @staticmethod
    def _event_epoch(event: dict[str, Any]) -> float:
        """The event's own timestamp as an epoch float; 0.0 when absent or unparsable, so an
        undated event never outranks a dated one."""
        stamp = event.get("eventTimestamp")
        if isinstance(stamp, datetime):
            return stamp.timestamp()
        if isinstance(stamp, int | float):
            return float(stamp)
        if isinstance(stamp, str):
            try:
                return datetime.fromisoformat(stamp).timestamp()
            except ValueError:
                return 0.0
        return 0.0

    @staticmethod
    def _payload_stamp(data: object) -> str:
        """OPTIONAL tiebreaker for two events with the same timestamp: the digest's own
        generated_at. "" when the snapshot doesn't carry one, so snapshots stored before this was
        read still load — it only ever breaks a tie, never decides on its own."""
        if not isinstance(data, dict):
            return ""
        stamp = (data.get("digest_result") or {}).get("generated_at")
        return stamp if isinstance(stamp, str) else ""

    def get_digest(self, digest_date: str) -> dict[str, Any] | None:
        session_id = f"{self.DIGEST_SESSION_PREFIX}-{digest_date}"
        try:
            data = self._load_session(session_id)
        except Exception as e:
            # RAISE, don't return None: a throttled/denied read used to read as "this day has no
            # digest", so the visual Lambda skipped delivery and reported success.
            raise MemoryReadError(f"Failed to load digest session '{session_id}': {e}") from e
        if data is None:
            logger.warning("No digest state in AgentCore Memory for session '%s'", session_id)
        else:
            logger.info("Loaded digest state from AgentCore Memory (session '%s')", session_id)
        return data

    @staticmethod
    def _extract_text(event: dict[str, Any]) -> str:
        for entry in event.get("payload", []):
            content = entry.get("conversational", {}).get("content", {})
            if "text" in content:
                return content["text"]
        return ""


def digest_memory_name(project_name: str, stage: str) -> str:
    """The AgentCore Memory name the foundation stack creates; its id is this name plus a suffix."""
    return f"{project_name}_{stage}_digest_state".replace("-", "_")


def _deployed_memory_id(config: Config, session: boto3.Session) -> str:
    """The stack's memory id, found by name. CDK sets MEMORY_ID only on the Lambdas, so a local run
    (main.py, research_cli.py) used to fall back to the local store and read none of the snapshots
    production writes: cross-day dedup saw no yesterday and recall_digest found nothing."""
    prefix = digest_memory_name(config.aws.project_name, config.aws.stage) + "-"
    try:
        pages = session.client("bedrock-agentcore-control").get_paginator("list_memories").paginate()
        return next((m["id"] for page in pages for m in page.get("memories", []) if m["id"].startswith(prefix)), "")
    except Exception as e:
        logger.warning("Could not look up AgentCore memory '%s*' (%s)", prefix, e)
        return ""


def create_memory_store(base_dir: Path | None = None) -> MemoryStore:
    """The AgentCore-backed store when the stack's memory exists (MEMORY_ID, or looked up by name for
    a local run), else the local store. The lookup and the reads share ONE session on the configured
    profile and config.aws.region, where the stack lives: a shell's AWS_PROFILE / AWS_REGION may point
    at another account or the Bedrock region, and a read client on them found the memory by name and
    then answered "Memory not found" for it."""
    config = get_config()
    session = boto3.Session(profile_name=available_boto_profile(config.aws.profile), region_name=config.aws.region)
    memory_id = os.environ.get("MEMORY_ID", "") or _deployed_memory_id(config, session)
    if memory_id:
        return AgentCoreMemoryStore(memory_id, region_name=config.aws.region, session=session)
    fallback = base_dir or Path("digest_state")
    logger.info("No AgentCore memory found — using LocalMemoryStore at '%s'", fallback)
    return LocalMemoryStore(fallback)
