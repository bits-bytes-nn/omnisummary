from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3

from .logger import logger
from .utils import aws_region

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
    """Persistence boundary for the structured digest snapshot.

    put_digest / get_latest_digest store and reload the digest state the follow-up
    agent reads back (replaces the file/S3 state store). Trend memory now lives in the
    structured trends.json owned by the TrendTracker, not here.
    """

    @abstractmethod
    def put_digest(self, digest_date: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_latest_digest(self) -> dict[str, Any] | None: ...

    @abstractmethod
    def get_digest(self, digest_date: str) -> dict[str, Any] | None:
        """The snapshot stored FOR that ISO date, or None when the date has none.

        Consumers that must act on a specific day (the visual Lambda publishes the digest for the
        date the pipeline generated) use this instead of get_latest_digest: 'load latest, then
        check its date' cannot work, because digest_result.generated_at is UTC and would
        false-mismatch the KST digest date on every pre-09:00 KST run.

        Raises MemoryReadError when the snapshot could not be READ (as opposed to not existing),
        so a publish path never mistakes an unreadable store for an empty day.
        """

    def get_recent_digests(self, n: int, exclude_date: str = "", after_date: str = "") -> list[dict[str, Any]]:
        """Return up to the n most recent digest snapshots (newest first), skipping the one for
        exclude_date and any dated strictly before after_date (ISO). Used to seed cross-day dedup
        from history so it works immediately, not only after the ledger is populated by a future
        run; exclude_date drops today's own snapshot (same-day re-run keeps its stories) and
        after_date bounds the seed to the SAME window the ledger uses, so a story that legitimately
        recurs after the TTL isn't suppressed by a stale snapshot. Base impl: just the latest."""
        latest = self.get_latest_digest()
        return [latest] if latest else []


class LocalMemoryStore(MemoryStore):
    """Filesystem-backed fallback for offline development."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def put_digest(self, digest_date: str, state: dict[str, Any]) -> None:
        path = self.base_dir / f"digest_{digest_date}.json"
        path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("Stored digest state locally at '%s'", path)

    def get_latest_digest(self) -> dict[str, Any] | None:
        files = sorted(self.base_dir.glob("digest_*.json"), reverse=True)
        if not files:
            logger.info("No local digest state found in '%s'", self.base_dir)
            return None
        logger.info("Loaded latest local digest state '%s'", files[0])
        return json.loads(files[0].read_text(encoding="utf-8"))

    def get_digest(self, digest_date: str) -> dict[str, Any] | None:
        path = self.base_dir / f"digest_{digest_date}.json"
        if not path.exists():
            logger.info("No local digest state for '%s' in '%s'", digest_date, self.base_dir)
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as e:
            raise MemoryReadError(f"Failed to read local digest state '{path}': {e}") from e

    def get_recent_digests(self, n: int, exclude_date: str = "", after_date: str = "") -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        skip = f"digest_{exclude_date}.json" if exclude_date else ""
        floor = f"digest_{after_date}.json" if after_date else ""
        files = [
            p
            for p in sorted(self.base_dir.glob("digest_*.json"), reverse=True)
            if p.name != skip and (not floor or p.name >= floor)
        ]
        for path in files[: max(0, n)]:
            try:
                out.append(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return out


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
    ) -> None:
        self.memory_id = memory_id
        self.actor_id = actor_id
        self._client = boto3.client("bedrock-agentcore", region_name=region_name or aws_region())

    MAX_EVENT_TEXT = 100_000
    # Safety cap on list_sessions pagination (100 sessions/page) so a runaway token loop can't
    # spin forever; 20 pages = 2000 sessions ≈ 5+ years of daily digests.
    MAX_SESSION_PAGES = 20
    # Events read per digest session. Small on purpose: get_recent_digests loads one page PER
    # session, so this is a constant factor on its cost that must not grow with history. A session
    # only ever holds more than one event when the same date's pipeline was re-run.
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
        """Bound the serialized state under AgentCore's per-event char limit by
        progressively shedding bulk: drop collected-item bodies, then truncate the
        item text fields the agent reads (it re-truncates anyway), then hard-cap.
        A single oversized digest must never abort the whole pipeline."""
        payload_text = json.dumps(state, ensure_ascii=False)
        if len(payload_text) <= self.MAX_EVENT_TEXT:
            return payload_text

        import copy

        trimmed = copy.deepcopy(state)
        trimmed["collected_items"] = {}
        payload_text = json.dumps(trimmed, ensure_ascii=False)
        if len(payload_text) <= self.MAX_EVENT_TEXT:
            logger.warning("Digest state exceeded %d chars; dropped collected_items bodies", self.MAX_EVENT_TEXT)
            return payload_text

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
            "collected_items": {},
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
        return json.dumps({"ranked_items": [], "digest_result": None, "collected_items": {}}, ensure_ascii=False)

    def _digest_session_ids(self) -> list[str]:
        # Paginate: AgentCore doesn't guarantee newest-first ordering, so a single 100-item page
        # can EXCLUDE the true latest session once >100 digest sessions exist (~3.3 months of daily
        # runs, and sessions are never deleted) — get_latest_digest would then serve a stale
        # snapshot. Follow NextToken to collect every digest session before sorting.
        ids: list[str] = []
        next_token: str | None = None
        for _ in range(self.MAX_SESSION_PAGES):
            kwargs: dict[str, Any] = {"memoryId": self.memory_id, "actorId": self.actor_id, "maxResults": 100}
            if next_token:
                kwargs["nextToken"] = next_token
            resp = self._client.list_sessions(**kwargs)
            ids.extend(
                s["sessionId"]
                for s in resp.get("sessionSummaries", [])
                if s["sessionId"].startswith(self.DIGEST_SESSION_PREFIX)
            )
            next_token = resp.get("nextToken")
            if not next_token:
                break
        else:
            logger.warning(
                "list_sessions hit the %d-page cap; latest-digest lookup may be incomplete", self.MAX_SESSION_PAGES
            )
        return sorted(ids, reverse=True)

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

    def get_latest_digest(self) -> dict[str, Any] | None:
        sessions = self._digest_session_ids()
        if not sessions:
            logger.info("No digest sessions found in AgentCore Memory")
            return None
        data = self._load_session(sessions[0])
        if data is not None:
            logger.info("Loaded latest digest state from AgentCore Memory (session '%s')", sessions[0])
        return data

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

    def get_recent_digests(self, n: int, exclude_date: str = "", after_date: str = "") -> list[dict[str, Any]]:
        skip = f"{self.DIGEST_SESSION_PREFIX}-{exclude_date}" if exclude_date else ""
        # Session ids sort as 'digest-YYYY-MM-DD', so a lexical >= on the suffix bounds by date.
        floor = f"{self.DIGEST_SESSION_PREFIX}-{after_date}" if after_date else ""
        out: list[dict[str, Any]] = []
        session_ids = [s for s in self._digest_session_ids() if s != skip and (not floor or s >= floor)]
        for session_id in session_ids[: max(0, n)]:
            try:
                data = self._load_session(session_id)
            except Exception as e:
                logger.warning("Failed to load digest session '%s': %s", session_id, e)
                continue
            if data is not None:
                out.append(data)
        return out

    @staticmethod
    def _extract_text(event: dict[str, Any]) -> str:
        for entry in event.get("payload", []):
            content = entry.get("conversational", {}).get("content", {})
            if "text" in content:
                return content["text"]
        return ""


def create_memory_store(base_dir: Path | None = None) -> MemoryStore:
    """Select the AgentCore-backed store when MEMORY_ID is configured, else local."""
    memory_id = os.environ.get("MEMORY_ID", "")
    if memory_id:
        return AgentCoreMemoryStore(memory_id)
    fallback = base_dir or Path("digest_state")
    logger.info("MEMORY_ID not set — using LocalMemoryStore at '%s'", fallback)
    return LocalMemoryStore(fallback)
