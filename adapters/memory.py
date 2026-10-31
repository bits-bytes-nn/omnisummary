from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3

from application.ports import MemoryReadError, MemoryStore
from shared.config import Config, get_config
from shared.logger import logger
from shared.naming import digest_memory_name
from shared.utils import available_boto_profile, aws_region

DEFAULT_ACTOR_ID = "omnisummary"


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
        # and would crash json.loads on read). Drop the ranked list first: the digest_result is what
        # the visual Lambda publishes, and the candidate list only feeds the comic's source text.
        logger.warning("Minimal digest state still over limit; dropping the ranked items")
        minimal["ranked_items"] = []
        payload_text = json.dumps(minimal, ensure_ascii=False)
        if len(payload_text) <= self.MAX_EVENT_TEXT:
            return payload_text

        logger.warning("Digest state still over limit; storing empty snapshot")
        return json.dumps({"ranked_items": [], "digest_result": None}, ensure_ascii=False)

    def _load_session(self, session_id: str) -> dict[str, Any] | None:
        """The NEWEST snapshot stored in a session, or None when it holds none.

        A session normally carries exactly one event, but a same-day re-run appends another and
        list_events does not promise an order — reading one page could therefore serve a stale attempt
        of a day that was re-run. Read every page and keep the newest instead. An
        event whose payload is unparseable is skipped; if that leaves nothing usable the error
        propagates, so get_digest reports an unreadable store rather than an empty day."""
        records: list[dict[str, Any]] = []
        token: str | None = None
        while True:
            page = self._client.list_events(
                memoryId=self.memory_id,
                actorId=self.actor_id,
                sessionId=session_id,
                includePayloads=True,
                **({"nextToken": token} if token else {}),
            )
            records.extend(page.get("events", []))
            token = page.get("nextToken")
            if not token:
                break
        best_key: tuple[float, str] | None = None
        best: dict[str, Any] | None = None
        error: Exception | None = None
        for record in records:
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


def _deployed_memory_id(config: Config, session: boto3.Session) -> str:
    """The stack's memory id, found by name. CDK sets MEMORY_ID only on the Lambdas, so a local run
    (main.py, research_cli.py) used to fall back to the local store and read none of the snapshots
    production writes: cross-day dedup saw no yesterday and recall_digest found nothing.

    "" only when the listing succeeded and matched nothing. A lookup that FAILED (expired SSO, denied,
    throttled) raises MemoryReadError: falling back to the local store then answered "nothing stored"
    for a date production had written."""
    prefix = digest_memory_name(config.aws.project_name, config.aws.stage) + "-"
    try:
        pages = session.client("bedrock-agentcore-control").get_paginator("list_memories").paginate()
        return next((m["id"] for page in pages for m in page.get("memories", []) if m["id"].startswith(prefix)), "")
    except Exception as e:
        raise MemoryReadError(f"Could not look up AgentCore memory '{prefix}*': {e}") from e


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
