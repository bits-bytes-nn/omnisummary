from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3

from .logger import logger

DEFAULT_ACTOR_ID = "omnisummary"


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
        region = region_name or os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2"))
        self._client = boto3.client("bedrock-agentcore", region_name=region)

    MAX_EVENT_TEXT = 100_000
    # When even the ranked set overflows, cap each item's stored body. The follow-up
    # agent re-truncates to ~2000 tokens via get_detail, so this loses nothing it uses.
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
        return json.dumps(minimal, ensure_ascii=False)[: self.MAX_EVENT_TEXT]

    def get_latest_digest(self) -> dict[str, Any] | None:
        sessions = self._client.list_sessions(memoryId=self.memory_id, actorId=self.actor_id, maxResults=100)
        digest_sessions = sorted(
            (
                s["sessionId"]
                for s in sessions.get("sessionSummaries", [])
                if s["sessionId"].startswith(self.DIGEST_SESSION_PREFIX)
            ),
            reverse=True,
        )
        if not digest_sessions:
            logger.info("No digest sessions found in AgentCore Memory")
            return None

        events = self._client.list_events(
            memoryId=self.memory_id,
            actorId=self.actor_id,
            sessionId=digest_sessions[0],
            maxResults=1,
            includePayloads=True,
        )
        records = events.get("events", [])
        if not records:
            return None
        text = self._extract_text(records[0])
        logger.info("Loaded latest digest state from AgentCore Memory (session '%s')", digest_sessions[0])
        return json.loads(text) if text else None

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
