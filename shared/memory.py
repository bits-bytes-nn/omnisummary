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
DIGEST_NAMESPACE_TEMPLATE = "/facts/{actor_id}/"


class MemoryStore(ABC):
    """Persistence boundary for digest state and cross-day agent recall.

    Two responsibilities:
    - put_digest / get_latest_digest: store and reload the structured digest snapshot
      the follow-up agent reads back (replaces the file/S3 state store).
    - record_trend / recall: write durable trend facts and semantically recall them
      across days (AgentCore long-term memory; no-op on local fallback).
    """

    @abstractmethod
    def put_digest(self, digest_date: str, state: dict[str, Any]) -> None: ...

    @abstractmethod
    def get_latest_digest(self) -> dict[str, Any] | None: ...

    @abstractmethod
    def record_trend(self, summary: str, *, session_id: str) -> None: ...

    @abstractmethod
    def recall(self, query: str, *, top_k: int = 5) -> list[str]: ...


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

    def record_trend(self, summary: str, *, session_id: str) -> None:
        path = self.base_dir / "trends.jsonl"
        line = json.dumps({"session_id": session_id, "summary": summary}, ensure_ascii=False)
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        path = self.base_dir / "trends.jsonl"
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        return [json.loads(line)["summary"] for line in lines[-top_k:]]


class AgentCoreMemoryStore(MemoryStore):
    """Bedrock AgentCore Memory-backed store (system of record in AWS).

    Digest snapshots are persisted as short-term session events under a stable
    actor; trend summaries additionally feed long-term semantic extraction for
    cross-day recall via retrieve_memory_records.
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

    def put_digest(self, digest_date: str, state: dict[str, Any]) -> None:
        session_id = f"{self.DIGEST_SESSION_PREFIX}-{digest_date}"
        payload_text = json.dumps(state, ensure_ascii=False)
        self._client.create_event(
            memoryId=self.memory_id,
            actorId=self.actor_id,
            sessionId=session_id,
            eventTimestamp=datetime.now(UTC),
            payload=[{"conversational": {"role": "ASSISTANT", "content": {"text": payload_text}}}],
        )
        logger.info("Stored digest state to AgentCore Memory (session '%s')", session_id)

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

    def record_trend(self, summary: str, *, session_id: str) -> None:
        self._client.create_event(
            memoryId=self.memory_id,
            actorId=self.actor_id,
            sessionId=session_id,
            eventTimestamp=datetime.now(UTC),
            payload=[{"conversational": {"role": "ASSISTANT", "content": {"text": summary}}}],
        )
        logger.info("Recorded trend to AgentCore Memory (session '%s')", session_id)

    def recall(self, query: str, *, top_k: int = 5) -> list[str]:
        namespace = DIGEST_NAMESPACE_TEMPLATE.format(actor_id=self.actor_id)
        try:
            response = self._client.retrieve_memory_records(
                memoryId=self.memory_id,
                namespace=namespace,
                searchCriteria={"searchQuery": query, "topK": top_k},
                maxResults=top_k,
            )
        except Exception as e:
            logger.warning("AgentCore recall failed: %s", e)
            return []
        return [self._extract_record_text(r) for r in response.get("memoryRecords", [])]

    @staticmethod
    def _extract_text(event: dict[str, Any]) -> str:
        for entry in event.get("payload", []):
            content = entry.get("conversational", {}).get("content", {})
            if "text" in content:
                return content["text"]
        return ""

    @staticmethod
    def _extract_record_text(record: dict[str, Any]) -> str:
        content = record.get("content", {})
        if isinstance(content, dict):
            return content.get("text", "")
        return str(content)


def create_memory_store(base_dir: Path | None = None) -> MemoryStore:
    """Select the AgentCore-backed store when MEMORY_ID is configured, else local."""
    memory_id = os.environ.get("MEMORY_ID", "")
    if memory_id:
        return AgentCoreMemoryStore(memory_id)
    fallback = base_dir or Path("digest_state")
    logger.info("MEMORY_ID not set — using LocalMemoryStore at '%s'", fallback)
    return LocalMemoryStore(fallback)
