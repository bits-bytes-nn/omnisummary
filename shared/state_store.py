from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from .logger import is_running_in_aws, logger

if TYPE_CHECKING:
    from .config import Config


class StateReadError(RuntimeError):
    """The state blob could NOT be read — distinct from "there is no history yet".

    Both used to come back as None, so a throttled/denied S3 GET read as an empty ledger and the
    next read-modify-write persisted that emptiness: the published-URL ledger, the recent-leads log,
    the visual-format window and the Threads idempotency marker were all blanked by one failed read.

    Every consumer handles this the same way: log at ERROR, treat the history as UNKNOWN, and SKIP
    the write. It must never reach a publish path — a lost digest is strictly worse than a run
    without history.
    """


class StateStore(ABC):
    """Blob store for the structured trends memory (read-modify-write each run).

    Distinct from shared.memory.MemoryStore by design — and NOT replaceable by it.
    trends.json is a deliberately time-varying document with explicit code-managed
    merge/cooling/archive of topic threads. AgentCore's managed strategies extract STABLE
    records (semantic facts, user preferences) or per-session summaries; even
    customMemoryStrategy can only *append to* those built-in prompts, not implement
    trend-thread maintenance. So trends.json stays the system of record for trends;
    MemoryStore holds the digest snapshot for the follow-up agent.
    """

    @abstractmethod
    def read(self, key: str) -> str | None: ...

    @abstractmethod
    def write(self, key: str, content: str) -> None: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    def read_json(self, key: str, default: Any = None) -> Any:
        """Read and parse a JSON blob; return default on missing/corrupt content.

        Raises StateReadError when the blob could not be READ at all — corrupt content is a
        recoverable "start fresh", an unreadable store is not."""
        raw = self.read(key)
        if not raw:
            return default
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            logger.warning("State key '%s' held invalid JSON; using default", key)
            return default

    def write_json(self, key: str, value: Any) -> None:
        self.write(key, json.dumps(value, ensure_ascii=False))


class LocalStateStore(StateStore):
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def read(self, key: str) -> str | None:
        path = self.base_dir / key
        if not path.exists():
            return None
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as e:
            raise StateReadError(f"Failed to read state '{path}': {e}") from e
        logger.debug("Read state from '%s' (%d chars)", path, len(content))
        return content

    def write(self, key: str, content: str) -> None:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.debug("Wrote state to '%s' (%d chars)", path, len(content))

    def exists(self, key: str) -> bool:
        return (self.base_dir / key).exists()


# S3 answers "no such object" as NoSuchKey on GET but as a bare 404 on HEAD (head_object has no
# response body to carry the code), so both spellings mean the same thing: genuinely absent.
_MISSING_KEY_CODES = frozenset({"NoSuchKey", "404", "NotFound"})


class S3StateStore(StateStore):
    def __init__(self, boto_session: Any, bucket_name: str, prefix: str = "") -> None:
        self.s3 = boto_session.client("s3")
        self.bucket = bucket_name
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        return f"{self.prefix}/{key}".lstrip("/")

    def read(self, key: str) -> str | None:
        s3_key = self._key(key)
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            content = response["Body"].read().decode("utf-8")
            logger.debug("Read state from 's3://%s/%s' (%d chars)", self.bucket, s3_key, len(content))
            return content
        except ClientError as e:
            code = e.response["Error"].get("Code", "")
            if code in _MISSING_KEY_CODES:
                return None
            # NOT None: "the object isn't there" and "S3 wouldn't tell me" are different answers,
            # and returning None for the second one let the next write persist an empty ledger.
            logger.error("Failed to read 's3://%s/%s': %s", self.bucket, s3_key, e)
            raise StateReadError(f"Failed to read 's3://{self.bucket}/{s3_key}': {e}") from e

    def write(self, key: str, content: str) -> None:
        s3_key = self._key(key)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=content.encode("utf-8"))
        logger.debug("Wrote state to 's3://%s/%s' (%d chars)", self.bucket, s3_key, len(content))

    def exists(self, key: str) -> bool:
        s3_key = self._key(key)
        try:
            self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError as e:
            if e.response["Error"].get("Code", "") in _MISSING_KEY_CODES:
                return False
            logger.error("Failed to stat 's3://%s/%s': %s", self.bucket, s3_key, e)
            raise StateReadError(f"Failed to stat 's3://{self.bucket}/{s3_key}': {e}") from e


def _boto_session(config: Config | None):
    """A session for the S3 store. In AWS the execution role is ambient, so take the default
    session; outside AWS honour the configured profile/region — otherwise a developer with
    STATE_BUCKET in .env would silently lose their credentials."""
    import boto3

    if is_running_in_aws() or config is None:
        return boto3.Session()
    return boto3.Session(profile_name=config.aws.profile or None, region_name=config.aws.region)


def create_state_store(config: Config | None = None) -> StateStore:
    """Select the S3-backed store whenever a state bucket is configured (STATE_BUCKET env, else
    config.aws.state_bucket_name), else the local filesystem fallback. Shared by the pipeline and
    the deep-research agent so both read/write the same trends.json.

    Gated on the BUCKET, not on is_running_in_aws(): that platform sniff meant any non-Lambda
    caller carrying STATE_BUCKET (the agent runtime, a container, a local run against the real
    bucket) silently wrote trends.json to the local filesystem and lost every trend thread."""
    from .constants import LocalPaths

    env_bucket = os.environ.get("STATE_BUCKET", "")
    if env_bucket:
        # S3_PREFIX is the digest-state prefix itself (set by the CDK Lambda env), unlike the
        # config's s3_prefix, which is the bucket ROOT and gets '/digest_state' appended.
        prefix = os.environ.get("S3_PREFIX", "digest_state")
        return S3StateStore(_boto_session(config), env_bucket, prefix=prefix)
    if config and config.aws.state_bucket_name:
        prefix = f"{config.aws.s3_prefix}/digest_state" if config.aws.s3_prefix else "digest_state"
        return S3StateStore(_boto_session(config), config.aws.state_bucket_name, prefix=prefix)
    return LocalStateStore(Path(LocalPaths.DIGEST_STATE_DIR.value))
