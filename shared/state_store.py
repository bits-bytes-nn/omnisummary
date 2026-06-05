from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError

from .logger import is_running_in_aws, logger

if TYPE_CHECKING:
    from .config import Config


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


class LocalStateStore(StateStore):
    def __init__(self, base_dir: str | Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def read(self, key: str) -> str | None:
        path = self.base_dir / key
        if not path.exists():
            return None
        content = path.read_text(encoding="utf-8")
        logger.debug("Read state from '%s' (%d chars)", path, len(content))
        return content

    def write(self, key: str, content: str) -> None:
        path = self.base_dir / key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        logger.debug("Wrote state to '%s' (%d chars)", path, len(content))

    def exists(self, key: str) -> bool:
        return (self.base_dir / key).exists()


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
            if e.response["Error"]["Code"] == "NoSuchKey":
                return None
            logger.warning("Failed to read 's3://%s/%s': %s", self.bucket, s3_key, e)
            return None

    def write(self, key: str, content: str) -> None:
        s3_key = self._key(key)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=content.encode("utf-8"))
        logger.debug("Wrote state to 's3://%s/%s' (%d chars)", self.bucket, s3_key, len(content))

    def exists(self, key: str) -> bool:
        s3_key = self._key(key)
        try:
            self.s3.head_object(Bucket=self.bucket, Key=s3_key)
            return True
        except ClientError:
            return False


def create_state_store(config: Config | None = None) -> StateStore:
    """Select the S3-backed store when running in AWS or when a state bucket is
    configured, else the local filesystem fallback. Shared by the pipeline and the
    follow-up agent so both read/write the same trends.json."""
    import boto3

    from .constants import LocalPaths

    if is_running_in_aws():
        bucket = os.environ.get("STATE_BUCKET", "")
        if bucket:
            prefix = os.environ.get("S3_PREFIX", "digest_state")
            return S3StateStore(boto3.Session(), bucket, prefix=prefix)
    if config and config.aws.state_bucket_name:
        prefix = f"{config.aws.s3_prefix}/digest_state" if config.aws.s3_prefix else "digest_state"
        return S3StateStore(
            boto3.Session(profile_name=config.aws.profile or None, region_name=config.aws.region),
            config.aws.state_bucket_name,
            prefix=prefix,
        )
    return LocalStateStore(Path(LocalPaths.DIGEST_STATE_DIR.value))
