from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from botocore.exceptions import ClientError

from .logger import logger


class StateStore(ABC):
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
