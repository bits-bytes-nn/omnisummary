from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from botocore.exceptions import BotoCoreError, ClientError

from application.ports import StateReadError, StateStore
from shared.logger import logger

if TYPE_CHECKING:
    from shared.config import Config


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
        except BotoCoreError as e:
            # Connection/timeout/streaming errors: just as unreadable, and every caller catches only
            # StateReadError.
            logger.error("Failed to read 's3://%s/%s': %s", self.bucket, s3_key, e)
            raise StateReadError(f"Failed to read 's3://{self.bucket}/{s3_key}': {e}") from e

    def write(self, key: str, content: str) -> None:
        s3_key = self._key(key)
        self.s3.put_object(Bucket=self.bucket, Key=s3_key, Body=content.encode("utf-8"))
        logger.debug("Wrote state to 's3://%s/%s' (%d chars)", self.bucket, s3_key, len(content))


def state_boto_session(config: Config | None):
    """A session for the state bucket (the S3 store, the hosted Threads images): ambient credentials, with the configured profile honoured only
    where it actually resolves.

    Gated on is_running_in_aws() this took the named-profile branch in any runtime that sets none of
    the Lambda/Batch/ECS variables the sniff looks for, where there is no ~/.aws/config, so every
    trends.json read and write failed on ProfileNotFound."""
    import boto3

    from adapters.aws import available_boto_profile

    if config is None:
        return boto3.Session()
    return boto3.Session(profile_name=available_boto_profile(config.aws.profile), region_name=config.aws.region or None)


def create_state_store(config: Config | None = None) -> StateStore:
    """Select the S3-backed store whenever a state bucket is configured (STATE_BUCKET env, else
    config.aws.state_bucket_name), else the local filesystem fallback. Shared by the pipeline and
    the deep-research agent so both read/write the same trends.json.

    Gated on the BUCKET, not on is_running_in_aws(): that platform sniff meant any non-Lambda
    caller carrying STATE_BUCKET (the agent runtime, a container, a local run against the real
    bucket) silently wrote trends.json to the local filesystem and lost every trend thread."""
    from shared.constants import LocalPaths

    env_bucket = os.environ.get("STATE_BUCKET", "")
    if env_bucket:
        # S3_PREFIX is the digest-state prefix itself (set by the CDK Lambda env), unlike the
        # config's s3_prefix, which is the bucket ROOT and gets '/digest_state' appended.
        prefix = os.environ.get("S3_PREFIX", "digest_state")
        return S3StateStore(state_boto_session(config), env_bucket, prefix=prefix)
    if config and config.aws.state_bucket_name:
        prefix = f"{config.aws.s3_prefix}/digest_state" if config.aws.s3_prefix else "digest_state"
        return S3StateStore(state_boto_session(config), config.aws.state_bucket_name, prefix=prefix)
    return LocalStateStore(Path(LocalPaths.DIGEST_STATE_DIR.value))
