import tempfile
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from shared.config import Config
from shared.state_store import LocalStateStore, S3StateStore, StateReadError, create_state_store


def _s3_session(client: MagicMock) -> MagicMock:
    session = MagicMock()
    session.client.return_value = client
    return session


def _body(content: bytes) -> dict:
    body = MagicMock()
    body.read.return_value = content
    return {"Body": body}


class TestLocalStateStore:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            store.write("test.txt", "hello world")
            assert store.read("test.txt") == "hello world"

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            assert not store.exists("missing.txt")
            store.write("exists.txt", "data")
            assert store.exists("exists.txt")

    def test_read_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            assert store.read("nonexistent.txt") is None

    def test_unreadable_file_raises_instead_of_reading_as_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            store.write("k.json", "[]")
            with patch("pathlib.Path.read_text", side_effect=OSError("EIO")):
                with pytest.raises(StateReadError):
                    store.read("k.json")

    def test_nested_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            store.write("subdir/file.txt", "nested")
            assert store.read("subdir/file.txt") == "nested"

    def test_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            store.write("file.txt", "v1")
            store.write("file.txt", "v2")
            assert store.read("file.txt") == "v2"

    def test_read_json_falls_back_on_corrupt_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            store.write("trends.json", "{not json")
            assert store.read_json("trends.json", default={"trends": []}) == {"trends": []}

    def test_write_json_roundtrips_non_ascii(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            store.write_json("k.json", {"제목": "값"})
            assert store.read_json("k.json") == {"제목": "값"}


class TestS3StateStore:
    def test_write_prefixes_key_and_encodes_utf8(self):
        client = MagicMock()
        store = S3StateStore(_s3_session(client), "bkt", prefix="omni/digest_state")
        store.write("trends.json", "내용")
        kwargs = client.put_object.call_args.kwargs
        assert kwargs["Bucket"] == "bkt"
        assert kwargs["Key"] == "omni/digest_state/trends.json"
        assert kwargs["Body"] == "내용".encode()

    def test_empty_prefix_writes_to_bucket_root(self):
        client = MagicMock()
        store = S3StateStore(_s3_session(client), "bkt")
        store.write("trends.json", "x")
        assert client.put_object.call_args.kwargs["Key"] == "trends.json"

    def test_read_decodes_body(self):
        client = MagicMock()
        client.get_object.return_value = _body("내용".encode())
        store = S3StateStore(_s3_session(client), "bkt", prefix="omni/digest_state")
        assert store.read("trends.json") == "내용"
        assert client.get_object.call_args.kwargs["Key"] == "omni/digest_state/trends.json"

    def test_missing_key_reads_none(self):
        client = MagicMock()
        client.get_object.side_effect = ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        store = S3StateStore(_s3_session(client), "bkt")
        assert store.read("trends.json") is None

    def test_other_client_error_is_distinguishable_from_no_history(self):
        # Regression: a denied/throttled GET returned None, exactly like "the key isn't there".
        # Every consumer then treated the history as empty and its next read-modify-write PERSISTED
        # that emptiness. An unreadable store must be its own, typed answer.
        client = MagicMock()
        client.get_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
        store = S3StateStore(_s3_session(client), "bkt")
        with patch("shared.state_store.logger.error") as err:
            with pytest.raises(StateReadError):
                store.read("trends.json")
        assert err.called
        with pytest.raises(StateReadError):
            store.read_json("trends.json", default={})

    def test_exists_true_and_false(self):
        client = MagicMock()
        store = S3StateStore(_s3_session(client), "bkt", prefix="p")
        assert store.exists("trends.json") is True
        client.head_object.side_effect = ClientError({"Error": {"Code": "404"}}, "HeadObject")
        assert store.exists("trends.json") is False

    def test_exists_raises_on_a_real_failure(self):
        # A denied HEAD used to read as "no such key", so the caller started a fresh trend memory
        # and then overwrote the real one.
        client = MagicMock()
        client.head_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "HeadObject")
        store = S3StateStore(_s3_session(client), "bkt")
        with pytest.raises(StateReadError):
            store.exists("trends.json")

    def test_read_json_parses_stored_json(self):
        client = MagicMock()
        client.get_object.return_value = _body(b'{"trends": []}')
        store = S3StateStore(_s3_session(client), "bkt")
        assert store.read_json("trends.json") == {"trends": []}


class TestCreateStateStore:
    def test_local_fallback_without_any_bucket(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        config = Config()
        config.aws.state_bucket_name = ""
        assert isinstance(create_state_store(config), LocalStateStore)

    def test_state_bucket_env_selects_s3_outside_aws(self, monkeypatch):
        # Regression: the store selection used to be gated on the platform sniff, so any non-Lambda
        # caller carrying STATE_BUCKET (agent runtime, container, local run against the real bucket)
        # silently wrote trends.json to the local filesystem and lost every trend thread.
        monkeypatch.setenv("STATE_BUCKET", "prod-bucket")
        monkeypatch.setenv("S3_PREFIX", "omni/digest_state")
        config = Config()
        config.aws.profile = "research"
        config.aws.region = "ap-northeast-2"
        with patch("shared.utils.available_boto_profile", return_value="research"):
            with patch("boto3.Session") as session:
                store = create_state_store(config)
        assert isinstance(store, S3StateStore)
        assert store.bucket == "prod-bucket"
        assert store.prefix == "omni/digest_state"
        # Credentials come from the configured profile wherever that profile actually resolves.
        assert session.call_args.kwargs == {"profile_name": "research", "region_name": "ap-northeast-2"}

    def test_an_unresolvable_profile_falls_back_to_ambient_credentials(self, monkeypatch):
        # The AgentCore runtime sets none of the variables the platform sniff looks for and its
        # container has no ~/.aws/config, so the old check took the profile_name="research" branch
        # there and every trends.json read/write died on ProfileNotFound. Ambient is the default now;
        # the profile is honoured only where it exists.
        monkeypatch.setenv("STATE_BUCKET", "prod-bucket")
        monkeypatch.delenv("S3_PREFIX", raising=False)
        config = Config()
        config.aws.profile = "research"
        config.aws.region = "ap-northeast-2"
        with patch("shared.utils.available_boto_profile", return_value=None):
            with patch("boto3.Session") as session:
                store = create_state_store(config)
        assert isinstance(store, S3StateStore)
        assert store.prefix == "digest_state"  # default when S3_PREFIX is unset
        assert session.call_args.kwargs == {"profile_name": None, "region_name": "ap-northeast-2"}

    def test_config_bucket_appends_digest_state_to_prefix(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        config = Config()
        config.aws.state_bucket_name = "cfg-bucket"
        config.aws.s3_prefix = "omnisummary"
        with patch("boto3.Session"):
            store = create_state_store(config)
        assert isinstance(store, S3StateStore)
        assert store.bucket == "cfg-bucket"
        assert store.prefix == "omnisummary/digest_state"

    def test_config_bucket_without_prefix(self, monkeypatch):
        monkeypatch.delenv("STATE_BUCKET", raising=False)
        config = Config()
        config.aws.state_bucket_name = "cfg-bucket"
        config.aws.s3_prefix = ""
        with patch("boto3.Session"):
            store = create_state_store(config)
        assert store.prefix == "digest_state"

    def test_env_bucket_wins_over_config_bucket(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "env-bucket")
        monkeypatch.setenv("S3_PREFIX", "root/digest_state")
        config = Config()
        config.aws.state_bucket_name = "cfg-bucket"
        with patch("boto3.Session"):
            store = create_state_store(config)
        assert store.bucket == "env-bucket"

    def test_no_config_still_works_in_aws(self, monkeypatch):
        monkeypatch.setenv("STATE_BUCKET", "prod-bucket")
        with patch("boto3.Session") as session:
            store = create_state_store()
        assert isinstance(store, S3StateStore)
        assert session.call_args.kwargs == {}
