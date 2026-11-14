import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from shared.constants import SSM_PLACEHOLDER, SSM_SECRET_ENV_VARS

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "put_secrets.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("put_secrets", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["put_secrets"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def put_secrets():
    return _load_module()


@pytest.fixture(autouse=True)
def _config_yaml_present(put_secrets, monkeypatch):
    # config/config.yaml is gitignored, so CI has none; main() refuses to run without it.
    monkeypatch.setattr(put_secrets, "CONFIG_PATH", _SCRIPT)


def _ssm(*, get_side_effect=None, put_side_effect=None) -> MagicMock:
    client = MagicMock()
    client.exceptions.ParameterNotFound = ClientError
    client.get_parameter.side_effect = get_side_effect
    client.put_parameter.side_effect = put_side_effect
    return client


def _validation_error() -> ClientError:
    return ClientError({"Error": {"Code": "ValidationException", "Message": "type change"}}, "PutParameter")


class TestMidWayAbort:
    """The loop used to die on the first parameter SSM refused, leaving every LATER secret on its
    CloudFormation placeholder — which reads as "unset" at runtime, so the affected feature silently
    stopped working with nothing in the output to say so."""

    def test_type_change_rejection_recreates_the_placeholder_as_a_secure_string(self, put_secrets, monkeypatch, capsys):
        names = list(SSM_SECRET_ENV_VARS)
        for env_var in SSM_SECRET_ENV_VARS.values():
            monkeypatch.setenv(env_var, "real-value")
        client = _ssm(put_side_effect=[_validation_error()] + [None] * (2 * len(names)))
        client.get_parameter.return_value = {"Parameter": {"Type": "String", "Value": SSM_PLACEHOLDER}}
        config = MagicMock()
        config.aws.project_name, config.aws.stage, config.aws.region = "omnisummary", "dev", "ap-northeast-2"
        with patch.object(put_secrets, "Config") as cfg:
            cfg.load.return_value = config
            with patch.object(
                put_secrets.boto3, "Session", return_value=MagicMock(client=MagicMock(return_value=client))
            ):
                with patch.object(sys, "argv", ["put_secrets.py"]):
                    rc = put_secrets.main()
        out = capsys.readouterr().out
        assert rc == 0  # every parameter landed, so this is not a failure
        # The placeholder parameter is DELETED and re-created ENCRYPTED — never left as plaintext.
        client.delete_parameter.assert_called_once_with(Name=f"/omnisummary/dev/{names[0]}")
        assert all(c.kwargs.get("Type") == "SecureString" for c in client.put_parameter.call_args_list)
        assert "value only" not in out
        # Every parameter was attempted: one rejection must not end the run.
        assert len({c.kwargs["Name"] for c in client.put_parameter.call_args_list}) == len(names)

    def test_a_string_holding_a_real_value_is_neither_deleted_nor_rewritten_as_plaintext(
        self, put_secrets, monkeypatch, capsys
    ):
        # The Threads token is rotated in place by the refresh Lambda, which cannot change the
        # parameter's type — so a String can hold a LIVE credential. Deleting it to encrypt it would
        # destroy the value, and writing the secret as plaintext is not "written": leave it and fail.
        names = list(SSM_SECRET_ENV_VARS)
        for env_var in SSM_SECRET_ENV_VARS.values():
            monkeypatch.setenv(env_var, "real-value")
        client = _ssm(put_side_effect=[_validation_error()] + [None] * (2 * len(names)))
        client.get_parameter.return_value = {"Parameter": {"Type": "String", "Value": "a-live-token"}}
        config = MagicMock()
        config.aws.project_name, config.aws.stage, config.aws.region = "omnisummary", "dev", "ap-northeast-2"
        with patch.object(put_secrets, "Config") as cfg:
            cfg.load.return_value = config
            with patch.object(
                put_secrets.boto3, "Session", return_value=MagicMock(client=MagicMock(return_value=client))
            ):
                with patch.object(sys, "argv", ["put_secrets.py"]):
                    rc = put_secrets.main()
        out = capsys.readouterr().out
        assert rc == 1
        client.delete_parameter.assert_not_called()
        assert all(c.kwargs.get("Type") == "SecureString" for c in client.put_parameter.call_args_list)
        assert f"aws ssm delete-parameter --name /omnisummary/dev/{names[0]}" in out
        assert "FAILED  : " in out and "written : " + names[0] not in out

    def test_a_deleted_parameter_that_cannot_be_rewritten_is_reported_as_failed(self, put_secrets, monkeypatch, capsys):
        # The one state worse than a plaintext secret: the placeholder is gone and nothing replaced
        # it. It must exit non-zero and print the exact command that restores the parameter.
        names = list(SSM_SECRET_ENV_VARS)
        for env_var in SSM_SECRET_ENV_VARS.values():
            monkeypatch.setenv(env_var, "real-value")
        denied = ClientError({"Error": {"Code": "AccessDeniedException"}}, "PutParameter")
        client = _ssm(put_side_effect=[_validation_error(), denied] + [None] * (2 * len(names)))
        client.get_parameter.return_value = {"Parameter": {"Type": "String", "Value": SSM_PLACEHOLDER}}
        config = MagicMock()
        config.aws.project_name, config.aws.stage, config.aws.region = "omnisummary", "dev", "ap-northeast-2"
        with patch.object(put_secrets, "Config") as cfg:
            cfg.load.return_value = config
            with patch.object(
                put_secrets.boto3, "Session", return_value=MagicMock(client=MagicMock(return_value=client))
            ):
                with patch.object(sys, "argv", ["put_secrets.py"]):
                    rc = put_secrets.main()
        out = capsys.readouterr().out
        assert rc == 1
        assert "DELETED but not re-created" in out
        assert f"aws ssm put-parameter --name /omnisummary/dev/{names[0]} --type SecureString" in out

    def test_a_hard_put_failure_is_reported_and_the_loop_continues(self, put_secrets, monkeypatch, capsys):
        names = list(SSM_SECRET_ENV_VARS)
        for env_var in SSM_SECRET_ENV_VARS.values():
            monkeypatch.setenv(env_var, "real-value")
        denied = ClientError({"Error": {"Code": "AccessDeniedException"}}, "PutParameter")
        client = _ssm(put_side_effect=[denied] + [None] * (2 * len(names)))
        client.get_parameter.return_value = {"Parameter": {"Type": "String", "Value": SSM_PLACEHOLDER}}
        config = MagicMock()
        config.aws.project_name, config.aws.stage, config.aws.region = "omnisummary", "dev", "ap-northeast-2"
        with patch.object(put_secrets, "Config") as cfg:
            cfg.load.return_value = config
            with patch.object(
                put_secrets.boto3, "Session", return_value=MagicMock(client=MagicMock(return_value=client))
            ):
                with patch.object(sys, "argv", ["put_secrets.py"]):
                    rc = put_secrets.main()
        out = capsys.readouterr().out
        assert rc == 1  # a secret that is NOT set must not exit 0
        assert "FAILED" in out
        assert len({c.kwargs["Name"] for c in client.put_parameter.call_args_list}) == len(names)


class TestVerify:
    def test_verify_is_read_only_and_flags_placeholders(self, put_secrets, capsys):
        client = _ssm()
        client.get_parameter.return_value = {"Parameter": {"Type": "String", "Value": SSM_PLACEHOLDER}}
        rc = put_secrets._verify(client, "omnisummary", "dev")
        out = capsys.readouterr().out
        assert rc == 1
        assert "PLACEHOLDER" in out
        client.put_parameter.assert_not_called()

    def test_verify_reports_ok_when_every_secret_is_set(self, put_secrets, capsys):
        client = _ssm()
        client.get_parameter.return_value = {"Parameter": {"Type": "SecureString", "Value": "real"}}
        rc = put_secrets._verify(client, "omnisummary", "dev")
        assert rc == 0
        assert "PLACEHOLDER" not in capsys.readouterr().out

    def test_verify_flags_a_set_but_unencrypted_parameter(self, put_secrets, capsys):
        # "Set" is not the same as "SecureString": a real value in a plain String parameter is the
        # state this script exists to remove, so it must be visible and exit non-zero.
        client = _ssm()
        client.get_parameter.return_value = {"Parameter": {"Type": "String", "Value": "a-live-token"}}
        rc = put_secrets._verify(client, "omnisummary", "dev")
        out = capsys.readouterr().out
        assert rc == 1
        assert "PLAINTEXT" in out
        client.put_parameter.assert_not_called()


class TestConfigGuard:
    def test_refuses_to_run_without_config_yaml(self, put_secrets, monkeypatch, tmp_path):
        # Config.load() falls back to the tracked template, whose project/stage name parameter paths
        # the deployed stack may never read.
        monkeypatch.setattr(put_secrets, "CONFIG_PATH", tmp_path / "config.yaml")
        with patch.object(put_secrets, "Config") as cfg:
            with patch.object(put_secrets.boto3, "client") as client:
                with patch.object(sys, "argv", ["put_secrets.py"]):
                    with pytest.raises(SystemExit, match="config.yaml is missing"):
                        put_secrets.main()
        cfg.load.assert_not_called()
        client.assert_not_called()


def _run_main(put_secrets, client, argv):
    config = MagicMock()
    config.aws.project_name, config.aws.stage, config.aws.region = "omnisummary", "dev", "ap-northeast-2"
    with patch.object(put_secrets, "Config") as cfg:
        cfg.load.return_value = config
        with patch.object(put_secrets.boto3, "Session", return_value=MagicMock(client=MagicMock(return_value=client))):
            with patch.object(sys, "argv", ["put_secrets.py", *argv]):
                return put_secrets.main()


class TestTheLiveValueIsKept:
    """The refresh Lambda rotates the Threads token in place: reasserting .env would restore an expired one."""

    @pytest.fixture(autouse=True)
    def _env(self, monkeypatch):
        for env_var in SSM_SECRET_ENV_VARS.values():
            monkeypatch.setenv(env_var, "local-value")

    def test_an_existing_secure_string_is_left_alone_without_force(self, put_secrets, capsys):
        client = _ssm()
        client.get_parameter.return_value = {"Parameter": {"Type": "SecureString", "Value": "rotated-value"}}
        assert _run_main(put_secrets, client, []) == 0
        client.put_parameter.assert_not_called()
        client.delete_parameter.assert_not_called()
        assert "live value differs from .env" in capsys.readouterr().out

    def test_force_overwrites_an_existing_secure_string(self, put_secrets):
        client = _ssm()
        client.get_parameter.return_value = {"Parameter": {"Type": "SecureString", "Value": "rotated-value"}}
        assert _run_main(put_secrets, client, ["--force"]) == 0
        assert len(client.put_parameter.call_args_list) == len(SSM_SECRET_ENV_VARS)
        assert all(c.kwargs["Value"] == "local-value" for c in client.put_parameter.call_args_list)

    def test_dry_run_writes_and_deletes_nothing(self, put_secrets, capsys):
        client = _ssm()
        client.get_parameter.return_value = {"Parameter": {"Type": "String", "Value": SSM_PLACEHOLDER}}
        assert _run_main(put_secrets, client, ["--dry-run"]) == 0
        client.put_parameter.assert_not_called()
        client.delete_parameter.assert_not_called()
        assert "would write SecureString" in capsys.readouterr().out
