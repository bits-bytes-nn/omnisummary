import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from shared.constants import ALL_SSM_SECRET_ENV_VARS, SSM_PLACEHOLDER

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

    def test_type_change_rejection_falls_back_to_a_value_only_write(self, put_secrets, monkeypatch, capsys):
        names = list(ALL_SSM_SECRET_ENV_VARS)
        for env_var in ALL_SSM_SECRET_ENV_VARS.values():
            monkeypatch.setenv(env_var, "real-value")
        client = _ssm(put_side_effect=[_validation_error()] + [None] * (2 * len(names)))
        client.get_parameter.return_value = {"Parameter": {"Type": "String", "Value": SSM_PLACEHOLDER}}
        config = MagicMock()
        config.aws.project_name, config.aws.stage, config.aws.region = "omnisummary", "dev", "ap-northeast-2"
        with patch.object(put_secrets, "Config") as cfg:
            cfg.load.return_value = config
            with patch.object(put_secrets.boto3, "client", return_value=client):
                with patch.object(sys, "argv", ["put_secrets.py"]):
                    rc = put_secrets.main()
        out = capsys.readouterr().out
        assert rc == 0  # every parameter landed, so this is not a failure
        # The retry drops Type (the same thing the refresh Lambda does), so the VALUE lands...
        retried = [c for c in client.put_parameter.call_args_list if "Type" not in c.kwargs]
        assert retried, "expected a value-only retry after the type-change rejection"
        # ...and the operator is told loudly how to make it a SecureString.
        assert "delete-parameter" in out
        # Every parameter was attempted: one rejection must not end the run.
        assert len({c.kwargs["Name"] for c in client.put_parameter.call_args_list}) == len(names)

    def test_a_hard_put_failure_is_reported_and_the_loop_continues(self, put_secrets, monkeypatch, capsys):
        names = list(ALL_SSM_SECRET_ENV_VARS)
        for env_var in ALL_SSM_SECRET_ENV_VARS.values():
            monkeypatch.setenv(env_var, "real-value")
        denied = ClientError({"Error": {"Code": "AccessDeniedException"}}, "PutParameter")
        client = _ssm(put_side_effect=[denied] + [None] * (2 * len(names)))
        client.get_parameter.return_value = {"Parameter": {"Type": "String", "Value": SSM_PLACEHOLDER}}
        config = MagicMock()
        config.aws.project_name, config.aws.stage, config.aws.region = "omnisummary", "dev", "ap-northeast-2"
        with patch.object(put_secrets, "Config") as cfg:
            cfg.load.return_value = config
            with patch.object(put_secrets.boto3, "client", return_value=client):
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
