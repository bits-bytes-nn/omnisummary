import importlib.util
import sys
from pathlib import Path

import pytest

from shared.constants import LanguageModelId

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "put_inference_profiles.py"
_REGION = "us-west-2"
_ACCOUNT = "123456789012"


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("put_inference_profiles", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["put_inference_profiles"] = module
    spec.loader.exec_module(module)
    return module


def _no_account() -> str:
    raise AssertionError("a foundation-model ARN has no account and must not look one up")


class TestCopyFromArn:
    @pytest.mark.parametrize("model", [LanguageModelId.GPT_V6_1_SOL, LanguageModelId.GPT_V6_1_TERRA])
    def test_a_bare_model_id_becomes_a_foundation_model_arn(self, script, model):
        # The resolver falls back to the bare id when no system-defined profile exists; an
        # inference-profile ARN around a bare id names nothing and CreateInferenceProfile rejects it.
        arn = script._copy_from_arn(model.value, model, _REGION, _no_account)
        assert arn == f"arn:aws:bedrock:{_REGION}::foundation-model/{model.value}"

    @pytest.mark.parametrize("prefix", ["global", "us"])
    def test_a_system_defined_profile_id_becomes_an_inference_profile_arn(self, script, prefix):
        model = LanguageModelId.GPT_V6_1_SOL
        arn = script._copy_from_arn(f"{prefix}.{model.value}", model, _REGION, lambda: _ACCOUNT)
        assert arn == f"arn:aws:bedrock:{_REGION}:{_ACCOUNT}:inference-profile/{prefix}.{model.value}"

    def test_an_arn_is_passed_through(self, script):
        model = LanguageModelId.GPT_V6_1_TERRA
        source = f"arn:aws:bedrock:{_REGION}:{_ACCOUNT}:inference-profile/global.{model.value}"
        assert script._copy_from_arn(source, model, _REGION, _no_account) == source


class TestMain:
    """--delete removes AWS resources; --dry-run must touch none."""

    @pytest.fixture
    def run(self, script, monkeypatch):
        from unittest.mock import MagicMock, patch

        from adapters.bedrock import BedrockCrossRegionModelHelper
        from shared.config import Config

        config = Config()
        config.aws.project_name, config.aws.stage = "omnisummary", "dev"
        monkeypatch.setattr(script, "CONFIG_PATH", _SCRIPT)
        live = BedrockCrossRegionModelHelper.application_profile_name(
            config.pipeline.digest_model, config.aws.project_name, config.aws.stage
        )
        client = MagicMock()
        client.get_paginator.return_value.paginate.return_value = [
            {"inferenceProfileSummaries": [{"inferenceProfileName": live, "inferenceProfileArn": "arn:live"}]}
        ]
        client.create_inference_profile.return_value = {"inferenceProfileArn": "arn:new"}
        session = MagicMock(client=MagicMock(return_value=client))

        def _run(*argv):
            with patch.object(script.Config, "load", return_value=config):
                with patch.object(script.boto3, "Session", return_value=session):
                    with patch.object(BedrockCrossRegionModelHelper, "_resolve", return_value="global.model"):
                        with patch.object(sys, "argv", ["put_inference_profiles.py", *argv]):
                            assert script.main() == 0
            return client

        return _run

    def test_delete_removes_only_the_existing_profiles_by_arn(self, run):
        client = run("--delete")
        client.delete_inference_profile.assert_called_once_with(inferenceProfileIdentifier="arn:live")
        client.create_inference_profile.assert_not_called()

    def test_delete_dry_run_deletes_nothing(self, run):
        client = run("--delete", "--dry-run")
        client.delete_inference_profile.assert_not_called()
        client.create_inference_profile.assert_not_called()

    def test_dry_run_creates_nothing(self, run):
        client = run("--dry-run")
        client.create_inference_profile.assert_not_called()
        client.delete_inference_profile.assert_not_called()

    def test_a_run_creates_the_missing_profiles_tagged_with_their_stage(self, run):
        client = run()
        stages = {
            next(t["value"] for t in c.kwargs["tags"] if t["key"] == "Stage")
            for c in client.create_inference_profile.call_args_list
        }
        assert stages == {"dev", "dev-local"}
