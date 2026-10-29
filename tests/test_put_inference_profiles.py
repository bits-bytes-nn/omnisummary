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
