import importlib.util
import sys
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "deploy.py"
_CONFIG = Path(__file__).resolve().parent.parent / "config" / "config-template.yaml"


@pytest.fixture
def deploy(monkeypatch, tmp_path):
    spec = importlib.util.spec_from_file_location("deploy_script", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["deploy_script"] = module
    spec.loader.exec_module(module)
    real_app = module.App
    monkeypatch.setattr(module, "CONFIG_PATH", _CONFIG)
    monkeypatch.setattr(module, "load_dotenv", lambda *_: None)
    monkeypatch.setenv("ALERT_EMAIL", "ops@example.com")
    monkeypatch.setenv("CDK_DEFAULT_ACCOUNT", "123456789012")
    monkeypatch.setattr(module, "App", lambda: real_app(outdir=str(tmp_path)))
    return module


def _stacks(tmp_path) -> set[str]:
    return {p.name.removesuffix(".template.json") for p in tmp_path.glob("*.template.json")}


class TestDigestImageRef:
    def test_without_a_pin_the_deploy_refuses(self, deploy, monkeypatch, tmp_path):
        # Unpinned, every Lambda would reference :latest and a deploy that changed anything else on a
        # function moved it to whatever image was pushed last. Synthesizing the foundation alone instead
        # dropped the exports the live application stack imports, so an update failed.
        monkeypatch.delenv("DIGEST_IMAGE_REF", raising=False)
        monkeypatch.delenv("FOUNDATION_ONLY", raising=False)
        with pytest.raises(SystemExit, match="DIGEST_IMAGE_REF is unset"):
            deploy.main()
        # describe-images --output text prints "None" when the tag is missing.
        monkeypatch.setenv("DIGEST_IMAGE_REF", "None")
        with pytest.raises(SystemExit, match="not a sha256 digest"):
            deploy.main()

    def test_foundation_only_synthesizes_the_foundation_for_a_fresh_account(self, deploy, monkeypatch, tmp_path):
        monkeypatch.delenv("DIGEST_IMAGE_REF", raising=False)
        monkeypatch.setenv("FOUNDATION_ONLY", "1")
        deploy.main()
        assert _stacks(tmp_path) == {"omnisummary-dev-foundation"}

    def test_a_pushed_digest_synthesizes_both_stacks_pinned_to_it(self, deploy, monkeypatch, tmp_path):
        digest = "sha256:" + "ab" * 32
        monkeypatch.setenv("DIGEST_IMAGE_REF", digest)
        monkeypatch.delenv("FOUNDATION_ONLY", raising=False)
        deploy.main()
        assert _stacks(tmp_path) == {"omnisummary-dev-foundation", "omnisummary-dev-application"}
        assert digest in (tmp_path / "omnisummary-dev-application.template.json").read_text()
