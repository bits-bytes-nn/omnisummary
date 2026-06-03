import base64
from unittest.mock import MagicMock, patch

import pytest

from agent.visuals import VisualGenerator, _parse_brief
from shared.constants import LanguageModelId
from shared.models import VisualBrief


class TestParseBrief:
    def test_extracts_embedded_json(self):
        raw = 'Here is the brief:\n```json\n{"title": "T", "caption": "C", "prompt": "draw X"}\n```\nDone.'
        brief = _parse_brief(raw)
        assert brief == VisualBrief(title="T", caption="C", prompt="draw X")

    def test_raises_without_json(self):
        with pytest.raises(ValueError):
            _parse_brief("no json here")

    def test_raises_on_missing_required_field(self):
        # A malformed brief missing required fields must surface as a ValueError, not
        # render garbage via silent .get() defaults.
        with pytest.raises(ValueError):
            _parse_brief('{"title": "only title"}')

    def test_raises_on_empty_field(self):
        # Empty title/caption/prompt are rejected at parse time so a runaway LLM parse
        # can't produce a garbage brief that only fails downstream at image generation.
        with pytest.raises(ValueError):
            _parse_brief('{"title": "", "caption": "c", "prompt": "p"}')

    def test_raises_on_overlong_prompt(self):
        long_prompt = "x" * 5000
        with pytest.raises(ValueError):
            _parse_brief(f'{{"title": "t", "caption": "c", "prompt": "{long_prompt}"}}')


def _generator() -> VisualGenerator:
    factory = MagicMock()
    factory.get_model.return_value = MagicMock()
    return VisualGenerator(factory, LanguageModelId.CLAUDE_V4_6_SONNET)


class TestVisualGenerator:
    @pytest.mark.asyncio
    async def test_generate_pipeline(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        gen = _generator()
        brief = VisualBrief(title="테스트", caption="요약", prompt="a one-page slide explaining X")

        async def fake_brief(instruction, source, context=""):
            return brief

        fake_img = base64.b64encode(b"PNGDATA").decode()
        openai_resp = MagicMock()
        openai_resp.data = [MagicMock(b64_json=fake_img)]
        fake_client = MagicMock()
        fake_client.images.generate.return_value = openai_resp

        with patch.object(gen, "brief", side_effect=fake_brief):
            with patch("openai.OpenAI", return_value=fake_client):
                image_bytes, out_brief = await gen.generate("a 1-page slide", "source text", "context")

        assert image_bytes == b"PNGDATA"
        assert out_brief.title == "테스트"
        assert fake_client.images.generate.called
        # the image prompt sent to OpenAI comes from brief.prompt
        assert fake_client.images.generate.call_args.kwargs["prompt"] == "a one-page slide explaining X"

    def test_render_requires_api_key(self):
        gen = _generator()
        with patch("agent.visuals.resolve_secret", return_value=""):
            with pytest.raises(RuntimeError):
                gen.render(VisualBrief(title="t", caption="c", prompt="anything"))

    def test_render_requires_prompt(self):
        gen = _generator()
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with pytest.raises(ValueError):
                gen.render(VisualBrief(title="t", caption="c", prompt=""))

    def test_render_raises_on_empty_image_data(self):
        gen = _generator()
        with patch("agent.visuals.resolve_secret", return_value="key"):
            resp = MagicMock()
            resp.data = []
            client = MagicMock()
            client.images.generate.return_value = resp
            with patch("openai.OpenAI", return_value=client):
                with pytest.raises(RuntimeError):
                    gen.render(VisualBrief(title="t", caption="c", prompt="draw"))

    def test_render_uses_configured_model_and_size(self):
        factory = MagicMock()
        factory.get_model.return_value = MagicMock()
        gen = VisualGenerator(
            factory, LanguageModelId.CLAUDE_V4_6_SONNET, image_model="custom-model", image_size="512x512"
        )
        fake_img = base64.b64encode(b"X").decode()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=fake_img)]
        client = MagicMock()
        client.images.generate.return_value = resp
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with patch("openai.OpenAI", return_value=client):
                gen.render(VisualBrief(title="t", caption="c", prompt="draw"))
        kwargs = client.images.generate.call_args.kwargs
        assert kwargs["model"] == "custom-model"
        assert kwargs["size"] == "512x512"
