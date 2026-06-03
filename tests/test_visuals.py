import base64
from unittest.mock import MagicMock, patch

import pytest

from agent.visuals import VisualGenerator, _parse_json_object
from shared.constants import LanguageModelId


class TestParseJsonObject:
    def test_extracts_embedded_json(self):
        raw = 'Here is the brief:\n```json\n{"title": "T", "prompt": "draw X"}\n```\nDone.'
        assert _parse_json_object(raw) == {"title": "T", "prompt": "draw X"}

    def test_raises_without_json(self):
        with pytest.raises(ValueError):
            _parse_json_object("no json here")


def _generator():
    factory = MagicMock()
    factory.get_model.return_value = MagicMock()
    return VisualGenerator(factory, LanguageModelId.CLAUDE_V4_6_SONNET)


class TestVisualGenerator:
    @pytest.mark.asyncio
    async def test_generate_pipeline(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        gen = _generator()
        brief = {"title": "테스트", "caption": "요약", "prompt": "a one-page slide explaining X"}

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
        assert out_brief["title"] == "테스트"
        assert fake_client.images.generate.called
        # the image prompt sent to OpenAI comes from brief["prompt"]
        assert fake_client.images.generate.call_args.kwargs["prompt"] == "a one-page slide explaining X"

    def test_render_requires_api_key(self):
        with patch("agent.visuals.resolve_secret", return_value=""):
            with pytest.raises(RuntimeError):
                VisualGenerator.render({"prompt": "anything"})

    def test_render_requires_prompt(self, monkeypatch):
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with pytest.raises(ValueError):
                VisualGenerator.render({"title": "no prompt field"})

    def test_render_raises_on_empty_image_data(self):
        with patch("agent.visuals.resolve_secret", return_value="key"):
            resp = MagicMock()
            resp.data = []
            client = MagicMock()
            client.images.generate.return_value = resp
            with patch("openai.OpenAI", return_value=client):
                with pytest.raises(RuntimeError):
                    VisualGenerator.render({"prompt": "draw"})
