import base64
from unittest.mock import MagicMock, patch

import pytest

from agent.visuals import COMIC_MODE, DIAGRAM_MODE, MODES, VisualGenerator, _parse_json_object


class TestParseJsonObject:
    def test_extracts_embedded_json(self):
        raw = 'Here is the comic:\n```json\n{"title": "T", "panels": []}\n```\nDone.'
        assert _parse_json_object(raw) == {"title": "T", "panels": []}

    def test_raises_without_json(self):
        with pytest.raises(ValueError):
            _parse_json_object("no json here")


class TestImagePrompts:
    def test_comic_prompt_includes_panels_and_captions(self):
        brief = {
            "style": "hand-drawn",
            "panels": [{"visual": "a robot", "caption": "안녕"}, {"visual": "a chip", "caption": "빠름"}],
        }
        prompt = COMIC_MODE.build_image_prompt(brief, 2)
        assert "2-panel" in prompt
        assert "a robot" in prompt
        assert "안녕" in prompt

    def test_diagram_prompt_uses_visual(self):
        brief = {"title": "T", "visual": "boxes connected by arrows showing a pipeline"}
        prompt = DIAGRAM_MODE.build_image_prompt(brief, 1)
        assert "pipeline" in prompt
        assert "diagram" in prompt.lower()


class TestBriefInputs:
    def test_comic_includes_panels(self):
        inputs = COMIC_MODE.brief_inputs("t", "c", 4)
        assert inputs["panels"] == "4"

    def test_diagram_omits_panels(self):
        inputs = DIAGRAM_MODE.brief_inputs("t", "c", 4)
        assert "panels" not in inputs


def _generator():
    factory = MagicMock()
    factory.get_model.return_value = MagicMock()
    from shared.constants import LanguageModelId

    return VisualGenerator(factory, LanguageModelId.CLAUDE_V4_6_SONNET)


class TestVisualGenerator:
    @pytest.mark.asyncio
    async def test_generate_rejects_unknown_mode(self):
        gen = _generator()
        with pytest.raises(ValueError):
            await gen.generate("t", "c", mode="hologram")

    @pytest.mark.asyncio
    async def test_generate_rejects_bad_panel_count(self):
        gen = _generator()
        with pytest.raises(ValueError):
            await gen.generate("t", "c", mode="comic", panels=99)

    @pytest.mark.asyncio
    async def test_generate_comic_pipeline(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        gen = _generator()
        brief = {"title": "테스트", "style": "x", "panels": [{"visual": "v", "caption": "c"}]}

        async def fake_brief(mode, title, content, panels):
            return brief

        fake_img = base64.b64encode(b"PNGDATA").decode()
        openai_resp = MagicMock()
        openai_resp.data = [MagicMock(b64_json=fake_img)]
        fake_client = MagicMock()
        fake_client.images.generate.return_value = openai_resp

        with patch.object(gen, "brief", side_effect=fake_brief):
            with patch("openai.OpenAI", return_value=fake_client):
                image_bytes, out_brief = await gen.generate("t", "c", mode="comic", panels=1)

        assert image_bytes == b"PNGDATA"
        assert out_brief["title"] == "테스트"
        assert fake_client.images.generate.called

    def test_render_requires_api_key(self):
        # Deterministic: resolve_secret returns "" (no env, no SSM) -> render must raise.
        with patch("agent.visuals.resolve_secret", return_value=""):
            with pytest.raises(RuntimeError):
                VisualGenerator.render(COMIC_MODE, {"panels": []}, 1)


class TestModes:
    def test_registry(self):
        assert set(MODES) == {"comic", "diagram"}
        assert MODES["comic"] is COMIC_MODE
        assert MODES["diagram"] is DIAGRAM_MODE
