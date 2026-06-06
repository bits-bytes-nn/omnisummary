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

    @pytest.mark.asyncio
    async def test_generate_retries_softened_on_moderation_block(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        gen = _generator()
        briefs = [
            VisualBrief(title="t1", caption="c1", prompt="edgy prompt"),
            VisualBrief(title="t2", caption="c2", prompt="softened prompt"),
        ]
        instructions: list[str] = []

        async def fake_brief(instruction, source, context=""):
            instructions.append(instruction)
            return briefs[len(instructions) - 1]

        fake_img = base64.b64encode(b"OK").decode()
        ok_resp = MagicMock()
        ok_resp.data = [MagicMock(b64_json=fake_img)]
        client = MagicMock()
        # first render raises moderation, second succeeds
        client.images.generate.side_effect = [
            Exception("Your request was rejected by the safety system: moderation_blocked"),
            ok_resp,
        ]

        with patch.object(gen, "brief", side_effect=fake_brief):
            with patch("openai.OpenAI", return_value=client):
                image_bytes, out_brief = await gen.generate("draw it", "src", "")

        assert image_bytes == b"OK"
        assert out_brief.title == "t2"  # the softened-retry brief was used
        assert len(instructions) == 2
        assert "safe-for-work" in instructions[1]

    @pytest.mark.asyncio
    async def test_generate_reraises_non_moderation_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        gen = _generator()

        async def fake_brief(instruction, source, context=""):
            return VisualBrief(title="t", caption="c", prompt="p")

        client = MagicMock()
        client.images.generate.side_effect = RuntimeError("network down")
        with patch.object(gen, "brief", side_effect=fake_brief):
            with patch("openai.OpenAI", return_value=client):
                with pytest.raises(RuntimeError, match="network down"):
                    await gen.generate("draw", "src", "")

    def test_render_requires_api_key(self):
        gen = _generator()
        with patch("agent.visuals.resolve_secret", return_value=""):
            with pytest.raises(RuntimeError):
                gen.render(VisualBrief(title="t", caption="c", prompt="anything"))

    def test_render_requires_prompt(self):
        gen = _generator()
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with pytest.raises(ValueError):
                gen.render(VisualBrief(title="t", caption="c", prompt="x").model_copy(update={"prompt": ""}))

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

    def test_is_moderation_error_string_fallback(self):
        assert VisualGenerator._is_moderation_error(Exception("... moderation_blocked ..."))
        assert VisualGenerator._is_moderation_error(Exception("rejected by the safety system"))
        assert not VisualGenerator._is_moderation_error(RuntimeError("network down"))

    def test_is_moderation_error_typed_openai_exception(self):
        from openai import BadRequestError

        exc = BadRequestError.__new__(BadRequestError)
        exc.body = {"code": "moderation_blocked", "type": "image_generation_user_error"}
        assert VisualGenerator._is_moderation_error(exc)

        other = BadRequestError.__new__(BadRequestError)
        other.body = {"code": "invalid_request", "type": "invalid_request_error"}
        assert not VisualGenerator._is_moderation_error(other)

    def test_render_uses_configured_model_and_orientation_size(self):
        factory = MagicMock()
        factory.get_model.return_value = MagicMock()
        gen = VisualGenerator(
            factory,
            LanguageModelId.CLAUDE_V4_6_SONNET,
            image_model="custom-model",
            image_sizes={"square": "1024x1024", "landscape": "1536x1024", "portrait": "1024x1536"},
        )
        fake_img = base64.b64encode(b"X").decode()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=fake_img)]
        client = MagicMock()
        client.images.generate.return_value = resp
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with patch("openai.OpenAI", return_value=client):
                gen.render(VisualBrief(title="t", caption="c", prompt="draw", orientation="landscape"))
        kwargs = client.images.generate.call_args.kwargs
        assert kwargs["model"] == "custom-model"
        assert kwargs["size"] == "1536x1024"  # orientation -> mapped size
