import base64
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.runnables import RunnableLambda
from pydantic import ValidationError

from agent.visuals import VisualGenerator
from shared.config import PipelineConfig
from shared.constants import LanguageModelId
from shared.models import VisualBrief


class TestVisualBriefValidation:
    # The brief is returned via Bedrock structured output (with_structured_output),
    # so schema enforcement lives in the VisualBrief model rather than a hand JSON parser.
    def test_rejects_empty_field(self):
        with pytest.raises(ValidationError):
            VisualBrief(title="", prompt="p", orientation="portrait")

    def test_a_long_four_panel_brief_is_accepted(self):
        # A 2026-10-11 four-panel brief ran past the old 4000-char cap (dall-e-3's) and the day
        # published text-only. The bound is the Images API limit for the GPT image models.
        assert VisualBrief(title="t", prompt="x" * 5000, orientation="portrait")
        with pytest.raises(ValidationError):
            VisualBrief(title="t", prompt="x" * 32001, orientation="portrait")

    def test_orientation_is_required(self):
        # A default let a brief that omitted the field silently become portrait.
        with pytest.raises(ValidationError):
            VisualBrief(title="t", prompt="p")


def _visual_kwargs(**overrides) -> dict:
    """VisualGenerator takes all ten visual knobs explicitly (it keeps NO defaults of its own —
    PipelineConfig is the single source of truth), so tests feed it the same config values
    production does."""
    cfg = PipelineConfig()
    kwargs: dict = {
        "image_model": cfg.image_model,
        "image_sizes": cfg.image_sizes,
        "source_max_tokens": cfg.visual_synopsis_source_max_tokens,
        "on_image_language": cfg.visual_on_image_language,
        "moderation_softening_instruction": cfg.visual_moderation_softening_instruction,
        "style_guidance": cfg.visual_synopsis_style_guidance,
        "style_aesthetic": cfg.visual_synopsis_style_aesthetic,
        "image_timeout_sec": cfg.visual_image_timeout_sec,
        "image_max_retries": cfg.visual_image_max_retries,
        "image_quality": cfg.visual_image_quality,
    }
    kwargs.update(overrides)
    return kwargs


def _generator(**overrides) -> VisualGenerator:
    factory = MagicMock()
    factory.get_model.return_value.with_structured_output.return_value = MagicMock()
    return VisualGenerator(factory, LanguageModelId.CLAUDE_V4_6_SONNET, **_visual_kwargs(**overrides))


class TestBrief:
    @pytest.mark.asyncio
    async def test_returns_structured_brief(self):
        # with_structured_output yields a validated VisualBrief; brief() returns it as-is,
        # with no text-JSON parsing in between.
        factory = MagicMock()
        out = VisualBrief(title="T", prompt="draw X", orientation="landscape")
        factory.get_model.return_value.with_structured_output.return_value = RunnableLambda(lambda _: out)
        factory.truncate_to_tokens.side_effect = lambda text, _: text
        gen = VisualGenerator(factory, LanguageModelId.CLAUDE_V4_6_SONNET, **_visual_kwargs())
        brief = await gen.brief("a 1-page slide", "source text")
        assert brief == out

    @pytest.mark.asyncio
    async def test_a_reply_without_the_tool_call_is_a_named_failure(self):
        # Models that reject forced tool_choice get the schema tool under "auto"; a text-only reply
        # parses to None, which must not surface as an opaque pydantic error.
        factory = MagicMock()
        factory.get_model.return_value.with_structured_output.return_value = RunnableLambda(lambda _: None)
        factory.truncate_to_tokens.side_effect = lambda text, _: text
        gen = VisualGenerator(factory, LanguageModelId.CLAUDE_V5_5_SONNET, **_visual_kwargs())
        with pytest.raises(ValueError, match="no VisualBrief tool call"):
            await gen.brief("a 1-page slide", "source text")


class TestVisualGenerator:
    @pytest.mark.asyncio
    async def test_generate_pipeline(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        gen = _generator()
        brief = VisualBrief(title="테스트", prompt="a one-page slide explaining X", orientation="portrait")

        async def fake_brief(instruction, source):
            return brief

        fake_img = base64.b64encode(b"PNGDATA").decode()
        openai_resp = MagicMock()
        openai_resp.data = [MagicMock(b64_json=fake_img)]
        fake_client = MagicMock()
        fake_client.images.generate.return_value = openai_resp

        with patch.object(gen, "brief", side_effect=fake_brief):
            with patch("openai.OpenAI", return_value=fake_client):
                image_bytes, out_brief = await gen.generate("a 1-page slide", "source text")

        assert image_bytes == b"PNGDATA"
        assert out_brief.title == "테스트"
        assert fake_client.images.generate.called
        # the image prompt sent to OpenAI comes from brief.prompt
        assert fake_client.images.generate.call_args.kwargs["prompt"] == "a one-page slide explaining X"

    def test_quality_is_omitted_unless_configured(self, monkeypatch):
        # Sending nothing leaves OpenAI's "auto", which picks between tiers whose per-image prices
        # differ ~4x. The default must stay omitted (no silent cost/quality change), and a configured
        # value must actually reach the API so the bill becomes deterministic.
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        brief = VisualBrief(title="T", prompt="draw", orientation="landscape")
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=base64.b64encode(b"P").decode())]
        resp.usage = MagicMock(input_tokens=11, output_tokens=2222, total_tokens=2233)

        for configured, expected in (("", None), ("medium", "medium")):
            client = MagicMock()
            client.images.generate.return_value = resp
            gen = _generator(image_quality=configured)
            with patch("openai.OpenAI", return_value=client):
                gen.render(brief)
            assert client.images.generate.call_args.kwargs.get("quality") == expected

    def test_render_reports_the_billed_token_counts(self, monkeypatch):
        # The response carries the tokens the image is billed on and they were discarded, leaving
        # spend as an estimate (published per-image price x a log count). Asserted on the logger
        # call rather than captured output: the project logger sets propagate=False and binds its
        # StreamHandler to the real stderr at import, so neither caplog nor capsys/capfd see it.
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        brief = VisualBrief(title="T", prompt="draw", orientation="landscape")
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=base64.b64encode(b"P").decode())]
        resp.usage = MagicMock(input_tokens=11, output_tokens=2222, total_tokens=2233)
        resp.quality = "high"
        client = MagicMock()
        client.images.generate.return_value = resp

        with patch("agent.visuals.logger") as log:
            with patch("openai.OpenAI", return_value=client):
                _generator(image_quality="").render(brief)
        rendered = [m for c in log.info.call_args_list if "Rendered" in (m := c.args[0] % c.args[1:])]
        assert rendered, "no render log line"
        assert "output=2222" in rendered[0]
        # Requested -> resolved. With the tier left EMPTY we send no `quality` and OpenAI picks one
        # itself, so only the RESPONSE says which of the ~4x-apart prices was billed. config.yaml
        # pins the tier for exactly that reason; this covers the explicit opt-out.
        assert "quality=auto->high" in rendered[0]

    def test_render_logs_unreported_when_the_response_omits_quality(self, monkeypatch):
        # Older/other SDK builds return no `quality`; the render must still succeed and the log must
        # say so plainly rather than printing a mock repr as if it were a tier.
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        brief = VisualBrief(title="T", prompt="draw", orientation="landscape")
        resp = MagicMock(spec=["data", "usage"])
        resp.data = [MagicMock(b64_json=base64.b64encode(b"P").decode())]
        resp.usage = MagicMock(input_tokens=1, output_tokens=2, total_tokens=3)
        client = MagicMock()
        client.images.generate.return_value = resp

        with patch("agent.visuals.logger") as log:
            with patch("openai.OpenAI", return_value=client):
                assert _generator(image_quality="").render(brief) == b"P"
        rendered = [m for c in log.info.call_args_list if "Rendered" in (m := c.args[0] % c.args[1:])]
        assert "quality=auto->unreported" in rendered[0]

    def test_a_pinned_tier_is_sent_and_logged(self, monkeypatch):
        # Left to OpenAI's "auto" the render log could only report `quality=auto->unreported`, so
        # nothing could say which of the ~4x-apart prices was bought — or how legible the on-image
        # labels would be.
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        brief = VisualBrief(title="T", prompt="draw", orientation="landscape")
        resp = MagicMock(spec=["data", "usage"])
        resp.data = [MagicMock(b64_json=base64.b64encode(b"P").decode())]
        resp.usage = MagicMock(input_tokens=1, output_tokens=2, total_tokens=3)
        client = MagicMock()
        client.images.generate.return_value = resp

        with patch("agent.visuals.logger") as log:
            with patch("openai.OpenAI", return_value=client):
                assert _generator(image_quality="high").render(brief) == b"P"
        assert client.images.generate.call_args.kwargs["quality"] == "high"
        rendered = [m for c in log.info.call_args_list if "Rendered" in (m := c.args[0] % c.args[1:])]
        assert "quality=high->" in rendered[0]

    def test_usage_summary_tolerates_a_missing_or_reshaped_usage(self):
        from agent.visuals import _usage_summary

        assert _usage_summary(None) == "unknown"
        assert _usage_summary(MagicMock(spec=[])) == "unknown"  # no token fields at all
        assert _usage_summary(MagicMock(spec=["output_tokens"], output_tokens=7)) == "output=7"

    def test_missing_usage_does_not_break_a_render(self, monkeypatch):
        # A usage-shape change in the SDK must never cost the day's image.
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        brief = VisualBrief(title="T", prompt="draw", orientation="portrait")
        resp = MagicMock(spec=["data"])
        resp.data = [MagicMock(b64_json=base64.b64encode(b"P").decode())]
        client = MagicMock()
        client.images.generate.return_value = resp
        with patch("openai.OpenAI", return_value=client):
            assert _generator().render(brief) == b"P"

    @pytest.mark.asyncio
    async def test_generate_retries_softened_on_moderation_block(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        gen = _generator()
        briefs = [
            VisualBrief(title="t1", prompt="edgy prompt", orientation="portrait"),
            VisualBrief(title="t2", prompt="softened prompt", orientation="portrait"),
        ]
        instructions: list[str] = []

        async def fake_brief(instruction, source):
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
                image_bytes, out_brief = await gen.generate("draw it", "src")

        assert image_bytes == b"OK"
        assert out_brief.title == "t2"  # the softened-retry brief was used
        assert len(instructions) == 2
        assert "safe-for-work" in instructions[1]

    @pytest.mark.asyncio
    async def test_generate_reraises_non_moderation_error(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        gen = _generator()

        async def fake_brief(instruction, source):
            return VisualBrief(title="t", prompt="p", orientation="portrait")

        client = MagicMock()
        client.images.generate.side_effect = RuntimeError("network down")
        with patch.object(gen, "brief", side_effect=fake_brief):
            with patch("openai.OpenAI", return_value=client):
                with pytest.raises(RuntimeError, match="network down"):
                    await gen.generate("draw", "src")

    def test_render_requires_api_key(self):
        gen = _generator()
        with patch("agent.visuals.resolve_secret", return_value=""):
            with pytest.raises(RuntimeError):
                gen.render(VisualBrief(title="t", prompt="anything", orientation="portrait"))

    def test_render_requires_prompt(self):
        gen = _generator()
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with pytest.raises(ValueError):
                gen.render(VisualBrief(title="t", prompt="x", orientation="portrait").model_copy(update={"prompt": ""}))

    def test_render_raises_on_empty_image_data(self):
        gen = _generator()
        with patch("agent.visuals.resolve_secret", return_value="key"):
            resp = MagicMock()
            resp.data = []
            client = MagicMock()
            client.images.generate.return_value = resp
            with patch("openai.OpenAI", return_value=client):
                with pytest.raises(RuntimeError):
                    gen.render(VisualBrief(title="t", prompt="draw", orientation="portrait"))

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
        gen = _generator(
            image_model="custom-model",
            image_sizes={"square": "1024x1024", "landscape": "1536x1024", "portrait": "1024x1536"},
        )
        fake_img = base64.b64encode(b"X").decode()
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=fake_img)]
        client = MagicMock()
        client.images.generate.return_value = resp
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with patch("openai.OpenAI", return_value=client) as openai_cls:
                gen.render(VisualBrief(title="t", prompt="draw", orientation="landscape"))
        kwargs = client.images.generate.call_args.kwargs
        assert kwargs["model"] == "custom-model"
        assert kwargs["size"] == "1536x1024"  # orientation -> mapped size
        # The OpenAI client is bounded by config, not the SDK defaults (600s x 2 retries), which
        # could outlive the visual Lambda's 15-min budget.
        client_kwargs = openai_cls.call_args.kwargs
        assert client_kwargs["timeout"] == PipelineConfig().visual_image_timeout_sec
        assert client_kwargs["max_retries"] == PipelineConfig().visual_image_max_retries

    def test_an_unmapped_orientation_raises_instead_of_rendering_a_shape_nobody_chose(self):
        # Unreachable through the shipped path: VisualBrief.orientation is a Literal and
        # PipelineConfig pins image_sizes' keys to the same vocabulary. It used to pick an arbitrary
        # configured size and rewrite the brief to match, so the format history learned a shape that
        # was never asked for.
        gen = _generator(image_sizes={"square": "1024x1024"})
        brief = VisualBrief(title="t", prompt="p", orientation="portrait")
        with pytest.raises(RuntimeError, match="No image size configured"):
            gen._resolve_size(brief)

    def test_an_empty_size_map_raises_instead_of_rendering_something_arbitrary(self):
        gen = _generator(image_sizes={})
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with pytest.raises(RuntimeError, match="No image size configured"):
                gen.render(VisualBrief(title="t", prompt="draw", orientation="portrait"))

    def test_generator_requires_every_visual_knob(self):
        # No duplicated defaults: a caller that forgets a knob fails loudly instead of silently
        # getting a stale in-code copy (style_aesthetic had already drifted from config).
        factory = MagicMock()
        with pytest.raises(TypeError):
            VisualGenerator(factory, LanguageModelId.CLAUDE_V4_6_SONNET, image_model="gpt-image-2")


def _image_response(data: bytes = b"IMG") -> MagicMock:
    resp = MagicMock()
    resp.data = [MagicMock(b64_json=base64.b64encode(data).decode())]
    return resp


class TestReferenceRender:
    def test_a_reference_sheet_turns_the_render_into_an_edit_with_the_note(self):
        from agent.visuals import REFERENCE_NOTE

        gen = _generator(image_quality="xhigh")
        client = MagicMock()
        client.images.edit.return_value = _image_response(b"COMIC")
        brief = VisualBrief(title="t", prompt="draw the strip", orientation="tall")
        with patch.object(gen, "_images_client", return_value=client):
            assert gen.render(brief, [b"SHEET"]) == b"COMIC"
        kwargs = client.images.edit.call_args.kwargs
        assert kwargs["prompt"] == REFERENCE_NOTE + "draw the strip"
        assert kwargs["image"][0][1] == b"SHEET"
        assert kwargs["size"] == PipelineConfig().image_sizes["tall"]
        client.images.generate.assert_not_called()

    def test_without_a_reference_it_is_a_plain_generate(self):
        gen = _generator()
        client = MagicMock()
        client.images.generate.return_value = _image_response()
        with patch.object(gen, "_images_client", return_value=client):
            gen.render(VisualBrief(title="t", prompt="p", orientation="square"))
        client.images.edit.assert_not_called()


class TestReviewCopy:
    def test_the_review_sees_a_full_resolution_jpeg_under_the_bedrock_image_cap(self):
        import io
        import os

        from PIL import Image

        from agent.visuals import _review_copy

        buffer = io.BytesIO()
        Image.frombytes("RGB", (848, 2544), os.urandom(848 * 2544 * 3)).save(buffer, format="PNG")
        png = buffer.getvalue()
        assert len(png) > 3_750_000
        copy = _review_copy(png)
        assert copy[:2] == b"\xff\xd8"
        assert len(copy) < 3_750_000
        with Image.open(io.BytesIO(copy)) as img:
            assert img.size == (848, 2544)


class TestReviewedRender:
    BRIEF = VisualBrief(title="t", prompt="p", orientation="tall")

    @pytest.fixture(autouse=True)
    def _raw_review_copy(self):
        with patch("agent.visuals._review_copy", side_effect=lambda image: image):
            yield

    @staticmethod
    def _gen(reviews: list) -> VisualGenerator:
        from unittest.mock import AsyncMock

        from shared.models import VisualReview

        gen = _generator()
        gen.reviewer = MagicMock()
        gen.reviewer.ainvoke = AsyncMock(side_effect=[VisualReview(issues=r) for r in reviews])
        return gen

    @pytest.mark.asyncio
    async def test_a_clean_render_is_kept_without_an_edit(self):
        gen = self._gen([[]])
        with patch.object(gen, "fix") as fix:
            assert await gen._reviewed(b"ORIG", self.BRIEF, None) == b"ORIG"
        fix.assert_not_called()
        image_block = gen.reviewer.ainvoke.call_args.args[0][1].content[0]
        assert image_block["mime_type"] == "image/jpeg"

    @pytest.mark.asyncio
    async def test_the_correction_is_kept_when_it_leaves_fewer_defects(self):
        gen = self._gen([["Panel 2: a sign floats"], []])
        with patch.object(gen, "fix", return_value=b"FIXED") as fix:
            assert await gen._reviewed(b"ORIG", self.BRIEF, None) == b"FIXED"
        assert fix.call_args.args[2] == ["Panel 2: a sign floats"]

    @pytest.mark.asyncio
    async def test_the_edit_targets_the_top_three_but_full_lists_decide(self):
        many = [f"Panel {i}: tag garbled" for i in range(1, 6)]
        gen = self._gen([many, many[:4]])
        with patch.object(gen, "fix", return_value=b"FIXED") as fix:
            assert await gen._reviewed(b"ORIG", self.BRIEF, None) == b"FIXED"
        assert fix.call_args.args[2] == many[:3]

    @pytest.mark.asyncio
    async def test_the_original_is_kept_when_the_correction_is_no_better(self):
        gen = self._gen([["Panel 1: x"], ["Panel 3: y"]])
        with patch.object(gen, "fix", return_value=b"FIXED"):
            assert await gen._reviewed(b"ORIG", self.BRIEF, None) == b"ORIG"

    @pytest.mark.asyncio
    async def test_a_failed_review_never_costs_the_visual(self):
        from unittest.mock import AsyncMock

        gen = _generator()
        gen.reviewer = MagicMock()
        gen.reviewer.ainvoke = AsyncMock(side_effect=RuntimeError("bedrock down"))
        with patch.object(gen, "fix") as fix:
            assert await gen._reviewed(b"ORIG", self.BRIEF, None) == b"ORIG"
        fix.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_correction_when_the_deadline_cannot_fit_another_render(self):
        import time

        gen = self._gen([["Panel 1: x"]])
        with patch.object(gen, "fix") as fix:
            assert await gen._reviewed(b"ORIG", self.BRIEF, time.monotonic() + 1) == b"ORIG"
        fix.assert_not_called()
