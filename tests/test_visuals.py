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
    # The brief is now returned via Bedrock structured output (with_structured_output),
    # so schema enforcement lives in the VisualBrief model rather than a hand JSON parser.
    def test_rejects_empty_field(self):
        with pytest.raises(ValidationError):
            VisualBrief(title="", caption="c", prompt="p")

    def test_rejects_overlong_prompt(self):
        with pytest.raises(ValidationError):
            VisualBrief(title="t", caption="c", prompt="x" * 5000)

    def test_strips_leaked_markup_from_human_facing_fields(self):
        # Regression (2026-08-17): the structured-output slip trailed the model's own scaffolding
        # into the caption, and `</caption>\n<parameter name="orientation">landscape` was published
        # verbatim on Threads. Tag-like markup must never survive into title/caption.
        brief = VisualBrief(
            title="제목<br>",
            caption='본문 끝.</caption>\n<parameter name="orientation">landscape',
            prompt="draw",
        )
        assert brief.title == "제목"
        # The bled value itself must go too. It used to survive because the bleed check compared
        # against the PARSED orientation (defaulted to 'portrait' here) rather than the field's
        # allowed values, so '본문 끝.\nlandscape' was published verbatim.
        assert brief.caption == "본문 끝."

    def test_strips_a_trailing_bled_orientation_value(self):
        # Observed on a 2026-08-18 local end-to-end run: the caption ended with a bare "\nportrait",
        # the tag-less form of the same structured-output slip that produced 08-17's
        # `</caption><parameter name="orientation">landscape`. The markup strip cannot see it.
        brief = VisualBrief(title="제목", caption="본문 끝이다.\nportrait", prompt="draw", orientation="portrait")
        assert brief.caption == "본문 끝이다."

    def test_drops_a_bled_value_that_is_not_the_chosen_orientation(self):
        # The 08-17 shape: the bleed names a DIFFERENT orientation than the parsed field. Candidates
        # come from the Literal itself (typing.get_args), so no hand-written word list can drift.
        brief = VisualBrief(title="제목", caption="본문 끝이다.\nsquare", prompt="draw", orientation="portrait")
        assert brief.caption == "본문 끝이다."

    def test_warns_only_when_a_value_was_actually_dropped(self):
        with patch("shared.models.logger") as log:
            VisualBrief(title="제목", caption="본문 끝이다.", prompt="draw", orientation="portrait")
        log.warning.assert_not_called()
        with patch("shared.models.logger") as log:
            VisualBrief(title="제목", caption="본문 끝이다.\nlandscape", prompt="draw", orientation="portrait")
        assert log.warning.called

    def test_keeps_the_word_when_it_is_part_of_the_prose(self):
        # Only a standalone FINAL line counts — prose that merely contains the word is untouched.
        brief = VisualBrief(
            title="portrait 모드 비교", caption="세로형 portrait 구도가 더 낫다.", prompt="draw", orientation="portrait"
        )
        assert brief.title == "portrait 모드 비교"
        assert brief.caption == "세로형 portrait 구도가 더 낫다."

    def test_keeps_prose_that_merely_contains_an_angle_bracket(self):
        # The strip must not eat comparisons/inequalities in ordinary prose.
        brief = VisualBrief(title="지연 <2ms", caption="a < b 인 경우", prompt="draw")
        assert brief.title == "지연 <2ms"
        assert brief.caption == "a < b 인 경우"


def _visual_kwargs(**overrides) -> dict:
    """VisualGenerator takes all ten visual knobs explicitly (it keeps NO defaults of its own —
    PipelineConfig is the single source of truth), so tests feed it the same config values
    production does."""
    cfg = PipelineConfig()
    kwargs: dict = {
        "image_model": cfg.image_model,
        "image_sizes": cfg.image_sizes,
        "source_max_tokens": cfg.visual_synopsis_source_max_tokens,
        "context_max_tokens": cfg.visual_synopsis_context_max_tokens,
        "caption_language": cfg.visual_caption_language,
        "on_image_language": cfg.visual_on_image_language,
        "moderation_softening_instruction": cfg.visual_moderation_softening_instruction,
        "style_guidance": cfg.visual_synopsis_style_guidance,
        "humor_guidance": cfg.visual_synopsis_humor_guidance,
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
        out = VisualBrief(title="T", caption="C", prompt="draw X", orientation="landscape")
        factory.get_model.return_value.with_structured_output.return_value = RunnableLambda(lambda _: out)
        factory.truncate_to_tokens.side_effect = lambda text, _: text
        gen = VisualGenerator(factory, LanguageModelId.CLAUDE_V4_6_SONNET, **_visual_kwargs())
        brief = await gen.brief("a 1-page slide", "source text", "context")
        assert brief == out


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

    def test_quality_is_omitted_unless_configured(self, monkeypatch):
        # Sending nothing leaves OpenAI's "auto", which picks between tiers whose per-image prices
        # differ ~4x. The default must stay omitted (no silent cost/quality change), and a configured
        # value must actually reach the API so the bill becomes deterministic.
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        brief = VisualBrief(title="T", caption="C", prompt="draw", orientation="landscape")
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
        brief = VisualBrief(title="T", caption="C", prompt="draw", orientation="landscape")
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=base64.b64encode(b"P").decode())]
        resp.usage = MagicMock(input_tokens=11, output_tokens=2222, total_tokens=2233)
        resp.quality = "high"
        client = MagicMock()
        client.images.generate.return_value = resp

        with patch("agent.visuals.logger") as log:
            with patch("openai.OpenAI", return_value=client):
                _generator().render(brief)
        rendered = [c.args[0] % c.args[1:] for c in log.info.call_args_list if "Rendered" in str(c.args[0])]
        assert rendered, "no render log line"
        assert "output=2222" in rendered[0]
        # Requested -> resolved. With nothing configured we send no `quality` and OpenAI picks a
        # tier itself, so only the RESPONSE says which of the ~4x-apart prices was billed.
        assert "quality=auto->high" in rendered[0]

    def test_render_logs_unreported_when_the_response_omits_quality(self, monkeypatch):
        # Older/other SDK builds return no `quality`; the render must still succeed and the log must
        # say so plainly rather than printing a mock repr as if it were a tier.
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        brief = VisualBrief(title="T", caption="C", prompt="draw", orientation="landscape")
        resp = MagicMock(spec=["data", "usage"])
        resp.data = [MagicMock(b64_json=base64.b64encode(b"P").decode())]
        resp.usage = MagicMock(input_tokens=1, output_tokens=2, total_tokens=3)
        client = MagicMock()
        client.images.generate.return_value = resp

        with patch("agent.visuals.logger") as log:
            with patch("openai.OpenAI", return_value=client):
                assert _generator().render(brief) == b"P"
        rendered = [c.args[0] % c.args[1:] for c in log.info.call_args_list if "Rendered" in str(c.args[0])]
        assert "quality=auto->unreported" in rendered[0]

    def test_usage_summary_tolerates_a_missing_or_reshaped_usage(self):
        from agent.visuals import _usage_summary

        assert _usage_summary(None) == "unknown"
        assert _usage_summary(MagicMock(spec=[])) == "unknown"  # no token fields at all
        assert _usage_summary(MagicMock(spec=["output_tokens"], output_tokens=7)) == "output=7"

    def test_missing_usage_does_not_break_a_render(self, monkeypatch):
        # A usage-shape change in the SDK must never cost the day's image.
        monkeypatch.setenv("OPENAI_API_KEY", "key")
        brief = VisualBrief(title="T", caption="C", prompt="draw")
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
                gen.render(VisualBrief(title="t", caption="c", prompt="draw", orientation="landscape"))
        kwargs = client.images.generate.call_args.kwargs
        assert kwargs["model"] == "custom-model"
        assert kwargs["size"] == "1536x1024"  # orientation -> mapped size
        # The OpenAI client is bounded by config, not the SDK defaults (600s x 2 retries), which
        # could outlive the visual Lambda's 15-min budget.
        client_kwargs = openai_cls.call_args.kwargs
        assert client_kwargs["timeout"] == PipelineConfig().visual_image_timeout_sec
        assert client_kwargs["max_retries"] == PipelineConfig().visual_image_max_retries

    def test_an_unmapped_orientation_warns_and_records_what_was_rendered(self):
        # image_sizes and the orientation vocabulary can only drift via a hand-built generator (the
        # config validator rejects it), and the silent `next(iter(...))` fallback then rendered a
        # shape nobody chose — which the format history recorded as if it had been picked.
        gen = _generator(image_sizes={"square": "1024x1024"})
        brief = VisualBrief(title="t", caption="c", prompt="draw", orientation="landscape")
        resp = MagicMock()
        resp.data = [MagicMock(b64_json=base64.b64encode(b"X").decode())]
        client = MagicMock()
        client.images.generate.return_value = resp
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with patch("openai.OpenAI", return_value=client):
                with patch("agent.visuals.logger") as log:
                    gen.render(brief)
        assert client.images.generate.call_args.kwargs["size"] == "1024x1024"
        assert any("No image size configured" in str(c.args) for c in log.warning.call_args_list)
        assert brief.orientation == "square"  # the history now learns the shape actually produced

    def test_an_empty_size_map_raises_instead_of_rendering_something_arbitrary(self):
        gen = _generator(image_sizes={})
        with patch("agent.visuals.resolve_secret", return_value="key"):
            with pytest.raises(RuntimeError, match="image_sizes"):
                gen.render(VisualBrief(title="t", caption="c", prompt="draw"))

    def test_generator_requires_every_visual_knob(self):
        # No duplicated defaults: a caller that forgets a knob fails loudly instead of silently
        # getting a stale in-code copy (style_aesthetic had already drifted from config).
        factory = MagicMock()
        with pytest.raises(TypeError):
            VisualGenerator(factory, LanguageModelId.CLAUDE_V4_6_SONNET, image_model="gpt-image-2")
