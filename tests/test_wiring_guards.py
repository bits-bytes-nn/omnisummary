"""Guards on how the pieces added on 2026-11-08 are wired: each test fails on a mutation a review run found surviving."""

import json
import sys
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aws_cdk import App, Environment
from aws_cdk.assertions import Template
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from application.digest_generator import DigestGenerator
from application.digest_pipeline import PipelineRun
from domain.models import CollectedItem, DigestContent, DigestItem, DigestResult, RankedItem
from shared.config import AWSConfig, Config, PipelineConfig
from shared.constants import SourceType
from tests.test_infrastructure import CONFIG_TEMPLATE


# D1: a label mismatch means the check did not land -> skipped=True
async def test_a_grounding_correction_with_unsent_labels_ships_flagged_unchecked():
    out = json.dumps({"violations": [{"claim": "x", "issue": "y"}], "corrected_digest": "ITEM 7 BODY: z"})
    factory = MagicMock()
    factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=out))
    gen = DigestGenerator(PipelineConfig(), factory)
    content = DigestContent(lead="리드.", items=[DigestItem(title="T", url="u", body="본문.")])
    result, skipped = await gen._verify_grounding(content, [])
    assert skipped and result.lead == "리드."


# D3/D4: generate() must carry the flag onto DigestResult
async def test_generate_carries_grounding_skipped_onto_the_result():
    emitted = json.dumps({"items": [{"title": "T", "url": "u", "body": "본문."}], "lead": "리드."})
    factory = MagicMock()
    factory.get_model.return_value = RunnableLambda(lambda _: AIMessage(content=emitted))
    gen = DigestGenerator(PipelineConfig(countdown_date=""), factory)
    gen._verify_grounding = AsyncMock(side_effect=lambda c, *_a, **_k: (c, True))
    ranked = [RankedItem(item=CollectedItem(item_id="i", source_type=SourceType.RSS, title="T", url="u"), score=0.8)]
    assert (await gen.generate(ranked, today=date(2030, 1, 1))).grounding_skipped is True


# H1: alert only on the skipped digest
def test_only_the_unchecked_digest_is_alerted():
    from entrypoints import digest_handler

    with patch.object(digest_handler, "publish_alert") as alert:
        digest_handler._maybe_alert_grounding(DigestResult(digest_text="x"), date(2026, 11, 8))
        assert alert.call_count == 0
        digest_handler._maybe_alert_grounding(DigestResult(digest_text="x", grounding_skipped=True), date(2026, 11, 8))
        assert alert.call_count == 1


# H2: _run must call the grounding alert with the pipeline's digest
async def test_run_alerts_a_digest_that_shipped_unchecked(monkeypatch):
    from entrypoints import digest_handler
    from tests.test_digest_handler import HealthReport, SourceHealth, SourceStatus, _config

    monkeypatch.setattr(digest_handler, "bootstrap", MagicMock())
    digest = DigestResult(
        digest_text="x",
        grounding_skipped=True,
        content=DigestContent(lead="l", items=[DigestItem(title="t", url="http://e/1", body="b")]),
    )
    items = [MagicMock()]
    health = HealthReport(sources=[SourceHealth(name="rss", item_count=1, status=SourceStatus.OK)])
    with (
        patch("entrypoints.digest_handler.already_published", return_value=False),
        patch("entrypoints.digest_handler.Config.load", return_value=_config()),
        patch("entrypoints.digest_handler.run_collectors_with_health", new=AsyncMock(return_value=(items, health))),
        patch("entrypoints.digest_handler._maybe_alert"),
        patch(
            "entrypoints.digest_handler.run_pipeline",
            new=AsyncMock(return_value=PipelineRun(items, [MagicMock()], digest)),
        ),
        patch("entrypoints.digest_handler.persist_digest"),
        patch("entrypoints.digest_handler._trigger_visual"),
        patch("entrypoints.digest_handler._emit_digest_items_metric"),
        patch.object(digest_handler, "publish_alert") as alert,
    ):
        await digest_handler._run()
    assert [c.args[0] for c in alert.call_args_list] == ["Grounding"]


# H3/H4: a re-run of a posted day reports the STORED story count for THAT date
def test_stored_story_count_reads_the_dates_snapshot(monkeypatch):
    from entrypoints import digest_handler

    store = MagicMock()
    store.get_digest.return_value.content = DigestContent(
        lead="l", items=[DigestItem(title=f"t{i}", url=f"u{i}", body="b") for i in range(5)]
    )
    monkeypatch.setattr(digest_handler, "bootstrap", MagicMock(memory_store=MagicMock(return_value=store)))
    config = MagicMock()
    assert digest_handler._stored_story_count(config, date(2026, 11, 8)) == 5
    store.get_digest.assert_called_once_with("2026-11-08")
    digest_handler.bootstrap.memory_store.assert_called_once_with(config)


# R7: a broken JSON-LD block before the page's own paid entity must not end the scan
def test_a_broken_ld_block_does_not_hide_a_later_paid_declaration():
    from adapters.research_backends import declares_paid

    html = (
        '<script type="application/ld+json">{broken</script>'
        '<script type="application/ld+json">{"@type":"NewsArticle","isAccessibleForFree":false}</script>'
    )
    assert declares_paid(html, "https://news.example/a")


# R9: the entity names the FINAL url after a redirect
async def test_a_paid_page_reached_through_a_redirect_is_caught(monkeypatch):
    import httpx

    import adapters.research_backends as rb

    paid = (
        '<script type="application/ld+json">{"isAccessibleForFree":false,"url":"https://site.example/final"}</script>'
    )

    def answer(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/short":
            return httpx.Response(301, headers={"Location": "https://site.example/final"})
        return httpx.Response(200, text=paid)

    real = httpx.AsyncClient
    monkeypatch.setattr(rb.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(answer), **kw))
    assert await rb.access_controlled(["https://site.example/short"]) == {"https://site.example/short"}


# O2: image_generation_user_error is a moderation block by type, not only by code
def test_image_generation_user_error_type_is_moderation():
    import httpx
    from openai import BadRequestError

    from adapters.openai_images import OpenAIImageRenderer

    resp = httpx.Response(400, request=httpx.Request("POST", "https://api.openai.com/v1/images/edits"))
    exc = BadRequestError("blocked", response=resp, body={"type": "image_generation_user_error", "code": None})
    assert OpenAIImageRenderer.is_moderation_error(exc)


# T2: the gateway hands its config down, or no comic is ever uploaded
async def test_the_gateway_posts_with_its_config_so_the_image_is_hosted():
    import adapters.threads as th

    config = MagicMock()
    config.pipeline.threads_topic_tag = ""
    with (
        patch.object(th, "state_location", return_value=("bucket", "root")),
        patch.object(th, "post_to_threads", new=AsyncMock()) as post,
    ):
        await th.ThreadsGateway(config).post_thread("root", ["r"], image_bytes=b"png")
    assert post.await_args.kwargs["config"] is config and post.await_args.kwargs["image_bucket"] == "bucket"


# M2: the name lookup runs on config.aws.region, not the Bedrock region
def test_the_memory_lookup_session_is_on_the_stacks_region(monkeypatch):
    from adapters import memory

    monkeypatch.delenv("MEMORY_ID", raising=False)
    config = Config.from_yaml(str(CONFIG_TEMPLATE))
    config.aws.region, config.aws.bedrock_region = "ap-northeast-2", "us-west-2"
    with (
        patch("adapters.memory.boto3.Session") as session,
        patch("adapters.memory._deployed_memory_id", return_value=""),
    ):
        memory.create_memory_store(config)
    assert session.call_args.kwargs["region_name"] == "ap-northeast-2"


def _foundation(email: str, budget: float) -> Template:
    from infrastructure.foundation_stack import OmniSummaryFoundationStack

    config = Config.from_yaml(str(CONFIG_TEMPLATE))
    config.aws.monthly_budget_usd = budget
    return (
        Template.from_stack(
            OmniSummaryFoundationStack(
                App(),
                "fnd-k",
                config=config,
                alert_email=email,
                env=Environment(account="123456789012", region=config.aws.region),
            )
        ),
        config,
    )


# F4/F7/F9
def test_spend_guards_need_an_email_and_watch_the_project_tag_at_a_days_budget():
    template, _ = _foundation("", 300)
    template.resource_count_is("AWS::Budgets::Budget", 0)
    template.resource_count_is("AWS::CE::AnomalyMonitor", 0)
    template, config = _foundation("a@example.com", 300)
    monitor = next(iter(template.find_resources("AWS::CE::AnomalyMonitor").values()))
    assert json.loads(monitor["Properties"]["MonitorSpecification"]) == {
        "Tags": {"Key": "Project", "Values": [config.aws.project_name]}
    }
    sub = next(iter(template.find_resources("AWS::CE::AnomalySubscription").values()))
    assert json.loads(sub["Properties"]["ThresholdExpression"])["Dimensions"]["Values"] == ["10.0"]


# S1/S2/S3: every model gets a stage profile AND a <stage>-local profile, tagged with that stage
def test_put_inference_profiles_creates_stage_and_local_profiles(monkeypatch, tmp_path):
    import importlib.util

    from tests.test_put_inference_profiles import _SCRIPT

    spec = importlib.util.spec_from_file_location("pip_kill", _SCRIPT)
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    config = Config.from_yaml(str(CONFIG_TEMPLATE))
    client = MagicMock()
    client.get_paginator.return_value.paginate.return_value = [{"inferenceProfileSummaries": []}]
    client.create_inference_profile.return_value = {"inferenceProfileArn": "arn"}
    session = MagicMock(client=MagicMock(return_value=client))
    monkeypatch.setattr(sys, "argv", ["put_inference_profiles.py"])
    monkeypatch.setattr(script, "CONFIG_PATH", tmp_path)
    monkeypatch.setattr(script.Config, "load", staticmethod(lambda *a, **k: config))
    monkeypatch.setattr(script.boto3, "Session", MagicMock(return_value=session))
    monkeypatch.setattr(script.BedrockCrossRegionModelHelper, "_resolve", staticmethod(lambda *a: "global.m"))
    assert script.main() == 0
    created = {
        c.kwargs["inferenceProfileName"]: {t["key"]: t["value"] for t in c.kwargs["tags"]}["Stage"]
        for c in client.create_inference_profile.call_args_list
    }
    stage = config.aws.stage
    models = script._configured_models(config)
    assert len(created) == 2 * len(models)
    assert sorted(set(created.values())) == sorted({stage, f"{stage}-local"})
    assert all(name.split("-", 1)[1].startswith(f"{tag}-") for name, tag in created.items())


# C1-C4: code defaults
def test_config_defaults():
    from pydantic import ValidationError

    assert AWSConfig().region == "ap-northeast-2"
    assert AWSConfig().monthly_budget_usd == 0.0
    assert PipelineConfig().item_text_max_tokens == 10000
    with pytest.raises(ValidationError):
        AWSConfig(monthly_budget_usd=-1)
