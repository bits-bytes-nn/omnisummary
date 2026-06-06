from agent.tool_state import DigestStateManager
from shared.constants import SourceType
from shared.models import CollectedItem, DigestContent, DigestItem, DigestResult, RankedItem


def _item(item_id: str) -> CollectedItem:
    return CollectedItem(
        item_id=item_id, source_type=SourceType.RSS, title=f"t-{item_id}", url=f"http://e.com/{item_id}", text="body"
    )


def test_export_state_keeps_only_ranked_collected_items():
    mgr = DigestStateManager()
    collected = [_item(f"i{n}") for n in range(10)]
    ranked = [RankedItem(item=collected[0], score=0.9), RankedItem(item=collected[1], score=0.8)]
    digest = DigestResult(digest_text="hi", ranked_items=ranked)
    mgr.store_digest(collected, ranked, digest)

    state = mgr.export_state()

    # all 10 collected, but only the 2 ranked-referenced ones are exported
    assert set(state["collected_items"].keys()) == {"i0", "i1"}
    assert len(state["ranked_items"]) == 2
    assert state["digest_result"]["digest_text"] == "hi"


def test_export_state_roundtrips_through_load():
    mgr = DigestStateManager()
    collected = [_item("a"), _item("b"), _item("c")]
    ranked = [RankedItem(item=collected[0], score=0.7)]
    mgr.store_digest(collected, ranked, DigestResult(digest_text="d", ranked_items=ranked))

    restored = DigestStateManager.load_from_dict(mgr.export_state())

    assert restored.get_item_count() == 1
    assert restored.get_item_by_number(1).item.item_id == "a"
    assert restored.get_item("a") is not None
    assert restored.get_item("b") is None  # trimmed out


def test_export_state_does_not_double_store_ranked_items():
    # The embedded digest_result must NOT re-embed the ranked list (it's stored at the top
    # level) — re-embedding doubled the snapshot and tripped the AgentCore 100k cap.
    mgr = DigestStateManager()
    collected = [_item("a")]
    ranked = [RankedItem(item=collected[0], score=0.7)]
    content = DigestContent(
        lead="lead", headline_index=1, items=[DigestItem(title="t", url="http://e.com/a", body="b")]
    )
    mgr.store_digest(collected, ranked, DigestResult(digest_text="d", ranked_items=ranked, content=content))

    state = mgr.export_state()
    assert state["digest_result"].get("ranked_items") in (None, [])  # excluded from the embed
    assert state["digest_result"]["content"]["headline_index"] == 1  # but content survives

    restored = DigestStateManager.load_from_dict(state)
    assert restored.get_content().lead == "lead"  # content round-trips
    assert restored.get_item_count() == 1  # ranked rebuilt from top-level list
