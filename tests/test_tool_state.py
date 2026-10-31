from domain.digest_snapshot import DigestStateManager
from domain.models import CollectedItem, DigestContent, DigestItem, DigestResult, RankedItem
from shared.constants import SourceType


def _item(item_id: str) -> CollectedItem:
    return CollectedItem(
        item_id=item_id, source_type=SourceType.RSS, title=f"t-{item_id}", url=f"http://e.com/{item_id}", text="body"
    )


def test_export_state_carries_only_the_ranked_items_and_the_digest():
    # Each RankedItem embeds its CollectedItem, so a separate collected-items map stored every
    # ranked item twice — and was what overflowed the AgentCore 100k-char event limit.
    mgr = DigestStateManager()
    ranked = [RankedItem(item=_item("i0"), score=0.9), RankedItem(item=_item("i1"), score=0.8)]
    mgr.store_digest(ranked, DigestResult(digest_text="hi", ranked_items=ranked))

    state = mgr.export_state()

    assert set(state) == {"ranked_items", "digest_result"}
    assert len(state["ranked_items"]) == 2
    assert state["digest_result"]["digest_text"] == "hi"


def test_export_state_roundtrips_through_load():
    mgr = DigestStateManager()
    ranked = [RankedItem(item=_item("a"), score=0.7)]
    mgr.store_digest(ranked, DigestResult(digest_text="d", ranked_items=ranked))

    restored = DigestStateManager.load_from_dict(mgr.export_state())

    ranked_back = restored.get_ranked_items()
    assert [ri.item.item_id for ri in ranked_back] == ["a"]
    assert ranked_back[0].score == 0.7  # the score survives, not just the item
    assert ranked_back[0].item.title == "t-a"


def test_a_snapshot_stored_with_collected_items_still_loads():
    # Snapshots written before the duplicate map was dropped stay readable (90-day memory expiry).
    mgr = DigestStateManager()
    ranked = [RankedItem(item=_item("a"), score=0.7)]
    mgr.store_digest(ranked, DigestResult(digest_text="d", ranked_items=ranked))
    legacy = {**mgr.export_state(), "collected_items": {"a": _item("a").model_dump(mode="json")}}

    assert [ri.item.item_id for ri in DigestStateManager.load_from_dict(legacy).get_ranked_items()] == ["a"]


def test_export_state_does_not_double_store_ranked_items():
    # The embedded digest_result must NOT re-embed the ranked list (it's stored at the top
    # level) — re-embedding doubled the snapshot and tripped the AgentCore 100k cap.
    mgr = DigestStateManager()
    ranked = [RankedItem(item=_item("a"), score=0.7)]
    content = DigestContent(lead="lead", items=[DigestItem(title="t", url="http://e.com/a", body="b")])
    mgr.store_digest(ranked, DigestResult(digest_text="d", ranked_items=ranked, content=content))

    state = mgr.export_state()
    assert state["digest_result"].get("ranked_items") in (None, [])  # excluded from the embed
    assert state["digest_result"]["content"]["lead"] == "lead"  # but content survives

    restored = DigestStateManager.load_from_dict(state)
    assert restored.get_content().lead == "lead"  # content round-trips
    assert len(restored.get_ranked_items()) == 1  # ranked rebuilt from top-level list
