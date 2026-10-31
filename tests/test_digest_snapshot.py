from domain.digest_snapshot import DigestSnapshot
from domain.models import CollectedItem, DigestContent, DigestItem, DigestResult, RankedItem
from shared.constants import SourceType


def _item(item_id: str) -> CollectedItem:
    return CollectedItem(
        item_id=item_id, source_type=SourceType.RSS, title=f"T{item_id}", url=f"https://e.com/{item_id}"
    )


def test_the_snapshot_carries_only_the_ranked_items_and_the_digest():
    ranked = [RankedItem(item=_item("a"), score=0.9)]
    state = DigestSnapshot(ranked, DigestResult(digest_text="hi", ranked_items=ranked)).to_dict()
    assert set(state) == {"ranked_items", "digest_result"}
    assert state["ranked_items"][0]["item"]["item_id"] == "a"


def test_it_roundtrips():
    ranked = [RankedItem(item=_item("a"), score=0.9)]
    content = DigestContent(lead="lead", items=[DigestItem(title="t", url="https://e.com/a", body="b")])
    restored = DigestSnapshot.from_dict(
        DigestSnapshot(ranked, DigestResult(digest_text="d", ranked_items=ranked, content=content)).to_dict()
    )
    assert [ri.item.item_id for ri in restored.ranked_items] == ["a"]
    assert restored.content is not None and restored.content.lead == "lead"


def test_a_legacy_snapshot_with_collected_items_still_loads():
    ranked = [RankedItem(item=_item("a"), score=0.9)]
    legacy = {
        **DigestSnapshot(ranked, DigestResult(digest_text="d")).to_dict(),
        "collected_items": {"a": _item("a").model_dump(mode="json")},
    }
    assert [ri.item.item_id for ri in DigestSnapshot.from_dict(legacy).ranked_items] == ["a"]


def test_the_ranked_items_are_stored_once():
    # Re-embedding them in the digest result doubled the snapshot past AgentCore's event limit.
    ranked = [RankedItem(item=_item("a"), score=0.9)]
    state = DigestSnapshot(ranked, DigestResult(digest_text="d", ranked_items=ranked)).to_dict()
    assert "ranked_items" not in state["digest_result"]


def test_an_empty_snapshot_has_no_content():
    assert DigestSnapshot.from_dict({}).content is None
