import hashlib

import pytest
from pydantic import ValidationError

from shared.constants import SourceType
from shared.models import CollectedItem, RankedItem


class TestCollectedItem:
    def test_fallback_item_id_from_url(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://example.com")
        expected = hashlib.sha256(b"http://example.com").hexdigest()[:16]
        assert item.item_id == expected

    def test_explicit_item_id_preserved(self):
        item = CollectedItem(item_id="my_id", source_type=SourceType.REDDIT, title="T", url="http://example.com")
        assert item.item_id == "my_id"

    def test_hash_by_url(self):
        a = CollectedItem(source_type=SourceType.REDDIT, title="A", url="http://a.com")
        b = CollectedItem(source_type=SourceType.REDDIT, title="B", url="http://a.com")
        assert hash(a) == hash(b)

    def test_eq_by_url(self):
        a = CollectedItem(source_type=SourceType.REDDIT, title="A", url="http://a.com")
        b = CollectedItem(source_type=SourceType.REDDIT, title="B", url="http://a.com")
        assert a == b

    def test_not_eq_different_url(self):
        a = CollectedItem(source_type=SourceType.REDDIT, title="A", url="http://a.com")
        b = CollectedItem(source_type=SourceType.REDDIT, title="A", url="http://b.com")
        assert a != b


class TestRankedItem:
    def test_score_in_range(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://a.com")
        ranked = RankedItem(item=item, score=0.5)
        assert ranked.score == 0.5

    def test_score_below_zero_rejected(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://a.com")
        with pytest.raises(ValidationError):
            RankedItem(item=item, score=-0.1)

    def test_score_above_one_rejected(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://a.com")
        with pytest.raises(ValidationError):
            RankedItem(item=item, score=1.1)

    def test_boundary_scores(self):
        item = CollectedItem(source_type=SourceType.REDDIT, title="T", url="http://a.com")
        assert RankedItem(item=item, score=0.0).score == 0.0
        assert RankedItem(item=item, score=1.0).score == 1.0
