from shared import format_collected_item
from shared.constants import SourceType
from shared.models import CollectedItem


def _item(text: str = "body text") -> CollectedItem:
    return CollectedItem(
        item_id="id1",
        source_type=SourceType.RSS,
        title="A Title",
        url="http://example.com",
        text=text,
        author="Alice",
    )


class TestFormatCollectedItem:
    def test_renders_header_fields_and_text(self):
        out = format_collected_item(
            _item(),
            index=2,
            max_tokens=1000,
            fields=[("Title", "A Title"), ("Source", "rss")],
        )
        assert out.startswith("=== Item 2 ===\n")
        assert "Title: A Title\n" in out
        assert "Source: rss\n" in out
        assert out.endswith("Text:\nbody text\n")

    def test_custom_text_label(self):
        out = format_collected_item(_item(), index=1, max_tokens=1000, fields=[], text_label="Content")
        assert "Content:\nbody text" in out

    def test_truncates_body_by_tokens(self):
        out = format_collected_item(_item("word " * 500), index=1, max_tokens=10, fields=[])
        body = out.split("Text:\n", 1)[1]
        assert len(body.split()) <= 10
