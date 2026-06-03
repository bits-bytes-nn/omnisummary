import tempfile

from shared.state_store import LocalStateStore


class TestLocalStateStore:
    def test_write_and_read(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            store.write("test.txt", "hello world")
            assert store.read("test.txt") == "hello world"

    def test_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            assert not store.exists("missing.txt")
            store.write("exists.txt", "data")
            assert store.exists("exists.txt")

    def test_read_missing_returns_none(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            assert store.read("nonexistent.txt") is None

    def test_nested_key(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            store.write("subdir/file.txt", "nested")
            assert store.read("subdir/file.txt") == "nested"

    def test_overwrite(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = LocalStateStore(tmpdir)
            store.write("file.txt", "v1")
            store.write("file.txt", "v2")
            assert store.read("file.txt") == "v2"
