import importlib
import json
import logging

import pytest

from shared.logger import (
    LoggerConfig,
    _CorrelationFilter,
    _JsonFormatter,
    configure_logger,
    get_correlation_id,
    set_correlation_id,
)

logger_mod = importlib.import_module("shared.logger")


@pytest.fixture(autouse=True)
def _reset_correlation_id():
    logger_mod._correlation_id.set("")
    yield
    logger_mod._correlation_id.set("")


def _record(msg: str = "hello %s", args=("world",)) -> logging.LogRecord:
    return logging.LogRecord("omnisummary", logging.INFO, "path", 10, msg, args, None)


class TestCorrelationId:
    def test_set_returns_value(self):
        value = set_correlation_id("abc123")
        assert value == "abc123"
        assert get_correlation_id() == "abc123"

    def test_set_generates_when_none(self):
        value = set_correlation_id(None)
        assert value
        assert len(value) == 12
        assert get_correlation_id() == value

    def test_filter_injects_correlation_id(self):
        set_correlation_id("req-xyz")
        record = _record()
        assert _CorrelationFilter().filter(record) is True
        assert record.correlation_id == "req-xyz"


class TestJsonFormatter:
    def test_emits_valid_json(self):
        set_correlation_id("cid-1")
        record = _record("collected %d items", (42,))
        _CorrelationFilter().filter(record)
        payload = json.loads(_JsonFormatter().format(record))
        assert payload["level"] == "INFO"
        assert payload["logger"] == "omnisummary"
        assert payload["message"] == "collected 42 items"
        assert payload["correlation_id"] == "cid-1"
        assert payload["timestamp"].endswith("+00:00")

    def test_omits_empty_correlation_id(self):
        logger_mod._correlation_id.set("")
        record = _record("plain", ())
        _CorrelationFilter().filter(record)
        payload = json.loads(_JsonFormatter().format(record))
        assert "correlation_id" not in payload

    def test_includes_exception(self):
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = logging.LogRecord("omnisummary", logging.ERROR, "p", 1, "failed", (), sys.exc_info())
        payload = json.loads(_JsonFormatter().format(record))
        assert "ValueError: boom" in payload["exception"]


class TestConfigureLogger:
    def test_json_logging_produces_json(self, capsys):
        cfg = LoggerConfig(name="test-json", level=logging.INFO, file_logging_enabled=False, json_logging=True)
        log = configure_logger(cfg)
        set_correlation_id("test-cid")
        log.info("structured %s", "msg")
        captured = capsys.readouterr()
        payload = json.loads(captured.err.strip())
        assert payload["message"] == "structured msg"
        assert payload["correlation_id"] == "test-cid"

    def test_human_logging_is_not_json(self, capsys):
        cfg = LoggerConfig(name="test-human", level=logging.INFO, file_logging_enabled=False, json_logging=False)
        log = configure_logger(cfg)
        set_correlation_id("hcid")
        log.info("plain message")
        captured = capsys.readouterr()
        assert "plain message" in captured.err
        assert "[hcid]" in captured.err


class TestFileHandlerNeverBreaksImport:
    """`logger = get_default_logger()` runs at module scope, so anything _add_file_handler touches
    happens before any application code. It used to mkdir the logs directory unguarded, which raised
    PermissionError on import — found by giving the container image a non-root user, but a read-only
    checkout or any sandbox hits the same wall. The file is a local convenience; the console handler
    already carries every record, and AWS skips the file path entirely."""

    def test_unwritable_logs_dir_degrades_to_console_only(self, capsys, monkeypatch):
        from pathlib import Path

        from shared.logger import _add_file_handler, _build_formatter

        def deny(*_a, **_kw):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(Path, "mkdir", deny)
        cfg = LoggerConfig(name="test-unwritable", level=logging.INFO, file_logging_enabled=True, json_logging=False)
        log = configure_logger(cfg)
        _add_file_handler(log, _build_formatter(cfg))  # must not raise
        assert not [h for h in log.handlers if isinstance(h, logging.FileHandler)]
        log.info("still logs to the console")
        assert "still logs to the console" in capsys.readouterr().err

    def test_unopenable_log_file_degrades_to_console_only(self, tmp_path, monkeypatch):
        from shared.logger import _add_file_handler, _build_formatter

        def deny(*_a, **_kw):
            raise OSError(30, "Read-only file system")

        # Patch the CONSTRUCTOR, not the name: _add_file_handler isinstance-checks FileHandler, and
        # replacing the class with a function would make that check raise instead of the open.
        monkeypatch.setattr(logging.FileHandler, "__init__", deny)
        cfg = LoggerConfig(name="test-unopenable", level=logging.INFO, file_logging_enabled=True, json_logging=False)
        log = configure_logger(cfg)
        _add_file_handler(log, _build_formatter(cfg))  # must not raise
        assert not [h for h in log.handlers if isinstance(h, logging.FileHandler)]
