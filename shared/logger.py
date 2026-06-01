import json
import logging
import os
import uuid
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path

from .constants import EnvVars, LocalPaths

_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="")


def is_running_in_aws() -> bool:
    aws_env_vars = [
        "AWS_EXECUTION_ENV",
        "AWS_LAMBDA_FUNCTION_NAME",
        "AWS_BATCH_JOB_ID",
        "ECS_CONTAINER_METADATA_URI",
    ]
    return any(env_var in os.environ for env_var in aws_env_vars)


def set_correlation_id(correlation_id: str | None = None) -> str:
    value = correlation_id or uuid.uuid4().hex[:12]
    _correlation_id.set(value)
    return value


def get_correlation_id() -> str:
    return _correlation_id.get()


class _CorrelationFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = _correlation_id.get()
        return True


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        correlation_id = getattr(record, "correlation_id", "")
        if correlation_id:
            payload["correlation_id"] = correlation_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


class LoggerConfig:
    def __init__(
        self,
        name: str,
        level: int,
        log_format: str = "%(asctime)s - %(levelname)s - %(name)s - [%(correlation_id)s] %(message)s",
        file_logging_enabled: bool = True,
        json_logging: bool = False,
    ):
        self.name = name
        self.level = level
        self.log_format = log_format
        self.file_logging_enabled = file_logging_enabled
        self.json_logging = json_logging


def _build_formatter(config: LoggerConfig) -> logging.Formatter:
    return _JsonFormatter() if config.json_logging else logging.Formatter(config.log_format)


def _add_console_handler(logger_obj: logging.Logger, formatter: logging.Formatter) -> None:
    if any(isinstance(h, logging.StreamHandler) for h in logger_obj.handlers):
        return
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger_obj.addHandler(console_handler)


def _add_file_handler(logger_obj: logging.Logger, formatter: logging.Formatter) -> None:
    logs_dir = Path(__file__).resolve().parent.parent / LocalPaths.LOGS_DIR.value
    logs_dir.mkdir(parents=True, exist_ok=True)
    base_filename = LocalPaths.LOGS_FILE.value
    name, ext = base_filename.rsplit(".", 1)
    timestamp = datetime.now().strftime("%Y-%m-%d")
    log_filename = f"{name}_{timestamp}.{ext}"
    log_file_path = logs_dir / log_filename
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path) for h in logger_obj.handlers):
        return
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger_obj.addHandler(file_handler)


def configure_logger(config: LoggerConfig) -> logging.Logger:
    logger_obj = logging.getLogger(config.name)
    logger_obj.setLevel(config.level)
    formatter = _build_formatter(config)
    logger_obj.handlers.clear()
    logger_obj.filters.clear()
    logger_obj.addFilter(_CorrelationFilter())
    _add_console_handler(logger_obj, formatter)
    if config.file_logging_enabled and not is_running_in_aws():
        _add_file_handler(logger_obj, formatter)
    logger_obj.propagate = False
    return logger_obj


def get_default_logger(name: str = "omnisummary") -> logging.Logger:
    log_level_str = os.getenv(EnvVars.LOG_LEVEL.value, "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    config = LoggerConfig(name=name, level=log_level, json_logging=is_running_in_aws())
    return configure_logger(config)


logger = get_default_logger()
