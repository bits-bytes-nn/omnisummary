import logging
import os
from datetime import datetime
from pathlib import Path

from .constants import EnvVars, LocalPaths


def is_running_in_aws() -> bool:
    aws_env_vars = [
        "AWS_EXECUTION_ENV",
        "AWS_LAMBDA_FUNCTION_NAME",
        "AWS_BATCH_JOB_ID",
        "ECS_CONTAINER_METADATA_URI",
    ]
    return any(env_var in os.environ for env_var in aws_env_vars)


class LoggerConfig:
    def __init__(
        self,
        name: str,
        level: int,
        log_format: str = "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        file_logging_enabled: bool = True,
    ):
        self.name = name
        self.level = level
        self.log_format = log_format
        self.file_logging_enabled = file_logging_enabled


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
    if any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(log_file_path)
        for h in logger_obj.handlers
    ):
        return
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger_obj.addHandler(file_handler)


def configure_logger(config: LoggerConfig) -> logging.Logger:
    logger_obj = logging.getLogger(config.name)
    logger_obj.setLevel(config.level)
    formatter = logging.Formatter(config.log_format)
    logger_obj.handlers.clear()
    _add_console_handler(logger_obj, formatter)
    if config.file_logging_enabled and not is_running_in_aws():
        _add_file_handler(logger_obj, formatter)
    logger_obj.propagate = False
    return logger_obj


def get_default_logger(name: str = "app") -> logging.Logger:
    log_level_str = os.getenv(EnvVars.LOG_LEVEL.value, "INFO").upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    config = LoggerConfig(name=name, level=log_level)
    return configure_logger(config)


logger = get_default_logger()
