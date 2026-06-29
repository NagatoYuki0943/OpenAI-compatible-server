from __future__ import annotations

import sys

from loguru import logger

from openai_compatible_server.config import Settings

LOG_FORMAT = (
    "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | "
    "request_id={extra[request_id]} | {name}:{function}:{line} | {message}"
)


def configure_logging(settings: Settings) -> None:
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    logger.remove()
    logger.configure(extra={"request_id": "-"})
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=LOG_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,
    )
    logger.add(
        settings.log_dir / "openai_server_{time:YYYY-MM-DD_HH-mm-ss}.log",
        level=settings.log_level,
        rotation=settings.log_rotation,
        retention=settings.log_retention,
        compression="zip",
        encoding="utf-8",
        enqueue=True,
        format=LOG_FORMAT,
        backtrace=False,
        diagnose=False,
    )
