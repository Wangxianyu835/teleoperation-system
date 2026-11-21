"""Common terminal, logging, and script-result helpers."""

from __future__ import annotations

import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from .config import PROJECT_ROOT, project_path


def configure_logging(
    level: str = "INFO",
    *,
    log_name: str | None = None,
) -> logging.Logger:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except (AttributeError, ValueError):
                pass
    logger = logging.getLogger("ffg_capture")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(message)s",
        datefmt="%H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    if log_name:
        log_dir = project_path("outputs/logs")
        log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"{log_name}.log",
            encoding="utf-8",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


class CallbackLogger:
    """Adapter accepted by the vendored Linker transport."""

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger

    def log(self, level: str, message: str) -> None:
        method = getattr(self.logger, level.lower(), self.logger.info)
        method(message)


def timestamp_label() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
    return target.resolve()


def print_banner(title: str) -> None:
    line = "=" * 72
    print(line)
    print(title)
    print(line)


def announce_artifact(label: str, path: str | Path | None) -> None:
    if path is None:
        return
    print(f"{label}: {Path(path).resolve()}")


def run_script(
    name: str,
    action: Callable[[], int | None],
    *,
    logger: logging.Logger,
) -> int:
    """Run a PyCharm entry point and turn failures into a clear exit status."""

    try:
        result = action()
        code = 0 if result is None else int(result)
    except KeyboardInterrupt:
        logger.warning("操作已被用户中止")
        code = 130
    except Exception as exc:
        logger.error("执行失败: %s", exc)
        logger.debug("完整异常信息:\n%s", traceback.format_exc())
        code = 1
    if code == 0:
        logger.info("%s 完成", name)
    else:
        logger.error("%s 未完成，退出码=%d", name, code)
    return code


def display_project_root() -> Path:
    return PROJECT_ROOT
