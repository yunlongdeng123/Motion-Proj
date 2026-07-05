"""轻量级日志辅助工具（若可用则使用 rich，否则使用标准库）。"""
from __future__ import annotations

import logging
import sys

_CONFIGURED = False


def get_logger(name: str = "motion_proj", level: int = logging.INFO) -> logging.Logger:
    global _CONFIGURED
    logger = logging.getLogger(name)
    if not _CONFIGURED:
        try:
            from rich.logging import RichHandler

            handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
            fmt = "%(message)s"
        except Exception:  # pragma: no cover - 环境中始终安装了 rich
            handler = logging.StreamHandler(sys.stdout)
            fmt = "[%(asctime)s] %(levelname)s %(name)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt, datefmt="%H:%M:%S"))
        root = logging.getLogger("motion_proj")
        root.handlers[:] = [handler]
        root.setLevel(level)
        root.propagate = False
        _CONFIGURED = True
    logger.setLevel(level)
    return logger
