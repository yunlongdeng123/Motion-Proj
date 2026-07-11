"""同一文件系统内的原子文件和目录提交。"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterator
from typing import Any


def atomic_write_text(path: str, value: str) -> None:
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=directory, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_write_json(path: str, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, ensure_ascii=False, default=str, allow_nan=False) + "\n")


@contextlib.contextmanager
def atomic_directory(target: str) -> Iterator[str]:
    """在临时目录完成写入；成功后一次性提交到目标路径。"""
    parent = os.path.dirname(os.path.abspath(target))
    os.makedirs(parent, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix=f".{os.path.basename(target)}.tmp-", dir=parent)
    try:
        yield tmp
        if os.path.exists(target):
            raise FileExistsError(target)
        os.replace(tmp, target)
        tmp = ""
    finally:
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp)
