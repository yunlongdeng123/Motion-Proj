"""代码、数据和环境的轻量稳定指纹。"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any


def sha256_json(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def file_fingerprint(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def directory_manifest_fingerprint(path: str, marker: str = "metadata.json") -> str:
    """只读取轻量 metadata/manifest，不加载 cache tensor。"""
    root = Path(path)
    rows = []
    if root.exists():
        for item in sorted(root.rglob(marker)):
            if not (item.parent / "COMPLETE").is_file() or ".stale-" in str(item):
                continue
            stat = item.stat()
            rows.append((str(item.relative_to(root)), stat.st_size, file_fingerprint(str(item))))
    return sha256_json(rows)


def git_state(root: str = ".") -> dict[str, str | bool]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(["git", "-C", root, *args], text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return "unknown"

    diff = run("diff", "--binary", "HEAD")
    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(diff and diff != "unknown"),
        "dirty_diff_hash": hashlib.sha256(diff.encode()).hexdigest() if diff != "unknown" else "unknown",
    }


def environment_fingerprint() -> dict[str, Any]:
    try:
        import torch

        torch_info = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception:
        torch_info = {}
    value = {"python": platform.python_version(), "platform": platform.platform(), **torch_info}
    value["fingerprint"] = sha256_json(value)
    return value
