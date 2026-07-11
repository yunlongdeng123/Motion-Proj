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


def git_state(root: str = ".") -> dict[str, Any]:
    def run(*args: str) -> str | None:
        try:
            return subprocess.check_output(
                ["git", "-C", root, *args],
                text=True,
                stderr=subprocess.DEVNULL,
            )
        except Exception:
            return None

    commit = run("rev-parse", "HEAD")
    repo_root = run("rev-parse", "--show-toplevel")
    diff = run("diff", "--binary", "HEAD")
    untracked = run("ls-files", "--others", "--exclude-standard", "--full-name", "-z")
    if commit is None or repo_root is None or diff is None or untracked is None:
        return {
            "commit": "unknown",
            "git_available": False,
            "dirty": True,
            "dirty_tracked": False,
            "dirty_untracked": False,
            "untracked_count": 0,
            "dirty_diff_hash": "unknown",
        }

    untracked_paths = sorted(path for path in untracked.split("\0") if path)
    untracked_manifest = []
    repository = Path(repo_root.strip())
    for relative_path in untracked_paths:
        path = repository / relative_path
        if path.is_symlink():
            digest = hashlib.sha256(os.readlink(path).encode()).hexdigest()
            kind = "symlink"
        elif path.is_file():
            digest = file_fingerprint(str(path))
            kind = "file"
        else:
            digest = "unreadable"
            kind = "other"
        untracked_manifest.append(
            {"path": relative_path, "kind": kind, "sha256": digest}
        )

    tracked_hash = hashlib.sha256(diff.encode()).hexdigest()
    worktree_hash = (
        sha256_json({"tracked_diff_sha256": tracked_hash, "untracked": untracked_manifest})
        if untracked_manifest
        else tracked_hash
    )
    return {
        "commit": commit.strip(),
        "git_available": True,
        "dirty": bool(diff or untracked_paths),
        "dirty_tracked": bool(diff),
        "dirty_untracked": bool(untracked_paths),
        "untracked_count": len(untracked_paths),
        "dirty_diff_hash": worktree_hash,
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
