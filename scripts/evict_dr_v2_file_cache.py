#!/usr/bin/env python
"""只回收显式指定任务输入所对应的 page-cache 区间。"""
from __future__ import annotations

import argparse
import os
from pathlib import Path


ALLOWED_PREFIXES = tuple(
    path.resolve()
    for path in (
        Path("/root/autodl-tmp/data/dynamic_editing_v2"),
        Path("/root/autodl-tmp/runs/dynamic_editing_v2"),
        Path("/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval"),
    )
)


def allowed(path: Path) -> bool:
    resolved = path.resolve()
    return any(resolved == prefix or prefix in resolved.parents for prefix in ALLOWED_PREFIXES)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    files = []
    for root in args.paths:
        if not allowed(root):
            raise ValueError(f"拒绝处理任务范围外路径: {root}")
        if root.is_file():
            files.append(root)
        elif root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    advised = 0
    advised_bytes = 0
    failures = []
    for path in files:
        try:
            descriptor = os.open(path, os.O_RDONLY)
            try:
                size = path.stat().st_size
                os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
                advised += 1
                advised_bytes += size
            finally:
                os.close(descriptor)
        except OSError as error:
            failures.append({"path": str(path), "error": str(error)})
    print(
        {
            "files": len(files),
            "advised": advised,
            "advised_bytes": advised_bytes,
            "failures": failures[:20],
            "failure_count": len(failures),
        }
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
