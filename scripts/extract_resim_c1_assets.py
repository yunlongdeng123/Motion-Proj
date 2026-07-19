"""从 nuScenes 共享 tar shards 中只提取 C1B-01 精确 manifest 的缺失成员。"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-plan", required=True)
    parser.add_argument("--archive-root", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--minimum-free-bytes", type=int, default=30 * 1024**3)
    args = parser.parse_args()
    plan_path, destination = Path(args.asset_plan).resolve(), Path(args.destination).resolve()
    archive_root, state_dir = Path(args.archive_root).resolve(), Path(args.state_dir).resolve()
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    required = sorted(set(map(str, plan["required"])))
    if any(Path(name).is_absolute() or ".." in Path(name).parts for name in required):
        raise SystemExit("asset plan 含不安全路径")
    destination.mkdir(parents=True, exist_ok=True)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock_handle = (state_dir / "extract.lock").open("w")
    fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    missing = [name for name in required if not (destination / name).is_file()]
    projected = shutil.disk_usage(destination).free - int(plan["estimated_missing_bytes"] * 1.5)
    if projected < args.minimum_free_bytes:
        raise SystemExit(f"磁盘安全门禁失败: projected_free={projected}")
    archives = sorted(archive_root.glob("v1.0-trainval??_blobs.tgz"))
    if len(archives) != 10:
        raise SystemExit(f"预期 10 个 trainval shards，实际 {len(archives)}")
    fingerprint = hashlib.sha256("\n".join(required).encode("utf-8")).hexdigest()
    for index, archive in enumerate(archives, start=1):
        if not missing:
            break
        list_path = state_dir / f"members-{fingerprint[:12]}-{index:02d}.txt"
        list_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
        log_path = state_dir / f"archive-{index:02d}-{fingerprint[:12]}.log"
        command = [
            "tar", "--extract", "--gzip", "--file", str(archive), "--directory", str(destination),
            "--skip-old-files", "--verbatim-files-from", "--files-from", str(list_path),
        ]
        with log_path.open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
        missing = [name for name in missing if not (destination / name).is_file()]
        print(json.dumps({
            "archive": archive.name, "tar_exit_code": result.returncode,
            "remaining": len(missing), "free_bytes": shutil.disk_usage(destination).free,
        }), flush=True)
        if shutil.disk_usage(destination).free < args.minimum_free_bytes:
            raise SystemExit("提取过程中触发 30 GiB 磁盘安全线")
    summary = {
        "asset_plan": str(plan_path), "asset_fingerprint": fingerprint,
        "required_count": len(required), "missing_count": len(missing), "missing": missing,
        "free_bytes": shutil.disk_usage(destination).free,
    }
    (state_dir / f"summary-{fingerprint[:12]}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if missing:
        raise SystemExit("精确资产提取不完整")


if __name__ == "__main__":
    main()
