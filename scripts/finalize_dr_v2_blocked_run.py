#!/usr/bin/env python3
"""补齐 V2 blocked run 的不可变运行合同，不改写 terminal.json。"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


TZ = dt.timezone(dt.timedelta(hours=8))
ALLOWED_TERMINAL = {"blocked", "rejected"}


def atomic_write(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def write_json_if_missing(path: Path, payload: object) -> None:
    if not path.exists():
        atomic_write(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    allowed_root = Path("/root/autodl-tmp/runs/dynamic_editing_v2").resolve()
    if allowed_root not in run_dir.parents:
        raise SystemExit(f"run 不在 V2 根目录下: {run_dir}")
    if run_dir.parent.name != args.task_id:
        raise SystemExit(
            f"task-id 与目录不一致: {args.task_id} != {run_dir.parent.name}"
        )

    terminal_path = run_dir / "terminal.json"
    terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
    if terminal.get("status") not in ALLOWED_TERMINAL:
        raise SystemExit(
            "只允许补齐 blocked/rejected run，当前状态="
            f"{terminal.get('status')!r}"
        )

    now = dt.datetime.now(TZ).isoformat()
    project_commit_path = run_dir / "source_snapshot" / "project_commit.txt"
    project_commit = (
        project_commit_path.read_text(encoding="utf-8").strip()
        if project_commit_path.exists()
        else None
    )
    stage_rows = []
    for path in sorted((run_dir / "stages").glob("*.json")):
        try:
            stage_rows.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError) as exc:
            stage_rows.append({"stage_file": path.name, "parse_error": str(exc)})

    write_json_if_missing(
        run_dir / "manifest.json",
        {
            "schema_version": 1,
            "task_id": args.task_id,
            "instance_id": run_dir.name,
            "created_at": now,
            "project_commit": project_commit,
            "seed": args.seed,
            "run_dir": str(run_dir),
            "status": terminal["status"],
            "failure": terminal.get("failure"),
            "finalizer": "scripts/finalize_dr_v2_blocked_run.py",
        },
    )
    write_json_if_missing(
        run_dir / "resolved.yaml",
        {
            "schema_version": 1,
            "task_id": args.task_id,
            "instance_id": run_dir.name,
            "seed": args.seed,
            "project_commit": project_commit,
            "terminal": terminal,
            "stages": stage_rows,
        },
    )

    resource_path = run_dir / "resource.jsonl"
    if not resource_path.exists():
        resource_files = sorted(
            str(path.relative_to(run_dir))
            for path in (run_dir / "environment").glob("resource_*.txt")
        )
        atomic_write(
            resource_path,
            json.dumps(
                {
                    "recorded_at": now,
                    "status": terminal["status"],
                    "raw_snapshots": resource_files,
                },
                ensure_ascii=False,
            )
            + "\n",
        )
    if not (run_dir / "metrics.jsonl").exists():
        atomic_write(run_dir / "metrics.jsonl", "")
    if not (run_dir / "summary.md").exists():
        atomic_write(
            run_dir / "summary.md",
            "\n".join(
                [
                    f"# {args.task_id} blocked run",
                    "",
                    f"- instance: `{run_dir.name}`",
                    f"- status: `{terminal['status']}`",
                    f"- failure: `{terminal.get('failure')}`",
                    f"- project commit: `{project_commit}`",
                    "- 结论：仅记录工程阻塞，不据此判断方法质量。",
                    "",
                ]
            ),
        )

    artifacts = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path.name.endswith(".partial"):
            continue
        if path.name == "artifacts.json":
            continue
        artifacts.append(
            {
                "path": str(path.relative_to(run_dir)),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
        )
    write_json_if_missing(
        run_dir / "artifacts.json",
        {"schema_version": 1, "created_at": now, "files": artifacts},
    )
    print(f"已补齐 blocked run 合同: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
