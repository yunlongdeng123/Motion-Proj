#!/usr/bin/env python3
"""Finalize one M3 run without overwriting an existing terminal artifact."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path


DEFAULT_ALLOWED_ROOT = Path("/root/autodl-tmp/runs/dynamic_editing_v2")
FINAL_STATUSES = {"blocked", "done"}
SUCCESS_STAGE_STATUSES = {"available", "done"}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    atomic_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def load_json(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_run_path(run_dir: Path, task_id: str, allowed_root: Path) -> Path:
    run_dir = run_dir.resolve()
    allowed_root = allowed_root.resolve()
    if allowed_root not in run_dir.parents:
        raise RuntimeError(f"run is outside the V2 root: {run_dir}")
    if run_dir.parent.name != task_id:
        raise RuntimeError(
            f"task-id does not match run parent: {task_id} != {run_dir.parent.name}"
        )
    return run_dir


def validate_required_stages(run_dir: Path, required_stages: list[str]) -> list[dict]:
    rows = []
    for name in required_stages:
        path = run_dir / "stages" / f"{name}.json"
        if not path.is_file():
            raise RuntimeError(f"required stage is missing: {path}")
        row = load_json(path)
        if row.get("status") not in SUCCESS_STAGE_STATUSES:
            raise RuntimeError(
                f"required stage did not succeed: {name} status={row.get('status')!r}"
            )
        rows.append(row)
    return rows


def finalize(
    *,
    run_dir: Path,
    task_id: str,
    status: str,
    summary: str,
    failure_code: str | None = None,
    failure_detail: str | None = None,
    evidence: list[str] | None = None,
    required_stages: list[str] | None = None,
    allowed_root: Path = DEFAULT_ALLOWED_ROOT,
) -> dict:
    if status not in FINAL_STATUSES:
        raise RuntimeError(f"unsupported final status: {status}")
    run_dir = validate_run_path(run_dir, task_id, allowed_root)
    evidence = evidence or []
    required_stages = required_stages or []
    artifacts_path = run_dir / "artifacts.json"
    if artifacts_path.exists():
        raise RuntimeError(f"run is already immutable: {artifacts_path}")

    terminal_path = run_dir / "terminal.json"
    terminal_before = load_json(terminal_path)
    if terminal_before.get("status") != "running":
        raise RuntimeError(
            f"only a running run can be finalized, got {terminal_before.get('status')!r}"
        )
    if status == "blocked" and (not failure_code or not failure_detail):
        raise RuntimeError("blocked finalization requires failure-code and failure-detail")
    if status == "done":
        validate_required_stages(run_dir, required_stages)

    finished_at = now()
    failure = (
        {
            "code": failure_code,
            "detail": failure_detail,
            "evidence": evidence,
        }
        if status == "blocked"
        else None
    )
    terminal = {
        "status": status,
        "updated_at": finished_at,
        "failure": failure,
    }
    atomic_json(terminal_path, terminal)

    manifest_path = run_dir / "manifest.json"
    manifest = load_json(manifest_path)
    manifest.update(
        {
            "status": status,
            "finished_at": finished_at,
            "failure": failure,
            "finalizer": "scripts/finalize_dr_v2_m3_run.py",
        }
    )
    atomic_json(manifest_path, manifest)

    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.exists():
        atomic_text(metrics_path, "")
    summary_lines = [
        f"# {task_id} {status} run",
        "",
        f"- instance: `{run_dir.name}`",
        f"- status: `{status}`",
    ]
    if failure:
        summary_lines.extend(
            [
                f"- failure code: `{failure_code}`",
                f"- failure detail: {failure_detail}",
            ]
        )
    if evidence:
        summary_lines.append("- evidence: " + ", ".join(f"`{item}`" for item in evidence))
    summary_lines.extend(["", summary.strip(), ""])
    atomic_text(run_dir / "summary.md", "\n".join(summary_lines))

    files = []
    for path in sorted(run_dir.rglob("*")):
        if not path.is_file() or path == artifacts_path or ".partial." in path.name:
            continue
        files.append(
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    artifacts = {
        "schema_version": 1,
        "created_at": finished_at,
        "status": status,
        "file_count": len(files),
        "files": files,
    }
    atomic_json(artifacts_path, artifacts)
    return {"terminal": terminal, "artifacts": artifacts}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--status", choices=sorted(FINAL_STATUSES), required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--failure-code")
    parser.add_argument("--failure-detail")
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--required-stage", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = finalize(
        run_dir=args.run_dir,
        task_id=args.task_id,
        status=args.status,
        summary=args.summary,
        failure_code=args.failure_code,
        failure_detail=args.failure_detail,
        evidence=args.evidence,
        required_stages=args.required_stage,
    )
    print(json.dumps(result["terminal"], ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
