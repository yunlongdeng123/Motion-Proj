#!/usr/bin/env python3
"""GPU-free source, environment, weight and data readiness audit."""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from motion_proj.cfbench.registry import load_registry


def _matches(pattern: str) -> list[str]:
    return sorted(glob.glob(pattern, recursive=True))


def _path_group(row: dict[str, Any], key: str) -> dict[str, Any]:
    spec = row.get(key, {})
    if not spec:
        return {"passed": False, "all_of": {}, "any_of": {}}
    all_of = [str(path) for path in spec.get("all_of", [])]
    any_of = [str(path) for path in spec.get("any_of", [])]
    all_hits = {pattern: bool(_matches(pattern)) for pattern in all_of}
    any_hits = {pattern: bool(_matches(pattern)) for pattern in any_of}
    passed = all(all_hits.values()) and (not any_of or any(any_hits.values()))
    return {"passed": passed, "all_of": all_hits, "any_of": any_hits}


def _source(row: dict[str, Any]) -> dict[str, Any]:
    spec = row["source"]
    path = Path(spec["path"])
    required = {name: (path / name).exists() for name in spec.get("required_paths", [])}
    head = None
    dirty = None
    if (path / ".git").exists():
        head_run = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        status_run = subprocess.run(
            ["git", "-C", str(path), "status", "--short"],
            capture_output=True,
            text=True,
            check=False,
        )
        if head_run.returncode == 0:
            head = head_run.stdout.strip()
        if status_run.returncode == 0:
            dirty = bool(status_run.stdout.strip())
    expected = spec.get("expected_commit")
    passed = path.is_dir() and all(required.values()) and (expected is None or head == expected)
    return {
        "passed": passed,
        "path": str(path),
        "head": head,
        "expected_commit": expected,
        "dirty": dirty,
        "required_paths": required,
    }


def audit_model(row: dict[str, Any]) -> dict[str, Any]:
    source = _source(row)
    environment = _path_group(row, "environment")
    weights = _path_group(row, "weights")
    data = _path_group(row, "data")
    prior_evidence = _path_group(row, "prior_evidence")
    if source["passed"] and environment["passed"] and weights["passed"] and data["passed"]:
        readiness = "ready_for_gpu_preflight"
    elif prior_evidence["passed"] and row.get("reuse_prior_evidence"):
        readiness = "prior_evidence_reusable"
    elif source["passed"] and not weights["passed"]:
        readiness = "source_only_missing_weights"
    elif source["passed"] and not environment["passed"]:
        readiness = "source_only_missing_environment"
    else:
        readiness = "blocked_source_or_data"
    return {
        "model_id": row["model_id"],
        "readiness": readiness,
        "source": source,
        "environment": environment,
        "weights": weights,
        "data": data,
        "prior_evidence": prior_evidence,
        "gpu_checks_run": False,
        "notes": row.get("resource_notes", []),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = load_registry(args.registry)
    rows = [audit_model(model) for model in registry["models"]]
    report = {
        "schema_version": "worldsim_v75_cfbench_preflight_v1",
        "mode": "gpu_free",
        "models": rows,
        "summary": {
            status: sum(row["readiness"] == status for row in rows)
            for status in sorted({row["readiness"] for row in rows})
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
