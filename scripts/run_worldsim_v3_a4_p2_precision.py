#!/usr/bin/env python
"""初始化 A4-P2 正式运行并冻结输入、代码与恢复账本。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Iterable, Mapping

from omegaconf import OmegaConf


PROJECT = Path("/root/autodl-tmp/motion_proj")
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p2_mixed_precision_protocol_v1.yaml"
RUN_ROOT = Path("/root/autodl-tmp/runs/worldsim_v3/WS-V3-A4-DEPLOYMENT-01")
_ACTIVE_RUN_DIR: Path | None = None


from scripts.validate_worldsim_v3_a4_p2_mixed_precision_protocol import (
    validate_inputs,
    validate_schema,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any], *, replace: bool = False) -> None:
    """在同一目录原子写入 JSON，默认禁止覆盖冻结证据。"""
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def directory_digest(directory: Path, pattern: str) -> dict[str, Any]:
    """按冻结的 sha256sum 文本清单算法计算目录摘要。"""
    paths = sorted(directory.glob(pattern), key=lambda path: path.name)
    payload = "".join(
        f"{sha256_file(path)}  ./{path.name}\n" for path in paths if path.is_file()
    )
    return {
        "sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "file_count": len(paths),
        "total_bytes": sum(path.stat().st_size for path in paths),
    }


def cgroup_memory_current() -> int | None:
    path = Path("/sys/fs/cgroup/memory.current")
    return int(path.read_text().strip()) if path.exists() else None


def cgroup_memory_events() -> dict[str, int]:
    path = Path("/sys/fs/cgroup/memory.events")
    if not path.exists():
        return {}
    return {
        key: int(value)
        for key, value in (line.split() for line in path.read_text().splitlines())
    }


def nvidia_compute_rows() -> list[dict[str, int]]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi 失败: {result.stderr.strip()}")
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        pid, used = (part.strip() for part in line.split(",", 1))
        rows.append({"pid": int(pid), "used_memory_mib": int(used)})
    return rows


def command_output(*command: str) -> str:
    return subprocess.check_output(command, cwd=PROJECT, text=True).strip()


def snapshot_sources(run_dir: Path, paths: Iterable[Path]) -> dict[str, str]:
    root = run_dir / "source_snapshot"
    hashes = {}
    for source in paths:
        relative = source.relative_to(PROJECT)
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        hashes[str(relative)] = sha256_file(target)
    return hashes


def write_stage(
    run_dir: Path,
    manifest: dict[str, Any],
    name: str,
    payload: Mapping[str, Any],
) -> None:
    """写入不可覆盖的阶段记录，并同步更新 manifest 指纹。"""
    path = run_dir / "stages" / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"A4-P2 completed stage overwrite forbidden: {path}")
    atomic_json(path, payload)
    manifest.setdefault("stage_hashes", {})[name] = sha256_file(path)
    atomic_json(run_dir / "manifest.json", manifest, replace=True)


def load_stage(run_dir: Path, manifest: Mapping[str, Any], name: str) -> dict[str, Any]:
    path = run_dir / "stages" / f"{name}.json"
    expected = manifest.get("stage_hashes", {}).get(name)
    if not expected or sha256_file(path) != expected:
        raise RuntimeError(f"A4-P2 completed stage hash drift: {name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "done" or payload.get("stage") != name:
        raise RuntimeError(f"A4-P2 completed stage payload invalid: {name}")
    return payload


def resolve_snapshot_paths(protocol_path: Path) -> list[Path]:
    return [
        protocol_path,
        PROJECT / "motion_proj/worldsim_v3/mixed_precision.py",
        PROJECT / "motion_proj/worldsim_v3/actor_metrics.py",
        PROJECT / "motion_proj/worldsim_v3/contribution_prune.py",
        PROJECT / "scripts/run_worldsim_v3_a4_p2_precision.py",
        PROJECT / "scripts/run_worldsim_v3_a4_p2_worker.py",
        PROJECT / "scripts/aggregate_worldsim_v3_a4_p2.py",
        PROJECT / "scripts/audit_worldsim_v3_a4_p2_resume.py",
        PROJECT / "scripts/finalize_worldsim_v3_a4_p2.py",
        PROJECT / "scripts/run_worldsim_v3_a4_p2_precision.sh",
        PROJECT / "scripts/validate_worldsim_v3_a4_p2_mixed_precision_protocol.py",
        PROJECT / "scripts/run_worldsim_v3_a4_p1_worker.py",
        PROJECT / "scripts/run_worldsim_v3_a4_p0_profile.py",
        PROJECT / "scripts/eval_worldsim_v3_a0_actor_metrics.py",
        PROJECT / "scripts/eval_worldsim_v3_a3_r1_heldout.py",
    ]


def main() -> None:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite A4-P2 run: {args.run_dir}")
    if args.run_dir.parent != RUN_ROOT:
        raise ValueError(f"A4-P2 run must be a direct child of {RUN_ROOT}")
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    input_audits = validate_inputs(protocol)
    gpu_rows = nvidia_compute_rows()
    if gpu_rows:
        raise RuntimeError(f"A4-P2 GPU preflight not idle: {gpu_rows}")
    disk_free = shutil.disk_usage(args.run_dir.parent).free
    floor = int(protocol["resource_ceilings"]["disk_free_floor_bytes"])
    if disk_free < floor:
        raise RuntimeError(f"A4-P2 disk preflight failed: {disk_free} < {floor}")

    args.run_dir.mkdir(parents=True)
    _ACTIVE_RUN_DIR = args.run_dir
    for relative in ("stages", "artifacts", "artifacts/candidates", "artifacts/quality"):
        (args.run_dir / relative).mkdir()
    source_hashes = snapshot_sources(args.run_dir, resolve_snapshot_paths(args.protocol))
    selected = protocol["selected_asset"]
    manifest = {
        "schema_version": 1,
        "status": "running",
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "scene": protocol["scene"],
        "seed": int(protocol["seed"]),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(args.protocol),
        "project_commit": command_output("git", "rev-parse", "HEAD"),
        "project_status": command_output("git", "status", "--short").splitlines(),
        "source_hashes": source_hashes,
        "input_audits": input_audits,
        "source_inputs_before": {
            name: sha256_file(Path(selected[name]["path"]))
            for name in ("checkpoint", "source_config", "actor_registry")
        },
        "stage_hashes": {},
        "preflight": {
            "gpu_compute_rows": gpu_rows,
            "disk_free_bytes": disk_free,
            "cgroup_memory_bytes": cgroup_memory_current(),
            "cgroup_memory_events": cgroup_memory_events(),
        },
    }
    atomic_json(args.run_dir / "manifest.json", manifest)

    evidence = protocol["p1_canonical_evidence"]
    p1_summary = json.loads(Path(evidence["summary"]["path"]).read_text(encoding="utf-8"))
    p1_terminal = json.loads(Path(evidence["terminal"]["path"]).read_text(encoding="utf-8"))
    input_stage = {
        "status": "done",
        "stage": "input_audit",
        "duration_seconds": 0.0,
        "input_audits": input_audits,
        "input_count": len(input_audits),
        "input_bytes": sum(
            int(row.get("bytes", row.get("total_bytes", 0)))
            for row in input_audits.values()
        ),
        "p1_status": p1_summary.get("status"),
        "p1_selected_arm": p1_summary.get("selection", {}).get("selected_arm"),
        "p1_fallback_exact_alias": p1_summary.get("selection", {}).get("fallback_exact_alias"),
        "p1_terminal": p1_terminal,
        "training_optimizer_or_source_mutation_performed": False,
        "minimum_rerun_unit": "input_audit_and_downstream",
    }
    write_stage(args.run_dir, manifest, "input_audit", input_stage)
    print(json.dumps({"status": "initialized", "run_dir": str(args.run_dir)}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.exists():
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A4_P2_INITIALIZATION_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
