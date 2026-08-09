#!/usr/bin/env python
"""执行 A4-P5 输入审计并物化 reference-only deployment registry。"""

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
PROTOCOL = PROJECT / "configs/worldsim_v3/a4_p5_registry_resume_protocol_v1.yaml"
_ACTIVE_RUN_DIR: Path | None = None


from scripts.validate_worldsim_v3_a4_p5_registry_resume_protocol import (
    validate_inputs,
    validate_schema,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any], *, replace: bool = False) -> None:
    if path.exists() and not replace:
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


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
    path = run_dir / "stages" / f"{name}.json"
    if path.exists():
        raise FileExistsError(f"A4-P5 completed stage overwrite forbidden: {path}")
    atomic_json(path, payload)
    manifest.setdefault("stage_hashes", {})[name] = sha256_file(path)
    atomic_json(run_dir / "manifest.json", manifest, replace=True)


def build_compact_registry(
    protocol: Mapping[str, Any], source_registry: Mapping[str, Any]
) -> dict[str, Any]:
    expected_embedded = protocol["selected_asset"]["actor_registry"][
        "embedded_registry_sha256"
    ]
    unhashed_source = dict(source_registry)
    unhashed_source.pop("actor_registry_sha256", None)
    if canonical_sha256(unhashed_source) != expected_embedded:
        raise RuntimeError("A4-P5 source actor registry embedded hash drift")
    actor_contract = protocol["registry_contract"]["actor_assets"]
    actors = []
    for actor in source_registry["actors"]:
        model_index = int(actor["rigid_model_index"])
        tensor_slice = actor["checkpoint_tensor_slice"]
        actors.append(
            {
                "asset_id": f"actor-{model_index:03d}",
                "rigid_model_index": model_index,
                "instance_token": actor["instance_token"],
                "class_name": actor["class_name"],
                "availability": actor["availability"],
                "selector": tensor_slice["selector"],
                "gaussian_count": int(tensor_slice["gaussian_count"]),
                "flat_indices_sha256": tensor_slice["flat_indices_sha256"],
                "source_registry_sha256": protocol["selected_asset"][
                    "actor_registry"
                ]["sha256"],
            }
        )
    actors.sort(key=lambda row: row["rigid_model_index"])
    required_fields = set(actor_contract["required_compact_fields"])
    if any(set(actor) != required_fields for actor in actors):
        raise RuntimeError("A4-P5 compact actor field drift")
    selected = protocol["selected_asset"]
    static_contract = protocol["registry_contract"]["static_asset"]
    payload = {
        "schema_version": protocol["registry_contract"]["schema_version"],
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "scene": protocol["scene"],
        "seed": int(protocol["seed"]),
        "source_assets": {
            name: {
                "path": selected[name]["path"],
                "sha256": selected[name]["sha256"],
                "bytes": int(selected[name]["bytes"]),
                "storage": "external_immutable_reference",
            }
            for name in ("checkpoint", "source_config", "actor_registry")
        },
        "static_assets": [
            {
                **static_contract,
                "checkpoint_path": selected["checkpoint"]["path"],
                "checkpoint_sha256": selected["checkpoint"]["sha256"],
            }
        ],
        "actor_assets": actors,
        "totals": {
            "static_asset_count": int(static_contract["asset_count"]),
            "static_gaussian_count": int(static_contract["gaussian_count"]),
            "actor_asset_count": len(actors),
            "available_actor_count": sum(
                actor["availability"] == "available" for actor in actors
            ),
            "unavailable_actor_count": sum(
                actor["availability"] != "available" for actor in actors
            ),
            "actor_gaussian_count": sum(actor["gaussian_count"] for actor in actors),
            "total_gaussian_count": int(static_contract["gaussian_count"])
            + sum(actor["gaussian_count"] for actor in actors),
        },
        "recovery": {
            "stage_order": protocol["recovery_contract"]["stage_order"],
            "minimum_rerun_units": protocol["recovery_contract"][
                "minimum_rerun_units"
            ],
            "completed_stage_policy": protocol["recovery_contract"][
                "completed_stage_policy"
            ],
        },
    }
    payload["registry_sha256"] = canonical_sha256(payload)
    return payload


def main() -> None:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(f"refusing to overwrite A4-P5 run: {args.run_dir}")
    protocol = OmegaConf.to_container(OmegaConf.load(args.protocol), resolve=True)
    validate_schema(protocol)
    input_audits = validate_inputs(protocol)
    gpu_rows = nvidia_compute_rows()
    if gpu_rows:
        raise RuntimeError(f"A4-P5 GPU preflight not idle: {gpu_rows}")
    disk_free = shutil.disk_usage(args.run_dir.parent).free
    if disk_free < int(protocol["resource_ceilings"]["disk_free_floor_bytes"]):
        raise RuntimeError(f"A4-P5 disk preflight failed: {disk_free}")
    args.run_dir.mkdir(parents=True)
    _ACTIVE_RUN_DIR = args.run_dir
    (args.run_dir / "stages").mkdir()
    (args.run_dir / "artifacts").mkdir()
    source_hashes = snapshot_sources(
        args.run_dir,
        [
            args.protocol,
            PROJECT / "scripts/run_worldsim_v3_a4_p5_registry.py",
            PROJECT / "scripts/run_worldsim_v3_a4_p5_reload_smoke.py",
            PROJECT / "scripts/aggregate_worldsim_v3_a4_p5.py",
            PROJECT / "scripts/audit_worldsim_v3_a4_p5_resume.py",
            PROJECT / "scripts/finalize_worldsim_v3_a4_p5.py",
            PROJECT / "scripts/run_worldsim_v3_a4_p5_registry.sh",
            PROJECT / "scripts/validate_worldsim_v3_a4_p5_registry_resume_protocol.py",
        ],
    )
    events_before = cgroup_memory_events()
    manifest = {
        "schema_version": 1,
        "status": "running",
        "task_id": protocol["task_id"],
        "profile_id": protocol["profile_id"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": sha256_file(args.protocol),
        "project_commit": command_output("git", "rev-parse", "HEAD"),
        "project_status": command_output("git", "status", "--short").splitlines(),
        "source_hashes": source_hashes,
        "input_audits": input_audits,
        "stage_hashes": {},
        "preflight": {
            "gpu_compute_rows": gpu_rows,
            "disk_free_bytes": disk_free,
            "cgroup_memory_bytes": cgroup_memory_current(),
            "cgroup_memory_events": events_before,
        },
    }
    atomic_json(args.run_dir / "manifest.json", manifest)
    input_stage = {
        "status": "done",
        "stage": "input_audit",
        "input_audits": input_audits,
        "input_count": len(input_audits),
        "input_bytes": sum(int(row["bytes"]) for row in input_audits.values()),
        "p0_terminal": json.loads(
            Path(protocol["p0_canonical_evidence"]["terminal"]["path"]).read_text(
                encoding="utf-8"
            )
        ),
        "minimum_rerun_unit": "input_audit_and_downstream",
    }
    write_stage(args.run_dir, manifest, "input_audit", input_stage)
    source_registry = json.loads(
        Path(protocol["selected_asset"]["actor_registry"]["path"]).read_text(
            encoding="utf-8"
        )
    )
    deployment_registry = build_compact_registry(protocol, source_registry)
    registry_path = args.run_dir / protocol["registry_contract"]["output"]
    atomic_json(registry_path, deployment_registry)
    if registry_path.stat().st_size > int(
        protocol["registry_contract"]["output_bytes_ceiling"]
    ):
        raise RuntimeError("A4-P5 compact registry exceeds frozen byte ceiling")
    materialize_stage = {
        "status": "done",
        "stage": "registry_materialize",
        "mode": protocol["registry_contract"]["mode"],
        "registry": {
            "path": str(registry_path.relative_to(args.run_dir)),
            "sha256": sha256_file(registry_path),
            "bytes": registry_path.stat().st_size,
            "canonical_registry_sha256": deployment_registry["registry_sha256"],
        },
        "totals": deployment_registry["totals"],
        "checkpoint_copy_or_rewrite_performed": False,
        "minimum_rerun_unit": "registry_materialize_and_downstream",
    }
    write_stage(args.run_dir, manifest, "registry_materialize", materialize_stage)
    manifest["materialize_complete"] = True
    manifest["checkpoint_sha256_after_materialize"] = sha256_file(
        Path(protocol["selected_asset"]["checkpoint"]["path"])
    )
    manifest["actor_registry_sha256_after_materialize"] = sha256_file(
        Path(protocol["selected_asset"]["actor_registry"]["path"])
    )
    manifest["run_bytes_after_materialize"] = directory_bytes(args.run_dir)
    atomic_json(args.run_dir / "manifest.json", manifest, replace=True)
    print(json.dumps({"status": "materialize_complete", "run_dir": str(args.run_dir)}))


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if _ACTIVE_RUN_DIR is not None:
            atomic_json(
                _ACTIVE_RUN_DIR / "terminal.json",
                {
                    "status": "blocked",
                    "failure": {
                        "code": "A4_P5_MATERIALIZE_FAILED",
                        "detail": f"{type(error).__name__}: {error}",
                    },
                },
                replace=True,
            )
        raise
