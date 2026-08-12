#!/usr/bin/env python3
"""构建 V3.3 S4 reference-only base + 小型空间 delta 资产包。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

_ACTIVE_RUN_DIR: Path | None = None

SUPPORTED_TASK_IDS = frozenset(
    {
        "WS-V33-S4-SPATIAL-DELTA-01",
        "WS-V4-B0-MATCHED-BASELINES-01",
        "WS-V4-M1-EVIDENCE-FIELD-01",
    }
)

from motion_proj.worldsim_v33.instance_field import load_instance_field  # noqa: E402
from motion_proj.worldsim_v33.roadpatch import (  # noqa: E402
    load_patch_delta,
    validate_patch_delta,
)
from motion_proj.worldsim_v33.spatial_delta import (  # noqa: E402
    PACKAGE_SCHEMA_VERSION,
    STACK_SCHEMA_VERSION,
    atomic_json,
    atomic_save_actor_insert_delta,
    atomic_save_npz,
    build_actor_insert_delta,
    build_erase_delta,
    load_actor_insert_delta,
    load_erase_delta,
    ordered_stack_manifest,
    sha256_arrays,
    sha256_file,
    validate_stack_manifest,
)


def verify_file(spec: Mapping[str, Any], role: str) -> dict[str, Any]:
    path = Path(spec["path"])
    if not path.is_file():
        raise FileNotFoundError(f"{role} 不存在: {path}")
    actual = sha256_file(path)
    if actual != spec["sha256"]:
        raise RuntimeError(
            f"{role} SHA 漂移: expected={spec['sha256']} actual={actual}"
        )
    return {"path": str(path), "sha256": actual, "bytes": path.stat().st_size}


def filter_background_delta(
    delta: Mapping[str, np.ndarray], *, target_role: str
) -> dict[str, np.ndarray]:
    selected = np.asarray(delta["target_role"]).astype(str) == str(target_role)
    if not selected.any():
        raise ValueError(f"RoadPatch 不含 target_role={target_role}")
    output = {name: np.asarray(value)[selected].copy() for name, value in delta.items()}
    validate_patch_delta(output)
    return output


def file_record(path: Path, package_root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(package_root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_stack(
    root: Path, *, stack_id: str, operations: list[dict[str, Any]]
) -> Path:
    path = root / "stacks" / f"{stack_id}.json"
    atomic_json(
        path,
        ordered_stack_manifest(stack_id=stack_id, operations=operations),
    )
    validate_stack_manifest(json.loads(path.read_text(encoding="utf-8")))
    return path


def available_stack_ids(*, has_background: bool, has_actor: bool) -> list[str]:
    stacks = ["base_only", "erase"]
    if has_background:
        stacks.append("erase_background")
    if has_actor:
        stacks.append("actor_override")
    if has_background and has_actor:
        stacks.append("full")
    return stacks


def main() -> int:
    global _ACTIVE_RUN_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if config.get("schema_version") not in {
        "worldsim_v33_s4_spatial_delta_v1",
        "worldsim_v4_v33_spatial_delta_v1",
    }:
        raise ValueError("S4 config schema version 漂移")
    if config.get("task_id") not in SUPPORTED_TASK_IDS:
        raise ValueError("S4 task_id 漂移")
    if tuple(config["composition"]["order"]) != (
        "ERASE",
        "INSERT_BACKGROUND",
        "INSERT_ACTOR",
        "RENDER_ONLY",
    ):
        raise ValueError("S4 composition order 漂移")
    run_dir = args.run_dir.resolve()
    _ACTIVE_RUN_DIR = run_dir
    artifacts = run_dir / "artifacts"
    package = artifacts / "worldsim_asset"
    if package.exists():
        raise FileExistsError(package)
    package.mkdir(parents=True)
    atomic_json(
        run_dir / "status.json",
        {
            "state": "running",
            "task_id": config["task_id"],
            "stage": "package",
            "started_unix": started,
        },
    )

    verified = {
        name: verify_file(spec, name)
        for name, spec in config["inputs"].items()
    }
    checkpoint_before = verified["checkpoint"]["sha256"]
    registry_before = verified["actor_registry"]["sha256"]
    field = load_instance_field(config["inputs"]["s1_instance_field"]["path"])
    actor = config["actor"]
    erase = build_erase_delta(
        field,
        instance_id=int(actor["dataset_instance_id"]),
        minimum_background_instance_opacity=float(
            config["composition"]["minimum_background_instance_opacity"]
        ),
    )
    erase_counts = {
        "Background": int(np.sum(np.asarray(erase["model_code"]) == 0)),
        "RigidNodes": int(np.sum(np.asarray(erase["model_code"]) == 1)),
    }
    expected_counts = config["gates"].get("erase_counts")
    if expected_counts is not None:
        expected_counts = {
            name: int(value) for name, value in expected_counts.items()
        }
    if expected_counts is not None and erase_counts != expected_counts:
        raise RuntimeError(
            f"ERASE count 漂移: expected={expected_counts} actual={erase_counts}"
        )

    has_background = "s2_roadpatch_delta" in config["inputs"]
    has_actor = "s3_actor_asset" in config["inputs"]
    background = None
    if has_background:
        parent_background = load_patch_delta(
            config["inputs"]["s2_roadpatch_delta"]["path"]
        )
        background = filter_background_delta(
            parent_background, target_role=str(actor["role"])
        )
    actor_delta = None
    if has_actor:
        with np.load(
            config["inputs"]["s3_actor_asset"]["path"], allow_pickle=False
        ) as payload:
            source_asset = {name: payload[name].copy() for name in payload.files}
        actor_delta = build_actor_insert_delta(
            source_asset,
            instance_id=int(actor["dataset_instance_id"]),
            instance_token=str(actor["instance_token"]),
            rigid_model_index=int(actor["rigid_model_index"]),
        )

    base_dir = package / "base"
    delete_dir = package / f"deltas/delete_actor_{actor['dataset_instance_id']}"
    actor_dir = package / f"deltas/actor_override_{actor['dataset_instance_id']}"
    base_dir.mkdir(parents=True)
    delete_dir.mkdir(parents=True)
    if has_actor:
        actor_dir.mkdir(parents=True)
    atomic_json(
        base_dir / "checkpoint.ref.json",
        {
            "kind": "immutable_external_reference",
            "role": "base_checkpoint",
            **verified["checkpoint"],
        },
    )
    atomic_json(
        base_dir / "actor_registry.ref.json",
        {
            "kind": "immutable_external_reference",
            "role": "actor_registry",
            **verified["actor_registry"],
        },
    )
    erase_indices = delete_dir / "erase_indices.npz"
    atomic_save_npz(erase_indices, erase)
    erase_loaded = load_erase_delta(erase_indices)
    erase_descriptor = {
        "operation_id": f"erase_actor_{actor['dataset_instance_id']}",
        "type": "ERASE",
        "reason": "S1 hard instance assignment removal without base row deletion",
        "instance_id": int(actor["dataset_instance_id"]),
        "instance_token": str(actor["instance_token"]),
        "mask_hash": str(np.asarray(erase_loaded["mask_hash"]).item()),
        "selection_policy": str(
            np.asarray(erase_loaded["selection_policy"]).item()
        ),
        "minimum_background_instance_opacity": float(
            np.asarray(
                erase_loaded["minimum_background_instance_opacity"]
            ).item()
        ),
        "counts": erase_counts,
        "payload": file_record(erase_indices, package),
        "base_rows_deleted": 0,
        "effective_opacity": 0.0,
    }
    atomic_json(delete_dir / "erase.json", erase_descriptor)
    background_manifest = None
    if background is not None:
        background_path = delete_dir / "background_patch.npz"
        atomic_save_npz(background_path, background)
        background_manifest = {
            "operation_id": f"insert_background_actor_{actor['dataset_instance_id']}",
            "type": "INSERT_BACKGROUND",
            "provenance": "GENERATED_BY_PATCH_REUSE",
            "target_role": str(actor["role"]),
            "rows": int(len(background["means"])),
            "arrays_sha256": sha256_arrays(background),
            "payload": file_record(background_path, package),
            "parent_delta": verified["s2_roadpatch_delta"],
        }
        atomic_json(delete_dir / "manifest.json", background_manifest)

    actor_loaded = None
    actor_manifest = None
    if actor_delta is not None:
        actor_path = actor_dir / "actor_override.npz"
        atomic_save_actor_insert_delta(actor_path, actor_delta)
        actor_loaded = load_actor_insert_delta(actor_path)
        actor_manifest = {
            "operation_id": f"insert_actor_{actor['dataset_instance_id']}",
            "type": "INSERT_ACTOR",
            "provenance": "GENERATED_ACTOR",
            "instance_id": int(actor["dataset_instance_id"]),
            "instance_token": str(actor["instance_token"]),
            "rigid_model_index": int(actor["rigid_model_index"]),
            "rows": int(len(actor_loaded["means"])),
            "arrays_sha256": sha256_arrays(actor_loaded),
            "payload": file_record(actor_path, package),
            "parent_asset": verified["s3_actor_asset"],
            "base_rows_deleted": 0,
        }
        atomic_json(actor_dir / "manifest.json", actor_manifest)

    erase_op = {
        "operation_id": erase_descriptor["operation_id"],
        "type": "ERASE",
        "manifest": (delete_dir / "erase.json").relative_to(package).as_posix(),
    }
    render_op = {"operation_id": "render_only", "type": "RENDER_ONLY"}
    stack_paths = [
        write_stack(package, stack_id="base_only", operations=[render_op]),
        write_stack(package, stack_id="erase", operations=[erase_op, render_op]),
    ]
    background_op = None
    if background_manifest is not None:
        background_op = {
            "operation_id": background_manifest["operation_id"],
            "type": "INSERT_BACKGROUND",
            "manifest": (delete_dir / "manifest.json").relative_to(package).as_posix(),
        }
        stack_paths.append(
            write_stack(
                package,
                stack_id="erase_background",
                operations=[erase_op, background_op, render_op],
            )
        )
    actor_op = None
    if actor_manifest is not None:
        actor_op = {
            "operation_id": actor_manifest["operation_id"],
            "type": "INSERT_ACTOR",
            "manifest": (actor_dir / "manifest.json").relative_to(package).as_posix(),
        }
        stack_paths.append(
            write_stack(
                package,
                stack_id="actor_override",
                operations=[erase_op, actor_op, render_op],
            )
        )
    if background_op is not None and actor_op is not None:
        stack_paths.append(
            write_stack(
                package,
                stack_id="full",
                operations=[erase_op, background_op, actor_op, render_op],
            )
        )
    actual_stack_ids = [path.stem for path in stack_paths]
    expected_stack_ids = available_stack_ids(
        has_background=has_background, has_actor=has_actor
    )
    if actual_stack_ids != expected_stack_ids:
        raise RuntimeError("S4 stack availability 漂移")
    if list(config["composition"].get("stacks", actual_stack_ids)) != actual_stack_ids:
        raise RuntimeError("S4 config/actual stack 集漂移")

    inventory = [
        file_record(path, package)
        for path in sorted(package.rglob("*"))
        if path.is_file()
    ]
    maximum_payload = max(row["bytes"] for row in inventory)
    if maximum_payload > int(config["gates"]["maximum_package_file_bytes"]):
        raise RuntimeError("S4 package 意外包含大文件或完整 checkpoint")
    if any(Path(row["path"]).suffix in {".pth", ".ckpt"} for row in inventory):
        raise RuntimeError("S4 reference-only package 禁止复制 checkpoint")
    package_manifest = {
        "schema_version": PACKAGE_SCHEMA_VERSION,
        "task_id": config["task_id"],
        "authoring_state": "base_plus_external_delta",
        "base": {
            "checkpoint": verified["checkpoint"],
            "actor_registry": verified["actor_registry"],
            "copied_into_package": False,
        },
        "actor": actor,
        "composition_order": config["composition"]["order"],
        "available_stacks": actual_stack_ids,
        "stage_abstentions": config.get("stage_abstentions", {}),
        "stacks": [file_record(path, package) for path in stack_paths],
        "inventory": inventory,
        "invariants": {
            "base_checkpoint_immutable": True,
            "actor_registry_immutable": True,
            "base_rows_deleted": 0,
            "individual_deltas_toggleable": True,
            "deterministic_operation_order": True,
            "duplicate_insert_indices": 0,
            "full_checkpoint_copy_count": 0,
            "all_insert_rows_have_provenance": True,
        },
    }
    package_manifest_path = package / "package_manifest.json"
    atomic_json(package_manifest_path, package_manifest)

    checkpoint_after = sha256_file(config["inputs"]["checkpoint"]["path"])
    registry_after = sha256_file(config["inputs"]["actor_registry"]["path"])
    if checkpoint_after != checkpoint_before or registry_after != registry_before:
        raise RuntimeError("S4 package authoring 修改了 base checkpoint/registry")
    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir()
    for source in (
        args.config.resolve(),
        Path(__file__).resolve(),
        PROJECT / "motion_proj/worldsim_v33/spatial_delta.py",
    ):
        shutil.copy2(source, source_snapshot / source.name)
    summary = {
        "schema_version": "worldsim_v33_s4_spatial_delta_package_summary_v1",
        "task_id": config["task_id"],
        "state": "completed",
        "verified_inputs": verified,
        "erase_counts": erase_counts,
        "background_insert_rows": (
            0 if background is None else int(len(background["means"]))
        ),
        "actor_insert_rows": (
            0 if actor_loaded is None else int(len(actor_loaded["means"]))
        ),
        "available_stacks": actual_stack_ids,
        "stage_abstentions": config.get("stage_abstentions", {}),
        "package_manifest": {
            **file_record(package_manifest_path, package),
            "absolute_path": str(package_manifest_path),
        },
        "package_bytes": sum(path.stat().st_size for path in package.rglob("*") if path.is_file()),
        "maximum_package_file_bytes": maximum_payload,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "actor_registry_sha256_before": registry_before,
        "actor_registry_sha256_after": registry_after,
        "torch_checkpoint_loaded": False,
        "training_performed": False,
        "elapsed_seconds": time.time() - started,
        "source_snapshot": {
            path.name: {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(source_snapshot.iterdir())
        },
    }
    summary_path = run_dir / "summary.json"
    atomic_json(summary_path, summary)
    atomic_json(
        run_dir / "status.json",
        {
            "state": "completed",
            "task_id": config["task_id"],
            "stage": "package",
            "summary_sha256": sha256_file(summary_path),
            "package_manifest_sha256": sha256_file(package_manifest_path),
            "completed_unix": time.time(),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as error:
        if _ACTIVE_RUN_DIR is not None and _ACTIVE_RUN_DIR.is_dir():
            atomic_json(
                _ACTIVE_RUN_DIR / "status.json",
                {
                    "state": "failed",
                    "stage": "package",
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "failed_unix": time.time(),
                },
            )
        raise
