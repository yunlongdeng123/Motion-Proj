"""WorldSim V6.1 ME-0：构建 SceneIR-O 与独立 Occupancy truth tiers。"""

from __future__ import annotations

import json
import multiprocessing
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from motion_proj.worldsim_v61.occupancy import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    VoxelGridSpec,
    build_observed_occupancy,
    content_sha256,
    sha256_file,
)


TASK_ID = "WS-V61-ME0-OCCIR-01"
RUNS_ROOT = Path("/root/autodl-tmp/runs")


class ME0ExperimentError(RuntimeError):
    """ME-0 正式实验合同失败。"""


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _resolve_runs_uri(uri: str) -> Path:
    if not uri.startswith("runs://"):
        raise ME0ExperimentError("只接受 runs URI")
    relative = Path(uri.removeprefix("runs://"))
    if ".." in relative.parts:
        raise ME0ExperimentError("runs URI 不得包含上级路径")
    return (RUNS_ROOT / relative).resolve()


def _verify(path: Path, expected_sha256: str) -> None:
    if not path.is_file() or sha256_file(path) != expected_sha256:
        raise ME0ExperimentError(f"冻结源漂移: {path}")


def _grid_spec(config: Mapping[str, Any], target_frame: int) -> VoxelGridSpec:
    grid = config["grid"]
    origin = np.asarray(grid["origin_m"], dtype=np.float64)
    extent = np.asarray(grid["extent_m"], dtype=np.float64)
    voxel = float(grid["voxel_size_m"])
    shape_float = (extent - origin) / voxel
    shape = tuple(int(round(value)) for value in shape_float)
    if not np.allclose(shape_float, shape, atol=1e-9, rtol=0.0):
        raise ME0ExperimentError("grid extent 不能整除 voxel size")
    return VoxelGridSpec(
        frame=f"target_lidar_{int(target_frame):03d}",
        origin_m=tuple(float(value) for value in origin),
        voxel_size_m=voxel,
        shape=shape,
    )


def _build_pair_task(task: Mapping[str, Any]) -> dict[str, Any]:
    config = task["config"]
    scene = str(task["scene"])
    target_frame = int(task["target_frame"])
    scene_root = Path(task["scene_root"])
    spec = _grid_spec(config, target_frame)
    result: dict[str, Any] = {"scene": scene, "target_frame": target_frame, "tiers": {}}
    for tier, frames in task["tier_frames"].items():
        arrays, audit = build_observed_occupancy(
            scene_root,
            target_frame,
            frames,
            spec,
            record_width=int(config["raw_lidar"]["point_record_float32_width"]),
            dynamic_box_margin_m=float(config["raw_lidar"]["dynamic_box_margin_m"]),
            maximum_free_rays_per_sweep=int(config["ray_carving"]["maximum_rays_per_sweep"]),
            maximum_range_m=float(config["ray_carving"]["maximum_range_m"]),
        )
        output_root = task.get("output_root")
        relative = f"evidence/{scene}/f{target_frame:03d}/{tier}.npz"
        if output_root:
            path = Path(output_root) / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, **arrays)
            audit["artifact_path"] = relative
            audit["artifact_sha256"] = sha256_file(path)
            audit["artifact_bytes"] = path.stat().st_size
        else:
            audit["artifact_path"] = None
            audit["artifact_sha256"] = None
            audit["artifact_bytes"] = None
        result["tiers"][tier] = audit
    return result


def _parallel_build(tasks: list[dict[str, Any]], maximum_workers: int) -> list[dict[str, Any]]:
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=maximum_workers, mp_context=context) as executor:
        return list(executor.map(_build_pair_task, tasks))


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise ME0ExperimentError("正式 ME-0 run 要求干净工作树")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise ME0ExperimentError("ME-0 task_id 漂移")

    p0_run = _resolve_runs_uri(config["sources"]["p0_run"])
    for name, expected in config["sources"]["p0_files"].items():
        _verify(p0_run / name, expected)
    p0_gate = json.loads((p0_run / "P0_GATE.json").read_text(encoding="utf-8"))
    r9_cases_path = Path(config["sources"]["r9_cases_path"])
    _verify(r9_cases_path, config["sources"]["r9_cases_sha256"])
    cases = _read_jsonl(r9_cases_path)

    run_root.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(run_root).free / 1024**3
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__occir-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        tasks: list[dict[str, Any]] = []
        for scene, scene_spec in sorted(config["raw_evidence"].items()):
            for target_frame in config["cohort"]["frame_indices"]:
                split = scene_spec["sweep_offsets_by_target"][int(target_frame)]
                tasks.append(
                    {
                        "config": {
                            "grid": config["grid"],
                            "raw_lidar": config["raw_lidar"],
                            "ray_carving": config["ray_carving"],
                        },
                        "scene": scene,
                        "target_frame": int(target_frame),
                        "scene_root": scene_spec["processed_scene_root"],
                        "tier_frames": {
                            "O_method": [int(target_frame) + int(value) for value in split["method"]],
                            "O_eval": [int(target_frame) + int(value) for value in split["eval"]],
                        },
                        "output_root": str(run_dir),
                    }
                )
        maximum_workers = min(int(config["resources"]["maximum_cpu_workers"]), len(tasks))
        repeat1 = _parallel_build(tasks, maximum_workers)
        repeat_tasks = [{**task, "output_root": None} for task in tasks]
        repeat2 = _parallel_build(repeat_tasks, maximum_workers)

        repeat1_index = {(row["scene"], row["target_frame"]): row for row in repeat1}
        repeat2_index = {(row["scene"], row["target_frame"]): row for row in repeat2}
        contracts: list[dict[str, Any]] = []
        sceneir_rows: list[dict[str, Any]] = []
        all_method_paths: set[str] = set()
        all_eval_paths: set[str] = set()
        all_method_hashes: set[str] = set()
        all_eval_hashes: set[str] = set()
        repeat_exact = True
        for key in sorted(repeat1_index):
            first = repeat1_index[key]
            second = repeat2_index[key]
            for tier in ("O_method", "O_eval"):
                audit = first["tiers"][tier]
                repeat_audit = second["tiers"][tier]
                exact = audit["content_sha256"] == repeat_audit["content_sha256"]
                repeat_exact &= exact
                payload_paths = {row["path"] for row in audit["source_files"]}
                payload_hashes = {row["sha256"] for row in audit["source_files"]}
                if tier == "O_method":
                    all_method_paths.update(payload_paths)
                    all_method_hashes.update(payload_hashes)
                else:
                    all_eval_paths.update(payload_paths)
                    all_eval_hashes.update(payload_hashes)
                contracts.append(
                    {
                        "schema_version": "worldsim_v61.occupancy_contract.v1",
                        "scene": first["scene"],
                        "target_frame": first["target_frame"],
                        "tier": tier,
                        "fresh_process_content_exact": exact,
                        **audit,
                    }
                )
            sceneir_rows.append(
                {
                    "schema_version": "worldsim_v61.sceneir_o_index.v1",
                    "scene": first["scene"],
                    "target_frame": first["target_frame"],
                    "appearance_state": "R10_cross_frontend_reconstructed_proposal_reference_only",
                    "occupancy_evidence": {
                        tier: {
                            "path": first["tiers"][tier]["artifact_path"],
                            "content_sha256": first["tiers"][tier]["content_sha256"],
                            "source_payload_sha256": first["tiers"][tier]["source_payload_sha256"],
                        }
                        for tier in ("O_method", "O_eval")
                    },
                    "collision_assets": "not_started",
                    "predicted_occupancy": "not_started",
                    "task_validity": "contract_only_no_quality_acceptance",
                }
            )

        binding_rows = []
        sceneir_index = {(row["scene"], row["target_frame"]): row for row in sceneir_rows}
        for case in cases:
            key = (case["scene"], int(case["frame_index"]))
            entry = sceneir_index[key]
            binding_rows.append(
                {
                    "schema_version": "worldsim_v61.case_evidence_binding.v1",
                    "case_id": case["case_id"],
                    "scene": case["scene"],
                    "frame_index": int(case["frame_index"]),
                    "frontend": case["frontend"],
                    "hole_type": case["hole_type"],
                    "O_method_content_sha256": entry["occupancy_evidence"]["O_method"]["content_sha256"],
                    "O_eval_content_sha256": entry["occupancy_evidence"]["O_eval"]["content_sha256"],
                }
            )

        state_nonzero = all(
            row["state_counts"]["unknown"] > 0
            and row["state_counts"]["free"] > 0
            and row["state_counts"]["occupied"] > 0
            for row in contracts
        )
        oriented_not_aabb = all(
            actor["oriented_voxel_count"] <= actor["corner_aabb_voxel_count"]
            for row in contracts
            for actor in row["actor_rows"]
        ) and any(row["strict_aabb_reduction_actor_count"] > 0 for row in contracts)
        actor_identity_lifecycle = all(
            row["actor_count"] == row["actor_identity_unique_count"]
            and row["actor_count"] > 0
            and all(actor["lifecycle_active"] for actor in row["actor_rows"])
            for row in contracts
        )
        removal_unknown = all(
            row["source_removal_unknown_count"] == row["actor_sparse_voxel_count"]
            for row in contracts
        )
        method_eval_content_distinct = all(
            row["occupancy_evidence"]["O_method"]["content_sha256"]
            != row["occupancy_evidence"]["O_eval"]["content_sha256"]
            for row in sceneir_rows
        )
        wall_seconds = time.monotonic() - started
        core_source = (repo_root / "motion_proj/worldsim_v61/occupancy.py").read_text(encoding="utf-8")
        checks = {
            "p0_authority_accepted": bool(p0_gate["checks"]["passed"]),
            "four_scene_frame_units_and_eight_tiers": len(sceneir_rows) == 4 and len(contracts) == 8,
            "case_binding_denominator_exact": len(binding_rows) == int(config["cohort"]["expected_case_count"])
            and len({row["case_id"] for row in binding_rows}) == len(binding_rows),
            "coordinate_roundtrip_exact": max(row["coordinate_roundtrip_max_abs_m"] for row in contracts)
            <= float(config["gate"]["maximum_roundtrip_error_m"]),
            "oriented_volume_without_corner_aabb_inflation": oriented_not_aabb,
            "actor_identity_and_lifecycle_exact": actor_identity_lifecycle,
            "free_occupied_unknown_all_nonzero": state_nonzero,
            "source_removal_restores_unknown": removal_unknown,
            "method_eval_paths_disjoint": all_method_paths.isdisjoint(all_eval_paths),
            "method_eval_payload_hashes_disjoint": all_method_hashes.isdisjoint(all_eval_hashes),
            "method_eval_content_distinct": method_eval_content_distinct,
            "fresh_process_content_exact": repeat_exact,
            "core_has_no_absolute_project_path": "/root/autodl-tmp" not in core_source,
            "voxel_size_not_legacy_0p4": float(config["grid"]["voxel_size_m"]) != 0.4,
            "confirmation_locked": bool(config["cohort"]["confirmation_locked"]),
            "no_gpu_generator_training_or_prediction": True,
            "disk_budget_sufficient": free_gib >= float(config["resources"]["minimum_disk_free_gib"]),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
        }
        checks["passed"] = all(checks.values())

        _write_jsonl(run_dir / "OCCUPANCY_CONTRACTS.jsonl", contracts)
        _write_jsonl(run_dir / "SCENEIR_O_INDEX.jsonl", sceneir_rows)
        _write_jsonl(run_dir / "CASE_BINDINGS.jsonl", binding_rows)
        _write_json(
            run_dir / "FRESH_PROCESS_AUDIT.json",
            {
                "schema_version": "worldsim_v61.fresh_process_audit.v1",
                "repeat_exact": repeat_exact,
                "repeat1": {
                    f"{row['scene']}__f{row['target_frame']:03d}": {
                        tier: row["tiers"][tier]["content_sha256"] for tier in ("O_method", "O_eval")
                    }
                    for row in repeat1
                },
                "repeat2": {
                    f"{row['scene']}__f{row['target_frame']:03d}": {
                        tier: row["tiers"][tier]["content_sha256"] for tier in ("O_method", "O_eval")
                    }
                    for row in repeat2
                },
            },
        )
        _write_json(
            run_dir / "SOURCE_TIER_AUDIT.json",
            {
                "schema_version": "worldsim_v61.source_tier_audit.v1",
                "method_unique_path_count": len(all_method_paths),
                "eval_unique_path_count": len(all_eval_paths),
                "method_eval_paths_disjoint": all_method_paths.isdisjoint(all_eval_paths),
                "method_eval_payload_hashes_disjoint": all_method_hashes.isdisjoint(all_eval_hashes),
                "method_path_index_sha256": content_sha256(sorted(all_method_paths)),
                "eval_path_index_sha256": content_sha256(sorted(all_eval_paths)),
            },
        )
        _write_json(
            run_dir / "ME0_GATE.json",
            {
                "schema_version": "worldsim_v61.me0_gate.v1",
                "checks": checks,
                "decision": "proceed_to_me1_oracle_occupancy" if checks["passed"] else "repair_sceneir_o_contract",
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v61.me0_resource_audit.v1",
                "gpu_used": False,
                "generator_started": False,
                "training_started": False,
                "prediction_started": False,
                "maximum_cpu_workers": maximum_workers,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
            },
        )
        status = "done" if checks["passed"] else "rejected"
        _write_json(
            run_dir / "SUMMARY.json",
            {
                "schema_version": "worldsim_v61.me0_summary.v1",
                "task_id": TASK_ID,
                "hypothesis_id": config["hypothesis_id"],
                "status": status,
                "hypothesis_outcome": "accepted_sceneir_o_contract" if checks["passed"] else "rejected",
                "source_commit": source_commit,
                "scene_frame_unit_count": len(sceneir_rows),
                "occupancy_tier_count": len(contracts),
                "case_binding_count": len(binding_rows),
                "maximum_coordinate_roundtrip_error_m": max(
                    row["coordinate_roundtrip_max_abs_m"] for row in contracts
                ),
                "fresh_process_exact": repeat_exact,
                "method_eval_paths_disjoint": all_method_paths.isdisjoint(all_eval_paths),
                "method_eval_payload_hashes_disjoint": all_method_hashes.isdisjoint(all_eval_hashes),
                "failure_ledger_delta": "none" if checks["passed"] else "pending_rejected_closeout",
                "next": "WS-V61-ME1-ORACLE-OCC-PROPOSAL-01" if checks["passed"] else "repair_sceneir_o_contract",
            },
        )
        tracked = [
            "OCCUPANCY_CONTRACTS.jsonl",
            "SCENEIR_O_INDEX.jsonl",
            "CASE_BINDINGS.jsonl",
            "FRESH_PROCESS_AUDIT.json",
            "SOURCE_TIER_AUDIT.json",
            "ME0_GATE.json",
            "RESOURCE_AUDIT.json",
            "SUMMARY.json",
        ]
        evidence = sorted((run_dir / "evidence").rglob("*.npz"))
        files = {
            name: {"bytes": (run_dir / name).stat().st_size, "sha256": sha256_file(run_dir / name)}
            for name in tracked
        }
        files.update(
            {
                str(path.relative_to(run_dir)): {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in evidence
            }
        )
        _write_json(
            run_dir / "MANIFEST.json",
            {"schema_version": "worldsim_v61.me0_manifest.v1", "files": files},
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.terminal.v1",
                "status": status,
                "manifest_sha256": sha256_file(run_dir / "MANIFEST.json"),
                "summary_sha256": sha256_file(run_dir / "SUMMARY.json"),
            },
        )
        print(run_dir, flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v61.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
