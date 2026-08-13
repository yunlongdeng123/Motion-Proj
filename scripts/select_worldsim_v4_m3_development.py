#!/usr/bin/env python3
"""聚合 M3 development denominator，并冻结轨迹与 warp 参数。"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v4.temporal_protocol import build_arm_trajectories  # noqa: E402
from scripts.run_worldsim_v4_m3_scene import (  # noqa: E402
    TASK_ID,
    actor_transforms,
    atomic_json,
    git_dirty,
    git_head,
    output_manifest,
    sha256_file,
    verify_binding,
    write_jsonl,
)


class M3DevelopmentSelectionError(RuntimeError):
    pass


def verify_run(binding: Mapping[str, Any], label: str) -> tuple[Path, dict[str, Any]]:
    run = Path(str(binding["path"])).resolve()
    summary_path = run / "summary.json"
    manifest_path = run / "manifest.json"
    if not summary_path.is_file() or not manifest_path.is_file():
        raise M3DevelopmentSelectionError(f"{label} 缺 summary/manifest")
    if sha256_file(summary_path) != binding["summary_sha256"]:
        raise M3DevelopmentSelectionError(f"{label} summary SHA 不匹配")
    if sha256_file(manifest_path) != binding["manifest_sha256"]:
        raise M3DevelopmentSelectionError(f"{label} manifest SHA 不匹配")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("test_quality_read") is not False:
        raise M3DevelopmentSelectionError(f"{label} 触碰了 test quality")
    return run, summary


def checkpoint_actor_transforms(
    scene_binding: Mapping[str, Any], frames: list[int]
) -> np.ndarray:
    checkpoint = verify_binding(scene_binding["checkpoint"], "trajectory checkpoint")
    state = torch.load(checkpoint, map_location="cpu")["models"]["RigidNodes"]

    class RigidState:
        pass

    rigid = RigidState()
    rigid.instances_trans = state["instances_trans"]
    rigid.instances_quats = state["instances_quats"]
    return actor_transforms(
        rigid, frames, int(scene_binding["actor"]["model_index"])
    )


def run(*, selection_config_path: Path, run_dir: Path) -> dict[str, Any]:
    started = time.monotonic()
    if run_dir.exists():
        raise FileExistsError(run_dir)
    if git_dirty():
        raise M3DevelopmentSelectionError("formal development selection 要求 clean worktree")
    run_dir.mkdir(parents=True)
    selection_config = yaml.safe_load(
        selection_config_path.read_text(encoding="utf-8")
    )
    m3_config_path = verify_binding(selection_config["m3_config"], "M3 config")
    inventory_path = verify_binding(
        selection_config["scene_inventory"], "M3 scene inventory"
    )
    m3_config = yaml.safe_load(m3_config_path.read_text(encoding="utf-8"))
    inventory = yaml.safe_load(inventory_path.read_text(encoding="utf-8"))
    development = {
        scene: binding
        for scene, binding in inventory["scenes"].items()
        if binding["partition"] == "development"
    }
    contract = selection_config["selection"]
    if len(development) != int(contract["development_scene_denominator"]):
        raise M3DevelopmentSelectionError("development scene denominator 不一致")
    scene_rows = []
    evaluable = []
    run_bindings = selection_config["development_scene_runs"]
    for scene, binding in development.items():
        if binding["status"] == "abstain":
            scene_rows.append(
                {
                    "scene": scene,
                    "status": "abstain",
                    "reason": binding["reason"],
                    "retained_in_denominator": True,
                    "quality_read": False,
                }
            )
            continue
        if scene not in run_bindings:
            raise M3DevelopmentSelectionError(f"ready scene 缺 development run: {scene}")
        _, summary = verify_run(run_bindings[scene], scene)
        if summary["scene"] != scene or summary["partition"] != "development":
            raise M3DevelopmentSelectionError(f"scene run identity 不一致: {scene}")
        if summary["status"] == "abstain":
            if summary["reason"] != "ABSTAIN_NO_RENDERED_CLIP_SUPPORT":
                raise M3DevelopmentSelectionError(f"未知 runtime abstain: {scene}")
            scene_rows.append(
                {
                    "scene": scene,
                    "status": "abstain",
                    "reason": summary["reason"],
                    "retained_in_denominator": True,
                    "quality_read": True,
                    "source_run": run_bindings[scene]["path"],
                }
            )
            continue
        if summary["status"] != "done" or not summary.get("rollback_exact"):
            raise M3DevelopmentSelectionError(f"evaluable scene 非 done/rollback exact: {scene}")
        support = summary["trajectory_audit"]["evidence_support_pixels"]
        if sum(support) <= 0:
            raise M3DevelopmentSelectionError(f"evaluable scene 支持为空: {scene}")
        frames = [int(value) for value in summary["clip"]["processed_keyframe_indices"]]
        evaluable.append(
            {
                "scene": scene,
                "source_run": run_bindings[scene]["path"],
                "source_transforms": checkpoint_actor_transforms(binding, frames),
                "support": np.asarray(support, dtype=np.float64),
            }
        )
        scene_rows.append(
            {
                "scene": scene,
                "status": "evaluable",
                "reason": None,
                "retained_in_denominator": True,
                "quality_read": True,
                "source_run": run_bindings[scene]["path"],
            }
        )
    if len(evaluable) < int(contract["minimum_evaluable_scene_count"]):
        raise M3DevelopmentSelectionError("development evaluable scene 不足")

    search = m3_config["trajectory"]["development_search"]
    candidates = []
    for control_points, regularization, retention in itertools.product(
        search["control_point_count"],
        search["acceleration_regularization"],
        search["evidence_retention"],
    ):
        per_scene = []
        for row in evaluable:
            trajectory = build_arm_trajectories(
                row["source_transforms"],
                support=row["support"],
                control_point_count=int(control_points),
                evidence_retention=float(retention),
                acceleration_regularization=float(regularization),
            )["CUBIC_BSPLINE_TEMPORAL_EVIDENCE"]
            per_scene.append(
                {
                    "scene": row["scene"],
                    "source_rmse_m": trajectory.source_rmse_m,
                    "acceleration_energy": trajectory.acceleration_energy,
                }
            )
        maximum_rmse = max(row["source_rmse_m"] for row in per_scene)
        candidates.append(
            {
                "control_point_count": int(control_points),
                "acceleration_regularization": float(regularization),
                "evidence_retention": float(retention),
                "per_scene": per_scene,
                "maximum_source_rmse_m": maximum_rmse,
                "mean_source_rmse_m": float(
                    np.mean([row["source_rmse_m"] for row in per_scene])
                ),
                "mean_acceleration_energy": float(
                    np.mean([row["acceleration_energy"] for row in per_scene])
                ),
                "rmse_gate_passed": maximum_rmse
                <= float(contract["maximum_scene_trajectory_rmse_m"]),
            }
        )
    eligible = [row for row in candidates if row["rmse_gate_passed"]]
    if not eligible:
        raise M3DevelopmentSelectionError("trajectory RMSE gate 无可选参数")
    selected_trajectory = min(
        eligible,
        key=lambda row: (
            row["mean_acceleration_energy"],
            row["mean_source_rmse_m"],
            row["control_point_count"],
            row["acceleration_regularization"],
            row["evidence_retention"],
        ),
    )
    _, warp = verify_run(selection_config["warp_sweep"], "warp sweep")
    if warp["status"] != "done" or warp["partition"] != "development":
        raise M3DevelopmentSelectionError("warp sweep identity 不一致")
    selected_alpha = float(warp["selected_warp_blend_alpha"])
    if selected_alpha not in [float(value) for value in search["warp_blend_alpha"]]:
        raise M3DevelopmentSelectionError("warp sweep 选择不在预注册 grid")
    improvement = float(warp["selected_relative_improvement"])
    warp_gate = improvement >= float(
        contract["temporal_error_relative_improvement_min"]
    )
    selected_parameters = {
        "control_point_count": selected_trajectory["control_point_count"],
        "acceleration_regularization": selected_trajectory[
            "acceleration_regularization"
        ],
        "evidence_retention": selected_trajectory["evidence_retention"],
        "warp_blend_alpha": selected_alpha,
    }
    development_freeze_passed = bool(warp_gate)
    write_jsonl(run_dir / "trajectory_candidates.jsonl", candidates)
    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir()
    for path in (selection_config_path, m3_config_path, inventory_path, Path(__file__)):
        shutil.copy2(path, source_snapshot / path.name)
    summary = {
        "schema_version": "worldsim_v4_m3_development_selection_summary_v1",
        "task_id": TASK_ID,
        "status": "done",
        "partition": "development",
        "scene_denominator": len(development),
        "evaluable_scene_count": len(evaluable),
        "abstain_scene_count": len(development) - len(evaluable),
        "scenes": scene_rows,
        "trajectory_candidate_count": len(candidates),
        "trajectory_eligible_count": len(eligible),
        "selected_trajectory": selected_trajectory,
        "warp_sweep_run": selection_config["warp_sweep"]["path"],
        "warp_relative_improvement": improvement,
        "warp_gate_passed": warp_gate,
        "selected_parameters": selected_parameters,
        "development_freeze_passed": development_freeze_passed,
        "limitations": [
            "仅 1/6 development scene 在冻结 clip/actor/三相机合同下可评",
            "scene-0994 的 RigidNodes actor 有效但冻结 clip 三相机渲染支持为零",
            "M1 posterior 已拒绝且未作为 M3 evidence memory 输入",
        ],
        "development_content_read": True,
        "development_optimization_read": True,
        "validation_content_read": False,
        "validation_optimization_read": False,
        "test_quality_read": False,
        "project_git_head": git_head(),
        "project_git_dirty": git_dirty(),
        "duration_seconds": time.monotonic() - started,
    }
    atomic_json(run_dir / "summary.json", summary)
    atomic_json(
        run_dir / "fingerprint.json",
        {
            "task_id": TASK_ID,
            "project_git_head": git_head(),
            "project_git_dirty": git_dirty(),
            "selection_config_sha256": sha256_file(selection_config_path),
            "m3_config_sha256": sha256_file(m3_config_path),
            "scene_inventory_sha256": sha256_file(inventory_path),
            "test_quality_read": False,
        },
    )
    manifest = output_manifest(run_dir)
    atomic_json(run_dir / "manifest.json", manifest)
    atomic_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "summary_sha256": sha256_file(run_dir / "summary.json"),
            "manifest_sha256": sha256_file(run_dir / "manifest.json"),
            "fingerprint_sha256": sha256_file(run_dir / "fingerprint.json"),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--selection-config",
        type=Path,
        default=PROJECT_ROOT
        / "configs/worldsim_v4/m3_development_selection_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(
        selection_config_path=args.selection_config.resolve(),
        run_dir=args.run_dir.resolve(),
    )
    print(
        json.dumps(
            {
                "status": summary["status"],
                "development_freeze_passed": summary[
                    "development_freeze_passed"
                ],
                "selected_parameters": summary["selected_parameters"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
