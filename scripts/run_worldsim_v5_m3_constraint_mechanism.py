#!/usr/bin/env python3
"""V5 M3 T2–T5 轨迹机制诊断；不读取图像、renderer 或 held-out quality。"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v5.constraint_projection import (
    KinematicLimits,
    PlanarTrajectory,
    desired_trajectory_rmse_m,
    minimum_jerk_smooth,
    project_road_contact,
    project_vehicle_kinematics,
    trajectory_metrics,
    v4_frozen_bspline_comparator,
)
from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    finalize_formal_run,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M3-CONSTRAINT-PROJECTED-TEMPORAL-01"
SCHEMA_VERSION = "worldsim_v5_m3_constraint_projection_mechanism_v1"


class M3ConstraintMechanismError(RuntimeError):
    """M3 trajectory mechanism 输入、分母或门禁漂移。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M3ConstraintMechanismError("M3 mechanism config schema 漂移")
    if payload.get("task_id") != TASK_ID or payload.get("status") != "running":
        raise M3ConstraintMechanismError("M3 mechanism task/status 漂移")
    scope = payload["scope"]
    for name in (
        "image_read",
        "lidar_blob_read",
        "renderer_started",
        "gpu_required",
        "development_render_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_quality_read",
        "parameter_search_performed",
        "method_arm_selection_performed",
        "obstacle_inventory_read",
    ):
        if scope.get(name) is not False:
            raise M3ConstraintMechanismError(f"trajectory-only scope 漂移: {name}")
    if (
        payload["gate"].get("request_count") != 16
        or payload["gate"].get("collision_gate_assessed") is not False
        or payload["gate"].get("render_gate_assessed") is not False
        or payload["gate"].get("method_selection_allowed") is not False
    ):
        raise M3ConstraintMechanismError("M3 mechanism gate 漂移")
    return payload


def quaternion_yaw(rotation_wxyz: list[float]) -> float:
    if len(rotation_wxyz) != 4 or not all(math.isfinite(float(v)) for v in rotation_wxyz):
        raise ValueError("rotation 必须为有限 wxyz quaternion")
    w, x, y, z = (float(value) for value in rotation_wxyz)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def build_desired_trajectory(
    selected: Mapping[str, Any], template: Mapping[str, Any]
) -> tuple[PlanarTrajectory, np.ndarray]:
    frames = list(selected["frames"])
    forward = np.asarray(template["forward_offset_m"], dtype=np.float64)
    lateral = np.asarray(template["lateral_offset_m"], dtype=np.float64)
    if len(frames) != 7 or forward.shape != (7,) or lateral.shape != (7,):
        raise M3ConstraintMechanismError("desired template/clip denominator 漂移")
    timestamps = np.asarray([int(row["timestamp"]) for row in frames], dtype=np.float64)
    times = (timestamps - timestamps[0]) / 1_000_000.0
    translations = np.asarray([row["translation"] for row in frames], dtype=np.float64)
    bottom_z = np.asarray([row["bottom_z"] for row in frames], dtype=np.float64)
    yaws = np.unwrap(np.asarray([quaternion_yaw(row["rotation"]) for row in frames]))
    heading = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)
    heading_left = np.stack([-np.sin(yaws), np.cos(yaws)], axis=1)
    positions = np.zeros((7, 3), dtype=np.float64)
    positions[:, :2] = (
        translations[:, :2]
        + forward[:, None] * heading
        + lateral[:, None] * heading_left
    )
    positions[:, 2] = bottom_z
    road_z = np.linspace(bottom_z[0], bottom_z[-1], 7)
    return PlanarTrajectory(times, positions, yaws), road_z


def _limits(config: Mapping[str, Any]) -> KinematicLimits:
    values = config["physical_constraints"]
    return KinematicLimits(
        maximum_speed_mps=float(values["maximum_speed_mps"]),
        maximum_acceleration_mps2=float(values["maximum_acceleration_mps2"]),
        maximum_deceleration_mps2=float(values["maximum_deceleration_mps2"]),
        maximum_yaw_rate_radps=float(values["maximum_yaw_rate_radps"]),
        maximum_lateral_acceleration_mps2=float(
            values["maximum_lateral_acceleration_mps2"]
        ),
        maximum_heading_velocity_mismatch_rad=float(
            values["maximum_heading_velocity_mismatch_rad"]
        ),
        maximum_contact_error_m=float(values["maximum_contact_error_m"]),
        collision_overlap_tolerance_m2=float(values["collision_overlap_tolerance_m2"]),
    )


def evaluate_request(
    selected: Mapping[str, Any], operation: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    desired, road_z = build_desired_trajectory(
        selected, config["desired_edit_templates"][operation]
    )
    implementation = config["implementation"]
    limits = _limits(config)
    t2 = v4_frozen_bspline_comparator(desired, **implementation["t2"])
    t3 = minimum_jerk_smooth(
        desired, jerk_weight=float(implementation["t3"]["third_difference_weight"])
    )
    t4 = project_road_contact(t3, road_z=road_z)
    t5, projection = project_vehicle_kinematics(
        t4,
        limits=limits,
        road_z=road_z,
        actor_radius_m=float(implementation["t5"]["actor_radius_m"]),
        maximum_iterations=int(implementation["t5"]["maximum_iterations"]),
        convergence_tolerance=float(implementation["t5"]["convergence_tolerance"]),
    )
    arms = {
        "T2_V4_FROZEN_SE3_BSPLINE": t2,
        "T3_MINIMUM_JERK": t3,
        "T4_MINIMUM_JERK_ROAD_CONTACT": t4,
        "T5_T4_VEHICLE_KINEMATICS": t5,
    }
    metrics = {}
    for name, trajectory in arms.items():
        endpoint_translation = float(
            np.max(
                np.linalg.norm(
                    trajectory.positions[[0, -1]] - desired.positions[[0, -1]], axis=1
                )
            )
        )
        endpoint_yaw = float(
            np.max(np.abs(trajectory.yaws[[0, -1]] - desired.yaws[[0, -1]]))
        )
        metrics[name] = {
            **trajectory_metrics(trajectory, limits=limits, road_z=road_z),
            "desired_trajectory_rmse_m": desired_trajectory_rmse_m(trajectory, desired),
            "endpoint_translation_error_m": endpoint_translation,
            "endpoint_yaw_error_rad": endpoint_yaw,
        }
    return {
        "scene": selected["scene"],
        "instance_token": selected["instance_token"],
        "operation": operation,
        "arms": metrics,
        "t5_projection": projection,
        "collision_gate_assessed": False,
        "render_gate_assessed": False,
    }


def decide(rows: list[Mapping[str, Any]], config: Mapping[str, Any]) -> dict[str, Any]:
    gate = config["gate"]
    if len(rows) != int(gate["request_count"]):
        raise M3ConstraintMechanismError("scene-operation request denominator 漂移")
    baseline = [
        int(row["arms"]["T2_V4_FROZEN_SE3_BSPLINE"]["total_violation_count"])
        for row in rows
    ]
    candidate = [
        int(row["arms"]["T5_T4_VEHICLE_KINEMATICS"]["total_violation_count"])
        for row in rows
    ]
    evaluable = [index for index, value in enumerate(baseline) if value > 0]
    safe = [index for index, value in enumerate(baseline) if value == 0]
    improved = sum(candidate[index] < baseline[index] for index in evaluable)
    safe_regression = sum(candidate[index] > 0 for index in safe)
    baseline_sum = sum(baseline[index] for index in evaluable)
    candidate_sum = sum(candidate[index] for index in evaluable)
    relative_reduction = (
        (baseline_sum - candidate_sum) / baseline_sum if baseline_sum > 0 else 0.0
    )
    endpoint_pass = sum(
        row["arms"]["T5_T4_VEHICLE_KINEMATICS"]["endpoint_translation_error_m"]
        <= float(config["physical_constraints"]["endpoint_translation_tolerance_m"])
        and row["arms"]["T5_T4_VEHICLE_KINEMATICS"]["endpoint_yaw_error_rad"]
        <= float(config["physical_constraints"]["endpoint_yaw_tolerance_rad"])
        for row in rows
    )
    contact_pass = sum(
        row["arms"]["T5_T4_VEHICLE_KINEMATICS"]["violation_counts"]["contact"] == 0
        for row in rows
    )
    minimum_improved = math.ceil(
        len(evaluable) * float(gate["improved_evaluable_fraction_min"])
    )
    mechanism_passed = (
        len(evaluable) >= int(gate["minimum_t2_violation_evaluable_request_count"])
        and improved >= minimum_improved
        and relative_reduction
        >= float(gate["aggregate_physical_violation_relative_reduction_min"])
        and safe_regression <= int(gate["t2_safe_request_regression_count_max"])
        and endpoint_pass >= int(gate["endpoint_pass_request_count_min"])
        and contact_pass >= int(gate["contact_pass_request_count_min"])
    )
    if len(evaluable) < int(gate["minimum_t2_violation_evaluable_request_count"]):
        conclusion = "m3_constraint_projection_insufficient_t2_violation_signal"
    elif mechanism_passed:
        conclusion = "m3_constraint_projection_mechanism_supported_collision_and_render_pending"
    else:
        conclusion = "m3_constraint_projection_mechanism_gate_rejected"
    return {
        "conclusion": conclusion,
        "mechanism_gate_passed": mechanism_passed,
        "request_count": len(rows),
        "t2_violation_evaluable_request_count": len(evaluable),
        "t2_safe_request_count": len(safe),
        "t5_improved_evaluable_request_count": improved,
        "minimum_improved_evaluable_request_count": minimum_improved,
        "t2_safe_request_regression_count": safe_regression,
        "t2_total_violation_count": sum(baseline),
        "t5_total_violation_count": sum(candidate),
        "aggregate_physical_violation_relative_reduction": relative_reduction,
        "endpoint_pass_request_count": endpoint_pass,
        "contact_pass_request_count": contact_pass,
        "collision_gate_assessed": False,
        "render_gate_assessed": False,
        "method_arm_selected": False,
        "parameter_search_performed": False,
        "matched_render_implementation_unlocked": mechanism_passed,
        "validation_unlocked": False,
        "test_unlocked": False,
    }


def _run_impl(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events = [{"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}]
    write_events(run_dir, events)
    started = time.perf_counter()
    bindings = {
        name: verify_file(config["clip_inventory"][name]["path"], config["clip_inventory"][name]["sha256"])
        for name in ("summary", "status", "inventory")
    }
    inventory = json.loads(Path(bindings["inventory"]["path"]).read_text())
    ready = [row for row in inventory["scenes"] if row["status"] == "ready"]
    if len(ready) != int(config["clip_inventory"]["ready_scene_count"]):
        raise M3ConstraintMechanismError("ready scene denominator 漂移")
    rows = [
        evaluate_request(scene["selected"], operation, config)
        for scene in ready
        for operation in config["desired_edit_templates"]["operation_order"]
    ]
    decision = decide(rows, config)
    diagnostics_path = run_dir / "artifacts/trajectory_diagnostics.json"
    atomic_json(diagnostics_path, rows)
    decision_path = run_dir / "artifacts/mechanism_decision.json"
    atomic_json(decision_path, decision)
    snapshot = copy_source_snapshot(
        run_dir,
        [
            config_path,
            PROJECT / "configs/worldsim_v5/m3_constraint_projection_development_v1.yaml",
            PROJECT / "motion_proj/worldsim_v5/constraint_projection.py",
            PROJECT / "scripts/run_worldsim_v5_m3_constraint_mechanism.py",
            PROJECT / "tests/test_worldsim_v5_m3_constraint_mechanism.py",
        ],
        PROJECT,
    )
    summary = {
        "schema_version": "worldsim_v5_m3_constraint_projection_mechanism_summary_v1",
        "task_id": TASK_ID,
        "task_status": "running",
        "status": "done",
        "phase": config["phase"],
        "source_commit": source_head,
        "conclusion": decision["conclusion"],
        "decision": decision,
        "scene_denominator": len(ready),
        "request_count": len(rows),
        "diagnostics_sha256": sha256_file(diagnostics_path),
        "decision_sha256": sha256_file(decision_path),
        "source_snapshot_count": len(snapshot),
        "duration_seconds": time.perf_counter() - started,
        "gpu_started": False,
        "image_read": False,
        "lidar_blob_read": False,
        "renderer_started": False,
        "development_render_quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_quality_read": False,
        "parameter_search_performed": False,
        "method_arm_selected": False,
    }
    events.append({"event": "run_done", "at_utc": utc_now(), **decision})
    events_record = write_events(run_dir, events)
    status = finalize_formal_run(
        run_dir=run_dir,
        task_id=TASK_ID,
        task_status="running",
        conclusion=decision["conclusion"],
        project_head=source_head,
        input_bindings=bindings,
        summary=summary,
        resolved_config_record=resolved,
        events_record=events_record,
    )
    return {**summary, "formal_status": status}


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    try:
        return _run_impl(config_path, run_dir)
    except Exception as error:
        if run_dir.is_dir() and not (run_dir / "status.json").exists():
            source_head = subprocess.check_output(
                ["git", "-C", str(PROJECT), "rev-parse", "HEAD"], text=True
            ).strip()
            atomic_json(
                run_dir / "status.json",
                {
                    "schema_version": "worldsim_v5_m3_constraint_projection_mechanism_status_v1",
                    "task_id": TASK_ID,
                    "task_status": "running",
                    "status": "blocked",
                    "source_commit": source_head,
                    "summary_sha256": None,
                    "manifest_sha256": None,
                    "reason": f"{type(error).__name__}: {error}",
                    "development_render_quality_read": False,
                    "validation_quality_read": False,
                    "test_quality_read": False,
                    "kitti_quality_read": False,
                    "gpu_started": False,
                    "finished_at_utc": utc_now(),
                },
            )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
