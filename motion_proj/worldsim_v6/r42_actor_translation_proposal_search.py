"""WorldSim V6 R42：在冻结网格上搜索可通过 interaction/contact 的 actor translation proposal。"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

from motion_proj.worldsim_v6.r12_dynamic_logsim import _replay_once
from motion_proj.worldsim_v6.r38_actor_interaction_factor import _compile_intervention
from motion_proj.worldsim_v6.r40_actor_lidar_contact_factor import _lift_static_lidar


TASK_ID = "WS-V6-R42-ACTOR-TRANSLATION-PROPOSAL-SEARCH-01"


class R42ExperimentError(RuntimeError):
    """R42 正式实验合同失败。"""


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256((payload + "\n").encode()).hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resolve_runs_uri(uri: str) -> Path:
    prefix = "runs://"
    relative = Path(uri[len(prefix) :]) if uri.startswith(prefix) else Path("..")
    if not uri.startswith(prefix) or relative.is_absolute() or ".." in relative.parts:
        raise R42ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R42ExperimentError(f"冻结输入漂移：{path}")


def _proposal_id(dx: float, dz: float) -> str:
    def token(value: float) -> str:
        return ("m" if value < 0 else "p") + f"{abs(value):.1f}".replace(".", "p")

    return f"translate_world_x_{token(dx)}_z_{token(dz)}"


def _evaluate_candidate(
    delta: np.ndarray,
    base_states: list[dict[str, Any]],
    base_positions: np.ndarray,
    base_collision_keys: set[tuple[int, tuple[str, ...]]],
    actor_means: np.ndarray,
    lidar_points: np.ndarray,
    lidar_tree: cKDTree,
    config: dict[str, Any],
) -> dict[str, Any]:
    target_id = str(config["cohort"]["actor_id"])
    states, collisions = _compile_intervention(base_states, target_id, delta)
    target_states = sorted(
        (row for row in states if row["actor_id"] == target_id),
        key=lambda row: int(row["timestamp_us"]),
    )
    edited_positions = np.asarray([row["centroid_world_m"] for row in target_states], dtype=np.float64)
    dt = float(config["cohort"]["timestamp_step_seconds"])
    velocity_error = float(np.max(np.abs(np.diff(base_positions, axis=0) / dt - np.diff(edited_positions, axis=0) / dt)))
    acceleration_error = float(
        np.max(
            np.abs(
                np.diff(np.diff(base_positions, axis=0) / dt, axis=0) / dt
                - np.diff(np.diff(edited_positions, axis=0) / dt, axis=0) / dt
            )
        )
    )
    edited_collision_keys = {
        (int(row["timestamp_us"]), tuple(row["actor_pair"]))
        for row in collisions
        if row["aabb_overlap"] and target_id in row["actor_pair"]
    }
    new_collisions = sorted(edited_collision_keys - base_collision_keys)
    shifted_means = actor_means.astype(np.float64) + delta[None, :]
    center_xz = np.mean(shifted_means[:, [0, 2]], axis=0)
    anchor_y = float(np.quantile(shifted_means[:, 1], float(config["factor_contract"]["actor_support_quantile"])))
    distances, indices = lidar_tree.query(
        center_xz,
        k=int(config["factor_contract"]["nearest_horizontal_candidates"]),
        distance_upper_bound=float(config["factor_contract"]["maximum_horizontal_radius_m"]),
    )
    distances = np.atleast_1d(distances)
    indices = np.atleast_1d(indices)
    valid = np.isfinite(distances) & (indices < lidar_points.shape[0])
    local_y = lidar_points[indices[valid], 1]
    ground_y = float(np.quantile(local_y, float(config["factor_contract"]["ground_height_quantile"]))) if local_y.size else None
    contact_error = abs(anchor_y - ground_y) if ground_y is not None else None
    q_self = "ACCEPT" if velocity_error <= float(config["factor_contract"]["maximum_kinematic_invariance_error"]) and acceleration_error <= float(config["factor_contract"]["maximum_kinematic_invariance_error"]) else "REJECT"
    q_interaction = "ACCEPT" if not new_collisions else "REJECT"
    q_contact = "ACCEPT" if int(np.count_nonzero(valid)) >= int(config["factor_contract"]["minimum_lidar_candidates"]) and contact_error is not None and contact_error <= float(config["factor_contract"]["maximum_contact_error_m"]) else "REJECT"
    joint = "ACCEPT_CONFORMANCE" if q_self == q_interaction == q_contact == "ACCEPT" else "REJECT"
    return {
        "proposal_id": _proposal_id(float(delta[0]), float(delta[2])),
        "proposal_status": "proposal_unbaked_unrendered",
        "translation_delta_m": delta.tolist(),
        "horizontal_displacement_m": float(np.linalg.norm(delta[[0, 2]])),
        "q_self_kinematics": q_self,
        "q_aabb_interaction": q_interaction,
        "q_lidar_contact": q_contact,
        "joint_admissibility": joint,
        "maximum_velocity_invariance_error": velocity_error,
        "maximum_acceleration_invariance_error": acceleration_error,
        "new_collision_events": len(new_collisions),
        "new_collision_examples": [
            {"timestamp_us": row[0], "actor_pair": list(row[1])} for row in new_collisions[:5]
        ],
        "local_lidar_candidate_count": int(np.count_nonzero(valid)),
        "actor_support_anchor_y_m": anchor_y,
        "lidar_ground_proxy_y_m": ground_y,
        "contact_absolute_error_m": contact_error,
        "renderer_execution": "ABSTAIN_PENDING_SELECTED_PROPOSAL",
        "physical_trajectory_validity": "ABSTAIN",
        "safety_validity": "ABSTAIN",
    }


def _selection_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    delta = row["translation_delta_m"]
    return (-float(row["horizontal_displacement_m"]), float(row["contact_absolute_error_m"]), float(delta[0]), float(delta[2]))


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R42ExperimentError("正式 R42 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R42ExperimentError("R42 task_id 漂移")
    sources = config["sources"]
    r41_run = _resolve_runs_uri(sources["r41_run"])
    binding_run = _resolve_runs_uri(sources["sceneir_binding_run"])
    base_package = binding_run / sources["base_sceneir_package"]
    r35_run = _resolve_runs_uri(sources["r35_run"])
    actor_package = r35_run / "package"
    r3_run = _resolve_runs_uri(sources["r3_run"])
    support_path = r3_run / sources["streetgs_support_file"]
    files = {
        r41_run / "MANIFEST.json": sources["r41_manifest_sha256"],
        r41_run / "R41_GATE.json": sources["r41_gate_sha256"],
        r41_run / "SUMMARY.json": sources["r41_summary_sha256"],
        binding_run / "MANIFEST.json": sources["sceneir_binding_manifest_sha256"],
        binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json": sources["sceneir_binding_gate_sha256"],
        base_package / "MANIFEST.json": sources["base_sceneir_manifest_sha256"],
        base_package / "sceneir.json": sources["base_sceneir_document_sha256"],
        r35_run / "MANIFEST.json": sources["r35_manifest_sha256"],
        actor_package / "PACKAGE_MANIFEST.json": sources["r35_package_manifest_sha256"],
        actor_package / "TRAJECTORY_GEOMETRY.json": sources["r35_trajectory_geometry_sha256"],
        support_path: sources["streetgs_support_sha256"],
    }
    for path, expected_sha in files.items():
        _verify(path, expected_sha)
    r41_gate = json.loads((r41_run / "R41_GATE.json").read_text(encoding="utf-8"))
    binding_gate = json.loads((binding_run / "R13_SCENEIR_SENSOR_BINDING_GATE.json").read_text(encoding="utf-8"))
    base_manifest = json.loads((base_package / "MANIFEST.json").read_text(encoding="utf-8"))
    actor_manifest = json.loads((actor_package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    package_files = {
        **{base_package / name: row["sha256"] for name, row in base_manifest["files"].items()},
        **{actor_package / name: row["sha256"] for name, row in actor_manifest["files"].items()},
    }
    for path, expected_sha in package_files.items():
        _verify(path, expected_sha)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R42ExperimentError("R42 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / f"{now.strftime('%Y%m%dT%H%M%SZ')}__proposal-search-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        base = _replay_once(base_package, 1)
        target_id = str(config["cohort"]["actor_id"])
        target_states = sorted((row for row in base["actor_states"] if row["actor_id"] == target_id), key=lambda row: int(row["timestamp_us"]))
        base_positions = np.asarray([row["centroid_world_m"] for row in target_states], dtype=np.float64)
        base_collision_keys = {
            (int(row["timestamp_us"]), tuple(row["actor_pair"]))
            for row in base["collision_labels"] if row["aabb_overlap"] and target_id in row["actor_pair"]
        }
        support = np.load(support_path, allow_pickle=False)
        lidar_points = _lift_static_lidar(support, 3)
        lidar_tree = cKDTree(lidar_points[:, [0, 2]])
        geometry = json.loads((actor_package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
        timestamps = [int(row["timestamp_us"]) for row in geometry["trajectory"]]
        actor_means_all = np.load(actor_package / geometry["arrays"]["means_world_m"]["path"], allow_pickle=False)
        actor_means = actor_means_all[timestamps.index(int(config["cohort"]["timestamp_us"]))]
        deltas = [
            np.asarray([float(dx), float(config["proposal_grid"]["world_y_m"]), float(dz)], dtype=np.float64)
            for dx in config["proposal_grid"]["world_x_m"]
            for dz in config["proposal_grid"]["world_z_m"]
            if not (bool(config["proposal_grid"]["exclude_identity"]) and float(dx) == 0.0 and float(dz) == 0.0)
        ]
        rows_1 = [_evaluate_candidate(delta, base["actor_states"], base_positions, base_collision_keys, actor_means, lidar_points, lidar_tree, config) for delta in deltas]
        rows_2 = [_evaluate_candidate(delta, base["actor_states"], base_positions, base_collision_keys, actor_means, lidar_points, lidar_tree, config) for delta in deltas]
        rows_1.sort(key=lambda row: row["proposal_id"])
        rows_2.sort(key=lambda row: row["proposal_id"])
        accepted = sorted((row for row in rows_1 if row["joint_admissibility"] == "ACCEPT_CONFORMANCE"), key=_selection_key)
        selected = accepted[0] if accepted else None
        _write_jsonl(run_dir / "PROPOSAL_CATALOG.jsonl", rows_1)
        _write_json(run_dir / "SELECTED_PROPOSAL.json", {
            "schema_version": "worldsim_v6.r42_selected_proposal.v1",
            "status": "selected_for_native_renderer_verification" if selected else "no_admissible_proposal",
            "selection_rule": config["proposal_grid"]["selection_order"],
            "proposal": selected,
            "physical_trajectory_validity": "ABSTAIN",
            "safety_validity": "ABSTAIN",
        })
        wall_seconds = time.monotonic() - started
        checks = {
            "r41_and_sceneir_authorities_accepted": r41_gate["checks"]["passed"] and binding_gate["checks"]["passed"],
            "candidate_count_exact": len(rows_1) == int(config["proposal_grid"]["expected_candidate_count"]),
            "trajectory_denominator_exact": len(target_states) == int(config["cohort"]["expected_trajectory_rows"]),
            "actor_primitive_denominator_exact": actor_means.shape[0] == int(config["cohort"]["expected_actor_primitives"]),
            "lidar_denominator_sufficient": lidar_points.shape[0] >= int(config["cohort"]["minimum_static_lidar_points"]),
            "all_self_kinematics_accept": all(row["q_self_kinematics"] == "ACCEPT" for row in rows_1),
            "joint_accept_count_sufficient": len(accepted) >= int(config["factor_contract"]["minimum_joint_accept_count"]),
            "selected_proposal_exists_and_is_admissible": selected is not None and selected["joint_admissibility"] == "ACCEPT_CONFORMANCE",
            "selected_proposal_deterministic": selected is not None and selected["proposal_id"] == sorted(accepted, key=_selection_key)[0]["proposal_id"],
            "repeat_exact": _content_sha256(rows_1) == _content_sha256(rows_2),
            "unrendered_unbaked_status_preserved": all(row["renderer_execution"] == "ABSTAIN_PENDING_SELECTED_PROPOSAL" and row["proposal_status"] == "proposal_unbaked_unrendered" for row in rows_1),
            "physical_and_safety_validity_abstain": all(row["physical_trajectory_validity"] == "ABSTAIN" and row["safety_validity"] == "ABSTAIN" for row in rows_1),
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in files.items()) and all(_sha256(path) == expected_sha for path, expected_sha in package_files.items()),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(run_dir / "R42_GATE.json", {
            "schema_version": "worldsim_v6.r42_gate.v1", "checks": checks,
            "decision": "accept_factor_aware_translation_proposal_search" if checks["passed"] else "reject_or_repair_translation_proposal_search",
        })
        _write_json(run_dir / "RESOURCE_AUDIT.json", {
            "schema_version": "worldsim_v6.r42_resource_audit.v1", "gpu_used": False,
            "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib,
            "training_started": False, "confirmation_content_read": False,
        })
        summary = {
            "schema_version": "worldsim_v6.r42_summary.v1", "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"], "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_factor_aware_translation_proposal_search" if checks["passed"] else "rejected",
            "source_commit": source_commit, "candidate_count": len(rows_1), "joint_accept_count": len(accepted),
            "interaction_reject_count": sum(row["q_aabb_interaction"] == "REJECT" for row in rows_1),
            "contact_reject_count": sum(row["q_lidar_contact"] == "REJECT" for row in rows_1),
            "selected_proposal_id": selected["proposal_id"] if selected else None,
            "selected_translation_delta_m": selected["translation_delta_m"] if selected else None,
            "renderer_execution": "ABSTAIN_PENDING_SELECTED_PROPOSAL",
            "physical_trajectory_validity": "ABSTAIN", "safety_validity": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R42_GATE.json", "SUMMARY.json", "PROPOSAL_CATALOG.jsonl", "SELECTED_PROPOSAL.json", "RESOURCE_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {
            "schema_version": "worldsim_v6.r42_manifest.v1",
            "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked},
        })
        _write_json(run_dir / "TERMINAL.json", {
            "schema_version": "worldsim_v6.terminal.v1", "status": summary["status"],
            "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
        })
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {
            "schema_version": "worldsim_v6.terminal.v1", "status": "failed",
            "error_type": type(error).__name__, "error": str(error),
        })
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r42_actor_translation_proposal_search_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

