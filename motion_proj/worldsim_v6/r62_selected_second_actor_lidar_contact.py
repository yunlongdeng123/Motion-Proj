"""WorldSim V6 R62：验证 R60 所选 actor2 编辑的 logged-LiDAR contact 因子。"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from scipy.spatial import cKDTree

from motion_proj.worldsim_v6.r40_actor_lidar_contact_factor import (
    _content_sha256, _evaluate, _git, _lift_static_lidar, _resolve_runs_uri,
    _sha256, _verify, _write_json, _write_jsonl,
)
from motion_proj.worldsim_v6.r46_detached_transform_package_logsim import _verify_package


TASK_ID = "WS-V6-R62-SELECTED-SECOND-ACTOR-LIDAR-CONTACT-01"


class R62ExperimentError(RuntimeError):
    """R62 正式实验合同失败。"""


def _array_bundle_sha256(values: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(values):
        value = np.ascontiguousarray(values[name])
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(value.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(value.shape).encode("ascii") + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def _extract_support(config_path: Path, upstream: Path, frame_index: int, camera_ids: list[int], downscale: int) -> dict[str, np.ndarray]:
    # 只加载冻结数据集，不实例化或训练渲染模型。
    run_root = config_path.parent
    backup = run_root / "backup"
    sys.path.insert(0, str(backup))
    sys.path.append(str(upstream))
    from datasets.driving_dataset import DrivingDataset
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(config_path)
    cfg.data.preload_device = "cpu"
    dataset = DrivingDataset(data_cfg=cfg.data)
    support: dict[str, np.ndarray] = {}
    for camera_id in camera_ids:
        image_infos, camera_infos = dataset.full_image_set.get_image(frame_index * len(camera_ids) + camera_id, downscale)
        prefix = f"cam{camera_id}"
        support[f"{prefix}_lidar_depth"] = image_infos["lidar_depth_map"].detach().cpu().numpy().astype(np.float32)
        support[f"{prefix}_dynamic_mask"] = image_infos["dynamic_masks"].detach().cpu().numpy().astype(bool)
        support[f"{prefix}_intrinsics"] = camera_infos["intrinsics"].detach().cpu().numpy().astype(np.float32)
        support[f"{prefix}_camera_to_world"] = camera_infos["camera_to_world"].detach().cpu().numpy().astype(np.float32)
    return support


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R62ExperimentError("正式 R62 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R62ExperimentError("R62 task_id 漂移")

    sources = config["sources"]
    r61_run = _resolve_runs_uri(sources["r61_run"])
    r60_run = _resolve_runs_uri(sources["r60_run"])
    r56_run = _resolve_runs_uri(sources["r56_run"])
    r40_run = _resolve_runs_uri(sources["r40_run"])
    package = r56_run / "package"
    geometry_path = package / "TRAJECTORY_GEOMETRY.json"
    selected_path = r60_run / "SELECTED_PROPOSAL.json"
    streetgs_config = Path(sources["streetgs_config"])
    upstream = Path(sources["streetgs_upstream_root"])
    frozen_files = {
        r61_run / "MANIFEST.json": sources["r61_manifest_sha256"],
        r61_run / "R61_GATE.json": sources["r61_gate_sha256"],
        r61_run / "SUMMARY.json": sources["r61_summary_sha256"],
        selected_path: sources["r60_selected_proposal_sha256"],
        r56_run / "MANIFEST.json": sources["r56_manifest_sha256"],
        r56_run / "R56_GATE.json": sources["r56_gate_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r56_package_manifest_sha256"],
        geometry_path: sources["r56_trajectory_geometry_sha256"],
        r40_run / "MANIFEST.json": sources["r40_manifest_sha256"],
        r40_run / "R40_GATE.json": sources["r40_gate_sha256"],
        r40_run / "SUMMARY.json": sources["r40_summary_sha256"],
        streetgs_config: sources["streetgs_config_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    if _git(upstream, "rev-parse", "HEAD") != sources["streetgs_upstream_commit"]:
        raise R62ExperimentError("StreetGS upstream commit 漂移")
    authorities_accepted = all(
        json.loads(path.read_text(encoding="utf-8"))["checks"]["passed"]
        for path in (r61_run / "R61_GATE.json", r56_run / "R56_GATE.json", r40_run / "R40_GATE.json")
    )
    selected = json.loads(selected_path.read_text(encoding="utf-8"))["selected"]
    selected_exact = bool(selected["proposal_id"] == "actor2_translate_x_-1.0_z_+0.0" and selected["translation_delta_m"] == config["interventions"][1]["translation_delta_m"] and selected["joint_decision"] == "ACCEPT" and int(selected["new_overlap_events"]) == 0)
    package_manifest = _verify_package(package)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R62ExperimentError("R62 磁盘资源不足")

    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = run_root / TASK_ID / f"{now}__actor2-lidar-contact-s{config['seed']}-r1"
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        cohort = config["cohort"]
        camera_ids = [int(value) for value in cohort["camera_ids"]]
        support_1 = _extract_support(streetgs_config, upstream, int(cohort["frame_index"]), camera_ids, int(cohort["camera_downscale"]))
        support_2 = _extract_support(streetgs_config, upstream, int(cohort["frame_index"]), camera_ids, int(cohort["camera_downscale"]))
        support_repeat_exact = _array_bundle_sha256(support_1) == _array_bundle_sha256(support_2)
        support_path = run_dir / "LIDAR_SUPPORT_FRAME098.npz"
        np.savez_compressed(support_path, **support_1)
        lidar_points = _lift_static_lidar(support_1, int(cohort["camera_count"]))

        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        timestamps = [int(row["timestamp_us"]) for row in geometry["trajectory"]]
        trajectory_index = timestamps.index(int(cohort["timestamp_us"]))
        means_by_time = np.load(package / geometry["arrays"]["means_world_m"]["path"], allow_pickle=False)
        actor_means = means_by_time[trajectory_index]
        validity = np.load(package / geometry["arrays"]["actor_frame_validity"]["path"], allow_pickle=False)
        horizontal = [int(value) for value in config["coordinate_contract"]["horizontal_axis_indices"]]
        tree = cKDTree(lidar_points[:, horizontal])
        rows_1 = [_evaluate(row, actor_means, lidar_points, tree, config) for row in config["interventions"]]
        rows_2 = [_evaluate(row, actor_means, lidar_points, tree, config) for row in config["interventions"]]
        decision_repeat_exact = _content_sha256(rows_1) == _content_sha256(rows_2)
        _write_jsonl(run_dir / "LIDAR_CONTACT_DECISIONS.jsonl", rows_1)

        wall_seconds = time.monotonic() - started
        checks: dict[str, Any] = {
            "r61_r56_r40_authorities_accepted": authorities_accepted,
            "r60_selected_proposal_exact": selected_exact,
            "streetgs_coordinate_contract_frozen": config["coordinate_contract"]["world_up_axis"] == "y" and horizontal == [0, 2],
            "frame_timestamp_actor_and_lifecycle_exact": int(cohort["frame_index"]) == 98 and int(cohort["timestamp_us"]) == 9800000 and geometry["frontend_model_index"] == int(cohort["expected_actor_model_index"]) and bool(validity[trajectory_index]) == bool(cohort["expected_frame_validity"]),
            "support_extraction_repeat_exact": support_repeat_exact,
            "static_lidar_denominator_sufficient": lidar_points.shape[0] >= int(cohort["minimum_static_lidar_points"]),
            "actor_primitive_denominator_exact": actor_means.shape[0] == int(cohort["expected_actor_primitives"]),
            "all_local_lidar_denominators_sufficient": all(row["local_lidar_candidate_count"] >= int(config["contact_query"]["minimum_lidar_candidates"]) for row in rows_1),
            "pre_registered_baseline_and_selected_contact_accept": all(row["q_lidar_contact"] == row["expected_decision"] for row in rows_1),
            "decision_repeat_exact": decision_repeat_exact,
            "semantic_road_and_physical_validity_abstain": all(row["q_semantic_road"] == "ABSTAIN" and row["physical_trajectory_validity"] == "ABSTAIN" for row in rows_1),
            "source_immutable": all(_sha256(path) == expected_sha for path, expected_sha in frozen_files.items()) and package_manifest == _verify_package(package),
            "wall_within_budget": wall_seconds <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True, "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        status = "done" if checks["passed"] else "rejected"
        _write_json(run_dir / "R62_GATE.json", {"schema_version": "worldsim_v6.r62_gate.v1", "checks": checks, "decision": "accept_selected_second_actor_logged_lidar_contact" if checks["passed"] else "reject_or_repair_selected_second_actor_lidar_contact"})
        _write_json(run_dir / "RESOURCE_AUDIT.json", {"schema_version": "worldsim_v6.r62_resource_audit.v1", "gpu_used": False, "wall_seconds": wall_seconds, "disk_free_gib_at_start": free_gib, "training_started": False, "confirmation_content_read": False})
        _write_json(run_dir / "SUPPORT_AUDIT.json", {"schema_version": "worldsim_v6.r62_support_audit.v1", "frame_index": int(cohort["frame_index"]), "camera_ids": camera_ids, "support_array_sha256": _array_bundle_sha256(support_1), "support_repeat_exact": support_repeat_exact, "static_lidar_point_count": int(lidar_points.shape[0]), "support_file_sha256": _sha256(support_path)})
        _write_json(run_dir / "SUMMARY.json", {
            "schema_version": "worldsim_v6.r62_summary.v1", "task_id": TASK_ID, "hypothesis_id": config["hypothesis_id"], "status": status,
            "hypothesis_outcome": "accepted_development_selected_second_actor_logged_lidar_contact" if checks["passed"] else "rejected", "source_commit": source_commit,
            "proposal_id": selected["proposal_id"], "frame_index": int(cohort["frame_index"]), "static_lidar_point_count": int(lidar_points.shape[0]),
            "logged_contact_error_m": rows_1[0]["contact_absolute_error_m"], "selected_contact_error_m": rows_1[1]["contact_absolute_error_m"],
            "logged_contact_decision": rows_1[0]["q_lidar_contact"], "selected_contact_decision": rows_1[1]["q_lidar_contact"],
            "semantic_road": "ABSTAIN", "physical_trajectory_validity": "ABSTAIN", "claim_boundary": config["claim_boundary"],
        })
        tracked = ["R62_GATE.json", "SUMMARY.json", "LIDAR_CONTACT_DECISIONS.jsonl", "LIDAR_SUPPORT_FRAME098.npz", "SUPPORT_AUDIT.json", "RESOURCE_AUDIT.json"]
        _write_json(run_dir / "MANIFEST.json", {"schema_version": "worldsim_v6.r62_manifest.v1", "files": {name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)} for name in tracked}})
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": status, "manifest_sha256": _sha256(run_dir / "MANIFEST.json"), "summary_sha256": _sha256(run_dir / "SUMMARY.json")})
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(run_dir / "TERMINAL.json", {"schema_version": "worldsim_v6.terminal.v1", "status": "failed", "error_type": type(error).__name__, "error": str(error)})
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, default=Path("configs/worldsim_v6/r62_selected_second_actor_lidar_contact_v1.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6"))
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
