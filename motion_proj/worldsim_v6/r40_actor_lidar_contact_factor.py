"""WorldSim V6 R40：用同前端 logged LiDAR 验证 frame57 actor contact support。"""

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

from motion_proj.worldsim_v6.r13_worldspace_route import _lift_points


TASK_ID = "WS-V6-R40-ACTOR-LIDAR-CONTACT-FACTOR-01"


class R40ExperimentError(RuntimeError):
    """R40 正式实验合同失败。"""


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
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
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
        raise R40ExperimentError("非法 runs URI")
    return (Path("/root/autodl-tmp/runs") / relative).resolve()


def _verify(path: Path, expected: str) -> None:
    if not path.is_file() or _sha256(path) != expected:
        raise R40ExperimentError(f"冻结输入漂移：{path}")


def _lift_static_lidar(support: Any, camera_count: int) -> np.ndarray:
    point_sets = []
    for camera_index in range(camera_count):
        prefix = f"cam{camera_index}"
        depth = np.asarray(support[f"{prefix}_lidar_depth"])
        valid = (depth > 0) & (~np.asarray(support[f"{prefix}_dynamic_mask"], dtype=bool))
        coordinates = np.argwhere(valid)
        points, _ = _lift_points(
            coordinates,
            depth,
            valid,
            np.asarray(support[f"{prefix}_intrinsics"]),
            np.asarray(support[f"{prefix}_camera_to_world"]),
        )
        point_sets.append(points)
    return np.concatenate(point_sets, axis=0).astype(np.float64)


def _evaluate(
    intervention: dict[str, Any],
    actor_means: np.ndarray,
    lidar_points: np.ndarray,
    tree: cKDTree,
    config: dict[str, Any],
) -> dict[str, Any]:
    vertical = int(config["coordinate_contract"]["vertical_axis_index"])
    horizontal = [int(value) for value in config["coordinate_contract"]["horizontal_axis_indices"]]
    delta = np.asarray(intervention["translation_delta_m"], dtype=np.float64)
    shifted = actor_means.astype(np.float64) + delta[None, :]
    center = np.mean(shifted[:, horizontal], axis=0)
    anchor = float(
        np.quantile(shifted[:, vertical], float(config["contact_query"]["actor_support_quantile"]))
    )
    distances, indices = tree.query(
        center,
        k=int(config["contact_query"]["nearest_horizontal_candidates"]),
        distance_upper_bound=float(config["contact_query"]["maximum_horizontal_radius_m"]),
    )
    distances = np.atleast_1d(distances)
    indices = np.atleast_1d(indices)
    valid = np.isfinite(distances) & (indices < lidar_points.shape[0])
    local_vertical = lidar_points[indices[valid], vertical]
    ground = (
        float(np.quantile(local_vertical, float(config["contact_query"]["ground_height_quantile"])))
        if local_vertical.size
        else None
    )
    error = abs(anchor - ground) if ground is not None else None
    accepted = int(np.count_nonzero(valid)) >= int(config["contact_query"]["minimum_lidar_candidates"])
    accepted = accepted and error is not None and error <= float(
        config["contact_query"]["maximum_contact_error_m"]
    )
    return {
        "intervention_id": intervention["id"],
        "translation_delta_m": delta.tolist(),
        "actor_support_anchor_y_m": anchor,
        "local_lidar_candidate_count": int(np.count_nonzero(valid)),
        "lidar_ground_proxy_y_m": ground,
        "contact_absolute_error_m": error,
        "q_lidar_contact": "ACCEPT" if accepted else "REJECT",
        "expected_decision": intervention["expected_lidar_contact_decision"],
        "q_semantic_road": "ABSTAIN",
        "physical_trajectory_validity": "ABSTAIN",
    }


def run_experiment(repo_root: Path, config_path: Path, run_root: Path) -> Path:
    started = time.monotonic()
    repo_root = repo_root.resolve()
    if _git(repo_root, "status", "--porcelain"):
        raise R40ExperimentError("正式 R40 run 禁止 dirty source")
    source_commit = _git(repo_root, "rev-parse", "HEAD")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("task_id") != TASK_ID:
        raise R40ExperimentError("R40 task_id 漂移")
    sources = config["sources"]
    r39_run = _resolve_runs_uri(sources["r39_run"])
    r35_run = _resolve_runs_uri(sources["r35_run"])
    r3_run = _resolve_runs_uri(sources["r3_run"])
    package = r35_run / "package"
    support_path = r3_run / sources["streetgs_support_file"]
    frozen_files = {
        r39_run / "MANIFEST.json": sources["r39_manifest_sha256"],
        r39_run / "R39_GATE.json": sources["r39_gate_sha256"],
        r39_run / "SUMMARY.json": sources["r39_summary_sha256"],
        r35_run / "MANIFEST.json": sources["r35_manifest_sha256"],
        package / "PACKAGE_MANIFEST.json": sources["r35_package_manifest_sha256"],
        package / "TRAJECTORY_GEOMETRY.json": sources["r35_trajectory_geometry_sha256"],
        r3_run / sources["streetgs_render_map"]: sources["streetgs_render_map_sha256"],
        r3_run / sources["streetgs_audit"]: sources["streetgs_audit_sha256"],
        support_path: sources["streetgs_support_sha256"],
    }
    for path, expected_sha in frozen_files.items():
        _verify(path, expected_sha)
    r39_gate = json.loads((r39_run / "R39_GATE.json").read_text(encoding="utf-8"))
    if r39_gate["checks"]["passed"] or r39_gate["decision"] != "reject_or_repair_actor_static_contact_factor":
        raise R40ExperimentError("R40 必须保留 R39 rejected authority")
    package_manifest = json.loads((package / "PACKAGE_MANIFEST.json").read_text(encoding="utf-8"))
    package_files = {
        package / relative: record["sha256"]
        for relative, record in package_manifest["files"].items()
    }
    for path, expected_sha in package_files.items():
        _verify(path, expected_sha)
    free_gib = shutil.disk_usage(run_root).free / (1024**3)
    if free_gib < float(config["resources"]["minimum_disk_free_gib"]):
        raise R40ExperimentError("R40 磁盘资源不足")

    now = datetime.now(timezone.utc)
    run_dir = run_root / TASK_ID / (
        f"{now.strftime('%Y%m%dT%H%M%SZ')}__lidar-contact-s{config['seed']}-r1"
    )
    run_dir.mkdir(parents=True, exist_ok=False)
    try:
        support = np.load(support_path, allow_pickle=False)
        lidar_points = _lift_static_lidar(support, int(config["cohort"]["camera_count"]))
        geometry = json.loads((package / "TRAJECTORY_GEOMETRY.json").read_text(encoding="utf-8"))
        timestamps = [int(row["timestamp_us"]) for row in geometry["trajectory"]]
        timestamp = int(config["cohort"]["timestamp_us"])
        trajectory_index = timestamps.index(timestamp)
        means_by_time = np.load(
            package / geometry["arrays"]["means_world_m"]["path"], allow_pickle=False
        )
        actor_means = means_by_time[trajectory_index]
        horizontal = [int(value) for value in config["coordinate_contract"]["horizontal_axis_indices"]]
        tree = cKDTree(lidar_points[:, horizontal])
        rows_1 = [
            _evaluate(row, actor_means, lidar_points, tree, config)
            for row in config["interventions"]
        ]
        rows_2 = [
            _evaluate(row, actor_means, lidar_points, tree, config)
            for row in config["interventions"]
        ]
        repeat_exact = _content_sha256(rows_1) == _content_sha256(rows_2)
        _write_jsonl(run_dir / "LIDAR_CONTACT_DECISIONS.jsonl", rows_1)
        wall_seconds = time.monotonic() - started
        checks = {
            "r39_rejection_preserved": not r39_gate["checks"]["passed"],
            "streetgs_coordinate_contract_frozen": config["coordinate_contract"]["world_up_axis"]
            == "y"
            and horizontal == [0, 2],
            "static_lidar_denominator_sufficient": lidar_points.shape[0]
            >= int(config["cohort"]["minimum_static_lidar_points"]),
            "actor_primitive_denominator_exact": actor_means.shape[0]
            == int(config["cohort"]["expected_actor_primitives"]),
            "all_local_lidar_denominators_sufficient": all(
                row["local_lidar_candidate_count"]
                >= int(config["contact_query"]["minimum_lidar_candidates"])
                for row in rows_1
            ),
            "pre_registered_directional_control": all(
                row["q_lidar_contact"] == row["expected_decision"] for row in rows_1
            ),
            "repeat_exact": repeat_exact,
            "semantic_road_and_physical_validity_abstain": all(
                row["q_semantic_road"] == "ABSTAIN"
                and row["physical_trajectory_validity"] == "ABSTAIN"
                for row in rows_1
            ),
            "source_immutable": all(
                _sha256(path) == expected_sha for path, expected_sha in frozen_files.items()
            )
            and all(_sha256(path) == expected_sha for path, expected_sha in package_files.items()),
            "wall_within_budget": wall_seconds
            <= float(config["resources"]["maximum_wall_seconds"]),
            "training_not_started": True,
            "confirmation_not_read": True,
        }
        checks["passed"] = all(checks.values())
        _write_json(
            run_dir / "R40_GATE.json",
            {
                "schema_version": "worldsim_v6.r40_gate.v1",
                "checks": checks,
                "decision": "accept_actor_logged_lidar_contact_factor"
                if checks["passed"]
                else "reject_or_repair_actor_lidar_contact_factor",
            },
        )
        _write_json(
            run_dir / "RESOURCE_AUDIT.json",
            {
                "schema_version": "worldsim_v6.r40_resource_audit.v1",
                "gpu_used": False,
                "wall_seconds": wall_seconds,
                "disk_free_gib_at_start": free_gib,
                "training_started": False,
                "confirmation_content_read": False,
            },
        )
        summary = {
            "schema_version": "worldsim_v6.r40_summary.v1",
            "task_id": TASK_ID,
            "hypothesis_id": config["hypothesis_id"],
            "status": "done" if checks["passed"] else "rejected",
            "hypothesis_outcome": "accepted_development_actor_logged_lidar_contact_factor"
            if checks["passed"]
            else "rejected",
            "source_commit": source_commit,
            "static_lidar_point_count": int(lidar_points.shape[0]),
            "contact_accept_count": sum(row["q_lidar_contact"] == "ACCEPT" for row in rows_1),
            "contact_reject_count": sum(row["q_lidar_contact"] == "REJECT" for row in rows_1),
            "physical_trajectory_validity": "ABSTAIN",
            "claim_boundary": config["claim_boundary"],
        }
        _write_json(run_dir / "SUMMARY.json", summary)
        tracked = ["R40_GATE.json", "SUMMARY.json", "LIDAR_CONTACT_DECISIONS.jsonl", "RESOURCE_AUDIT.json"]
        _write_json(
            run_dir / "MANIFEST.json",
            {
                "schema_version": "worldsim_v6.r40_manifest.v1",
                "files": {
                    name: {"bytes": (run_dir / name).stat().st_size, "sha256": _sha256(run_dir / name)}
                    for name in tracked
                },
            },
        )
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": summary["status"],
                "manifest_sha256": _sha256(run_dir / "MANIFEST.json"),
                "summary_sha256": _sha256(run_dir / "SUMMARY.json"),
            },
        )
        print(str(run_dir), flush=True)
        return run_dir
    except Exception as error:
        _write_json(
            run_dir / "TERMINAL.json",
            {
                "schema_version": "worldsim_v6.terminal.v1",
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/worldsim_v6/r40_actor_lidar_contact_factor_v1.yaml"),
    )
    parser.add_argument(
        "--run-root", type=Path, default=Path("/root/autodl-tmp/runs/worldsim_v6")
    )
    args = parser.parse_args()
    run_experiment(args.repo_root, args.config, args.run_root)
    return 0

