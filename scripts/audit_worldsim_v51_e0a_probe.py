#!/usr/bin/env python3
"""独立复算并审计 Stage E r020 的 E0a 结构密度结论。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

import numpy as np
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))


SCENES = ("scene-0471", "scene-1087", "scene-0379")
LEVELS = ("fine_q50", "medium_q75", "coarse_q90")


class AuditError(RuntimeError):
    """E0a 冻结证据不再满足预注册契约。"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as table:
        return {name: np.asarray(table[name]) for name in table.files}


def _assert_close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(float(actual), float(expected), rtol=0.0, atol=1e-12):
        raise AuditError(f"数值漂移: {label}: {actual} != {expected}")


def _assert_payload(actual: Any, expected: Any, label: str) -> None:
    if isinstance(expected, Mapping):
        if not isinstance(actual, Mapping) or set(actual) != set(expected):
            raise AuditError(f"字段漂移: {label}")
        for key in expected:
            _assert_payload(actual[key], expected[key], f"{label}/{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AuditError(f"列表漂移: {label}")
        for index, value in enumerate(expected):
            _assert_payload(actual[index], value, f"{label}/{index}")
        return
    if isinstance(expected, bool) or expected is None or isinstance(expected, str):
        if actual != expected:
            raise AuditError(f"值漂移: {label}: {actual!r} != {expected!r}")
        return
    if isinstance(expected, (int, float)):
        _assert_close(actual, expected, label)
        return
    if actual != expected:
        raise AuditError(f"值漂移: {label}")


def _verify_inventory(run_dir: Path, manifest: dict[str, Any]) -> dict[str, int]:
    expected = {row["path"]: row for row in manifest["inventory"]}
    observed = {
        path.relative_to(run_dir).as_posix(): path
        for path in sorted(run_dir.rglob("*"))
        if path.is_file() and path.name not in {"manifest.json", "status.json"}
    }
    if set(observed) != set(expected):
        raise AuditError("run manifest 覆盖范围漂移")
    total_bytes = 0
    for relative, path in observed.items():
        record = expected[relative]
        size = path.stat().st_size
        if size != int(record["bytes"]) or _sha256(path) != record["sha256"]:
            raise AuditError(f"run manifest 身份漂移: {relative}")
        total_bytes += size
    return {"entry_count": len(observed), "bytes": total_bytes}


def _view_names(config: dict[str, Any]) -> list[str]:
    observation = config["observation_contract"]
    return [
        f"f{int(frame):03d}_c{int(camera)}.npz"
        for frame in observation["frames"]
        for camera in observation["cameras"]
    ]


def _observation_visibility(
    graph_config: dict[str, Any], gaussian_count: int, view_names: list[str]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    manifest_spec = graph_config["inputs"]["unary_manifest"]
    manifest_path = Path(manifest_spec["path"])
    if _sha256(manifest_path) != manifest_spec["sha256"]:
        raise AuditError("冻结 unary manifest 身份漂移")
    manifest = _json(manifest_path)
    inventory = {row["path"]: row for row in manifest["inventory"]}
    unary_run = manifest_path.parent
    visibility = np.zeros((gaussian_count, len(view_names)), dtype=np.float32)
    reports = []
    for view_index, name in enumerate(view_names):
        relative = f"artifacts/observations/{name}"
        path = unary_run / relative
        record = inventory.get(relative)
        if (
            record is None
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or _sha256(path) != record["sha256"]
        ):
            raise AuditError(f"冻结 observation 身份漂移: {relative}")
        table = _npz(path)
        gaussian_id = np.asarray(table["gaussian_id"], dtype=np.int64)
        observed_visibility = np.asarray(table["visibility"], dtype=np.float32)
        valid = (
            np.asarray(table["sam_probability_available"], dtype=bool)
            & np.asarray(table["mask_quality_accepted"], dtype=bool)
            & (np.asarray(table["reliability"], dtype=np.float32) > 0.0)
            & (observed_visibility > 0.0)
        )
        if gaussian_id.ndim != 1 or np.unique(gaussian_id).size != gaussian_id.size:
            raise AuditError(f"observation Gaussian ID 漂移: {relative}")
        if observed_visibility.shape != gaussian_id.shape or valid.shape != gaussian_id.shape:
            raise AuditError(f"observation 行分母漂移: {relative}")
        if np.any(gaussian_id < 0) or np.any(gaussian_id >= gaussian_count):
            raise AuditError(f"observation Gaussian ID 越界: {relative}")
        if not np.isfinite(observed_visibility).all() or np.any(
            (observed_visibility < 0.0) | (observed_visibility > 1.0)
        ):
            raise AuditError(f"observation visibility 漂移: {relative}")
        visibility[gaussian_id[valid], view_index] = observed_visibility[valid]
        reports.append(
            {
                "view": name[:-4],
                "manifest_sha256": record["sha256"],
                "row_count": int(gaussian_id.size),
                "valid_row_count": int(np.count_nonzero(valid)),
            }
        )
    return visibility, reports


def _voxel_assignment(centers: np.ndarray, voxel_size_m: float) -> np.ndarray:
    keys = np.floor(np.asarray(centers, dtype=np.float64) / float(voxel_size_m)).astype(
        np.int64
    )
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    return inverse.astype(np.int64, copy=False)


def _density_report(
    assignment: np.ndarray, visibility: np.ndarray, unary: np.ndarray
) -> dict[str, Any]:
    node_id = np.asarray(assignment, dtype=np.int64)
    node_count = int(node_id.max()) + 1
    if not np.array_equal(np.unique(node_id), np.arange(node_count, dtype=np.int64)):
        raise AuditError("node ID 不是从零开始的稠密编号")
    member_count = np.bincount(node_id, minlength=node_count).astype(np.int64)
    observed = np.asarray(visibility, dtype=np.float32) > 0.0
    raw_view_count = np.count_nonzero(observed, axis=1).astype(np.int64)
    node_view_count = np.zeros(node_count, dtype=np.int64)
    for view_index in range(observed.shape[1]):
        node_view_count += (
            np.bincount(
                node_id,
                weights=observed[:, view_index].astype(np.int8),
                minlength=node_count,
            )
            > 0
        )
    assigned_union = node_view_count[node_id]
    raw_zero = raw_view_count == 0
    rescued = raw_zero & (assigned_union > 0)
    unary32 = np.asarray(unary, dtype=np.float32)
    actor = unary32 >= 0.9
    background = unary32 <= 0.1
    actor_nodes = np.bincount(node_id, weights=actor.astype(np.int8), minlength=node_count) > 0
    background_nodes = (
        np.bincount(node_id, weights=background.astype(np.int8), minlength=node_count) > 0
    )
    conflict_nodes = actor_nodes & background_nodes
    conflict_gaussians = int(member_count[conflict_nodes].sum(dtype=np.int64))
    posterior_sum = np.bincount(node_id, weights=unary32, minlength=node_count)
    posterior_square_sum = np.bincount(
        node_id, weights=np.square(unary32, dtype=np.float32), minlength=node_count
    )
    posterior_mean = posterior_sum / member_count
    posterior_variance = np.maximum(
        posterior_square_sum / member_count - np.square(posterior_mean), 0.0
    )
    raw_zero_count = int(np.count_nonzero(raw_zero))
    return {
        "gaussian_count": int(node_id.size),
        "node_count": node_count,
        "node_reduction_ratio": float(1.0 - node_count / node_id.size),
        "singleton_node_fraction": float(np.mean(member_count == 1, dtype=np.float64)),
        "member_count_percentiles": {
            str(percentile): float(np.percentile(member_count, percentile))
            for percentile in (0, 50, 90, 99, 100)
        },
        "raw_mean_observed_views_per_gaussian": float(
            raw_view_count.mean(dtype=np.float64)
        ),
        "node_mean_union_observed_views": float(node_view_count.mean(dtype=np.float64)),
        "gaussian_weighted_node_union_observed_views": float(
            assigned_union.mean(dtype=np.float64)
        ),
        "observation_union_gain_per_gaussian": float(
            (assigned_union - raw_view_count).mean(dtype=np.float64)
        ),
        "raw_zero_observation_gaussian_count": raw_zero_count,
        "rescued_zero_observation_gaussian_count": int(np.count_nonzero(rescued)),
        "rescued_zero_observation_gaussian_fraction": float(
            np.count_nonzero(rescued) / raw_zero_count if raw_zero_count else 0.0
        ),
        "actor_seed_node_count": int(np.count_nonzero(actor_nodes)),
        "background_seed_node_count": int(np.count_nonzero(background_nodes)),
        "conflicting_seed_node_count": int(np.count_nonzero(conflict_nodes)),
        "conflicting_seed_gaussian_count": conflict_gaussians,
        "conflicting_seed_gaussian_fraction": float(conflict_gaussians / node_id.size),
        "gaussian_weighted_within_node_unary_variance": float(
            posterior_variance[node_id].mean(dtype=np.float64)
        ),
    }


def _edge_report(
    centers: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    quantiles: list[float],
) -> dict[str, Any]:
    distance = np.linalg.norm(centers[source] - centers[target], axis=1).astype(np.float32)
    if not np.isfinite(distance).all():
        raise AuditError("冻结 KNN 含非有限边长")
    positive = distance[distance > 0.0]
    sizes = np.quantile(positive, quantiles, method="linear").astype(np.float64)
    return {
        "edge_count": int(distance.size),
        "positive_edge_count": int(positive.size),
        "zero_edge_count": int(distance.size - positive.size),
        "minimum_edge_length_m": float(positive.min()),
        "maximum_edge_length_m": float(positive.max()),
        "mean_edge_length_m": float(positive.mean(dtype=np.float64)),
        "quantiles": [float(value) for value in quantiles],
        "voxel_sizes_m": [float(value) for value in sizes],
    }


def _gate(scene_reports: list[dict[str, Any]]) -> dict[str, Any]:
    scenes = []
    for report in scene_reports:
        levels = []
        passing = []
        for level in report["levels"]:
            checks = {
                "node_count_reduced": level["node_count"] < report["raw"]["node_count"],
                "observation_union_strictly_improved": level[
                    "gaussian_weighted_node_union_observed_views"
                ]
                > report["raw"]["raw_mean_observed_views_per_gaussian"],
                "unobserved_gaussians_strictly_rescued": level[
                    "rescued_zero_observation_gaussian_count"
                ]
                > 0,
            }
            passed = bool(all(checks.values()))
            levels.append({"level": level["level"], "pass": passed, "checks": checks})
            if passed:
                passing.append(level["level"])
        scenes.append(
            {
                "scene": report["scene"],
                "pass": bool(passing),
                "passing_levels": passing,
                "levels": levels,
            }
        )
    return {
        "pass": bool(all(scene["pass"] for scene in scenes)),
        "scene_count": len(scenes),
        "passing_scene_count": sum(scene["pass"] for scene in scenes),
        "scenes": scenes,
    }


def audit(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config = _yaml(config_path)
    status = _json(run_dir / "status.json")
    summary = _json(run_dir / "summary.json")
    manifest = _json(run_dir / "manifest.json")
    conclusion = "e0a_structural_density_gate_pass_preregister_e0b_same_propagation"
    if status.get("status") != "done" or manifest.get("status") != "done":
        raise AuditError("r020 terminal 必须保持 done")
    if summary.get("status") != "done" or summary.get("conclusion") != conclusion:
        raise AuditError("r020 summary 结论漂移")
    if status.get("conclusion") != conclusion:
        raise AuditError("r020 status 结论漂移")
    if (run_dir / "resolved_config.yaml").read_text(
        encoding="utf-8"
    ) != config_path.read_text(encoding="utf-8"):
        raise AuditError("resolved config 不是 byte exact")
    inventory = _verify_inventory(run_dir, manifest)
    source_tree = subprocess.check_output(
        ["git", "-C", str(PROJECT), "rev-parse", f"{summary['source_commit']}^{{tree}}"],
        text=True,
    ).strip()
    if source_tree != summary["source_tree"] or status["source_commit"] != summary["source_commit"]:
        raise AuditError("r020 source commit/tree 漂移")

    prereg_path = PROJECT / config["frozen_stage_d_inputs"]["preregistration"]["path"]
    if _sha256(prereg_path) != config["frozen_stage_d_inputs"]["preregistration"]["sha256"]:
        raise AuditError("冻结 Stage D preregistration 漂移")
    prereg = _yaml(prereg_path)
    scene_inputs = {row["scene"]: row for row in prereg["historical_inputs"]}
    view_names = _view_names(config)
    audited_scenes = []
    recomputed_scenes = []
    for scene_index, scene_name in enumerate(SCENES):
        spec = scene_inputs[scene_name]
        recorded = summary["scene_reports"][scene_index]
        persisted = _json(run_dir / f"artifacts/reports/{scene_name}.json")
        _assert_payload(persisted, recorded, f"{scene_name}/persisted_report")
        if recorded["scene"] != scene_name or recorded["observation_view_count"] != 15:
            raise AuditError(f"scene/view 分母漂移: {scene_name}")

        b3_path = Path(spec["b3_unary"]["path"])
        if _sha256(b3_path) != spec["b3_unary"]["sha256"]:
            raise AuditError(f"B3 身份漂移: {scene_name}")
        b3 = _npz(b3_path)
        gaussian_count = int(spec["expected_gaussian_count"])
        gaussian_id = np.asarray(b3["gaussian_id"], dtype=np.int64)
        centers = np.asarray(b3["center"], dtype=np.float64)
        unary = np.asarray(b3["unary_posterior"], dtype=np.float32)
        if not np.array_equal(gaussian_id, np.arange(gaussian_count, dtype=np.int64)):
            raise AuditError(f"Gaussian 分母漂移: {scene_name}")

        graph_config_path = PROJECT / spec["v5_graph_config"]["path"]
        if _sha256(graph_config_path) != spec["v5_graph_config"]["sha256"]:
            raise AuditError(f"graph config 身份漂移: {scene_name}")
        visibility, observation_reports = _observation_visibility(
            _yaml(graph_config_path), gaussian_count, view_names
        )
        _assert_payload(observation_reports, recorded["observation_views"], f"{scene_name}/views")

        edge_path = Path(spec["v5_graph_run"]["path"]) / "artifacts/graph/edges.npz"
        if _sha256(edge_path) != spec["v5_graph_run"]["files"]["artifacts/graph/edges.npz"]:
            raise AuditError(f"KNN edge 身份漂移: {scene_name}")
        edges = _npz(edge_path)
        edge_report = _edge_report(
            centers,
            np.asarray(edges["source_gaussian_id"], dtype=np.int64),
            np.asarray(edges["target_gaussian_id"], dtype=np.int64),
            [float(value) for value in config["method"]["voxel_size_quantiles"]],
        )
        _assert_payload(edge_report, recorded["edge_length_report"], f"{scene_name}/edges")

        raw = _density_report(np.arange(gaussian_count, dtype=np.int64), visibility, unary)
        _assert_payload(raw, recorded["raw"], f"{scene_name}/raw")
        levels = []
        for level_index, level_name in enumerate(LEVELS):
            path = run_dir / f"artifacts/sidecars/{scene_name}/{level_name}.npz"
            sidecar = _npz(path)
            if set(sidecar) != {"gaussian_id", "node_id", "voxel_size_m", "node_count"}:
                raise AuditError(f"sidecar 字段漂移: {scene_name}/{level_name}")
            if sidecar["gaussian_id"].dtype != np.int64 or sidecar["node_id"].dtype != np.int64:
                raise AuditError(f"sidecar ID dtype 漂移: {scene_name}/{level_name}")
            if not np.array_equal(sidecar["gaussian_id"], gaussian_id):
                raise AuditError(f"sidecar Gaussian ID 漂移: {scene_name}/{level_name}")
            voxel_size = float(sidecar["voxel_size_m"])
            _assert_close(
                voxel_size,
                edge_report["voxel_sizes_m"][level_index],
                f"{scene_name}/{level_name}/voxel_size",
            )
            recomputed_assignment = _voxel_assignment(centers, voxel_size)
            if not np.array_equal(sidecar["node_id"], recomputed_assignment):
                raise AuditError(f"sidecar assignment 漂移: {scene_name}/{level_name}")
            node_count = int(sidecar["node_count"])
            if node_count != int(recomputed_assignment.max()) + 1:
                raise AuditError(f"sidecar node_count 漂移: {scene_name}/{level_name}")
            level = _density_report(recomputed_assignment, visibility, unary)
            level.update(
                {
                    "level": level_name,
                    "edge_length_quantile": float(
                        config["method"]["voxel_size_quantiles"][level_index]
                    ),
                    "voxel_size_m": voxel_size,
                    "assignment_array_repeat_exact": True,
                    "sidecar_sha256": _sha256(path),
                }
            )
            _assert_payload(level, recorded["levels"][level_index], f"{scene_name}/{level_name}")
            levels.append(level)
        recomputed_scenes.append({"scene": scene_name, "raw": raw, "levels": levels})
        audited_scenes.append(
            {
                "scene": scene_name,
                "gaussian_count": gaussian_count,
                "view_count": len(observation_reports),
                "zero_edge_count": edge_report["zero_edge_count"],
                "level_node_counts": {
                    level["level"]: level["node_count"] for level in levels
                },
                "sidecar_assignments_exact": True,
                "density_metrics_exact": True,
            }
        )

    gate = _gate(recomputed_scenes)
    _assert_payload(gate, summary["e0a_gate"], "e0a_gate")
    if not gate["pass"] or gate["passing_scene_count"] != 3:
        raise AuditError("E0a gate 不再通过")
    repeat = run_dir / "artifacts/repeatability/scene0471_fine_q50.npz"
    canonical = run_dir / "artifacts/sidecars/scene-0471/fine_q50.npz"
    if repeat.read_bytes() != canonical.read_bytes():
        raise AuditError("首个 sidecar repeatability 不再 byte exact")
    if _sha256(canonical) != summary["repeatability"]["sha256"]:
        raise AuditError("repeatability SHA 漂移")

    resource_samples = [
        json.loads(line)
        for line in (run_dir / "artifacts/resource_samples.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line
    ]
    valid = [row for row in resource_samples if "monitor_error" not in row]
    resources = summary["resources"]
    if len(resource_samples) != resources["sample_count"] or len(valid) != resources["sample_count"]:
        raise AuditError("resource sample 分母漂移")
    if max(int(row["gpu_used_mib"]) for row in valid) != resources["nvidia_peak_mib"]:
        raise AuditError("NVIDIA peak 复算漂移")
    if max(int(row["cgroup_memory_current_bytes"]) for row in valid) != resources[
        "cgroup_memory_peak_bytes"
    ]:
        raise AuditError("cgroup peak 复算漂移")
    if not all(summary["resource_checks"].values()):
        raise AuditError("resource gate 漂移")

    for lock in (
        "quality_read",
        "propagation_executed",
        "parameter_search",
        "e1_panogs_execution",
        "e2_ag2aussian_execution",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
    ):
        if summary.get(lock) is not False:
            raise AuditError(f"禁止项漂移: {lock}")
    if summary.get("m2_status") != "pending" or summary.get("m3_status") != "pending":
        raise AuditError("M2/M3 状态漂移")

    event_names = [
        json.loads(line)["event"]
        for line in (run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    if event_names != ["run_started", "scene_completed", "scene_completed", "scene_completed", "run_completed"]:
        raise AuditError("事件序列漂移")

    return {
        "status": "passed",
        "conclusion": "r020_e0a_sidecars_density_metrics_and_gate_independently_replayed_exact",
        "run": str(run_dir),
        "source_commit": summary["source_commit"],
        "source_tree": summary["source_tree"],
        "config_sha256": _sha256(config_path),
        "inventory": inventory,
        "scene_audits": audited_scenes,
        "gate": gate,
        "repeatability": summary["repeatability"],
        "resources": resources,
        "locks": {
            "quality_read": False,
            "propagation_executed": False,
            "parameter_search": False,
            "e1_panogs_execution": False,
            "e2_ag2aussian_execution": False,
            "m2_status": "pending",
            "m3_status": "pending",
        },
        "hashes": {
            name: _sha256(run_dir / name)
            for name in (
                "summary.json",
                "status.json",
                "manifest.json",
                "events.jsonl",
                "artifacts/resources.json",
                "artifacts/resource_samples.jsonl",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_e_e0a_superprimitive_probe_v2.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.config.resolve(), args.run_dir.resolve())
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
