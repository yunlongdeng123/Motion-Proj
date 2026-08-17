#!/usr/bin/env python3
"""Run the no-quality E0a voxel super-primitive observation-density probe."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np
import scipy


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v5.evidence_schema import atomic_save_npz
from motion_proj.worldsim_v51.protocol import ProtocolError, V51_BRANCH, load_yaml, sha256_file
from motion_proj.worldsim_v51.superprimitive_probe import (
    edge_length_quantile_voxel_sizes,
    evaluate_e0a_density_gate,
    observation_density_report,
    voxel_assignments,
)
from scripts.run_worldsim_v51_h_uplift import (
    ResourceMonitor,
    _git,
    _inventory,
    _nvidia_used_mib,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
)


SCHEMA = "worldsim_v51_stage_e_e0a_superprimitive_probe_v2"
TASK_ID = "WS-V51-M1-E-NODE-ELEVATION-01"
SCENES = ("scene-0471", "scene-1087", "scene-0379")
LEVELS = ("fine_q50", "medium_q75", "coarse_q90")


def _repo_git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _view_names(config: dict[str, Any]) -> list[str]:
    observation = config["observation_contract"]
    return [
        f"f{int(frame):03d}_c{int(camera)}.npz"
        for frame in observation["frames"]
        for camera in observation["cameras"]
    ]


def load_observation_visibility(
    *,
    unary_run: Path,
    unary_manifest_path: Path,
    unary_manifest_sha256: str,
    gaussian_count: int,
    view_names: list[str],
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """Load only availability/reliability/visibility; never access SAM probabilities."""

    if sha256_file(unary_manifest_path) != unary_manifest_sha256:
        raise ProtocolError("E0a unary manifest identity drift")
    manifest = json.loads(unary_manifest_path.read_text(encoding="utf-8"))
    inventory = {record["path"]: record for record in manifest["inventory"]}
    visibility = np.zeros((gaussian_count, len(view_names)), dtype=np.float32)
    reports = []
    for view_index, name in enumerate(view_names):
        relative = f"artifacts/observations/{name}"
        record = inventory.get(relative)
        path = unary_run / relative
        if (
            record is None
            or not path.is_file()
            or path.stat().st_size != int(record["bytes"])
            or sha256_file(path) != record["sha256"]
        ):
            raise ProtocolError(f"E0a observation identity drift: {relative}")
        with np.load(path, allow_pickle=False) as table:
            gaussian_id = np.asarray(table["gaussian_id"], dtype=np.int64)
            observed_visibility = np.asarray(table["visibility"], dtype=np.float32)
            valid = (
                np.asarray(table["sam_probability_available"], dtype=bool)
                & np.asarray(table["mask_quality_accepted"], dtype=bool)
                & (np.asarray(table["reliability"], dtype=np.float32) > 0.0)
                & (observed_visibility > 0.0)
            )
            if observed_visibility.shape != gaussian_id.shape or valid.shape != gaussian_id.shape:
                raise ProtocolError(f"E0a observation row alignment drift: {relative}")
            if gaussian_id.ndim != 1 or np.unique(gaussian_id).size != gaussian_id.size:
                raise ProtocolError(f"E0a observation Gaussian ids are not unique: {relative}")
            if np.any(gaussian_id < 0) or np.any(gaussian_id >= gaussian_count):
                raise ProtocolError(f"E0a observation Gaussian id out of range: {relative}")
            if not np.isfinite(observed_visibility).all() or np.any(
                (observed_visibility < 0.0) | (observed_visibility > 1.0)
            ):
                raise ProtocolError(f"E0a observation visibility drift: {relative}")
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


def validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA or config.get("task_id") != TASK_ID:
        raise ProtocolError("E0a schema/task drift")
    if config.get("status") != "running" or config.get("phase") != (
        "e0a_simple_voxel_structural_observation_density_probe"
    ):
        raise ProtocolError("E0a phase/status drift")
    if int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("E0a seed drift")

    recovery = config["recovery"]
    parent_config = PROJECT / recovery["parent_config"]["path"]
    if not parent_config.is_file() or sha256_file(parent_config) != recovery["parent_config"]["sha256"]:
        raise ProtocolError("E0a recovery parent-config drift")
    blocked_run = Path(recovery["blocked_run"]["path"])
    for name, expected in recovery["blocked_run"]["hashes"].items():
        path = blocked_run / name
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolError(f"E0a blocked-run evidence drift: {name}")
    blocked_status = json.loads((blocked_run / "status.json").read_text(encoding="utf-8"))
    if (
        blocked_status.get("status") != "blocked"
        or blocked_status.get("source_commit") != recovery["blocked_run"]["source_commit"]
        or blocked_status.get("error") != recovery["blocked_run"]["required_error"]
    ):
        raise ProtocolError("E0a blocked-run terminal drift")
    edge_audit = recovery["edge_audit"]
    audit_script = PROJECT / edge_audit["script"]["path"]
    if not audit_script.is_file() or sha256_file(audit_script) != edge_audit["script"]["sha256"]:
        raise ProtocolError("E0a edge-audit script drift")
    audit_report = Path(edge_audit["report"]["path"])
    if (
        not audit_report.is_file()
        or audit_report.stat().st_size != int(edge_audit["report"]["bytes"])
        or sha256_file(audit_report) != edge_audit["report"]["sha256"]
    ):
        raise ProtocolError("E0a edge-audit report drift")

    rejection_spec = config["stage_d_rejection_freeze"]
    rejection_path = PROJECT / rejection_spec["path"]
    if not rejection_path.is_file() or sha256_file(rejection_path) != rejection_spec["sha256"]:
        raise ProtocolError("E0a Stage D rejection freeze drift")
    rejection = load_yaml(rejection_path)
    if rejection.get("status") != rejection_spec["required_status"]:
        raise ProtocolError("E0a Stage D terminal drift")
    if rejection["canonical_run"]["conclusion"] != rejection_spec["required_conclusion"]:
        raise ProtocolError("E0a Stage D conclusion drift")
    if rejection["governance"]["next_task"] != rejection_spec["required_next_task"]:
        raise ProtocolError("E0a route authorization drift")

    frozen_paths = {}
    for name, spec in config["frozen_stage_d_inputs"].items():
        path = PROJECT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"E0a frozen Stage D input drift: {name}")
        frozen_paths[name] = path
    operator_freeze = load_yaml(frozen_paths["operator_freeze"])
    if operator_freeze.get("status") != "done" or operator_freeze["locks"]["quality_read"] is not False:
        raise ProtocolError("E0a operator freeze terminal/quality drift")

    panogs = config["paper_route_context"]["panogs"]
    if (
        config["paper_route_context"].get("e0_classification")
        != "internal_simple_voxel_control_not_panogs_reproduction"
        or config["paper_route_context"].get("e1_status")
        != "locked_until_e0a_and_e0b_pass"
        or config["paper_route_context"].get("e2_status") != "locked_until_e0_gate"
    ):
        raise ProtocolError("E0a paper-route classification drift")
    repo = Path(panogs["repository_path"])
    if not (repo / ".git").is_dir():
        raise ProtocolError("E0a PanoGS repository is missing")
    if _repo_git(repo, "rev-parse", "HEAD") != panogs["commit"]:
        raise ProtocolError("E0a PanoGS commit drift")
    if _repo_git(repo, "rev-parse", "HEAD^{tree}") != panogs["tree"]:
        raise ProtocolError("E0a PanoGS tree drift")
    if _repo_git(repo, "status", "--porcelain"):
        raise ProtocolError("E0a PanoGS repository is dirty")
    for relative, expected in panogs["source_hashes"].items():
        path = repo / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolError(f"E0a PanoGS source drift: {relative}")
    paper_path = Path(panogs["paper_path"])
    if (
        not paper_path.is_file()
        or paper_path.stat().st_size != int(panogs["paper_bytes"])
        or sha256_file(paper_path) != panogs["paper_sha256"]
    ):
        raise ProtocolError("E0a PanoGS paper drift")

    method = config["method"]
    expected_method = {
        "arm": "E0A",
        "node_source": "frozen_raw_gaussian_centers",
        "grouping": "world_origin_axis_aligned_voxel",
        "voxel_size_source": "frozen_directed_knn_edge_length_quantile",
        "zero_length_edge_policy": "exclude_from_scale_quantiles_preserve_all_gaussians",
        "voxel_size_quantiles": [0.5, 0.75, 0.9],
        "level_names": list(LEVELS),
        "selected_level": None,
        "learned_anchor": False,
        "base_model_membership_consumed": False,
        "dino_consumed": False,
        "motion_consumed": False,
        "sam_probability_consumed": False,
        "quality_target_consumed": False,
        "propagation_executed": False,
        "parameter_search": False,
    }
    if method != expected_method:
        raise ProtocolError("E0a method contract drift")
    observation = config["observation_contract"]
    if (
        observation.get("frames") != [0, 40, 80, 120, 160]
        or observation.get("cameras") != [0, 1, 2]
        or int(observation.get("expected_view_count_per_scene", -1)) != 15
        or observation.get("use_evaluation_views") is not False
    ):
        raise ProtocolError("E0a observation contract drift")
    gate = config["e0a_gate"]
    if (
        int(gate.get("expected_scene_count", -1)) != 3
        or gate.get("pass_action")
        != "freeze_e0a_sidecars_then_preregister_e0b_same_propagation_control"
        or gate.get("fail_action")
        != "reject_node_elevation_stop_e1_e2_advance_gaussian_grouping"
    ):
        raise ProtocolError("E0a gate drift")
    if tuple(config["execution"]["scenes"]) != SCENES:
        raise ProtocolError("E0a scene order drift")
    if config.get("failure_ledger_refs") != [
        "V51-F31",
        "V51-F37",
        "V51-F38",
        "V51-F39",
        "V51-F40",
    ]:
        raise ProtocolError("E0a failure-ledger binding drift")
    runtime = config["runtime"]
    observed_runtime = {
        "python": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if observed_runtime != {name: runtime[name] for name in observed_runtime}:
        raise ProtocolError(f"E0a runtime drift: {observed_runtime}")
    locks = config["locks"]
    for name in (
        "parse_v5_quality_diagnostics",
        "render_evaluation_views",
        "h_quality_read",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "final_heldout_quality_read",
        "kitti_method_tuning",
        "parameter_search",
        "e0b_propagation_execution",
        "e1_panogs_execution",
        "e2_ag2aussian_execution",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"E0a lock drift: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("E0a M2/M3 status drift")
    prereg = load_yaml(frozen_paths["preregistration"])
    operator_config = load_yaml(frozen_paths["operator_config"])
    frozen_observation = operator_config["observation_contract"]
    for name in ("frames", "cameras", "expected_view_count_per_scene", "use_evaluation_views"):
        if frozen_observation.get(name) != observation.get(name):
            raise ProtocolError(f"E0a inherited observation drift: {name}")
    return config, prereg, operator_config


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config, prereg, operator_config = validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"E0a run directory exists: {run_dir}")
    if _git(PROJECT, "branch", "--show-current") != V51_BRANCH:
        raise ProtocolError("E0a must run on the V5.1 branch")
    if _git(PROJECT, "status", "--porcelain"):
        raise ProtocolError("E0a requires a clean worktree")
    nvidia_start = _nvidia_used_mib()
    if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
        raise ProtocolError(f"E0a GPU not idle at start: {nvidia_start} MiB")

    run_dir.mkdir(parents=True)
    source_commit = _git(PROJECT, "rev-parse", "HEAD")
    source_tree = _git(PROJECT, "rev-parse", "HEAD^{tree}")
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    events = [{"event": "run_started", "at_utc": _utc_now(), "source_commit": source_commit}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "status.json", {"status": "running", "task_id": TASK_ID})

    monitor = ResourceMonitor(config["resources"]["monitor_interval_seconds"])
    memory_events_before = Path("/sys/fs/cgroup/memory.events").read_text(encoding="utf-8")
    monitor.start()
    started = time.perf_counter()
    try:
        scene_inputs = {scene["scene"]: scene for scene in prereg["historical_inputs"]}
        view_names = _view_names(config)
        scene_reports = []
        repeatability_report = None
        for scene_name in SCENES:
            scene_started = time.perf_counter()
            spec = scene_inputs[scene_name]
            b3_path = Path(spec["b3_unary"]["path"])
            if sha256_file(b3_path) != spec["b3_unary"]["sha256"]:
                raise ProtocolError(f"E0a B3 identity drift: {scene_name}")
            with np.load(b3_path, allow_pickle=False) as table:
                gaussian_id = np.asarray(table["gaussian_id"], dtype=np.int64).copy()
                centers = np.asarray(table["center"], dtype=np.float64).copy()
                unary = np.asarray(table["unary_posterior"], dtype=np.float32).copy()
            gaussian_count = int(spec["expected_gaussian_count"])
            if not np.array_equal(gaussian_id, np.arange(gaussian_count, dtype=np.int64)):
                raise ProtocolError(f"E0a Gaussian denominator drift: {scene_name}")

            graph_config_path = PROJECT / spec["v5_graph_config"]["path"]
            if sha256_file(graph_config_path) != spec["v5_graph_config"]["sha256"]:
                raise ProtocolError(f"E0a graph config identity drift: {scene_name}")
            graph_config = load_yaml(graph_config_path)
            unary_manifest = graph_config["inputs"]["unary_manifest"]
            visibility, observation_reports = load_observation_visibility(
                unary_run=Path(unary_manifest["path"]).parent,
                unary_manifest_path=Path(unary_manifest["path"]),
                unary_manifest_sha256=unary_manifest["sha256"],
                gaussian_count=gaussian_count,
                view_names=view_names,
            )
            edge_path = Path(spec["v5_graph_run"]["path"]) / "artifacts/graph/edges.npz"
            expected_edge_sha = spec["v5_graph_run"]["files"]["artifacts/graph/edges.npz"]
            if sha256_file(edge_path) != expected_edge_sha:
                raise ProtocolError(f"E0a edge identity drift: {scene_name}")
            with np.load(edge_path, allow_pickle=False) as edges:
                source = np.asarray(edges["source_gaussian_id"], dtype=np.int64).copy()
                target = np.asarray(edges["target_gaussian_id"], dtype=np.int64).copy()
            sizes, edge_report = edge_length_quantile_voxel_sizes(
                centers, source, target, config["method"]["voxel_size_quantiles"]
            )
            raw_assignment = np.arange(gaussian_count, dtype=np.int64)
            raw_report = observation_density_report(raw_assignment, visibility, unary)
            levels = []
            for level_index, (level_name, voxel_size) in enumerate(zip(LEVELS, sizes)):
                assignment, voxel_keys = voxel_assignments(centers, float(voxel_size))
                repeat_assignment, repeat_keys = voxel_assignments(centers, float(voxel_size))
                if not np.array_equal(assignment, repeat_assignment) or not np.array_equal(
                    voxel_keys, repeat_keys
                ):
                    raise ProtocolError(f"E0a full assignment repeat drift: {scene_name}/{level_name}")
                level_report = observation_density_report(assignment, visibility, unary)
                level_report.update(
                    {
                        "level": level_name,
                        "edge_length_quantile": float(
                            config["method"]["voxel_size_quantiles"][level_index]
                        ),
                        "voxel_size_m": float(voxel_size),
                        "assignment_array_repeat_exact": True,
                    }
                )
                output = run_dir / f"artifacts/sidecars/{scene_name}/{level_name}.npz"
                payload = {
                    "gaussian_id": gaussian_id,
                    "node_id": assignment,
                    "voxel_size_m": np.asarray(float(voxel_size), dtype=np.float64),
                    "node_count": np.asarray(voxel_keys.shape[0], dtype=np.int64),
                }
                atomic_save_npz(output, payload)
                if repeatability_report is None:
                    repeat_path = run_dir / "artifacts/repeatability/scene0471_fine_q50.npz"
                    atomic_save_npz(repeat_path, payload)
                    if output.read_bytes() != repeat_path.read_bytes():
                        raise ProtocolError("E0a first sidecar NPZ repeat is not byte exact")
                    repeatability_report = {
                        "scene": scene_name,
                        "level": level_name,
                        "gaussian_count": gaussian_count,
                        "node_count": int(voxel_keys.shape[0]),
                        "byte_exact": True,
                        "sha256": sha256_file(output),
                    }
                level_report["sidecar_sha256"] = sha256_file(output)
                levels.append(level_report)
                del assignment, repeat_assignment, voxel_keys, repeat_keys
                gc.collect()
            scene_report = {
                "scene": scene_name,
                "gaussian_count": gaussian_count,
                "observation_view_count": len(observation_reports),
                "observation_views": observation_reports,
                "edge_length_report": edge_report,
                "raw": raw_report,
                "levels": levels,
                "wall_seconds": time.perf_counter() - scene_started,
            }
            _write_json(run_dir / f"artifacts/reports/{scene_name}.json", scene_report)
            scene_reports.append(scene_report)
            events.append({"event": "scene_completed", "at_utc": _utc_now(), "scene": scene_name})
            _write_jsonl(run_dir / "events.jsonl", events)
            del centers, unary, visibility, source, target, raw_assignment
            gc.collect()

        gate = evaluate_e0a_density_gate(
            scene_reports, expected_scene_count=int(config["e0a_gate"]["expected_scene_count"])
        )
        terminal_status = "done" if gate["pass"] else "rejected"
        conclusion = (
            "e0a_structural_density_gate_pass_preregister_e0b_same_propagation"
            if gate["pass"]
            else "e0a_density_rejected_stop_e1_e2_advance_gaussian_grouping"
        )
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid_samples = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid_samples:
            raise ProtocolError("E0a resource monitor produced no valid sample")
        memory_events_after = Path("/sys/fs/cgroup/memory.events").read_text(encoding="utf-8")
        resources = {
            "nvidia_start_mib": nvidia_start,
            "nvidia_peak_mib": max(int(row["gpu_used_mib"]) for row in valid_samples),
            "cgroup_memory_peak_bytes": max(
                int(row["cgroup_memory_current_bytes"]) for row in valid_samples
            ),
            "sample_count": len(monitor.samples),
            "monitor_error_count": len(monitor.samples) - len(valid_samples),
            "wall_seconds": time.perf_counter() - started,
            "memory_events_unchanged": memory_events_before == memory_events_after,
        }
        ceilings = config["resources"]
        resource_checks = {
            "nvidia_peak": resources["nvidia_peak_mib"]
            <= int(ceilings["maximum_nvidia_peak_mib"]),
            "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"]
            <= int(ceilings["maximum_cgroup_memory_bytes"]),
            "wall": resources["wall_seconds"] <= float(ceilings["maximum_wall_seconds"]),
            "monitor": resources["monitor_error_count"] == 0,
            "memory_events": resources["memory_events_unchanged"],
        }
        _write_json(run_dir / "artifacts/resources.json", resources)
        if not all(resource_checks.values()):
            raise ProtocolError(f"E0a resource gate failed: {resource_checks}")
        summary = {
            "schema_version": "worldsim_v51_e0a_superprimitive_probe_summary_v1",
            "task_id": TASK_ID,
            "status": terminal_status,
            "conclusion": conclusion,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "scene_reports": scene_reports,
            "e0a_gate": gate,
            "repeatability": repeatability_report,
            "resources": resources,
            "resource_checks": resource_checks,
            "quality_read": False,
            "propagation_executed": False,
            "parameter_search": False,
            "e1_panogs_execution": False,
            "e2_ag2aussian_execution": False,
            "screening_quality_read": False,
            "confirmation_quality_read": False,
            "validation_quality_read": False,
            "test_quality_read": False,
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "summary.json", summary)
        events.append({"event": "run_completed", "at_utc": _utc_now(), "status": terminal_status})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "manifest.json",
            {
                "schema_version": "worldsim_v51_e0a_superprimitive_probe_manifest_v1",
                "task_id": TASK_ID,
                "status": terminal_status,
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_e0a_superprimitive_probe_status_v1",
                "task_id": TASK_ID,
                "status": terminal_status,
                "conclusion": conclusion,
                "source_commit": source_commit,
            },
        )
        return summary
    except BaseException as error:
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        events.append(
            {"event": "run_blocked", "at_utc": _utc_now(), "error": f"{type(error).__name__}: {error}"}
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_e0a_superprimitive_probe_status_v1",
                "task_id": TASK_ID,
                "status": "blocked",
                "error": f"{type(error).__name__}: {error}",
                "source_commit": source_commit,
            },
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT / "configs/worldsim_v51/stage_e_e0a_superprimitive_probe_v2.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
