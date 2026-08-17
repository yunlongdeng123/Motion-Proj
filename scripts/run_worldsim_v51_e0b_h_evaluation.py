#!/usr/bin/env python3
"""在冻结 H views 上评估 U2/B3 G0、raw D0 与 voxel E0B。"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import scipy
import torch


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v5.evidence_schema import atomic_save_npz
from motion_proj.worldsim_v51.progressive_evaluation import METRICS
from motion_proj.worldsim_v51.protocol import ProtocolError, V51_BRANCH, load_yaml, sha256_file
from motion_proj.worldsim_v51.superprimitive_evaluation import evaluate_e0b_h_gate
from scripts.run_worldsim_v5_m1_graph_diagnostic import (
    _aggregate_metrics,
    _build_runtime,
    _collect_gaussians,
    _global_layout,
    _load_npz,
    _metric_row,
    _runtime_helpers,
    _verify_unary_contract,
    rasterize_ownership_probability,
)
from scripts.run_worldsim_v51_d0_h_evaluation import (
    _verified_baseline,
    _verified_graph_inventory,
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
from scripts.worldsim_v5_forensics_common import verify_file


SCHEMA = "worldsim_v51_stage_e_e0b_h_evaluation_v1"
TASK_ID = "WS-V51-M1-E-NODE-ELEVATION-01"
SCENES = ("scene-0471", "scene-1087", "scene-0379")
ARMS = ("U2_B3_G0", "D0", "E0B")


def validate_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA:
        raise ProtocolError("E0b H evaluation schema drift")
    if config.get("task_id") != TASK_ID or config.get("status") != "running":
        raise ProtocolError("E0b H evaluation task/status drift")
    if config.get("phase") != "e0b_h_matched_evaluation" or int(
        config.get("seed", -1)
    ) != 20260814:
        raise ProtocolError("E0b H evaluation phase/seed drift")

    frozen = []
    for key in ("operator_freeze", "d0_h_rejection_freeze"):
        spec = config[key]
        path = PROJECT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"E0b H frozen input identity drift: {key}")
        payload = load_yaml(path)
        if payload.get("status") != spec["required_status"]:
            raise ProtocolError(f"E0b H frozen input status drift: {key}")
        frozen.append(payload)
    operator, rejection = frozen
    if operator["canonical_run"]["conclusion"] != config["operator_freeze"][
        "required_conclusion"
    ]:
        raise ProtocolError("E0b H operator conclusion drift")
    if rejection["canonical_run"]["conclusion"] != config["d0_h_rejection_freeze"][
        "required_conclusion"
    ]:
        raise ProtocolError("E0b H D0 rejection conclusion drift")

    for name, spec in config["reference_implementation"].items():
        path = PROJECT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"E0b H reference identity drift: {name}")
    arms = config["arms"]
    if (
        arms.get("primary_comparator") != "U2_B3_G0"
        or arms.get("mechanism_comparator") != "D0"
        or arms.get("candidate") != "E0B"
        or arms.get("persisted_probability_dtype") != "float16"
    ):
        raise ProtocolError("E0b H arm contract drift")

    d0_run = Path(config["d0_h_run"]["path"])
    d0_manifest_path = d0_run / "manifest.json"
    if not d0_manifest_path.is_file() or sha256_file(d0_manifest_path) != config[
        "d0_h_run"
    ]["manifest_sha256"]:
        raise ProtocolError("E0b H frozen D0 manifest identity drift")
    d0_manifest = json.loads(d0_manifest_path.read_text(encoding="utf-8"))
    if d0_manifest.get("status") != config["d0_h_run"]["required_status"]:
        raise ProtocolError("E0b H frozen D0 terminal drift")
    d0_inventory = {row["path"]: row for row in d0_manifest["inventory"]}

    scenes = config["scenes"]
    if tuple(scene["scene"] for scene in scenes) != SCENES:
        raise ProtocolError("E0b H scene order drift")
    if [int(scene["expected_evaluation_view_count"]) for scene in scenes] != [8, 1, 3]:
        raise ProtocolError("E0b H view denominator drift")
    operator_scenes = {scene["scene"]: scene for scene in operator["scenes"]}
    for scene in scenes:
        name = scene["scene"]
        if scene["e0b_sidecar"]["sha256"] != operator_scenes[name]["sidecar_sha256"]:
            raise ProtocolError(f"E0b H sidecar binding drift: {name}")
        path = Path(scene["e0b_sidecar"]["path"])
        if not path.is_file() or sha256_file(path) != scene["e0b_sidecar"]["sha256"]:
            raise ProtocolError(f"E0b H sidecar identity drift: {name}")

    evaluation = config["evaluation"]
    expected_evaluation = {
        "target_declaration": "frozen_v5_sam_binary_mask_proxy_not_ground_truth",
        "target_usage": "evaluation_only_never_method_input",
        "expected_total_view_count": 12,
        "probability_threshold": 0.5,
        "boundary_tolerance_px": 3,
        "ece_bins": 15,
        "metric_source_precision": "persisted_float16_for_all_arms",
        "scene_aggregation": "equal_view",
        "cross_scene_aggregation": "equal_scene",
        "metrics": list(METRICS),
    }
    for name, expected in expected_evaluation.items():
        if evaluation.get(name) != expected:
            raise ProtocolError(f"E0b H evaluation contract drift: {name}")
    primary_gate = config["primary_h_gate_vs_u2_b3_g0"]
    mechanism_gate = config["mechanism_h_gate_vs_d0"]
    if {
        name: primary_gate[name]
        for name in (
            "scene_count",
            "minimum_positive_boundary_f1_scenes",
            "minimum_scene_balanced_boundary_f1_delta_exclusive",
            "minimum_scene_balanced_iou_delta",
            "maximum_scene_balanced_false_negative_semantic_mass_delta",
        )
    } != {
        "scene_count": 3,
        "minimum_positive_boundary_f1_scenes": 2,
        "minimum_scene_balanced_boundary_f1_delta_exclusive": 0.0,
        "minimum_scene_balanced_iou_delta": 0.0,
        "maximum_scene_balanced_false_negative_semantic_mass_delta": 0.02,
    }:
        raise ProtocolError("E0b H primary gate drift")
    if {
        name: mechanism_gate[name]
        for name in (
            "scene_count",
            "minimum_nonnegative_boundary_f1_scenes",
            "minimum_scene_balanced_boundary_f1_delta_exclusive",
            "minimum_scene_balanced_iou_delta",
            "maximum_scene_balanced_false_negative_semantic_mass_delta",
        )
    } != {
        "scene_count": 3,
        "minimum_nonnegative_boundary_f1_scenes": 2,
        "minimum_scene_balanced_boundary_f1_delta_exclusive": 0.0,
        "minimum_scene_balanced_iou_delta": 0.0,
        "maximum_scene_balanced_false_negative_semantic_mass_delta": 0.0,
    }:
        raise ProtocolError("E0b H mechanism gate drift")
    if config["decision"] != {
        "pass_requires": "primary_and_mechanism_gates",
        "pass_action": "freeze_e0b_then_preregister_e1_panogs_faithful_port",
        "fail_action": "reject_simple_node_elevation_stop_e1_e2_advance_gaussian_grouping",
    }:
        raise ProtocolError("E0b H decision drift")

    runtime = config["runtime"]
    observed_runtime = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if observed_runtime != {name: runtime[name] for name in observed_runtime}:
        raise ProtocolError(f"E0b H runtime drift: {observed_runtime}")
    for name in (
        "parameter_search",
        "e0b_recompute",
        "d0_recompute",
        "baseline_recompute",
        "target_change",
        "view_change",
        "screening_quality_read",
        "confirmation_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "final_heldout_quality_read",
        "kitti_method_tuning",
        "e1_panogs_execution",
        "e2_ag2aussian_execution",
    ):
        if config["locks"].get(name) is not False:
            raise ProtocolError(f"E0b H lock drift: {name}")
    if config["locks"].get("m2_status") != "pending" or config["locks"].get(
        "m3_status"
    ) != "pending":
        raise ProtocolError("E0b H M2/M3 status drift")
    return config, operator, d0_manifest, d0_inventory


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config, _, _, d0_inventory = validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"E0b H run directory exists: {run_dir}")
    if _git(PROJECT, "branch", "--show-current") != V51_BRANCH:
        raise ProtocolError("E0b H evaluation must run on V5.1 branch")
    if _git(PROJECT, "status", "--porcelain"):
        raise ProtocolError("E0b H evaluation requires a clean worktree")
    if not torch.cuda.is_available():
        raise ProtocolError("E0b H evaluation requires CUDA")
    nvidia_start = _nvidia_used_mib()
    if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
        raise ProtocolError(f"E0b H GPU not idle at start: {nvidia_start} MiB")

    run_dir.mkdir(parents=True)
    source_commit = _git(PROJECT, "rev-parse", "HEAD")
    source_tree = _git(PROJECT, "rev-parse", "HEAD^{tree}")
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    events = [{"event": "run_started", "at_utc": _utc_now(), "source_commit": source_commit}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "status.json", {"status": "running", "task_id": TASK_ID})

    device = torch.device(device_name)
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    monitor = ResourceMonitor(config["resources"]["monitor_interval_seconds"])
    monitor.start()
    started = time.perf_counter()
    checkpoint_records = []
    try:
        scene_reports = []
        get_view_data, _, release_trainer_render_info = _runtime_helpers()
        d0_run = Path(config["d0_h_run"]["path"])
        for scene in config["scenes"]:
            graph_config, graph_inventory = _verified_graph_inventory(scene)
            verified_inputs = {
                name: verify_file(value["path"], value["sha256"])
                for name, value in graph_config["inputs"].items()
            }
            _, _, evaluation_rows, _ = _verify_unary_contract(graph_config, verified_inputs)
            if len(evaluation_rows) != int(scene["expected_evaluation_view_count"]):
                raise ProtocolError(f"E0b H evaluation rows drift: {scene['scene']}")
            e0b_path = Path(scene["e0b_sidecar"]["path"])
            e0b = _load_npz(e0b_path)
            e0b_posterior = np.asarray(e0b["e0b_posterior"], dtype=np.float32)
            checkpoint_path = Path(verified_inputs["formal_checkpoint"]["path"])
            checkpoint_before = sha256_file(checkpoint_path)
            _, dataset, trainer = _build_runtime(graph_config, device)
            base_model, base_index, _, _ = _global_layout(trainer)
            b3_table = _load_npz(Path(verified_inputs["unary_b3_gaussians"]["path"]))
            if not np.array_equal(e0b["gaussian_id"], np.arange(e0b_posterior.size)):
                raise ProtocolError(f"E0b H Gaussian ids drift: {scene['scene']}")
            if (
                e0b_posterior.size != base_model.size
                or not np.array_equal(base_model, b3_table["base_model"])
                or not np.array_equal(base_index, b3_table["base_index"])
            ):
                raise ProtocolError(f"E0b H live Gaussian layout drift: {scene['scene']}")

            evaluation_by_arm = {arm: [] for arm in ARMS}
            graph_run = Path(scene["graph_run"]["path"])
            for view_index, row in enumerate(evaluation_rows):
                frame = int(row["frame"])
                camera_id = int(row["camera_id"])
                baseline = _verified_baseline(
                    graph_run,
                    graph_inventory,
                    config["arms"]["frozen_graph_directory_map"]["U2_B3_G0"],
                    frame,
                    camera_id,
                )
                d0 = _verified_baseline(
                    d0_run,
                    d0_inventory,
                    f"{scene['scene']}/D0",
                    frame,
                    camera_id,
                )
                if not np.array_equal(baseline["target"], d0["target"]):
                    raise ProtocolError(f"E0b H matched target drift: {scene['scene']}")
                target = np.asarray(baseline["target"], dtype=bool)
                image_infos, camera_infos, *_ = get_view_data(dataset, frame, camera_id, device)
                try:
                    with torch.inference_mode():
                        processed_camera, gaussians = _collect_gaussians(
                            trainer, image_infos, camera_infos
                        )
                        alpha, _ = rasterize_ownership_probability(
                            means=gaussians.means,
                            quats=gaussians.quats,
                            scales=gaussians.scales,
                            base_opacities=gaussians.opacities,
                            probability=e0b_posterior,
                            viewmats=torch.linalg.inv(processed_camera.camtoworlds)[None, ...],
                            intrinsics=processed_camera.Ks[None, ...],
                            width=int(processed_camera.W),
                            height=int(processed_camera.H),
                            near_plane=float(trainer.render_cfg.near_plane),
                            far_plane=float(trainer.render_cfg.far_plane),
                            packed=bool(trainer.render_cfg.packed),
                            radius_clip=float(trainer.render_cfg.get("radius_clip", 0.0)),
                            antialiased=bool(trainer.render_cfg.antialiased),
                        )
                        e0b_probability = alpha.detach().cpu().numpy().astype(np.float16)
                finally:
                    release_trainer_render_info(trainer)
                if e0b_probability.shape != target.shape:
                    raise ProtocolError(f"E0b H render shape drift: {scene['scene']}")
                output_path = (
                    run_dir
                    / f"artifacts/evaluation/{scene['scene']}/E0B"
                    / f"f{frame:03d}_c{camera_id}.npz"
                )
                atomic_save_npz(
                    output_path,
                    {"probability": e0b_probability, "target": target.astype(np.int8)},
                )
                probabilities = {
                    "U2_B3_G0": np.asarray(baseline["probability"], dtype=np.float32),
                    "D0": np.asarray(d0["probability"], dtype=np.float32),
                    "E0B": e0b_probability.astype(np.float32),
                }
                for arm, probability in probabilities.items():
                    metrics = _metric_row(
                        probability,
                        target,
                        threshold=float(config["evaluation"]["probability_threshold"]),
                        boundary_tolerance=int(
                            config["evaluation"]["boundary_tolerance_px"]
                        ),
                        ece_bins=int(config["evaluation"]["ece_bins"]),
                    )
                    evaluation_by_arm[arm].append(
                        {"frame": frame, "camera_id": camera_id, **metrics}
                    )
                print(
                    f"E0b H {scene['scene']} {view_index + 1}/{len(evaluation_rows)} "
                    f"frame={frame} camera={camera_id}",
                    flush=True,
                )
            checkpoint_after = sha256_file(checkpoint_path)
            if checkpoint_after != checkpoint_before:
                raise ProtocolError(f"E0b H checkpoint mutation: {scene['scene']}")
            checkpoint_records.append(
                {
                    "scene": scene["scene"],
                    "sha256_before": checkpoint_before,
                    "sha256_after": checkpoint_after,
                }
            )
            aggregate = {
                arm: _aggregate_metrics(
                    [{name: float(row[name]) for name in METRICS} for row in rows]
                )
                for arm, rows in evaluation_by_arm.items()
            }
            report = {
                "scene": scene["scene"],
                "evaluation_view_count": len(evaluation_rows),
                "evaluation_rows": evaluation_by_arm,
                "evaluation_aggregate": aggregate,
                "delta_vs_u2_b3_g0": {
                    arm: {
                        name: float(aggregate[arm][name] - aggregate["U2_B3_G0"][name])
                        for name in METRICS
                    }
                    for arm in ("D0", "E0B")
                },
                "delta_e0b_vs_d0": {
                    name: float(aggregate["E0B"][name] - aggregate["D0"][name])
                    for name in METRICS
                },
                "checkpoint_sha256_before": checkpoint_before,
                "checkpoint_sha256_after": checkpoint_after,
                "target_declaration": config["evaluation"]["target_declaration"],
            }
            _write_json(run_dir / f"artifacts/reports/{scene['scene']}.json", report)
            scene_reports.append(report)
            del trainer, dataset, e0b, e0b_posterior, b3_table
            gc.collect()
            torch.cuda.empty_cache()
            events.append({"event": "scene_completed", "at_utc": _utc_now(), "scene": scene["scene"]})
            _write_jsonl(run_dir / "events.jsonl", events)

        gate = evaluate_e0b_h_gate(
            scene_reports,
            config["primary_h_gate_vs_u2_b3_g0"],
            config["mechanism_h_gate_vs_d0"],
        )
        method_status = "passed_h" if gate["pass"] else "rejected"
        conclusion = (
            "e0b_h_gate_pass_unlock_e1_panogs_preregistration"
            if gate["pass"]
            else "e0b_rejected_stop_e1_e2_advance_gaussian_grouping"
        )
        report = {
            "schema_version": "worldsim_v51_e0b_h_evaluation_report_v1",
            "task_id": TASK_ID,
            "method_status": method_status,
            "conclusion": conclusion,
            "scene_reports": scene_reports,
            "h_gate": gate,
            "checkpoint_records": checkpoint_records,
            "h_quality_read": True,
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
        _write_json(run_dir / "artifacts/h_evaluation_report.json", report)
        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid_samples = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid_samples:
            raise ProtocolError("E0b H resource monitor produced no valid sample")
        resources = {
            "nvidia_start_mib": nvidia_start,
            "nvidia_peak_mib": max(int(row["gpu_used_mib"]) for row in valid_samples),
            "torch_allocated_peak_mib": torch.cuda.max_memory_allocated(device) / 1024**2,
            "torch_reserved_peak_mib": torch.cuda.max_memory_reserved(device) / 1024**2,
            "cgroup_memory_peak_bytes": max(
                int(row["cgroup_memory_current_bytes"]) for row in valid_samples
            ),
            "sample_count": len(monitor.samples),
            "monitor_error_count": len(monitor.samples) - len(valid_samples),
            "wall_seconds": time.perf_counter() - started,
        }
        _write_json(run_dir / "artifacts/resources.json", resources)
        ceilings = config["resources"]
        resource_checks = {
            "nvidia_peak": resources["nvidia_peak_mib"]
            <= int(ceilings["maximum_nvidia_peak_mib"]),
            "torch_allocated_peak": resources["torch_allocated_peak_mib"]
            <= float(ceilings["maximum_torch_allocated_peak_mib"]),
            "torch_reserved_peak": resources["torch_reserved_peak_mib"]
            <= float(ceilings["maximum_torch_reserved_peak_mib"]),
            "cgroup_memory_peak": resources["cgroup_memory_peak_bytes"]
            <= int(ceilings["maximum_cgroup_memory_bytes"]),
            "wall": resources["wall_seconds"] <= float(ceilings["maximum_wall_seconds"]),
            "monitor": resources["monitor_error_count"] == 0,
        }
        if not all(resource_checks.values()):
            raise ProtocolError(f"E0b H resource gate failed: {resource_checks}")

        terminal_status = "done" if gate["pass"] else "rejected"
        summary = {
            "schema_version": "worldsim_v51_e0b_h_evaluation_summary_v1",
            "task_id": TASK_ID,
            "status": terminal_status,
            "conclusion": conclusion,
            "source_commit": source_commit,
            "source_tree": source_tree,
            "h_gate": gate,
            "scene_reports": scene_reports,
            "resources": resources,
            "resource_checks": resource_checks,
            "target_declaration": config["evaluation"]["target_declaration"],
            "h_quality_read": True,
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
                "schema_version": "worldsim_v51_e0b_h_evaluation_manifest_v1",
                "task_id": TASK_ID,
                "status": terminal_status,
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_e0b_h_evaluation_status_v1",
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
                "schema_version": "worldsim_v51_e0b_h_evaluation_status_v1",
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
        default=PROJECT / "configs/worldsim_v51/stage_e_e0b_h_evaluation_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.run_dir.resolve(), args.device)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
