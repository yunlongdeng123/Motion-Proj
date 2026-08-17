#!/usr/bin/env python3
"""Evaluate frozen D0 against matched U2/B3 G0 and V5 G3 on H only."""

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
from motion_proj.worldsim_v51.progressive_evaluation import (
    METRICS,
    evaluate_progressive_h_gate,
)
from motion_proj.worldsim_v51.protocol import ProtocolError, V51_BRANCH, load_yaml, sha256_file
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


SCHEMA = "worldsim_v51_stage_d_progressive_h_evaluation_v1"
TASK_ID = "WS-V51-M1-D-PROGRESSIVE-01"
SCENES = ("scene-0471", "scene-1087", "scene-0379")
ARMS = ("U2_B3_G0", "U2_B3_G_V5", "D0")


def validate_config(config_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA:
        raise ProtocolError("D0 H evaluation schema drift")
    if config.get("task_id") != TASK_ID or config.get("status") != "running":
        raise ProtocolError("D0 H evaluation task/status drift")
    if config.get("phase") != "d0_h_matched_evaluation" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("D0 H evaluation phase/seed drift")

    freeze_spec = config["operator_freeze"]
    freeze_path = PROJECT / freeze_spec["path"]
    if not freeze_path.is_file() or sha256_file(freeze_path) != freeze_spec["sha256"]:
        raise ProtocolError("D0 operator freeze identity drift")
    freeze = load_yaml(freeze_path)
    if freeze.get("status") != freeze_spec["required_status"]:
        raise ProtocolError("D0 operator freeze terminal drift")
    if freeze["canonical_run"]["conclusion"] != freeze_spec["required_conclusion"]:
        raise ProtocolError("D0 operator freeze conclusion drift")

    for name, spec in config["reference_implementation"].items():
        path = PROJECT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"D0 H reference implementation drift: {name}")
    arms = config["arms"]
    if (
        arms.get("primary_comparator") != "U2_B3_G0"
        or arms.get("strong_external_baseline") != "U2_B3_G_V5"
        or arms.get("candidate") != "D0"
    ):
        raise ProtocolError("D0 H arm contract drift")
    if arms.get("persisted_probability_dtype") != "float16":
        raise ProtocolError("D0 H persisted precision drift")

    scenes = config["scenes"]
    if tuple(scene["scene"] for scene in scenes) != SCENES:
        raise ProtocolError("D0 H scene order drift")
    if [int(scene["expected_evaluation_view_count"]) for scene in scenes] != [8, 1, 3]:
        raise ProtocolError("D0 H view denominator drift")
    for scene, frozen_scene in zip(scenes, freeze["scenes"]):
        if scene["scene"] != frozen_scene["scene"]:
            raise ProtocolError("D0 H freeze scene order drift")
        if scene["d0_sidecar"]["sha256"] != frozen_scene["sidecar_sha256"]:
            raise ProtocolError(f"D0 H sidecar binding drift: {scene['scene']}")

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
            raise ProtocolError(f"D0 H evaluation contract drift: {name}")
    gate = config["h_gate"]
    expected_gate = {
        "scene_count": 3,
        "minimum_positive_boundary_f1_scenes": 2,
        "minimum_scene_balanced_boundary_f1_delta_exclusive": 0.0,
        "minimum_scene_balanced_iou_delta": 0.0,
        "maximum_scene_balanced_false_negative_semantic_mass_delta": 0.02,
        "pass_action": "freeze_d0_then_run_s_exact_once_without_parameter_change",
        "fail_action": "reject_progressive_skip_d1_then_advance_super_primitive_or_anchor",
    }
    for name, expected in expected_gate.items():
        if gate.get(name) != expected:
            raise ProtocolError(f"D0 H gate drift: {name}")
    expected_resources = {
        "maximum_nvidia_at_start_mib": 512,
        "maximum_nvidia_peak_mib": 24000,
        "maximum_torch_allocated_peak_mib": 24000,
        "maximum_torch_reserved_peak_mib": 24000,
        "maximum_cgroup_memory_bytes": 85899345920,
        "maximum_wall_seconds": 3600,
    }
    for name, expected in expected_resources.items():
        if config["resources"].get(name) != expected:
            raise ProtocolError(f"D0 H resource contract drift: {name}")

    runtime = config["runtime"]
    observed_runtime = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if observed_runtime != {name: runtime[name] for name in observed_runtime}:
        raise ProtocolError(f"D0 H runtime drift: {observed_runtime}")
    locks = config["locks"]
    for name in (
        "parameter_search",
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
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"D0 H lock drift: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("M2/M3 must remain pending")
    return config, freeze


def _verified_graph_inventory(scene: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    graph_config_path = PROJECT / scene["graph_config"]["path"]
    if sha256_file(graph_config_path) != scene["graph_config"]["sha256"]:
        raise ProtocolError(f"D0 H graph config identity drift: {scene['scene']}")
    graph_config = load_yaml(graph_config_path)
    run = Path(scene["graph_run"]["path"])
    manifest_path = run / "manifest.json"
    if sha256_file(manifest_path) != scene["graph_run"]["manifest_sha256"]:
        raise ProtocolError(f"D0 H graph manifest identity drift: {scene['scene']}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "done":
        raise ProtocolError(f"D0 H graph run terminal drift: {scene['scene']}")
    return graph_config, {row["path"]: row for row in manifest["inventory"]}


def _verified_baseline(
    run: Path,
    inventory: dict[str, Any],
    arm_directory: str,
    frame: int,
    camera_id: int,
) -> dict[str, np.ndarray]:
    relative = f"artifacts/evaluation/{arm_directory}/f{frame:03d}_c{camera_id}.npz"
    record = inventory.get(relative)
    path = run / relative
    if (
        record is None
        or not path.is_file()
        or path.stat().st_size != int(record["bytes"])
    ):
        raise ProtocolError(f"D0 H baseline artifact missing: {relative}")
    if sha256_file(path) != record["sha256"]:
        raise ProtocolError(f"D0 H baseline artifact drift: {relative}")
    return _load_npz(path)


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config, _ = validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"D0 H run directory exists: {run_dir}")
    if _git(PROJECT, "branch", "--show-current") != V51_BRANCH:
        raise ProtocolError("D0 H evaluation must run on V5.1 branch")
    if _git(PROJECT, "status", "--porcelain"):
        raise ProtocolError("D0 H evaluation requires a clean worktree")
    if not torch.cuda.is_available():
        raise ProtocolError("D0 H evaluation requires CUDA")
    nvidia_start = _nvidia_used_mib()
    if nvidia_start > int(config["resources"]["maximum_nvidia_at_start_mib"]):
        raise ProtocolError(
            f"D0 H GPU not idle at start: {nvidia_start} MiB"
        )

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
        for scene in config["scenes"]:
            graph_config, graph_inventory = _verified_graph_inventory(scene)
            verified_inputs = {
                name: verify_file(value["path"], value["sha256"])
                for name, value in graph_config["inputs"].items()
            }
            _, _, evaluation_rows, _ = _verify_unary_contract(
                graph_config, verified_inputs
            )
            if len(evaluation_rows) != int(scene["expected_evaluation_view_count"]):
                raise ProtocolError(f"D0 H evaluation rows drift: {scene['scene']}")
            d0_path = Path(scene["d0_sidecar"]["path"])
            if sha256_file(d0_path) != scene["d0_sidecar"]["sha256"]:
                raise ProtocolError(f"D0 H sidecar identity drift: {scene['scene']}")
            d0 = _load_npz(d0_path)
            d0_posterior = np.asarray(d0["d0_posterior"], dtype=np.float32)
            checkpoint_path = Path(verified_inputs["formal_checkpoint"]["path"])
            checkpoint_before = sha256_file(checkpoint_path)
            _, dataset, trainer = _build_runtime(graph_config, device)
            base_model, base_index, _, _ = _global_layout(trainer)
            b3_table = _load_npz(Path(verified_inputs["unary_b3_gaussians"]["path"]))
            with np.load(scene["d0_sidecar"]["path"], allow_pickle=False) as table:
                if not np.array_equal(table["gaussian_id"], np.arange(d0_posterior.size)):
                    raise ProtocolError(f"D0 H Gaussian ids drift: {scene['scene']}")
            if (
                d0_posterior.size != base_model.size
                or not np.array_equal(base_model, b3_table["base_model"])
                or not np.array_equal(base_index, b3_table["base_index"])
            ):
                raise ProtocolError(f"D0 H live Gaussian layout drift: {scene['scene']}")

            evaluation_by_arm = {arm: [] for arm in ARMS}
            graph_run = Path(scene["graph_run"]["path"])
            for view_index, row in enumerate(evaluation_rows):
                frame = int(row["frame"])
                camera_id = int(row["camera_id"])
                baselines = {
                    arm: _verified_baseline(
                        graph_run,
                        graph_inventory,
                        config["arms"]["frozen_graph_directory_map"][arm],
                        frame,
                        camera_id,
                    )
                    for arm in ("U2_B3_G0", "U2_B3_G_V5")
                }
                if not np.array_equal(
                    baselines["U2_B3_G0"]["target"],
                    baselines["U2_B3_G_V5"]["target"],
                ):
                    raise ProtocolError(f"D0 H baseline target drift: {scene['scene']}")
                target = np.asarray(baselines["U2_B3_G0"]["target"], dtype=bool)
                image_infos, camera_infos, *_ = get_view_data(
                    dataset, frame, camera_id, device
                )
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
                            probability=d0_posterior,
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
                        d0_probability = alpha.detach().cpu().numpy().astype(np.float16)
                finally:
                    release_trainer_render_info(trainer)
                if d0_probability.shape != target.shape:
                    raise ProtocolError(f"D0 H render shape drift: {scene['scene']}")
                output_path = (
                    run_dir
                    / f"artifacts/evaluation/{scene['scene']}/D0"
                    / f"f{frame:03d}_c{camera_id}.npz"
                )
                atomic_save_npz(
                    output_path,
                    {"probability": d0_probability, "target": target.astype(np.int8)},
                )
                arm_probabilities = {
                    "U2_B3_G0": np.asarray(
                        baselines["U2_B3_G0"]["probability"], dtype=np.float32
                    ),
                    "U2_B3_G_V5": np.asarray(
                        baselines["U2_B3_G_V5"]["probability"], dtype=np.float32
                    ),
                    "D0": d0_probability.astype(np.float32),
                }
                for arm, probability in arm_probabilities.items():
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
                    f"D0 H {scene['scene']} {view_index + 1}/{len(evaluation_rows)} "
                    f"frame={frame} camera={camera_id}",
                    flush=True,
                )
            checkpoint_after = sha256_file(checkpoint_path)
            if checkpoint_after != checkpoint_before:
                raise ProtocolError(f"D0 H checkpoint mutation: {scene['scene']}")
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
                    for arm in ("U2_B3_G_V5", "D0")
                },
                "checkpoint_sha256_before": checkpoint_before,
                "checkpoint_sha256_after": checkpoint_after,
                "target_declaration": config["evaluation"]["target_declaration"],
            }
            _write_json(run_dir / f"artifacts/reports/{scene['scene']}.json", report)
            scene_reports.append(report)
            del trainer, dataset, d0, d0_posterior, b3_table
            gc.collect()
            torch.cuda.empty_cache()
            events.append({"event": "scene_completed", "at_utc": _utc_now(), "scene": scene["scene"]})
            _write_jsonl(run_dir / "events.jsonl", events)

        gate = evaluate_progressive_h_gate(scene_reports, config["h_gate"])
        method_status = "passed_h" if gate["pass"] else "rejected"
        conclusion = (
            "d0_h_gate_pass_freeze_then_s_exact_once"
            if gate["pass"]
            else "d0_progressive_rejected_skip_d1_advance_super_primitive_or_anchor"
        )
        report = {
            "schema_version": "worldsim_v51_d0_h_evaluation_report_v1",
            "task_id": TASK_ID,
            "method_status": method_status,
            "conclusion": conclusion,
            "scene_reports": scene_reports,
            "h_gate": gate,
            "checkpoint_records": checkpoint_records,
            "parameter_search": False,
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
            raise ProtocolError("D0 H resource monitor produced no valid sample")
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
            raise ProtocolError(f"D0 H resource gate failed: {resource_checks}")

        terminal_status = "done" if gate["pass"] else "rejected"
        summary = {
            "schema_version": "worldsim_v51_d0_h_evaluation_summary_v1",
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
            "parameter_search": False,
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
        manifest = {
            "schema_version": "worldsim_v51_d0_h_evaluation_manifest_v1",
            "task_id": TASK_ID,
            "status": terminal_status,
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_d0_h_evaluation_status_v1",
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
            {
                "event": "run_blocked",
                "at_utc": _utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_d0_h_evaluation_status_v1",
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
        default=PROJECT / "configs/worldsim_v51/stage_d_progressive_h_evaluation_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.run_dir.resolve(), args.device)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
