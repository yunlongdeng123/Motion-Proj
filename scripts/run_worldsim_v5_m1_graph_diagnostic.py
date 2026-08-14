#!/usr/bin/env python3
"""复用 r037 冻结 unary，执行 scene0471 的小型 graph mechanism 诊断。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v4.evidence_metrics import probability_metrics
from motion_proj.worldsim_v5.evidence_schema import (
    atomic_save_npz,
    sha256_file,
    validate_gaussian_table,
)
from motion_proj.worldsim_v5.ownership_renderer import rasterize_ownership_probability
from motion_proj.worldsim_v5.physical_graph import (
    build_physical_graph,
    diffuse_graph_probability,
    graph_affinity_diagnostics,
)
from scripts.run_worldsim_v5_m1_unary_diagnostic import (
    _aggregate_metrics,
    _binary_iou,
    _boundary_f1,
    _build_runtime,
    _collect_gaussians,
    _global_layout,
    _negative_log_likelihood,
    _runtime_helpers,
)
from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    inventory_files,
    prepare_formal_run,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M1-STRUCTURED-OWNERSHIP-01"
SCHEMA_VERSION = "worldsim_v5_m1_graph_diagnostic_v1"
UNARY_NAMES = ("B1", "B3")
GRAPH_NAMES = ("G0", "G1", "G2", "G3")


class GraphDiagnosticError(RuntimeError):
    """冻结输入、graph 或 render replay 漂移。"""


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise GraphDiagnosticError("graph diagnostic config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "structured_graph_mechanism_smoke"
    ):
        raise GraphDiagnosticError("graph diagnostic task/phase/status 漂移")
    if tuple(payload["graph"]["unary_inputs"]) != UNARY_NAMES:
        raise GraphDiagnosticError("graph unary input 集合或顺序漂移")
    if tuple(payload["graph"]["arms"]) != GRAPH_NAMES:
        raise GraphDiagnosticError("graph arm 集合或顺序漂移")
    if payload["graph"].get("base_model_consumed_by_graph") is not False:
        raise GraphDiagnosticError("graph 禁止消费 base_model proxy")
    if payload["evaluation"].get("automatic_validation_unlock") is not False:
        raise GraphDiagnosticError("graph diagnostic 禁止自动解锁 validation")
    if payload["evaluation"].get("automatic_semantic_split_unlock") is not False:
        raise GraphDiagnosticError("graph diagnostic 禁止自动解锁 semantic split")
    return payload


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.array(data[name], copy=True) for name in data.files}


def _metric_row(
    probability: np.ndarray,
    target: np.ndarray,
    *,
    threshold: float,
    boundary_tolerance: int,
    ece_bins: int,
) -> dict[str, float]:
    metrics = probability_metrics(
        np.asarray(probability, dtype=np.float32),
        np.asarray(target, dtype=np.float32),
        bins=int(ece_bins),
    )
    metrics.update(
        iou_at_frozen_threshold=_binary_iou(probability >= threshold, target),
        boundary_f1=_boundary_f1(
            probability >= threshold, target, tolerance=int(boundary_tolerance)
        ),
        nll=_negative_log_likelihood(probability, target),
    )
    return metrics


def _write_terminal(
    run_dir: Path,
    *,
    status: str,
    source_head: str,
    summary_sha256: str | None,
    manifest_sha256: str | None,
    reason: str | None = None,
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v5_m1_graph_status_v1",
            "task_id": TASK_ID,
            "status": status,
            "source_commit": source_head,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def _verify_unary_contract(
    config: Mapping[str, Any], inputs: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]], Path]:
    summary = json.loads(Path(inputs["unary_summary"]["path"]).read_text(encoding="utf-8"))
    status = json.loads(Path(inputs["unary_status"]["path"]).read_text(encoding="utf-8"))
    fingerprint = json.loads(
        Path(inputs["unary_fingerprint"]["path"]).read_text(encoding="utf-8")
    )
    manifest = json.loads(
        Path(inputs["unary_manifest"]["path"]).read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        Path(inputs["unary_diagnostics"]["path"]).read_text(encoding="utf-8")
    )
    if (
        summary.get("task_id") != TASK_ID
        or summary.get("status") != "done"
        or summary.get("phase") != "structured_unary_mechanism_smoke"
        or summary.get("scene") != config["scene"]["name"]
        or summary.get("graph_inference_started") is not False
        or summary.get("parameter_search_performed") is not False
        or summary.get("validation_quality_read") is not False
        or summary.get("heldout_quality_read") is not False
    ):
        raise GraphDiagnosticError("r037 summary contract 漂移")
    if (
        status.get("status") != "done"
        or status.get("summary_sha256") != inputs["unary_summary"]["sha256"]
        or status.get("manifest_sha256") != inputs["unary_manifest"]["sha256"]
        or fingerprint.get("source_clean") is not True
        or manifest.get("status") != "done"
        or diagnostics.get("graph_inference_started") is not False
        or diagnostics.get("parameter_search_performed") is not False
    ):
        raise GraphDiagnosticError("r037 terminal/fingerprint contract 漂移")
    inventory = {row["path"]: row for row in manifest["inventory"]}
    unary_root = Path(inputs["unary_manifest"]["path"]).parent
    for unary, input_name in (
        ("B1", "unary_b1_gaussians"),
        ("B3", "unary_b3_gaussians"),
    ):
        relative = f"artifacts/gaussians/{unary}.npz"
        if (
            relative not in inventory
            or inventory[relative]["sha256"] != inputs[input_name]["sha256"]
            or Path(inputs[input_name]["path"]).resolve() != (unary_root / relative).resolve()
        ):
            raise GraphDiagnosticError(f"r037 {unary} Gaussian inventory 漂移")
    rows = diagnostics.get("evaluation_rows", {}).get("B1", [])
    b3_rows = diagnostics.get("evaluation_rows", {}).get("B3", [])
    if (
        len(rows) != int(summary["accepted_evaluation_view_count"])
        or len(rows) != 8
        or len(b3_rows) != 8
        or int(summary["abstained_evaluation_view_count"]) != 7
    ):
        raise GraphDiagnosticError("r037 evaluation denominator 漂移")
    for row in [*rows, *b3_rows]:
        relative = str(row["path"])
        record = inventory.get(relative)
        path = unary_root / relative
        if (
            record is None
            or record["sha256"] != row["sha256"]
            or sha256_file(path) != row["sha256"]
        ):
            raise GraphDiagnosticError(f"r037 evaluation artifact 漂移: {relative}")
    return summary, diagnostics, rows, unary_root


def _shared_geometry_contract(
    left: Mapping[str, np.ndarray], right: Mapping[str, np.ndarray]
) -> None:
    for name in (
        "gaussian_id",
        "base_model",
        "base_index",
        "center",
        "covariance",
        "normal_proxy",
        "normal_available",
    ):
        if not np.array_equal(left[name], right[name]):
            raise GraphDiagnosticError(f"B1/B3 shared geometry 漂移: {name}")


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved_record = write_resolved_config(run_dir, config)
    events: list[dict[str, Any]] = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}
    ]
    write_events(run_dir, events)
    try:
        inputs = {
            name: verify_file(value["path"], value["sha256"])
            for name, value in config["inputs"].items()
        }
        unary_summary, unary_diagnostics, evaluation_rows, unary_root = (
            _verify_unary_contract(config, inputs)
        )
        checkpoint_path = Path(inputs["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint_path)
        tables = {
            "B1": _load_npz(Path(inputs["unary_b1_gaussians"]["path"])),
            "B3": _load_npz(Path(inputs["unary_b3_gaussians"]["path"])),
        }
        for table in tables.values():
            validate_gaussian_table(table)
        _shared_geometry_contract(tables["B1"], tables["B3"])
        graph_cfg = config["graph"]
        started = time.perf_counter()
        edge_table, graph_metadata = build_physical_graph(
            tables["B1"],
            candidate_k=int(graph_cfg["candidate_k"]),
            normal_affinity_power=float(graph_cfg["normal_affinity_power"]),
            workers=int(graph_cfg["query_workers"]),
            chunk_size=int(graph_cfg["edge_chunk_size"]),
        )
        source = edge_table["source_gaussian_id"]
        target = edge_table["target_gaussian_id"]
        b3_barrier = np.clip(
            1.0
            - np.maximum(
                tables["B3"]["boundary_ambiguity"][source],
                tables["B3"]["boundary_ambiguity"][target],
            ),
            0.0,
            1.0,
        ).astype(np.float32)
        edge_table["boundary_barrier_B1"] = edge_table["boundary_barrier"].copy()
        edge_table["boundary_barrier_B3"] = b3_barrier
        edge_table["full_affinity_B1"] = edge_table["edge_affinity"].copy()
        edge_table["full_affinity_B3"] = (
            edge_table["physical_affinity"] * b3_barrier
        ).astype(np.float32)
        affinity_by_graph = {
            "G1": edge_table["euclidean_affinity"],
            "G2": edge_table["physical_affinity"],
        }
        posteriors: dict[str, np.ndarray] = {}
        diffusion_reports: dict[str, dict[str, float | int]] = {}
        for unary in UNARY_NAMES:
            initial = np.asarray(tables[unary]["unary_posterior"], dtype=np.float32)
            posteriors[f"{unary}_G0"] = initial.copy()
            for graph in ("G1", "G2", "G3"):
                affinity = (
                    edge_table[f"full_affinity_{unary}"]
                    if graph == "G3"
                    else affinity_by_graph[graph]
                )
                posterior, report = diffuse_graph_probability(
                    initial,
                    source_gaussian_id=source,
                    target_gaussian_id=target,
                    edge_affinity=affinity,
                    effective_evidence_count=tables[unary][
                        "effective_evidence_count"
                    ],
                    diffusion_rate=float(graph_cfg["diffusion_rate"]),
                    iterations=int(graph_cfg["diffusion_iterations"]),
                    minimum_affinity_sum=float(graph_cfg["minimum_affinity_sum"]),
                )
                key = f"{unary}_{graph}"
                posteriors[key] = posterior
                diffusion_reports[key] = report

        proxy_target = tables["B1"]["base_model"] == "RigidNodes"
        evaluation_cfg = config["evaluation"]
        gaussian_metrics = {
            key: _metric_row(
                value,
                proxy_target,
                threshold=float(evaluation_cfg["probability_threshold"]),
                boundary_tolerance=int(evaluation_cfg["boundary_tolerance_px"]),
                ece_bins=int(evaluation_cfg["ece_bins"]),
            )
            for key, value in posteriors.items()
        }
        gaussian_delta_vs_g0 = {
            f"{unary}_{graph}": {
                metric: float(
                    gaussian_metrics[f"{unary}_{graph}"][metric]
                    - gaussian_metrics[f"{unary}_G0"][metric]
                )
                for metric in gaussian_metrics[f"{unary}_{graph}"]
            }
            for unary in UNARY_NAMES
            for graph in ("G1", "G2", "G3")
        }
        topology: dict[str, dict[str, float | int]] = {}
        for unary in UNARY_NAMES:
            for graph in ("G1", "G2", "G3"):
                affinity = (
                    edge_table[f"full_affinity_{unary}"]
                    if graph == "G3"
                    else affinity_by_graph[graph]
                )
                key = f"{unary}_{graph}"
                topology[key] = graph_affinity_diagnostics(
                    source_gaussian_id=source,
                    target_gaussian_id=target,
                    edge_affinity=affinity,
                    probability=posteriors[key],
                    proxy_target=proxy_target,
                )

        edge_path = run_dir / "artifacts/graph/edges.npz"
        atomic_save_npz(edge_path, edge_table)
        posterior_path = run_dir / "artifacts/graph/posteriors.npz"
        atomic_save_npz(
            posterior_path,
            {"gaussian_id": tables["B1"]["gaussian_id"], **posteriors},
        )
        if not torch.cuda.is_available():
            raise GraphDiagnosticError("graph render diagnostic 需要 CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        maximum_start = int(config["resources"]["maximum_gpu_allocated_at_start_mib"])
        if torch.cuda.memory_allocated(device) > maximum_start * 1024**2:
            raise GraphDiagnosticError("graph diagnostic GPU preflight 非空闲")
        torch.cuda.reset_peak_memory_stats(device)
        get_view_data, _, release_trainer_render_info = _runtime_helpers()
        _, dataset, trainer = _build_runtime(config, device)
        base_model, base_index, background_count, rigid_count = _global_layout(trainer)
        if (
            not np.array_equal(base_model, tables["B1"]["base_model"])
            or not np.array_equal(base_index, tables["B1"]["base_index"])
        ):
            raise GraphDiagnosticError("live checkpoint Gaussian layout 与 r037 漂移")

        evaluation_by_arm: dict[str, list[dict[str, Any]]] = {
            key: [] for key in posteriors
        }
        g0_replay_exact: dict[str, bool] = {unary: True for unary in UNARY_NAMES}
        for view_index, row in enumerate(evaluation_rows):
            frame = int(row["frame"])
            camera_id = int(row["camera_id"])
            frozen_b1_path = unary_root / str(row["path"])
            frozen_b1 = _load_npz(frozen_b1_path)
            target_mask = np.asarray(frozen_b1["target"], dtype=bool)
            frozen_by_unary = {"B1": frozen_b1}
            b3_relative = str(row["path"]).replace(
                "artifacts/evaluation/B1/", "artifacts/evaluation/B3/"
            )
            frozen_by_unary["B3"] = _load_npz(unary_root / b3_relative)
            if not np.array_equal(
                frozen_by_unary["B3"]["target"], frozen_b1["target"]
            ):
                raise GraphDiagnosticError("r037 B1/B3 evaluation target 漂移")
            image_infos, camera_infos, *_ = get_view_data(
                dataset, frame, camera_id, device
            )
            try:
                with torch.inference_mode():
                    processed_camera, gaussians = _collect_gaussians(
                        trainer, image_infos, camera_infos
                    )
                    for key, posterior in posteriors.items():
                        alpha, _ = rasterize_ownership_probability(
                            means=gaussians.means,
                            quats=gaussians.quats,
                            scales=gaussians.scales,
                            base_opacities=gaussians.opacities,
                            probability=posterior,
                            viewmats=torch.linalg.inv(processed_camera.camtoworlds)[
                                None, ...
                            ],
                            intrinsics=processed_camera.Ks[None, ...],
                            width=int(processed_camera.W),
                            height=int(processed_camera.H),
                            near_plane=float(trainer.render_cfg.near_plane),
                            far_plane=float(trainer.render_cfg.far_plane),
                            packed=bool(trainer.render_cfg.packed),
                            radius_clip=float(
                                trainer.render_cfg.get("radius_clip", 0.0)
                            ),
                            antialiased=bool(trainer.render_cfg.antialiased),
                        )
                        probability = alpha.detach().cpu().numpy().astype(np.float32)
                        if probability.shape != target_mask.shape:
                            raise GraphDiagnosticError("graph render/target shape 漂移")
                        unary, graph = key.split("_", 1)
                        if graph == "G0":
                            exact = np.array_equal(
                                probability.astype(np.float16),
                                frozen_by_unary[unary]["probability"],
                            )
                            g0_replay_exact[unary] &= bool(exact)
                            if not exact:
                                raise GraphDiagnosticError(
                                    f"{unary} G0 未 exact replay r037 float16"
                                )
                        metrics = _metric_row(
                            probability,
                            target_mask,
                            threshold=float(
                                evaluation_cfg["probability_threshold"]
                            ),
                            boundary_tolerance=int(
                                evaluation_cfg["boundary_tolerance_px"]
                            ),
                            ece_bins=int(evaluation_cfg["ece_bins"]),
                        )
                        output = (
                            run_dir
                            / "artifacts/evaluation"
                            / key
                            / f"f{frame:03d}_c{camera_id}.npz"
                        )
                        atomic_save_npz(
                            output,
                            {
                                "probability": probability.astype(np.float16),
                                "target": target_mask.astype(np.int8),
                            },
                        )
                        evaluation_by_arm[key].append(
                            {
                                "frame": frame,
                                "camera_id": camera_id,
                                **metrics,
                                "path": str(output.relative_to(run_dir)),
                                "sha256": sha256_file(output),
                            }
                        )
            finally:
                release_trainer_render_info(trainer)
            print(
                f"M1 graph evaluation {view_index + 1}/{len(evaluation_rows)} "
                f"frame={frame} camera={camera_id}",
                flush=True,
            )

        evaluation_aggregate = {
            key: _aggregate_metrics(
                [
                    {
                        name: value
                        for name, value in row.items()
                        if isinstance(value, float)
                    }
                    for row in rows
                ]
            )
            for key, rows in evaluation_by_arm.items()
        }
        evaluation_delta_vs_g0 = {
            f"{unary}_{graph}": {
                metric: float(
                    evaluation_aggregate[f"{unary}_{graph}"][metric]
                    - evaluation_aggregate[f"{unary}_G0"][metric]
                )
                for metric in evaluation_aggregate[f"{unary}_{graph}"]
            }
            for unary in UNARY_NAMES
            for graph in ("G1", "G2", "G3")
        }
        checkpoint_after = sha256_file(checkpoint_path)
        if checkpoint_after != checkpoint_before:
            raise GraphDiagnosticError("formal checkpoint 在 graph diagnostic 后 mutation")
        duration = time.perf_counter() - started
        diagnostics = {
            "schema_version": "worldsim_v5_m1_graph_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": config["scene"]["name"],
            "gaussian_counts": {
                "Background": background_count,
                "RigidNodes": rigid_count,
            },
            "graph_metadata": graph_metadata,
            "graph_does_not_consume_base_model": True,
            "diffusion_reports": diffusion_reports,
            "topology_diagnostics": topology,
            "gaussian_metrics": gaussian_metrics,
            "gaussian_delta_vs_g0": gaussian_delta_vs_g0,
            "evaluation_rows": evaluation_by_arm,
            "evaluation_aggregate": evaluation_aggregate,
            "evaluation_delta_vs_g0": evaluation_delta_vs_g0,
            "g0_replay_r037_float16_exact": g0_replay_exact,
            "parameter_search_performed": False,
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "semantic_split_started": False,
        }
        diagnostics_path = run_dir / "artifacts/diagnostics.json"
        atomic_json(diagnostics_path, diagnostics)
        snapshot = copy_source_snapshot(
            run_dir,
            [
                config_path,
                PROJECT / "scripts/run_worldsim_v5_m1_graph_diagnostic.py",
                PROJECT / "scripts/run_worldsim_v5_m1_unary_diagnostic.py",
                PROJECT / "scripts/worldsim_v5_forensics_common.py",
                PROJECT / "scripts/eval_worldsim_v3_a3_r1_heldout.py",
                PROJECT / "motion_proj/worldsim_v4/evidence_metrics.py",
                PROJECT / "motion_proj/worldsim_v5/evidence_schema.py",
                PROJECT / "motion_proj/worldsim_v5/ownership_renderer.py",
                PROJECT / "motion_proj/worldsim_v5/physical_graph.py",
                PROJECT / "tests/test_run_worldsim_v5_m1_graph_diagnostic.py",
                PROJECT / "tests/test_worldsim_v5_physical_graph.py",
            ],
            PROJECT,
        )
        summary = {
            "schema_version": "worldsim_v5_m1_graph_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "phase": "structured_graph_mechanism_smoke",
            "scene": config["scene"]["name"],
            "source_commit": source_head,
            "unary_source_commit": unary_summary["source_commit"],
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "gaussian_count": int(tables["B1"]["gaussian_id"].size),
            "edge_count": int(graph_metadata["edge_count"]),
            "candidate_k": int(graph_metadata["candidate_k"]),
            "evaluation_view_count": len(evaluation_rows),
            "duration_seconds": duration,
            "peak_gpu_memory_mib": int(
                torch.cuda.max_memory_allocated(device) / 1024**2
            ),
            "edge_artifact_sha256": sha256_file(edge_path),
            "posterior_artifact_sha256": sha256_file(posterior_path),
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "g0_replay_r037_float16_exact": g0_replay_exact,
            "arm_gaussian_metrics": gaussian_metrics,
            "arm_gaussian_delta_vs_g0": gaussian_delta_vs_g0,
            "arm_evaluation_aggregate": evaluation_aggregate,
            "arm_evaluation_delta_vs_g0": evaluation_delta_vs_g0,
            "topology_diagnostics": topology,
            "graph_inference_started": True,
            "method_inference_started": True,
            "parameter_search_performed": False,
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "semantic_split_started": False,
            "automatic_validation_unlock": False,
        }
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        fingerprint = {
            "schema_version": "worldsim_v5_m1_graph_fingerprint_v1",
            "task_id": TASK_ID,
            "source_commit": source_head,
            "source_clean": True,
            "resolved_config": resolved_record,
            "inputs": inputs,
            "runtime": {
                "drivestudio_commit": config["runtime"]["drivestudio_commit"],
                "drivestudio_status": config["runtime"][
                    "drivestudio_expected_status"
                ],
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(device),
            },
            "source_snapshot": snapshot,
        }
        fingerprint_path = run_dir / "fingerprint.json"
        atomic_json(fingerprint_path, fingerprint)
        events.append({"event": "run_done", "at_utc": utc_now()})
        write_events(run_dir, events)
        manifest = {
            "schema_version": "worldsim_v5_m1_graph_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
        }
        manifest_path = run_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        _write_terminal(
            run_dir,
            status="done",
            source_head=source_head,
            summary_sha256=sha256_file(summary_path),
            manifest_sha256=sha256_file(manifest_path),
        )
        return summary
    except Exception as error:
        events.append(
            {
                "event": "run_blocked",
                "at_utc": utc_now(),
                "reason": f"{type(error).__name__}: {error}",
            }
        )
        write_events(run_dir, events)
        _write_terminal(
            run_dir,
            status="blocked",
            source_head=source_head,
            summary_sha256=None,
            manifest_sha256=None,
            reason=f"{type(error).__name__}: {error}",
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    result = run(args.config.resolve(), args.run_dir.resolve(), args.device)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
