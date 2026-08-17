#!/usr/bin/env python3
"""在 E0a fine voxel node 上运行与 D0 相同的 no-quality progressive propagation。"""

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


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v5.evidence_schema import atomic_save_npz
from motion_proj.worldsim_v51.progressive_propagation import (
    affinity_matrices_from_topologies,
    build_exact_logical_topologies,
    progressive_region_growing,
)
from motion_proj.worldsim_v51.protocol import ProtocolError, V51_BRANCH, load_yaml, sha256_file
from motion_proj.worldsim_v51.superprimitive_propagation import (
    aggregate_node_evidence,
    broadcast_node_result,
    quotient_directed_edges,
)
from scripts.run_worldsim_v51_d0_progressive_operator import (
    ResourceMonitor,
    _git,
    _inventory,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
    load_observation_matrices,
)


SCHEMA = "worldsim_v51_stage_e_e0b_same_propagation_v1"
TASK_ID = "WS-V51-M1-E-NODE-ELEVATION-01"
SCENES = ("scene-0471", "scene-1087", "scene-0379")


def _load_frozen_yaml(config: dict[str, Any], name: str) -> dict[str, Any]:
    spec = config[name]
    path = PROJECT / spec["path"]
    if not path.is_file() or sha256_file(path) != spec["sha256"]:
        raise ProtocolError(f"E0b frozen input identity drift: {name}")
    payload = load_yaml(path)
    if payload.get("status") != spec["required_status"]:
        raise ProtocolError(f"E0b frozen input terminal drift: {name}")
    return payload


def validate_config(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    config = load_yaml(config_path)
    if config.get("schema_version") != SCHEMA:
        raise ProtocolError("E0b schema drift")
    if config.get("task_id") != TASK_ID or config.get("status") != "running":
        raise ProtocolError("E0b task/status drift")
    if config.get("phase") != "e0b_fine_q50_same_propagation_without_quality_read":
        raise ProtocolError("E0b phase drift")
    if int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("E0b seed drift")

    e0a = _load_frozen_yaml(config, "e0a_freeze")
    d0 = _load_frozen_yaml(config, "stage_d_operator_freeze")
    rejection = _load_frozen_yaml(config, "stage_d_rejection_freeze")
    if e0a["canonical_run"]["conclusion"] != config["e0a_freeze"]["required_conclusion"]:
        raise ProtocolError("E0b E0a conclusion drift")
    if bool(e0a["gate"]["pass"]) is not True:
        raise ProtocolError("E0b requires a passing E0a structural gate")
    if d0["canonical_run"]["conclusion"] != config["stage_d_operator_freeze"][
        "required_conclusion"
    ]:
        raise ProtocolError("E0b D0 operator conclusion drift")
    if rejection["canonical_run"]["conclusion"] != config["stage_d_rejection_freeze"][
        "required_conclusion"
    ]:
        raise ProtocolError("E0b D0 rejection conclusion drift")

    for name, spec in config["frozen_implementation"].items():
        path = PROJECT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"E0b implementation identity drift: {name}")

    selection = config["level_selection"]
    expected_rank = ["fine_q50", "medium_q75", "coarse_q90"]
    if (
        selection.get("eligible_levels") != expected_rank
        or selection.get("rank_order") != expected_rank
        or selection.get("selected_level") != "fine_q50"
        or float(selection.get("selected_quantile", -1.0)) != 0.5
        or selection.get("policy") != "first_eligible_level_minimum_intervention"
    ):
        raise ProtocolError("E0b H-independent level selection drift")
    for flag in (
        "quality_metric_consumed",
        "density_gain_used_for_ranking",
        "seed_conflict_used_for_ranking",
        "parameter_search",
    ):
        if selection.get(flag) is not False:
            raise ProtocolError(f"E0b forbidden selection signal drift: {flag}")
    for scene in SCENES:
        passing = e0a["gate"]["passing_levels_per_scene"][scene]
        if passing != expected_rank:
            raise ProtocolError(f"E0b E0a eligible-level drift: {scene}")
    if next(level for level in expected_rank if all(
        level in e0a["gate"]["passing_levels_per_scene"][scene] for scene in SCENES
    )) != selection["selected_level"]:
        raise ProtocolError("E0b selected level is not the first globally eligible level")

    method = config["method"]
    expected_method = {
        "arm": "E0B",
        "invariant_baseline": "U2_B3",
        "mechanism_comparator": "D0_raw_gaussian_progressive",
        "node": "e0a_fine_q50_world_origin_voxel",
        "node_unary": "unweighted_member_arithmetic_mean_of_u2_b3_posterior",
        "node_view_probability": "member_visibility_weighted_mean_of_valid_sam_probability",
        "node_view_visibility": "maximum_valid_member_visibility",
        "topology": "quotient_of_frozen_v5_directed_knn_drop_self_deduplicate_then_symmetrize",
        "maximum_logical_distance": 2,
        "semantic_distribution": "l2_normalized_binary_vector_one_minus_p_and_p",
        "pair_similarity": "cosine",
        "view_confidence": "common_valid_visibility_product",
        "region_affinity": "member_count_and_logical_distance_weighted_mean",
        "logical_distance_decay": 0.5,
        "progressive_thresholds": [0.9, 0.8, 0.7, 0.6, 0.5],
        "actor_seed_minimum": 0.9,
        "background_seed_maximum": 0.1,
        "unknown_probability": 0.5,
        "parameter_search": False,
    }
    for name, expected in expected_method.items():
        if method.get(name) != expected:
            raise ProtocolError(f"E0b method drift: {name}")

    observation = config["observation_contract"]
    if (
        observation.get("frames") != [0, 40, 80, 120, 160]
        or observation.get("cameras") != [0, 1, 2]
        or int(observation.get("expected_view_count_per_scene", -1)) != 15
        or observation.get("use_evaluation_views") is not False
    ):
        raise ProtocolError("E0b observation contract drift")
    if tuple(config["execution"]["scenes"]) != SCENES:
        raise ProtocolError("E0b scene order drift")
    if config["execution"]["maximum_rounds_per_threshold"] is not None:
        raise ProtocolError("E0b must grow to a fixed point")

    scene_specs = config["scenes"]
    if tuple(scene["scene"] for scene in scene_specs) != SCENES:
        raise ProtocolError("E0b scene binding drift")
    e0a_scene_map = {scene["scene"]: scene for scene in e0a["scenes"]}
    d0_scene_map = {scene["scene"]: scene for scene in d0["scenes"]}
    for scene in scene_specs:
        name = scene["scene"]
        if int(scene["gaussian_count"]) != int(e0a_scene_map[name]["gaussian_count"]):
            raise ProtocolError(f"E0b Gaussian count drift: {name}")
        if int(scene["node_count"]) != int(e0a_scene_map[name]["levels"]["fine_q50"]["node_count"]):
            raise ProtocolError(f"E0b node count drift: {name}")
        if scene["node_sidecar"]["sha256"] != e0a_scene_map[name]["levels"][
            "fine_q50"
        ]["sidecar_sha256"]:
            raise ProtocolError(f"E0b node sidecar binding drift: {name}")
        if scene["d0_sidecar"]["sha256"] != d0_scene_map[name]["sidecar_sha256"]:
            raise ProtocolError(f"E0b D0 sidecar binding drift: {name}")
        for artifact in ("node_sidecar", "d0_sidecar"):
            path = Path(scene[artifact]["path"])
            if not path.is_file() or sha256_file(path) != scene[artifact]["sha256"]:
                raise ProtocolError(f"E0b scene artifact identity drift: {name}/{artifact}")

    future_gate = config["future_h_gate_preregistration_boundary"]
    if (
        future_gate.get("primary_comparator") != "U2_B3_G0"
        or future_gate.get("mechanism_comparator") != "D0"
        or future_gate.get("candidate") != "E0B"
        or future_gate.get("quality_read_in_this_run") is not False
    ):
        raise ProtocolError("E0b future H boundary drift")
    mechanism_gate = future_gate["additional_mechanism_gate"]
    if mechanism_gate != {
        "minimum_nonnegative_boundary_f1_scenes_vs_d0": 2,
        "minimum_scene_balanced_boundary_f1_delta_vs_d0_exclusive": 0.0,
        "minimum_scene_balanced_iou_delta_vs_d0": 0.0,
        "maximum_scene_balanced_false_negative_semantic_mass_delta_vs_d0": 0.0,
    }:
        raise ProtocolError("E0b future mechanism gate drift")

    runtime = config["runtime"]
    observed_runtime = {
        "python": sys.executable,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if observed_runtime != {name: runtime[name] for name in observed_runtime}:
        raise ProtocolError(f"E0b runtime drift: {observed_runtime}")
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
        "e1_panogs_execution",
        "e2_ag2aussian_execution",
    ):
        if config["locks"].get(name) is not False:
            raise ProtocolError(f"E0b lock drift: {name}")
    if config["locks"].get("m2_status") != "pending" or config["locks"].get(
        "m3_status"
    ) != "pending":
        raise ProtocolError("E0b M2/M3 status drift")
    return config, e0a, d0


def _view_names(config: dict[str, Any]) -> list[str]:
    observation = config["observation_contract"]
    return [
        f"f{int(frame):03d}_c{int(camera)}.npz"
        for frame in observation["frames"]
        for camera in observation["cameras"]
    ]


def _historical_inputs(e0a: dict[str, Any]) -> dict[str, dict[str, Any]]:
    e0a_config_path = PROJECT / e0a["source"]["config"]["path"]
    if sha256_file(e0a_config_path) != e0a["source"]["config"]["sha256"]:
        raise ProtocolError("E0b E0a source config drift")
    e0a_config = load_yaml(e0a_config_path)
    prereg_spec = e0a_config["frozen_stage_d_inputs"]["preregistration"]
    prereg_path = PROJECT / prereg_spec["path"]
    if not prereg_path.is_file() or sha256_file(prereg_path) != prereg_spec["sha256"]:
        raise ProtocolError("E0b Stage D preregistration drift")
    prereg = load_yaml(prereg_path)
    return {scene["scene"]: scene for scene in prereg["historical_inputs"]}


def _repeatability_probe(
    node_unary: np.ndarray,
    quotient_source: np.ndarray,
    quotient_target: np.ndarray,
    node_probability: np.ndarray,
    node_visibility: np.ndarray,
    config: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    prefix_count = min(
        int(config["execution"]["repeatability_probe"]["node_prefix_count"]),
        int(node_unary.size),
    )
    retained = (quotient_source < prefix_count) & (quotient_target < prefix_count)
    if not np.any(retained):
        raise ProtocolError("E0b repeatability prefix has no quotient edges")
    paths = []
    payloads = []
    for repeat in range(2):
        topologies = build_exact_logical_topologies(
            quotient_source[retained],
            quotient_target[retained],
            node_count=prefix_count,
            maximum_logical_distance=int(config["method"]["maximum_logical_distance"]),
        )
        affinities, _ = affinity_matrices_from_topologies(
            topologies,
            node_probability[:prefix_count],
            node_visibility[:prefix_count],
            chunk_size=int(config["execution"]["affinity_chunk_size"]),
        )
        result = progressive_region_growing(
            node_unary[:prefix_count],
            topologies=topologies,
            affinities=affinities,
            progressive_thresholds=config["method"]["progressive_thresholds"],
            logical_distance_decay=float(config["method"]["logical_distance_decay"]),
            actor_seed_minimum=float(config["method"]["actor_seed_minimum"]),
            background_seed_maximum=float(config["method"]["background_seed_maximum"]),
            unknown_probability=float(config["method"]["unknown_probability"]),
        )
        payload = {
            "assignment_level": result["assignment_level"],
            "e0b_label": result["labels"],
            "e0b_posterior": result["posterior"],
            "node_id": np.arange(prefix_count, dtype=np.int64),
        }
        path = run_dir / f"artifacts/repeatability/node_prefix_{repeat}.npz"
        atomic_save_npz(path, payload)
        paths.append(path)
        payloads.append(payload)
    for field in payloads[0]:
        if not np.array_equal(payloads[0][field], payloads[1][field]):
            raise ProtocolError(f"E0b repeatability array drift: {field}")
    if paths[0].read_bytes() != paths[1].read_bytes():
        raise ProtocolError("E0b repeatability NPZ is not byte exact")
    return {
        "node_prefix_count": prefix_count,
        "quotient_directed_edge_count": int(np.count_nonzero(retained)),
        "byte_exact": True,
        "sha256": sha256_file(paths[0]),
    }


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    config, e0a, _ = validate_config(config_path)
    if run_dir.exists():
        raise ProtocolError(f"E0b run directory exists: {run_dir}")
    if _git("branch", "--show-current") != V51_BRANCH:
        raise ProtocolError("E0b must run on the V5.1 branch")
    if _git("status", "--porcelain"):
        raise ProtocolError("E0b requires a clean worktree")

    run_dir.mkdir(parents=True)
    source_commit = _git("rev-parse", "HEAD")
    source_tree = _git("rev-parse", "HEAD^{tree}")
    _write_text(run_dir / "resolved_config.yaml", config_path.read_text(encoding="utf-8"))
    events = [{"event": "run_started", "at_utc": _utc_now(), "source_commit": source_commit}]
    _write_jsonl(run_dir / "events.jsonl", events)
    _write_json(run_dir / "status.json", {"status": "running", "task_id": TASK_ID})

    monitor = ResourceMonitor(config["resources"]["monitor_interval_seconds"])
    memory_events_before = Path("/sys/fs/cgroup/memory.events").read_text(encoding="utf-8")
    started = time.perf_counter()
    monitor.start()
    try:
        historical = _historical_inputs(e0a)
        scene_config = {scene["scene"]: scene for scene in config["scenes"]}
        view_names = _view_names(config)
        reports = []
        repeatability = None
        for scene_name in SCENES:
            scene_started = time.perf_counter()
            spec = historical[scene_name]
            binding = scene_config[scene_name]
            b3_path = Path(spec["b3_unary"]["path"])
            if sha256_file(b3_path) != spec["b3_unary"]["sha256"]:
                raise ProtocolError(f"E0b B3 identity drift: {scene_name}")
            with np.load(b3_path, allow_pickle=False) as b3:
                gaussian_id = np.asarray(b3["gaussian_id"], dtype=np.int64).copy()
                unary = np.asarray(b3["unary_posterior"], dtype=np.float32).copy()
            gaussian_count = int(binding["gaussian_count"])
            if not np.array_equal(gaussian_id, np.arange(gaussian_count, dtype=np.int64)):
                raise ProtocolError(f"E0b Gaussian denominator drift: {scene_name}")

            node_path = Path(binding["node_sidecar"]["path"])
            with np.load(node_path, allow_pickle=False) as table:
                if not np.array_equal(table["gaussian_id"], gaussian_id):
                    raise ProtocolError(f"E0b node sidecar Gaussian drift: {scene_name}")
                node_id = np.asarray(table["node_id"], dtype=np.int64).copy()
                node_count = int(table["node_count"])
            if node_count != int(binding["node_count"]):
                raise ProtocolError(f"E0b node denominator drift: {scene_name}")

            d0_path = Path(binding["d0_sidecar"]["path"])
            with np.load(d0_path, allow_pickle=False) as table:
                if not np.array_equal(table["gaussian_id"], gaussian_id):
                    raise ProtocolError(f"E0b D0 Gaussian drift: {scene_name}")
                d0_posterior = np.asarray(table["d0_posterior"], dtype=np.float32).copy()

            graph_config_path = PROJECT / spec["v5_graph_config"]["path"]
            if sha256_file(graph_config_path) != spec["v5_graph_config"]["sha256"]:
                raise ProtocolError(f"E0b graph config identity drift: {scene_name}")
            graph_config = load_yaml(graph_config_path)
            unary_manifest = graph_config["inputs"]["unary_manifest"]
            manifest_path = Path(unary_manifest["path"])
            probability, visibility, observation_reports = load_observation_matrices(
                unary_run=manifest_path.parent,
                unary_manifest_path=manifest_path,
                unary_manifest_sha256=unary_manifest["sha256"],
                gaussian_count=gaussian_count,
                view_names=view_names,
            )

            edge_path = Path(spec["v5_graph_run"]["path"]) / "artifacts/graph/edges.npz"
            if sha256_file(edge_path) != spec["v5_graph_run"]["files"][
                "artifacts/graph/edges.npz"
            ]:
                raise ProtocolError(f"E0b raw edge identity drift: {scene_name}")
            with np.load(edge_path, allow_pickle=False) as edges:
                raw_source = np.asarray(edges["source_gaussian_id"], dtype=np.int64).copy()
                raw_target = np.asarray(edges["target_gaussian_id"], dtype=np.int64).copy()

            aggregation_started = time.perf_counter()
            node_unary, node_probability, node_visibility, aggregation_report = (
                aggregate_node_evidence(unary, probability, visibility, node_id)
            )
            quotient_source, quotient_target, quotient_report = quotient_directed_edges(
                raw_source, raw_target, node_id
            )
            aggregation_seconds = time.perf_counter() - aggregation_started

            topology_started = time.perf_counter()
            topologies = build_exact_logical_topologies(
                quotient_source,
                quotient_target,
                node_count=node_count,
                maximum_logical_distance=int(config["method"]["maximum_logical_distance"]),
            )
            topology_seconds = time.perf_counter() - topology_started
            affinity_started = time.perf_counter()
            affinities, affinity_reports = affinity_matrices_from_topologies(
                topologies,
                node_probability,
                node_visibility,
                chunk_size=int(config["execution"]["affinity_chunk_size"]),
            )
            affinity_seconds = time.perf_counter() - affinity_started

            if repeatability is None and config["execution"]["repeatability_probe"]["enabled"]:
                repeatability = _repeatability_probe(
                    node_unary,
                    quotient_source,
                    quotient_target,
                    node_probability,
                    node_visibility,
                    config,
                    run_dir,
                )

            propagation_started = time.perf_counter()
            result = progressive_region_growing(
                node_unary,
                topologies=topologies,
                affinities=affinities,
                progressive_thresholds=config["method"]["progressive_thresholds"],
                logical_distance_decay=float(config["method"]["logical_distance_decay"]),
                actor_seed_minimum=float(config["method"]["actor_seed_minimum"]),
                background_seed_maximum=float(config["method"]["background_seed_maximum"]),
                unknown_probability=float(config["method"]["unknown_probability"]),
                maximum_rounds_per_threshold=config["execution"][
                    "maximum_rounds_per_threshold"
                ],
            )
            e0b_label, e0b_posterior, assignment_level = broadcast_node_result(
                node_id,
                result["labels"],
                result["posterior"],
                result["assignment_level"],
            )
            propagation_seconds = time.perf_counter() - propagation_started
            output_path = run_dir / f"artifacts/sidecars/{scene_name}.npz"
            atomic_save_npz(
                output_path,
                {
                    "assignment_level": assignment_level,
                    "d0_raw_posterior": d0_posterior,
                    "e0b_label": e0b_label,
                    "e0b_posterior": e0b_posterior,
                    "gaussian_id": gaussian_id,
                    "node_id": node_id,
                    "u2_b3_posterior": unary,
                },
            )
            report = {
                "scene": scene_name,
                "gaussian_count": gaussian_count,
                "node_count": node_count,
                "selected_level": "fine_q50",
                "observation_views": observation_reports,
                "aggregation": aggregation_report,
                "quotient": quotient_report,
                "logical_neighbor_counts": [int(matrix.nnz) for matrix in topologies],
                "affinity_reports": affinity_reports,
                "propagation": result["report"],
                "gaussian_readout": {
                    "changed_posterior_vs_d0_gaussian_count": int(
                        np.count_nonzero(e0b_posterior != d0_posterior)
                    ),
                    "changed_posterior_vs_d0_gaussian_fraction": float(
                        np.mean(e0b_posterior != d0_posterior, dtype=np.float64)
                    ),
                    "node_constant": True,
                },
                "seconds": {
                    "aggregation_and_quotient": aggregation_seconds,
                    "topology": topology_seconds,
                    "affinity": affinity_seconds,
                    "propagation_and_readout": propagation_seconds,
                    "total": time.perf_counter() - scene_started,
                },
                "sidecar": {
                    "path": output_path.relative_to(run_dir).as_posix(),
                    "bytes": output_path.stat().st_size,
                    "sha256": sha256_file(output_path),
                },
                "quality_read": False,
            }
            if report["seconds"]["total"] > float(
                config["resources"]["maximum_wall_seconds_per_scene"]
            ):
                raise ProtocolError(f"E0b scene wall limit exceeded: {scene_name}")
            _write_json(run_dir / f"artifacts/reports/{scene_name}.json", report)
            reports.append(report)
            events.append({"event": "scene_completed", "at_utc": _utc_now(), "scene": scene_name})
            _write_jsonl(run_dir / "events.jsonl", events)
            del probability, visibility, raw_source, raw_target, quotient_source, quotient_target
            del node_probability, node_visibility, node_unary, topologies, affinities, result
            del e0b_label, e0b_posterior, assignment_level, unary, d0_posterior, node_id
            gc.collect()

        monitor.stop()
        _write_jsonl(run_dir / "artifacts/resource_samples.jsonl", monitor.samples)
        valid_samples = [row for row in monitor.samples if "monitor_error" not in row]
        if not valid_samples:
            raise ProtocolError("E0b resource monitor produced no valid sample")
        resources = {
            "sample_count": len(monitor.samples),
            "monitor_error_count": len(monitor.samples) - len(valid_samples),
            "peak_cgroup_memory_bytes": max(
                int(row["cgroup_memory_current_bytes"]) for row in valid_samples
            ),
            "peak_gpu_used_mib": max(int(row["gpu_used_mib"]) for row in valid_samples),
            "memory_events_before": memory_events_before,
            "memory_events_after": Path("/sys/fs/cgroup/memory.events").read_text(
                encoding="utf-8"
            ),
            "total_wall_seconds": time.perf_counter() - started,
        }
        _write_json(run_dir / "artifacts/resources.json", resources)
        if resources["monitor_error_count"] != 0:
            raise ProtocolError("E0b resource monitor reported errors")
        if resources["peak_cgroup_memory_bytes"] > int(
            config["resources"]["maximum_cgroup_memory_bytes"]
        ):
            raise ProtocolError("E0b cgroup memory limit exceeded")
        if resources["peak_gpu_used_mib"] > int(
            config["resources"]["maximum_gpu_used_mib"]
        ):
            raise ProtocolError("E0b unexpected GPU usage")
        if resources["total_wall_seconds"] > float(
            config["resources"]["maximum_total_wall_seconds"]
        ):
            raise ProtocolError("E0b total wall limit exceeded")
        if resources["memory_events_before"] != resources["memory_events_after"]:
            raise ProtocolError("E0b cgroup memory events changed")

        summary = {
            "schema_version": "worldsim_v51_e0b_same_propagation_summary_v1",
            "task_id": TASK_ID,
            "status": "done",
            "conclusion": config["operator_success_conclusion"],
            "source_commit": source_commit,
            "source_tree": source_tree,
            "selected_level": "fine_q50",
            "scene_reports": reports,
            "repeatability_probe": repeatability,
            "resources": resources,
            "quality_read": False,
            "h_quality_read": False,
            "screening_quality_read": False,
            "confirmation_quality_read": False,
            "validation_quality_read": False,
            "test_quality_read": False,
            "parameter_search": False,
            "e1_panogs_execution": False,
            "e2_ag2aussian_execution": False,
            "m2_status": "pending",
            "m3_status": "pending",
        }
        _write_json(run_dir / "summary.json", summary)
        events.append({"event": "run_completed", "at_utc": _utc_now(), "status": "done"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "manifest.json",
            {
                "schema_version": "worldsim_v51_e0b_same_propagation_manifest_v1",
                "task_id": TASK_ID,
                "status": "done",
                "inventory": _inventory(run_dir),
            },
        )
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_e0b_same_propagation_status_v1",
                "task_id": TASK_ID,
                "status": "done",
                "conclusion": config["operator_success_conclusion"],
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
                "schema_version": "worldsim_v51_e0b_same_propagation_status_v1",
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
        default=PROJECT / "configs/worldsim_v51/stage_e_e0b_same_propagation_v1.yaml",
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = run(args.config.resolve(), args.run_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
