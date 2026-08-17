#!/usr/bin/env python3
"""Evaluate frozen H B0/B1 uplift with preregistered proxy and heldout metrics."""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import shutil
import signal
import sys
import time
from typing import Any

import numpy as np
import scipy
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v51.feature_evaluation import (
    actor_feature_metrics,
    evaluate_h_gate,
    reproject_feature_arms,
    repeatability_against_aggregate,
    single_view_gaussian_feature,
)
from motion_proj.worldsim_v51.feature_sidecar import array_sha256
from motion_proj.worldsim_v51.protocol import ProtocolError, V51_BRANCH, load_yaml, sha256_file
from scripts.run_worldsim_v51_h_uplift import (
    ResourceMonitor,
    _build_scene_runtime,
    _git,
    _inventory,
    _nvidia_used_mib,
    _render_intersections,
    _utc_now,
    _write_json,
    _write_jsonl,
    _write_text,
)


def _load_bound_manifest(spec: dict[str, Any], label: str) -> dict[str, Any]:
    path = Path(spec["run_path"]) / spec["manifest_path"]
    if not path.is_file() or sha256_file(path) != spec["manifest_sha256"]:
        raise ProtocolError(f"H evaluation {label} manifest identity drift")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("record_chain_sha256") != spec["record_chain_sha256"]:
        raise ProtocolError(f"H evaluation {label} record chain drift")
    return manifest


def _load_evaluation_config(config_path: Path) -> dict[str, Any]:
    raw = load_yaml(config_path)
    if raw.get("schema_version") != "worldsim_v51_stage_b_h_evaluation_recovery_v1":
        return raw
    if raw.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01" or raw.get("status") != "running":
        raise ProtocolError("H evaluation recovery task/status drift")
    if set(raw) != {
        "schema_version",
        "task_id",
        "status",
        "base_config",
        "recovery",
        "resource_overrides",
        "failure_ledger_refs",
        "failure_ledger_delta",
    }:
        raise ProtocolError("H evaluation recovery fields drift")
    base_spec = raw["base_config"]
    base_path = PROJECT / base_spec["path"]
    if not base_path.is_file() or sha256_file(base_path) != base_spec["sha256"]:
        raise ProtocolError("H evaluation recovery base config identity drift")
    base = load_yaml(base_path)
    if base.get("schema_version") != base_spec["schema_version"]:
        raise ProtocolError("H evaluation recovery base schema drift")
    recovery = raw["recovery"]
    blocked_run = Path(recovery["blocked_run"])
    if recovery.get("blocked_status") != "blocked" or recovery.get(
        "blocked_quality_read_for_recovery"
    ) is not False:
        raise ProtocolError("H evaluation recovery blocked contract drift")
    if recovery.get("reuse_blocked_outputs") is not False:
        raise ProtocolError("H evaluation recovery must not reuse blocked outputs")
    for relative, expected in (
        ("status.json", recovery["blocked_status_sha256"]),
        ("artifacts/resources.json", recovery["blocked_resources_sha256"]),
        ("artifacts/h_evaluation_report.json", recovery["blocked_report_sha256"]),
    ):
        path = blocked_run / relative
        if not path.is_file() or sha256_file(path) != expected:
            raise ProtocolError(f"H evaluation recovery blocked identity drift: {relative}")
    if recovery.get("allowed_change") != "raise_nvidia_and_torch_resource_ceiling_only":
        raise ProtocolError("H evaluation recovery allowed change drift")
    overrides = raw["resource_overrides"]
    if overrides != {
        "maximum_nvidia_peak_mib": 24000,
        "maximum_torch_reserved_peak_mib": 24000,
    }:
        raise ProtocolError("H evaluation recovery resource override drift")
    if base["resources"]["maximum_nvidia_peak_mib"] != 22528 or base["resources"][
        "maximum_torch_reserved_peak_mib"
    ] != 22528:
        raise ProtocolError("H evaluation recovery original ceilings drift")
    merged = copy.deepcopy(base)
    merged["resources"].update(overrides)
    merged["recovery"] = copy.deepcopy(recovery)
    merged["failure_ledger_refs"] = list(raw["failure_ledger_refs"])
    merged["failure_ledger_delta"] = list(raw["failure_ledger_delta"])
    return merged


def _view_record_map(
    manifest: dict[str, Any], *, scenes: list[str], frames: list[int], cameras: list[int], label: str
) -> dict[str, dict[str, Any]]:
    records = list(manifest["records"])
    expected = {
        f"{scene}:{int(frame)}:{int(camera)}"
        for scene in scenes
        for frame in frames
        for camera in cameras
    }
    result: dict[str, dict[str, Any]] = {}
    root = None
    for record in records:
        key = f"{record['scene']}:{int(record['frame'])}:{int(record['camera'])}"
        if key in result:
            raise ProtocolError(f"H evaluation duplicate {label} view: {key}")
        result[key] = record
    if set(result) != expected:
        raise ProtocolError(f"H evaluation {label} view grid incomplete")
    return result


def validate_config(
    config_path: Path,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
]:
    config = _load_evaluation_config(config_path)
    if config.get("schema_version") != "worldsim_v51_stage_b_h_evaluation_v1":
        raise ProtocolError("H evaluation schema drift")
    if config.get("task_id") != "WS-V51-M1-B-LUDVIG-UPLIFT-01":
        raise ProtocolError("H evaluation task drift")
    if config.get("status") != "running" or int(config.get("seed", -1)) != 20260814:
        raise ProtocolError("H evaluation status/seed drift")
    for name in ("uplift_freeze", "evidence_feature_freeze", "evaluation_feature_freeze"):
        spec = config[name]
        path = PROJECT / spec["path"]
        if not path.is_file() or sha256_file(path) != spec["sha256"]:
            raise ProtocolError(f"H evaluation freeze identity drift: {name}")
        if load_yaml(path).get("status") != spec["required_status"]:
            raise ProtocolError(f"H evaluation freeze status drift: {name}")

    uplift_manifest = _load_bound_manifest(config["uplift_freeze"], "uplift")
    evidence_manifest = _load_bound_manifest(config["evidence_feature_freeze"], "evidence")
    evaluation_manifest = _load_bound_manifest(config["evaluation_feature_freeze"], "evaluation")
    if config["evidence_feature_freeze"]["pca_state_sha256"] != config[
        "evaluation_feature_freeze"
    ]["pca_state_sha256"]:
        raise ProtocolError("H evaluation PCA identity differs across splits")

    scene_names = [scene["name"] for scene in config["scenes"]]
    if scene_names != ["scene-0471", "scene-1087", "scene-0379"]:
        raise ProtocolError("H evaluation scene order drift")
    view = config["view_contract"]
    expected_view = {
        "evidence_frames": [0, 40, 80, 120, 160],
        "evaluation_frames": [2, 42, 82, 122, 162],
        "final_heldout_remainder": 4,
        "cameras": [0, 1, 2],
        "views_per_split_per_scene": 15,
        "evidence_view_count": 45,
        "evaluation_view_count": 45,
        "image_index_formula": "frame_times_3_plus_camera",
        "model_native_renderer_size_wh": [800, 450],
        "patch_grid_shape": [40, 64, 114],
        "feature_dimension": 40,
        "reference_frame": 80,
        "reference_camera": 1,
    }
    for name, expected in expected_view.items():
        if view.get(name) != expected:
            raise ProtocolError(f"H evaluation view contract drift: {name}")
    if any(int(frame) % 5 != 2 for frame in view["evaluation_frames"]):
        raise ProtocolError("H evaluation remainder drift")
    if any(int(frame) % 5 == 4 for frame in view["evidence_frames"] + view["evaluation_frames"]):
        raise ProtocolError("H evaluation touched final heldout remainder")
    evidence_by_view = _view_record_map(
        evidence_manifest,
        scenes=scene_names,
        frames=view["evidence_frames"],
        cameras=view["cameras"],
        label="evidence",
    )
    evaluation_by_view = _view_record_map(
        evaluation_manifest,
        scenes=scene_names,
        frames=view["evaluation_frames"],
        cameras=view["cameras"],
        label="evaluation",
    )
    for label, record_map, spec in (
        ("evidence", evidence_by_view, config["evidence_feature_freeze"]),
        ("evaluation", evaluation_by_view, config["evaluation_feature_freeze"]),
    ):
        for key, record in record_map.items():
            path = Path(spec["run_path"]) / record["path"]
            if not path.is_file() or sha256_file(path) != record["file_sha256"]:
                raise ProtocolError(f"H evaluation {label} sidecar identity drift: {key}")
            if record["pca_state_sha256"] != spec["pca_state_sha256"]:
                raise ProtocolError(f"H evaluation {label} PCA sidecar drift: {key}")

    uplift_by_arm: dict[str, dict[str, Any]] = {}
    if int(uplift_manifest.get("record_count", -1)) != 6:
        raise ProtocolError("H evaluation uplift arm denominator drift")
    for record in uplift_manifest["records"]:
        key = f"{record['scene']}:{record['arm']}"
        if key in uplift_by_arm:
            raise ProtocolError(f"H evaluation duplicate uplift arm: {key}")
        path = Path(config["uplift_freeze"]["run_path"]) / record["path"]
        if not path.is_file() or path.stat().st_size != int(record["bytes"]):
            raise ProtocolError(f"H evaluation uplift sidecar missing: {key}")
        if sha256_file(path) != record["file_sha256"]:
            raise ProtocolError(f"H evaluation uplift sidecar SHA drift: {key}")
        uplift_by_arm[key] = record
    if set(uplift_by_arm) != {f"{scene}:{arm}" for scene in scene_names for arm in ("B0", "B1")}:
        raise ProtocolError("H evaluation uplift scene/arm grid incomplete")

    for scene in config["scenes"]:
        for name in ("formal_summary", "formal_checkpoint", "source_config"):
            spec = scene[name]
            path = Path(spec["path"])
            if not path.is_file() or sha256_file(path) != spec["sha256"]:
                raise ProtocolError(f"H evaluation scene input drift: {scene['name']}/{name}")
            if name == "formal_checkpoint" and path.stat().st_size != int(spec["bytes"]):
                raise ProtocolError(f"H evaluation checkpoint bytes drift: {scene['name']}")
        if int(scene["background_gaussians"]) + int(scene["rigid_gaussians"]) != int(
            scene["total_gaussians"]
        ):
            raise ProtocolError(f"H evaluation Gaussian layout drift: {scene['name']}")

    operator = config["operator"]
    expected_operator = {
        "contribution_source": "motion_proj.worldsim_v5.renderer_intersections",
        "formula": "alpha_times_transmittance_before_alpha",
        "minimum_intersection_contribution": 1e-4,
        "minimum_gaussian_view_mass": 1e-3,
        "minimum_reprojection_pixel_mass": 1e-3,
        "epsilon": 1e-8,
        "sparse_backend": "scipy_csr_float64",
        "pixel_feature_sampling": "lazy_bilinear_align_corners_false",
        "optional_pruning": False,
    }
    for name, expected in expected_operator.items():
        if operator.get(name) != expected:
            raise ProtocolError(f"H evaluation operator drift: {name}")
    evaluation = config["evaluation"]
    expected_evaluation = {
        "membership_declaration": "model_membership_proxy_not_ground_truth",
        "membership_usage": "evaluation_only_never_method_pca_or_uplift_input",
        "actor_source": "RigidNodes.point_ids",
        "actor_active_source": "RigidNodes.instances_fv_at_reference_frame",
        "background_source": "Background_model_rows",
        "geometry_reference": "model_world_means_at_frame80_camera1",
        "background_match": "euclidean_nearest_covered_background_ckdtree_workers1",
        "minimum_actor_covered_gaussians": 32,
        "maximum_pairs_per_actor": 4096,
        "pair_sampler": "enumerate_all_if_at_most_cap_else_pcg64_unique_unordered_rejection",
        "pair_seed_derivation": "sha256_first8_big_endian_mask63_of_base_scene_actor",
        "actor_weighting_within_scene": "equal_actor",
        "view_weighting_within_scene": "equal_view",
        "scene_weighting": "equal_scene",
        "cosine_epsilon": 1e-8,
        "no_eligible_actor_policy": "scene_abstain",
        "exact_common_b0_b1_reprojection_pixels": True,
        "minimum_valid_reprojection_views_per_scene": 15,
    }
    for name, expected in expected_evaluation.items():
        if evaluation.get(name) != expected:
            raise ProtocolError(f"H evaluation metric contract drift: {name}")
    gate = config["h_gate"]
    expected_gate = {
        "scene_count": 3,
        "minimum_evaluable_scenes": 2,
        "minimum_positive_b1_margin_scenes": 2,
        "minimum_scene_balanced_b1_margin_exclusive": 0.0,
        "minimum_scene_balanced_rigid_coverage": 0.60,
        "minimum_scene_balanced_heldout_b1_minus_b0": -0.01,
        "pass_unlocks": "screening_exact_once_only",
        "fail_action": "reject_ludvig_uplift_and_raw_graph_then_advance_frozen_route",
    }
    for name, expected in expected_gate.items():
        if gate.get(name) != expected:
            raise ProtocolError(f"H evaluation gate drift: {name}")

    runtime = config["runtime"]
    observed_runtime = {
        "python": sys.executable,
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }
    if observed_runtime != {name: runtime[name] for name in observed_runtime}:
        raise ProtocolError(f"H evaluation runtime drift: {observed_runtime}")
    upstream = Path(runtime["drivestudio_checkout"])
    if _git(upstream, "rev-parse", "HEAD") != runtime["drivestudio_commit"]:
        raise ProtocolError("H evaluation DriveStudio commit drift")
    if _git(upstream, "status", "--short") != runtime["drivestudio_expected_status"]:
        raise ProtocolError("H evaluation DriveStudio patch status drift")
    patch = Path(runtime["compatibility_patch"]["path"])
    if not patch.is_file() or sha256_file(patch) != runtime["compatibility_patch"]["sha256"]:
        raise ProtocolError("H evaluation compatibility patch drift")

    locks = config["locks"]
    for name in (
        "h_evidence_feature_read",
        "h_evaluation_feature_read",
        "h_uplift_feature_read",
        "h_renderer_start",
        "membership_proxy_read_evaluation_only",
        "method_quality_read_h_only",
    ):
        if locks.get(name) is not True:
            raise ProtocolError(f"H evaluation authorization drift: {name}")
    for name in (
        "proxy_as_method_input",
        "pca_fit",
        "uplift_recompute",
        "screening_pixels_read",
        "screening_quality_read",
        "confirmation_pixels_read",
        "confirmation_quality_read",
        "final_heldout_pixels_read",
        "final_heldout_quality_read",
        "validation_quality_read",
        "test_quality_read",
        "kitti_method_tuning",
    ):
        if locks.get(name) is not False:
            raise ProtocolError(f"H evaluation lock drift: {name}")
    if locks.get("m2_status") != "pending" or locks.get("m3_status") != "pending":
        raise ProtocolError("H evaluation M2/M3 must remain pending")
    return config, evidence_by_view, evaluation_by_view, uplift_by_arm


def _load_patch(record: dict[str, Any], root: Path) -> np.ndarray:
    path = root / record["path"]
    if sha256_file(path) != record["file_sha256"]:
        raise ProtocolError(f"H evaluation feature file drift: {path}")
    with np.load(path, allow_pickle=False) as archive:
        if archive.files != ["feature"]:
            raise ProtocolError(f"H evaluation feature fields drift: {path}")
        feature = np.asarray(archive["feature"], dtype=np.float32)
    if list(feature.shape) != record["shape"] or str(feature.dtype) != record["dtype"]:
        raise ProtocolError(f"H evaluation feature shape/dtype drift: {path}")
    if not np.isfinite(feature).all() or array_sha256(feature) != record["content_sha256"]:
        raise ProtocolError(f"H evaluation feature content drift: {path}")
    return feature


def _load_gaussian_arms(
    config: dict[str, Any], scene: dict[str, Any], records: dict[str, dict[str, Any]]
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], np.ndarray]:
    root = Path(config["uplift_freeze"]["run_path"])
    features: dict[str, np.ndarray] = {}
    weights: dict[str, np.ndarray] = {}
    supported_counts: dict[str, np.ndarray] = {}
    for arm in ("B0", "B1"):
        record = records[f"{scene['name']}:{arm}"]
        path = root / record["path"]
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"feature", "weight", "supported_view_count"}:
                raise ProtocolError(f"H evaluation uplift fields drift: {scene['name']}/{arm}")
            feature = np.asarray(archive["feature"])
            weight = np.asarray(archive["weight"])
            supported = np.asarray(archive["supported_view_count"])
        expected_shape = (int(scene["total_gaussians"]), 40)
        if feature.shape != expected_shape or feature.dtype != np.float32:
            raise ProtocolError(f"H evaluation uplift feature drift: {scene['name']}/{arm}")
        if weight.shape != (expected_shape[0],) or weight.dtype != np.float64:
            raise ProtocolError(f"H evaluation uplift weight drift: {scene['name']}/{arm}")
        if supported.shape != (expected_shape[0],) or supported.dtype != np.int32:
            raise ProtocolError(f"H evaluation uplift support drift: {scene['name']}/{arm}")
        if not np.isfinite(feature).all() or not np.isfinite(weight).all() or np.any(weight < 0.0):
            raise ProtocolError(f"H evaluation uplift non-finite: {scene['name']}/{arm}")
        if array_sha256(feature) != record["feature_content_sha256"]:
            raise ProtocolError(f"H evaluation uplift feature content drift: {scene['name']}/{arm}")
        if array_sha256(weight) != record["weight_content_sha256"]:
            raise ProtocolError(f"H evaluation uplift weight content drift: {scene['name']}/{arm}")
        if array_sha256(supported) != record["supported_view_count_content_sha256"]:
            raise ProtocolError(f"H evaluation uplift support content drift: {scene['name']}/{arm}")
        features[arm] = feature
        weights[arm] = weight
        supported_counts[arm] = supported
    if not np.array_equal(weights["B0"] > 0.0, weights["B1"] > 0.0):
        raise ProtocolError(f"H evaluation B0/B1 coverage differs: {scene['name']}")
    if not np.array_equal(supported_counts["B0"], supported_counts["B1"]):
        raise ProtocolError(f"H evaluation B0/B1 support count differs: {scene['name']}")
    return features, weights, supported_counts["B0"]


def _reference_geometry(trainer: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    background = trainer.models["Background"]
    rigid = trainer.models["RigidNodes"]
    background_world = background._means.detach().cpu().numpy().astype(np.float64, copy=True)
    with torch.inference_mode():
        rigid_world = rigid.transform_means(rigid._means).detach().cpu().numpy().astype(np.float64, copy=True)
    actor_id = rigid.point_ids[..., 0].detach().cpu().numpy().astype(np.int64, copy=True)
    active = rigid.instances_fv[rigid.cur_frame].detach().cpu().numpy().astype(bool, copy=True)
    return background_world, rigid_world, actor_id, active


def execute(
    config: dict[str, Any],
    evidence_by_view: dict[str, dict[str, Any]],
    evaluation_by_view: dict[str, dict[str, Any]],
    uplift_by_arm: dict[str, dict[str, Any]],
    run_dir: Path,
) -> dict[str, Any]:
    view = config["view_contract"]
    operator = config["operator"]
    evaluation = config["evaluation"]
    evidence_root = Path(config["evidence_feature_freeze"]["run_path"])
    evaluation_root = Path(config["evaluation_feature_freeze"]["run_path"])
    scene_reports: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    completed_views = 0
    for scene in config["scenes"]:
        scene_started = time.monotonic()
        checkpoint_path = Path(scene["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint_path)
        dataset, trainer = _build_scene_runtime(config, scene)
        features, weights, supported_view_count = _load_gaussian_arms(config, scene, uplift_by_arm)
        covered_by_arm = {arm: weights[arm] > 0.0 for arm in ("B0", "B1")}
        common_covered = covered_by_arm["B0"] & covered_by_arm["B1"]

        evidence_reports: list[dict[str, Any]] = []
        reference_geometry = None
        for frame in view["evidence_frames"]:
            for camera in view["cameras"]:
                key = f"{scene['name']}:{int(frame)}:{int(camera)}"
                patch_grid = _load_patch(evidence_by_view[key], evidence_root)
                gids, pixels, contribution, render = _render_intersections(
                    config, dataset, trainer, frame=int(frame), camera=int(camera)
                )
                if int(frame) == int(view["reference_frame"]) and int(camera) == int(
                    view["reference_camera"]
                ):
                    reference_geometry = _reference_geometry(trainer)
                single = single_view_gaussian_feature(
                    gaussian_id=gids,
                    pixel_id=pixels,
                    contribution_weight=contribution,
                    patch_grid=patch_grid,
                    gaussian_count=int(scene["total_gaussians"]),
                    image_height=int(render["height"]),
                    image_width=int(render["width"]),
                    minimum_intersection_contribution=float(
                        operator["minimum_intersection_contribution"]
                    ),
                    minimum_gaussian_view_mass=float(operator["minimum_gaussian_view_mass"]),
                    epsilon=float(operator["epsilon"]),
                )
                arms = {
                    arm: repeatability_against_aggregate(
                        aggregate_feature=features[arm],
                        aggregate_covered=covered_by_arm[arm],
                        view_gaussian_id=single["gaussian_id"],
                        view_feature=single["feature"],
                        epsilon=float(evaluation["cosine_epsilon"]),
                    )
                    for arm in ("B0", "B1")
                }
                evidence_reports.append(
                    {
                        "frame": int(frame),
                        "camera": int(camera),
                        "single_view_covered_gaussian_count": single["covered_gaussian_count"],
                        "arms": arms,
                    }
                )
                completed_views += 1
                _write_json(
                    run_dir / "artifacts/evaluation_progress.json",
                    {
                        "schema_version": "worldsim_v51_h_evaluation_progress_v1",
                        "completed_view_count": completed_views,
                        "expected_view_count": 90,
                        "last_view": {"split": "evidence", "scene": scene["name"], "frame": frame, "camera": camera},
                    },
                )
                del gids, pixels, contribution, patch_grid, single
                gc.collect()
        if reference_geometry is None:
            raise ProtocolError(f"H evaluation reference geometry missing: {scene['name']}")
        repeatability = {}
        for arm in ("B0", "B1"):
            values = [row["arms"][arm]["mean_cosine"] for row in evidence_reports]
            if any(value is None for value in values):
                raise ProtocolError(f"H evaluation repeatability denominator empty: {scene['name']}/{arm}")
            repeatability[arm] = {
                "valid_view_count": len(values),
                "scene_mean_cosine": float(np.mean(values, dtype=np.float64)),
            }

        background_world, rigid_world, actor_id, active_actor = reference_geometry
        actor_metrics = {
            arm: actor_feature_metrics(
                feature=features[arm],
                covered=covered_by_arm[arm],
                background_count=int(scene["background_gaussians"]),
                rigid_actor_id=actor_id,
                active_actor=active_actor,
                background_world_position=background_world,
                rigid_world_position=rigid_world,
                scene=scene["name"],
                seed=int(config["seed"]),
                minimum_actor_gaussians=int(evaluation["minimum_actor_covered_gaussians"]),
                maximum_pairs_per_actor=int(evaluation["maximum_pairs_per_actor"]),
                cosine_epsilon=float(evaluation["cosine_epsilon"]),
            )
            for arm in ("B0", "B1")
        }

        heldout_reports: list[dict[str, Any]] = []
        for frame in view["evaluation_frames"]:
            for camera in view["cameras"]:
                key = f"{scene['name']}:{int(frame)}:{int(camera)}"
                patch_grid = _load_patch(evaluation_by_view[key], evaluation_root)
                gids, pixels, contribution, render = _render_intersections(
                    config, dataset, trainer, frame=int(frame), camera=int(camera)
                )
                reprojection = reproject_feature_arms(
                    features_by_arm=features,
                    common_covered=common_covered,
                    gaussian_id=gids,
                    pixel_id=pixels,
                    contribution_weight=contribution,
                    patch_grid=patch_grid,
                    image_height=int(render["height"]),
                    image_width=int(render["width"]),
                    minimum_intersection_contribution=float(
                        operator["minimum_intersection_contribution"]
                    ),
                    minimum_pixel_mass=float(operator["minimum_reprojection_pixel_mass"]),
                    cosine_epsilon=float(evaluation["cosine_epsilon"]),
                )
                heldout_reports.append({"frame": int(frame), "camera": int(camera), **reprojection})
                completed_views += 1
                _write_json(
                    run_dir / "artifacts/evaluation_progress.json",
                    {
                        "schema_version": "worldsim_v51_h_evaluation_progress_v1",
                        "completed_view_count": completed_views,
                        "expected_view_count": 90,
                        "last_view": {"split": "evaluation", "scene": scene["name"], "frame": frame, "camera": camera},
                    },
                )
                del gids, pixels, contribution, patch_grid
                gc.collect()
        valid_heldout = [row for row in heldout_reports if row["B1_minus_B0"] is not None]
        if valid_heldout:
            scene_b0 = float(np.mean([row["B0_mean_cosine"] for row in valid_heldout], dtype=np.float64))
            scene_b1 = float(np.mean([row["B1_mean_cosine"] for row in valid_heldout], dtype=np.float64))
            scene_delta = scene_b1 - scene_b0
        else:
            scene_b0 = scene_b1 = scene_delta = None
        heldout = {
            "valid_view_count": len(valid_heldout),
            "scene_B0_mean_cosine": scene_b0,
            "scene_B1_mean_cosine": scene_b1,
            "scene_B1_minus_B0": scene_delta,
            "view_reports": heldout_reports,
        }
        background_count = int(scene["background_gaussians"])
        rigid_count = int(scene["rigid_gaussians"])
        coverage = {
            "background": float(common_covered[:background_count].sum() / background_count),
            "rigid": float(common_covered[background_count:].sum() / rigid_count),
            "global": float(common_covered.mean()),
        }
        evaluable = bool(
            actor_metrics["B1"]["eligible_actor_count"] > 0
            and len(valid_heldout) >= int(evaluation["minimum_valid_reprojection_views_per_scene"])
        )
        checkpoint_after = sha256_file(checkpoint_path)
        if checkpoint_before != checkpoint_after:
            raise ProtocolError(f"H evaluation checkpoint drift: {scene['name']}")
        checkpoint_records.append(
            {"scene": scene["name"], "before": checkpoint_before, "after": checkpoint_after, "immutable": True}
        )
        scene_report = {
            "scene": scene["name"],
            "scene_index": int(scene["index"]),
            "membership_declaration": evaluation["membership_declaration"],
            "membership_usage": evaluation["membership_usage"],
            "evaluable": evaluable,
            "coverage": coverage,
            "repeatability": repeatability,
            "actor_metrics": actor_metrics,
            "heldout_reprojection": heldout,
            "supported_view_count_min": int(supported_view_count[common_covered].min()),
            "supported_view_count_max": int(supported_view_count.max()),
            "evidence_view_reports": evidence_reports,
            "seconds": time.monotonic() - scene_started,
        }
        _write_json(run_dir / f"artifacts/scene_reports/{scene['name']}.json", scene_report)
        scene_reports.append(scene_report)
        del trainer, dataset, features, weights, supported_view_count
        del background_world, rigid_world, actor_id, active_actor, reference_geometry
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    gate = evaluate_h_gate(scene_reports, config["h_gate"])
    report = {
        "schema_version": "worldsim_v51_h_evaluation_report_v1",
        "membership_declaration": config["evaluation"]["membership_declaration"],
        "membership_usage": config["evaluation"]["membership_usage"],
        "processed_scene_count": len(scene_reports),
        "processed_evidence_view_count": 45,
        "processed_evaluation_view_count": 45,
        "scene_reports": scene_reports,
        "checkpoint_records": checkpoint_records,
        "h_gate": gate,
    }
    _write_json(run_dir / "artifacts/h_evaluation_report.json", report)
    return report


def run(config_path: Path, run_dir: Path) -> dict[str, Any]:
    branch = _git(PROJECT, "branch", "--show-current")
    head = _git(PROJECT, "rev-parse", "HEAD")
    if branch != V51_BRANCH or _git(PROJECT, "status", "--short"):
        raise ProtocolError("H evaluation formal run requires a clean V5.1 worktree")
    config, evidence_by_view, evaluation_by_view, uplift_by_arm = validate_config(config_path)
    _write_text(
        run_dir / "resolved_config.yaml",
        yaml.safe_dump(
            {
                "h_evaluation": config,
                "evidence_feature_record_count": len(evidence_by_view),
                "evaluation_feature_record_count": len(evaluation_by_view),
                "uplift_arm_record_count": len(uplift_by_arm),
            },
            allow_unicode=True,
            sort_keys=False,
        ),
    )
    resources = config["resources"]
    gpu_start = _nvidia_used_mib()
    disk_available = shutil.disk_usage("/root/autodl-tmp").free
    if gpu_start > int(resources["maximum_nvidia_used_at_start_mib"]):
        raise ProtocolError("H evaluation GPU start is not idle")
    if disk_available < int(resources["minimum_disk_available_bytes"]):
        raise ProtocolError("H evaluation disk available is insufficient")
    torch.set_num_threads(int(config["runtime"]["cpu_threads"]))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor(float(resources["sample_interval_seconds"]))
    monitor.start()
    timeout_seconds = int(resources["timeout_seconds"])

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"H evaluation exceeded {timeout_seconds} s")

    previous_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    started = time.monotonic()
    try:
        report = execute(config, evidence_by_view, evaluation_by_view, uplift_by_arm, run_dir)
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)
        monitor.stop()
        _write_jsonl(run_dir / "resource_samples.jsonl", monitor.samples)
    valid = [row for row in monitor.samples if "gpu_used_mib" in row]
    if not valid:
        raise ProtocolError("H evaluation resource monitor has no valid sample")
    resource = {
        "gpu_used_at_start_mib": gpu_start,
        "nvidia_smi_peak_used_mib": max(row["gpu_used_mib"] for row in valid),
        "torch_peak_allocated_mib": torch.cuda.max_memory_allocated() / (1024**2),
        "torch_peak_reserved_mib": torch.cuda.max_memory_reserved() / (1024**2),
        "cgroup_memory_peak_bytes": max(row["cgroup_memory_current_bytes"] for row in valid),
        "disk_available_at_start_bytes": disk_available,
        "sample_count": len(valid),
        "monitor_error_count": len(monitor.samples) - len(valid),
        "duration_seconds": time.monotonic() - started,
    }
    _write_json(run_dir / "artifacts/resources.json", resource)
    if resource["nvidia_smi_peak_used_mib"] > int(resources["maximum_nvidia_peak_mib"]):
        raise ProtocolError("H evaluation NVIDIA peak exceeded")
    if resource["torch_peak_reserved_mib"] > int(resources["maximum_torch_reserved_peak_mib"]):
        raise ProtocolError("H evaluation Torch reserved peak exceeded")
    if resource["cgroup_memory_peak_bytes"] > int(resources["maximum_cgroup_peak_bytes"]):
        raise ProtocolError("H evaluation cgroup peak exceeded")
    _write_jsonl(
        run_dir / "metrics.jsonl",
        [
            {
                "metric": "scene_h_evaluation",
                "scene": row["scene"],
                "evaluable": row["evaluable"],
                "rigid_coverage": row["coverage"]["rigid"],
                "b1_actor_background_margin": row["actor_metrics"]["B1"]["scene_margin"],
                "heldout_b1_minus_b0": row["heldout_reprojection"]["scene_B1_minus_B0"],
            }
            for row in report["scene_reports"]
        ]
        + [{"metric": "h_gate", **report["h_gate"]}, {"metric": "resource_terminal", **resource}],
    )
    passed = bool(report["h_gate"]["pass"])
    summary = {
        "schema_version": "worldsim_v51_h_evaluation_summary_v1",
        "task_id": config["task_id"],
        "status": "done" if passed else "rejected",
        "conclusion": (
            "h_ludvig_uplift_passed_pre_registered_evaluation_gate"
            if passed
            else "h_ludvig_uplift_and_raw_graph_rejected_by_pre_registered_evaluation_gate"
        ),
        "source_commit": head,
        "source_branch": branch,
        "worktree_clean": True,
        "config_sha256": sha256_file(config_path),
        "report": report,
        "resource": resource,
        "membership_declaration": config["evaluation"]["membership_declaration"],
        "membership_proxy_read_evaluation_only": True,
        "proxy_as_method_input": False,
        "h_method_quality_read": True,
        "screening_quality_read": False,
        "confirmation_quality_read": False,
        "final_heldout_quality_read": False,
        "validation_quality_read": False,
        "test_quality_read": False,
        "kitti_method_tuning": False,
        "m2_status": "pending",
        "m3_status": "pending",
        "failure_ledger_refs": config["failure_ledger_refs"],
        "failure_ledger_delta": "pending_post_run_closeout" if not passed else "V51-F15_resolved",
        "created_at_utc": _utc_now(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "fingerprint.json",
        {
            "schema_version": "worldsim_v51_h_evaluation_fingerprint_v1",
            "task_id": config["task_id"],
            "source_commit": head,
            "source_branch": branch,
            "config_sha256": summary["config_sha256"],
            "uplift_freeze_sha256": config["uplift_freeze"]["sha256"],
            "evidence_feature_freeze_sha256": config["evidence_feature_freeze"]["sha256"],
            "evaluation_feature_freeze_sha256": config["evaluation_feature_freeze"]["sha256"],
            "membership_declaration": config["evaluation"]["membership_declaration"],
            "seed": int(config["seed"]),
        },
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=PROJECT / "configs/worldsim_v51/stage_b_h_evaluation_v2.yaml"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=False)
    events = [{"event": "run_started", "at_utc": _utc_now()}]
    _write_jsonl(run_dir / "events.jsonl", events)
    try:
        summary = run(args.config.resolve(), run_dir)
        events.append({"event": "run_terminal", "at_utc": _utc_now(), "status": summary["status"]})
        _write_jsonl(run_dir / "events.jsonl", events)
        manifest = {
            "schema_version": "worldsim_v51_h_evaluation_manifest_v1",
            "task_id": summary["task_id"],
            "status": summary["status"],
            "inventory": _inventory(run_dir),
        }
        _write_json(run_dir / "manifest.json", manifest)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_h_evaluation_status_v1",
                "task_id": summary["task_id"],
                "status": summary["status"],
                "source_commit": summary["source_commit"],
                "summary_sha256": sha256_file(run_dir / "summary.json"),
                "manifest_sha256": sha256_file(run_dir / "manifest.json"),
                "finished_at_utc": _utc_now(),
            },
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    except Exception as error:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
        events.append({"event": "run_blocked", "at_utc": _utc_now(), "reason": f"{type(error).__name__}: {error}"})
        _write_jsonl(run_dir / "events.jsonl", events)
        _write_json(
            run_dir / "status.json",
            {
                "schema_version": "worldsim_v51_h_evaluation_status_v1",
                "task_id": "WS-V51-M1-B-LUDVIG-UPLIFT-01",
                "status": "blocked",
                "reason": f"{type(error).__name__}: {error}",
                "finished_at_utc": _utc_now(),
            },
        )
        raise


if __name__ == "__main__":
    main()
