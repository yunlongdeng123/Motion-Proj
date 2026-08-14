#!/usr/bin/env python3
"""正式评估 reference-blind 跨视图 depth scaffold 与冻结 dense Gaussianization。"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import cv2
import numpy as np
from scipy.ndimage import binary_dilation
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v32.inpainting_adapter import completion_points_from_view
from motion_proj.worldsim_v33.spatial_delta import atomic_save_npz
from motion_proj.worldsim_v4.repair_assets import temporary_repair_composition
from motion_proj.worldsim_v4.repair_builders import completion_points_to_repair_asset
from motion_proj.worldsim_v5.cross_view_scaffold import (
    CrossViewScaffoldError,
    frozen_source_views,
    fuse_cross_view_scaffold,
    lidar_agreement_audit,
    project_background_depth_stack,
)
from motion_proj.worldsim_v5.geometry_repair import (
    GeometryRepairError,
    depth_error_summary,
    fit_inverse_depth_surface,
)
from scripts.run_worldsim_v5_m2_actor_geometry import _load_requests, _mask
from scripts.run_worldsim_v5_m2_staged_geometry_diagnostic import (
    build_all_rigid_erase_delta,
)
from scripts.run_worldsim_v5_m2_surface_ablation import _runtime
from scripts.worldsim_v5_forensics_common import (
    atomic_json,
    copy_source_snapshot,
    inventory_files,
    prepare_formal_run,
    sha256_file,
    utc_now,
    verify_file,
    write_events,
    write_resolved_config,
)


TASK_ID = "WS-V5-M2-GEOMETRY-FIRST-REPAIR-01"
SCHEMA_VERSION = "worldsim_v5_m2_cross_view_scaffold_v1"


class M2CrossViewError(RuntimeError):
    """跨视图 scaffold 的冻结输入、回放或正式运行合约失败。"""


@dataclass(frozen=True)
class PriorRun:
    root: Path
    rows: dict[tuple[int, int, int], dict[str, Any]]


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise M2CrossViewError("cross-view config schema 漂移")
    if (
        payload.get("task_id") != TASK_ID
        or payload.get("status") != "running"
        or payload.get("phase") != "g4_cross_view_depth_scaffold_development"
        or payload["request_protocol"]["unit"] != "one_actor_one_view_one_hole"
        or payload["scaffold"]["id"] != "G4_CROSS_VIEW_DEPTH_SCAFFOLD"
        or payload["gaussianization"]["stride"] != 1
        or payload["projection"]["target_depth_passed_to_projection"] != "all_nan"
    ):
        raise M2CrossViewError("cross-view task/request/scaffold 合约漂移")
    for name in (
        "validation_quality_read",
        "heldout_quality_read",
        "test_quality_read",
        "kitti_quality_read",
        "parameter_search_performed",
        "method_arm_selection_performed",
        "router_refit_performed",
    ):
        if payload["scope"].get(name) is not False:
            raise M2CrossViewError(f"cross-view restriction 漂移: {name}")
    if (
        payload["request_protocol"]["target_reference_interior_available_to_candidate"]
        is not False
        or payload["reference"]["independent_geometry_claim_allowed"] is not False
        or payload["projection"]["target_depth_consistency_filter_enabled"] is not False
        or payload["projection"]["lidar_used_to_modify_candidate"] is not False
    ):
        raise M2CrossViewError("target reference/LiDAR/claim 边界漂移")
    if 0 in payload["source_views"]["temporal_offsets"]:
        raise M2CrossViewError("source views 不得含 target offset 0")
    return payload


def candidate_decision(
    rows: list[Mapping[str, Any]], gate: Mapping[str, Any]
) -> dict[str, Any]:
    done = [row for row in rows if row.get("status") == "done"]
    raw_deltas = [
        float(row["candidate"]["raw_geometry_error"]["mae_m"])
        - float(row["baseline"]["g0_raw_geometry_error"]["mae_m"])
        for row in done
    ]
    post_deltas = [
        float(row["candidate"]["post_geometry_error"]["mae_m"])
        - float(row["baseline"]["dense_post_geometry_error"]["mae_m"])
        for row in done
    ]
    post_minus_raw = [
        float(row["candidate"]["post_geometry_error"]["mae_m"])
        - float(row["candidate"]["raw_geometry_error"]["mae_m"])
        for row in done
    ]
    raw_improved = sum(
        value <= -float(gate["minimum_raw_improvement_m"])
        for value in raw_deltas
    )
    post_improved = sum(
        value <= -float(gate["minimum_post_improvement_m"])
        for value in post_deltas
    )
    raw_safe = sum(
        float(row["candidate"]["raw_geometry_error"]["mae_m"])
        <= float(gate["geometry_safe_mae_m"])
        for row in done
    )
    post_safe = sum(
        float(row["candidate"]["post_geometry_error"]["mae_m"])
        <= float(gate["geometry_safe_mae_m"])
        for row in done
    )

    def aggregate(values: list[float]) -> dict[str, float]:
        return {
            "mean": float(np.mean(values)) if values else math.nan,
            "median": float(np.median(values)) if values else math.nan,
            "minimum": float(np.min(values)) if values else math.nan,
            "maximum": float(np.max(values)) if values else math.nan,
        }

    raw_stats = aggregate(raw_deltas)
    post_stats = aggregate(post_deltas)
    representation_stats = aggregate(post_minus_raw)
    relative_passed = (
        len(done) >= int(gate["minimum_evaluable_request_count"])
        and raw_improved >= int(gate["minimum_raw_improvement_request_count"])
        and post_improved >= int(gate["minimum_post_improvement_request_count"])
        and raw_stats["mean"] < float(gate["require_mean_raw_delta_below_m"])
        and raw_stats["median"] < float(gate["require_median_raw_delta_below_m"])
        and post_stats["mean"] < float(gate["require_mean_post_delta_below_m"])
        and post_stats["median"] < float(gate["require_median_post_delta_below_m"])
    )
    absolute_safe_passed = (
        raw_safe >= int(gate["minimum_geometry_safe_request_count"])
        and post_safe >= int(gate["minimum_geometry_safe_request_count"])
    )
    representation_passed = (
        representation_stats["mean"]
        <= float(gate["maximum_mean_post_minus_raw_mae_m"])
        and representation_stats["median"]
        <= float(gate["maximum_median_post_minus_raw_mae_m"])
    )
    if relative_passed and absolute_safe_passed and representation_passed:
        conclusion = "g4_cross_view_scaffold_geometry_safe_on_model_proxy"
    elif relative_passed and representation_passed:
        conclusion = "g4_cross_view_scaffold_relative_supported_absolute_safe_gate_failed"
    elif not representation_passed:
        conclusion = "g4_cross_view_scaffold_gaussianization_nonregression_failed"
    else:
        conclusion = "g4_cross_view_scaffold_relative_gate_rejected"
    return {
        "conclusion": conclusion,
        "evaluable_request_count": len(done),
        "baseline_replay_exact_count": sum(
            row.get("baseline_replay_exact") is True for row in done
        ),
        "raw_improvement_request_count": raw_improved,
        "post_improvement_request_count": post_improved,
        "raw_geometry_safe_request_count": raw_safe,
        "post_geometry_safe_request_count": post_safe,
        "raw_candidate_minus_g0_m": raw_stats,
        "post_candidate_minus_dense_baseline_m": post_stats,
        "candidate_post_minus_raw_mae_m": representation_stats,
        "relative_gate_passed": relative_passed,
        "absolute_geometry_safe_gate_passed": absolute_safe_passed,
        "gaussianization_nonregression_gate_passed": representation_passed,
        "method_arm_selected": False,
        "validation_unlocked": False,
        "router_refit_performed": False,
        "reference_scope": "model_derived_proxy_not_independent_ground_truth",
    }


def _prior(binding: Mapping[str, Any], expected: int) -> PriorRun:
    path = Path(binding["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {
        (int(row["frame"]), int(row["camera_id"]), int(row["actor_id"])): row
        for row in payload["rows"]
    }
    if len(rows) != expected:
        raise M2CrossViewError(f"prior request denominator 漂移: {path}")
    return PriorRun(path.parent.parent, rows)


def _terminal(
    run_dir: Path,
    *,
    status: str,
    source_head: str,
    summary_sha256: str | None,
    manifest_sha256: str | None,
    reason: str | None,
) -> None:
    atomic_json(
        run_dir / "status.json",
        {
            "schema_version": "worldsim_v5_m2_cross_view_status_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": status,
            "source_commit": source_head,
            "summary_sha256": summary_sha256,
            "manifest_sha256": manifest_sha256,
            "reason": reason,
            "finished_at_utc": utc_now(),
        },
    )


def _render_candidate(
    *,
    depth: np.ndarray,
    valid: np.ndarray,
    observed: np.ndarray,
    inpaint: np.ndarray,
    base: Mapping[str, Any],
    trainer: Any,
    dataset: Any,
    device: torch.device,
    frame: int,
    camera: int,
    actor_id: int,
    erase_delta: Mapping[str, np.ndarray],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    from scripts.run_worldsim_v32_s2_3dgic import render_snapshot

    gaussian = config["gaussianization"]
    points = completion_points_from_view(
        rgb=inpaint,
        depth=depth,
        mask=valid,
        observed_cross_view=observed,
        intrinsics=np.asarray(base["intrinsics"]),
        camera_to_world=np.asarray(base["camera_to_world"]),
        stride=int(gaussian["stride"]),
        scale_multiplier=float(gaussian["scale_multiplier"]),
        minimum_scale_m=float(gaussian["minimum_scale_m"]),
        maximum_scale_m=float(gaussian["maximum_scale_m"]),
    )
    if points.means.shape[0] == 0:
        raise M2CrossViewError("G4 candidate Gaussian 为空")
    asset = completion_points_to_repair_asset(
        points,
        candidate_id=f"scene0471_f{frame:03d}_c{camera}_g4_cross_view_a{actor_id:03d}",
        method="DONOR",
        provenance="cross_view_background_depth_scaffold",
        features_rest_shape=tuple(trainer.models["Background"]._features_rest.shape[1:]),
        opacity=float(gaussian["opacity"]),
        target_frame=frame,
        target_camera_id=camera,
    )
    started = time.perf_counter()
    with temporary_repair_composition(
        trainer.models, erase_delta=erase_delta, asset=asset
    ) as composition:
        rendered = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=frame,
            camera_id=camera,
            device=device,
        )
    return {
        "post": np.asarray(rendered["depth"], dtype=np.float32),
        "background_opacity": np.asarray(
            rendered["background_opacity"], dtype=np.float32
        ),
        "gaussian_count": int(points.means.shape[0]),
        "scale_min_m": float(np.min(points.scales)),
        "scale_mean_m": float(np.mean(points.scales)),
        "scale_max_m": float(np.max(points.scales)),
        "compute_seconds": time.perf_counter() - started,
        "composition_audit": composition,
    }


def run(config_path: Path, run_dir: Path, device_name: str) -> dict[str, Any]:
    config = load_config(config_path)
    source_head = prepare_formal_run(run_dir, TASK_ID, PROJECT)
    resolved = write_resolved_config(run_dir, config)
    events: list[dict[str, Any]] = [
        {"event": "run_started", "at_utc": utc_now(), "source_commit": source_head}
    ]
    write_events(run_dir, events)
    try:
        inputs = {
            name: verify_file(binding["path"], binding["sha256"])
            for name, binding in config["inputs"].items()
        }
        g0_summary = json.loads(Path(inputs["g0_actor_summary"]["path"]).read_text())
        gaussian_summary = json.loads(
            Path(inputs["gaussianization_summary"]["path"]).read_text()
        )
        unlock = config["unlock_binding"]
        supported = gaussian_summary["mechanism_decision"][
            "supported_diagnostic_arms"
        ]
        if (
            g0_summary.get("conclusion") != unlock["g0_conclusion"]
            or gaussian_summary.get("conclusion")
            != unlock["gaussianization_conclusion"]
            or supported != unlock["required_supported_diagnostic_arms"]
            or unlock["forbidden_supported_diagnostic_arm"] in supported
            or gaussian_summary["mechanism_decision"]["baseline_replay_exact_count"]
            != int(unlock["baseline_replay_exact_count"])
            or gaussian_summary.get("method_arm_selected") is not False
        ):
            raise M2CrossViewError("r005/r011 unlock binding 漂移")
        mask_root, requests = _load_requests(config)
        protocol = config["request_protocol"]
        expected = int(protocol["expected_request_count"])
        if len(requests) != expected:
            raise M2CrossViewError("actor request denominator 漂移")
        accepted = sum(request.get("accepted") is True for request in requests)
        rejected = len(requests) - accepted
        if (
            accepted != int(protocol["expected_accepted_mask_count"])
            or rejected != int(protocol["expected_rejected_mask_count"])
        ):
            raise M2CrossViewError("actor accepted/rejected denominator 漂移")
        g0_prior = _prior(inputs["g0_actor_diagnostics"], expected)
        gaussian_prior = _prior(inputs["gaussianization_diagnostics"], expected)
        checkpoint = Path(inputs["formal_checkpoint"]["path"])
        checkpoint_before = sha256_file(checkpoint)
        if not torch.cuda.is_available():
            raise M2CrossViewError("cross-view scaffold 需要 CUDA")
        device = torch.device(device_name)
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        _, dataset, trainer = _runtime(config, device)
        from scripts.run_worldsim_v32_s2_3dgic import render_snapshot

        erase_delta = build_all_rigid_erase_delta(trainer.models["RigidNodes"])
        grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
        for request in requests:
            grouped[(int(request["frame"]), int(request["camera_id"]))].append(
                request
            )
        render_cache: dict[tuple[int, int], dict[str, Any]] = {}

        def snapshot(frame: int, camera: int) -> dict[str, Any]:
            key = (int(frame), int(camera))
            if key not in render_cache:
                value = render_snapshot(
                    trainer=trainer,
                    dataset=dataset,
                    frame=key[0],
                    camera_id=key[1],
                    device=device,
                )
                value["frame"] = key[0]
                value["camera_id"] = key[1]
                render_cache[key] = value
            return render_cache[key]

        rows: list[dict[str, Any]] = []
        fallback_cfg = config["g0_fallback"]
        source_cfg = config["source_views"]
        projection_cfg = config["projection"]
        scaffold_cfg = config["scaffold"]
        for view_index, ((frame, camera), view_requests) in enumerate(
            sorted(grouped.items())
        ):
            base = snapshot(frame, camera)
            reference = np.asarray(base["background_depth"], dtype=np.float32)
            source_keys = frozen_source_views(
                target_frame=frame,
                camera_id=camera,
                temporal_offsets=source_cfg["temporal_offsets"],
                minimum_frame=int(config["scene"]["minimum_frame"]),
                maximum_frame=int(config["scene"]["maximum_frame"]),
            )
            supports = [snapshot(*key) for key in source_keys]
            projected = project_background_depth_stack(
                supports=supports,
                target_shape=reference.shape,
                target_intrinsics=np.asarray(base["intrinsics"]),
                target_camera_to_world=np.asarray(base["camera_to_world"]),
                minimum_source_opacity=float(
                    projection_cfg["minimum_source_background_opacity"]
                ),
                source_stride=int(projection_cfg["source_stride"]),
            )
            projected_lidar = project_background_depth_stack(
                supports=supports,
                target_shape=reference.shape,
                target_intrinsics=np.asarray(base["intrinsics"]),
                target_camera_to_world=np.asarray(base["camera_to_world"]),
                minimum_source_opacity=0.0,
                source_stride=int(projection_cfg["source_stride"]),
                depth_key="measured_lidar_depth",
                require_background_opacity=False,
            )
            for request in sorted(
                view_requests, key=lambda item: int(item["actor_id"])
            ):
                actor_id = int(request["actor_id"])
                key = (frame, camera, actor_id)
                row: dict[str, Any] = {
                    "frame": frame,
                    "camera_id": camera,
                    "actor_id": actor_id,
                }
                g0_row = g0_prior.rows[key]
                gaussian_row = gaussian_prior.rows[key]
                if request["accepted"] is not True:
                    if (
                        g0_row.get("status") != "abstain"
                        or gaussian_row.get("status") != "abstain"
                    ):
                        raise M2CrossViewError("prior rejected request 漂移")
                    row.update(
                        {"status": "abstain", "reason": "ABSTAIN_SAM_MASK_REJECTED"}
                    )
                    rows.append(row)
                    continue
                try:
                    target = binary_dilation(
                        _mask(mask_root, request),
                        iterations=int(protocol["target_mask_dilation_pixels"]),
                    )
                    inner = binary_dilation(
                        target,
                        iterations=int(protocol["support_ring_inner_pixels"]),
                    )
                    outer = binary_dilation(
                        target,
                        iterations=int(protocol["support_ring_outer_pixels"]),
                    )
                    support = outer & ~inner & ~np.asarray(base["dynamic_mask"], bool)
                    withheld = reference.copy()
                    withheld[target] = np.nan
                    fallback = fit_inverse_depth_surface(
                        depth=withheld,
                        support_mask=support,
                        target_mask=target,
                        intrinsics=np.asarray(base["intrinsics"]),
                        model="G0_ROBUST_PLANE",
                        minimum_support_points=int(
                            fallback_cfg["minimum_support_points"]
                        ),
                        huber_delta=float(fallback_cfg["huber_delta"]),
                        maximum_iterations=int(
                            fallback_cfg["maximum_iterations"]
                        ),
                        minimum_depth_m=float(fallback_cfg["minimum_depth_m"]),
                        maximum_depth_m=float(fallback_cfg["maximum_depth_m"]),
                    )
                    scaffold = fuse_cross_view_scaffold(
                        fallback_depth=fallback.depth,
                        target_mask=target,
                        projected=projected,
                        minimum_support_views=int(
                            scaffold_cfg["minimum_support_views"]
                        ),
                        maximum_absolute_disagreement_m=float(
                            scaffold_cfg["maximum_absolute_disagreement_m"]
                        ),
                        maximum_relative_disagreement=float(
                            scaffold_cfg["maximum_relative_disagreement"]
                        ),
                        maximum_extrapolation_pixels=float(
                            scaffold_cfg["maximum_extrapolation_pixels"]
                        ),
                    )
                    if not np.all(scaffold.valid[target]):
                        raise M2CrossViewError("G4 candidate target depth 不完整")
                    inpaint = cv2.inpaint(
                        cv2.cvtColor(
                            np.asarray(base["groundtruth"], np.uint8),
                            cv2.COLOR_RGB2BGR,
                        ),
                        target.astype(np.uint8) * 255,
                        3.0,
                        cv2.INPAINT_TELEA,
                    )
                    inpaint = cv2.cvtColor(inpaint, cv2.COLOR_BGR2RGB)
                    candidate = _render_candidate(
                        depth=scaffold.depth,
                        valid=scaffold.valid,
                        observed=scaffold.direct_support,
                        inpaint=inpaint,
                        base=base,
                        trainer=trainer,
                        dataset=dataset,
                        device=device,
                        frame=frame,
                        camera=camera,
                        actor_id=actor_id,
                        erase_delta=erase_delta,
                        config=config,
                    )
                    rollback = render_snapshot(
                        trainer=trainer,
                        dataset=dataset,
                        frame=frame,
                        camera_id=camera,
                        device=device,
                    )
                    if not np.array_equal(rollback["rgb"], base["rgb"]):
                        raise M2CrossViewError("G4 composition rollback 非 exact")
                    g0_artifact = g0_prior.root / g0_row["artifact"]["path"]
                    gaussian_artifact = (
                        gaussian_prior.root / gaussian_row["artifact"]["path"]
                    )
                    verify_file(g0_artifact, g0_row["artifact"]["sha256"])
                    verify_file(
                        gaussian_artifact, gaussian_row["artifact"]["sha256"]
                    )
                    with np.load(g0_artifact, allow_pickle=False) as old_g0, np.load(
                        gaussian_artifact, allow_pickle=False
                    ) as old_gaussian:
                        common = old_g0["common_evaluation_mask"].astype(bool)
                        g0_raw = old_g0["raw_surface_depth"].astype(np.float32)
                        dense_post = old_gaussian["dense_post_depth"].astype(
                            np.float32
                        )
                        replay_exact = (
                            np.array_equal(old_g0["target_mask"].astype(bool), target)
                            and np.array_equal(
                                old_g0["support_mask"].astype(bool), support
                            )
                            and np.array_equal(
                                old_gaussian["target_mask"].astype(bool), target
                            )
                            and np.array_equal(
                                old_gaussian["common_evaluation_mask"].astype(bool),
                                common,
                            )
                            and np.array_equal(
                                old_g0["reference_depth"],
                                reference.astype(np.float16),
                                equal_nan=True,
                            )
                            and np.array_equal(
                                old_g0["raw_surface_depth"],
                                fallback.depth.astype(np.float16),
                                equal_nan=True,
                            )
                            and gaussian_row.get("baseline_replay_exact") is True
                        )
                    if not replay_exact:
                        raise M2CrossViewError(f"r005/r011 baseline replay 非 exact: {key}")
                    if not np.all(
                        np.isfinite(candidate["post"][common])
                        & (candidate["post"][common] > 1e-4)
                    ):
                        raise M2CrossViewError("G4 post render 在冻结 common mask 非有限")
                    baseline_raw = depth_error_summary(g0_raw, reference, common)
                    baseline_post = depth_error_summary(
                        dense_post, reference, common
                    )
                    candidate_raw = depth_error_summary(
                        scaffold.depth, reference, common
                    )
                    candidate_post = depth_error_summary(
                        candidate["post"], reference, common
                    )
                    representation_gap = depth_error_summary(
                        candidate["post"], scaffold.depth, common
                    )
                    projection_audit = projected.audit(target)
                    scaffold_audit = scaffold.audit(target)
                    lidar_audit = lidar_agreement_audit(
                        scaffold=scaffold,
                        lidar_projected=projected_lidar,
                        target_mask=target,
                    )
                    artifact = (
                        run_dir
                        / "artifacts/requests"
                        / f"f{frame:03d}_c{camera}_a{actor_id:03d}.npz"
                    )
                    atomic_save_npz(
                        artifact,
                        {
                            "target_mask": target.astype(np.int8),
                            "support_mask": support.astype(np.int8),
                            "common_evaluation_mask": common.astype(np.int8),
                            "reference_depth": reference.astype(np.float16),
                            "g0_raw_depth": g0_raw.astype(np.float16),
                            "dense_baseline_post_depth": dense_post.astype(np.float16),
                            "g4_raw_depth": scaffold.depth.astype(np.float16),
                            "g4_post_depth": candidate["post"].astype(np.float16),
                            "direct_support": scaffold.direct_support.astype(np.int8),
                            "extrapolated_support": scaffold.extrapolated_support.astype(
                                np.int8
                            ),
                            "support_count": scaffold.support_count.astype(np.int8),
                            "disagreement_m": scaffold.disagreement_m.astype(np.float16),
                        },
                    )
                    row.update(
                        {
                            "status": "done",
                            "target_pixels": int(target.sum()),
                            "common_evaluation_pixels": int(common.sum()),
                            "baseline_replay_exact": True,
                            "target_reference_interior_available_to_candidate": False,
                            "target_reference_interior_evaluation_only": True,
                            "source_views": [list(item) for item in source_keys],
                            "projection_audit": projection_audit,
                            "scaffold_audit": scaffold_audit,
                            "lidar_audit": lidar_audit,
                            "fallback_surface": fallback.audit(),
                            "baseline": {
                                "g0_raw_geometry_error": baseline_raw,
                                "dense_post_geometry_error": baseline_post,
                            },
                            "candidate": {
                                "raw_geometry_error": candidate_raw,
                                "post_geometry_error": candidate_post,
                                "representation_gap": representation_gap,
                                "gaussian_count": candidate["gaussian_count"],
                                "scale_min_m": candidate["scale_min_m"],
                                "scale_mean_m": candidate["scale_mean_m"],
                                "scale_max_m": candidate["scale_max_m"],
                                "compute_seconds": candidate["compute_seconds"],
                                "composition_audit": candidate[
                                    "composition_audit"
                                ],
                            },
                            "reference": {
                                "source": config["reference"]["source"],
                                "independent_geometry_claim_allowed": False,
                            },
                            "artifact": {
                                "path": str(artifact.relative_to(run_dir)),
                                "sha256": sha256_file(artifact),
                            },
                        }
                    )
                except (GeometryRepairError, CrossViewScaffoldError) as error:
                    row.update({"status": "abstain", "reason": str(error)})
                rows.append(row)
            print(
                f"M2 G4 {view_index + 1}/{len(grouped)} frame={frame} camera={camera} sources={len(source_keys)} requests={len(view_requests)}",
                flush=True,
            )
        decision = candidate_decision(rows, config["candidate_gate"])
        if decision["baseline_replay_exact_count"] != decision[
            "evaluable_request_count"
        ]:
            raise M2CrossViewError("r005/r011 replay denominator 不完整")
        diagnostics = {
            "schema_version": "worldsim_v5_m2_cross_view_diagnostics_v1",
            "task_id": TASK_ID,
            "status": "done",
            "scene": config["scene"]["name"],
            "request_unit": "one_actor_one_view_one_hole",
            "rows": rows,
            "candidate_decision": decision,
            "target_reference_interior_available_to_candidate": False,
            "method_arm_selected": False,
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "test_quality_read": False,
            "kitti_quality_read": False,
            "parameter_search_performed": False,
            "router_refit_performed": False,
        }
        diagnostics_path = run_dir / "artifacts/diagnostics.json"
        atomic_json(diagnostics_path, diagnostics)
        checkpoint_after = sha256_file(checkpoint)
        if checkpoint_after != checkpoint_before:
            raise M2CrossViewError("formal checkpoint 被 G4 修改")
        snapshot_files = [
            config_path,
            PROJECT / "scripts/run_worldsim_v5_m2_cross_view_scaffold.py",
            PROJECT / "motion_proj/worldsim_v5/cross_view_scaffold.py",
            PROJECT / "motion_proj/worldsim_v5/geometry_repair.py",
            PROJECT / "tests/test_worldsim_v5_m2_cross_view_scaffold.py",
        ]
        snapshot = copy_source_snapshot(run_dir, snapshot_files, PROJECT)
        summary = {
            "schema_version": "worldsim_v5_m2_cross_view_summary_v1",
            "task_id": TASK_ID,
            "task_status": "running",
            "status": "done",
            "phase": config["phase"],
            "scene": config["scene"]["name"],
            "source_commit": source_head,
            "conclusion": decision["conclusion"],
            "request_count": len(rows),
            "evaluable_request_count": decision["evaluable_request_count"],
            "abstain_request_count": len(rows) - decision["evaluable_request_count"],
            "candidate_decision": decision,
            "target_reference_interior_available_to_candidate": False,
            "method_arm_selected": False,
            "validation_unlocked": False,
            "reference_scope": "model_derived_proxy_not_independent_ground_truth",
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "diagnostics_sha256": sha256_file(diagnostics_path),
            "duration_seconds": time.perf_counter() - started,
            "peak_gpu_memory_mib": int(
                torch.cuda.max_memory_allocated(device) / 1024**2
            ),
            "validation_quality_read": False,
            "heldout_quality_read": False,
            "test_quality_read": False,
            "kitti_quality_read": False,
            "parameter_search_performed": False,
            "router_refit_performed": False,
        }
        summary_path = run_dir / "summary.json"
        atomic_json(summary_path, summary)
        atomic_json(
            run_dir / "fingerprint.json",
            {
                "schema_version": "worldsim_v5_m2_cross_view_fingerprint_v1",
                "task_id": TASK_ID,
                "source_commit": source_head,
                "source_clean": True,
                "resolved_config": resolved,
                "inputs": inputs,
                "runtime": {
                    "drivestudio_commit": config["runtime"][
                        "drivestudio_commit"
                    ],
                    "drivestudio_status": config["runtime"][
                        "drivestudio_expected_status"
                    ],
                    "torch": torch.__version__,
                    "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(device),
                },
                "source_snapshot": snapshot,
            },
        )
        events.append({"event": "run_done", "at_utc": utc_now(), **decision})
        write_events(run_dir, events)
        manifest = {
            "schema_version": "worldsim_v5_m2_cross_view_run_manifest_v1",
            "task_id": TASK_ID,
            "status": "done",
            "inventory": inventory_files(run_dir, {"manifest.json", "status.json"}),
        }
        manifest_path = run_dir / "manifest.json"
        atomic_json(manifest_path, manifest)
        _terminal(
            run_dir,
            status="done",
            source_head=source_head,
            summary_sha256=sha256_file(summary_path),
            manifest_sha256=sha256_file(manifest_path),
            reason=None,
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
        _terminal(
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
