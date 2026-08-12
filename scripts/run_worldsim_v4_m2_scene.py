#!/usr/bin/env python3
"""构建并真实渲染一个 nuScenes M2 scene 的 matched repair candidates。"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping

import cv2
import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import binary_dilation
from scipy.spatial import cKDTree
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from motion_proj.worldsim_v32.inpainting_adapter import (  # noqa: E402
    completion_points_from_view,
)
from motion_proj.worldsim_v33.roadpatch import (  # noqa: E402
    build_hole_anchor,
    build_patch_index,
    materialize_patch_delta,
    search_donors,
)
from motion_proj.worldsim_v33.spatial_delta import (  # noqa: E402
    atomic_json,
    load_erase_delta,
    sha256_file,
    temporary_spatial_composition,
)
from motion_proj.worldsim_v4.repair_assets import (  # noqa: E402
    atomic_save_repair_asset,
    temporary_repair_composition,
)
from motion_proj.worldsim_v4.repair_builders import (  # noqa: E402
    completion_points_to_repair_asset,
    normalized_repair_risks,
    roadpatch_delta_to_repair_asset,
)
from motion_proj.worldsim_v4.repair_candidates import RepairCandidate  # noqa: E402
from motion_proj.worldsim_v4.repair_risk import (  # noqa: E402
    RepairRiskWeights,
    score_repair_candidate,
)
from scripts.lift_worldsim_v32_semantics import build_runtime  # noqa: E402
from scripts.run_worldsim_v32_s2_3dgic import (  # noqa: E402
    combine_cross_view_splats,
    load_binary_mask,
    lpips_uint8,
    make_completion,
    psnr_uint8,
    render_snapshot,
    ssim_uint8,
)


TASK_ID = "WS-V4-M2-REPAIR-ROUTER-01"


class M2SceneRunError(RuntimeError):
    pass


def _json_safe(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, payload: Any) -> None:
    atomic_json(path, _json_safe(payload))


def _snapshot_input_scene_config(config_path: Path, run_dir: Path) -> dict[str, Any]:
    source = config_path.resolve()
    if not source.is_file():
        raise M2SceneRunError(f"materialized scene config does not exist: {source}")
    snapshot_dir = run_dir / "source_snapshot"
    snapshot_dir.mkdir(exist_ok=True)
    target = snapshot_dir / "materialized_scene_config.yaml"
    shutil.copy2(source, target)
    return {
        "source_path": str(source),
        "snapshot_path": str(target),
        "sha256": sha256_file(target),
        "bytes": target.stat().st_size,
    }


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_dirty() -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "status", "--porcelain"], text=True
        ).strip()
    )


def _verify(binding: Mapping[str, Any], label: str) -> Path:
    path = Path(binding["path"])
    if not path.is_file():
        raise M2SceneRunError(f"{label} 不存在: {path}")
    actual = sha256_file(path)
    if actual != binding["sha256"]:
        raise M2SceneRunError(
            f"{label} SHA 漂移: expected={binding['sha256']} actual={actual}"
        )
    if "bytes" in binding and path.stat().st_size != int(binding["bytes"]):
        raise M2SceneRunError(f"{label} bytes 漂移")
    return path


def preflight(config: Mapping[str, Any], *, phase: str) -> dict[str, Any]:
    if (
        config.get("schema_version") != "worldsim_v4_m2_scene_v1"
        or config.get("task_id") != TASK_ID
        or config.get("partition") != "development"
        or config.get("heldout_content_read") is not False
        or config.get("test_quality_read") is not False
    ):
        raise M2SceneRunError("M2 scene contract 漂移")
    if phase == "formal" and _git_dirty():
        raise M2SceneRunError("M2 formal run 要求 clean git worktree")
    verified: dict[str, Any] = {}
    for name, binding in config.get("inputs", {}).items():
        if isinstance(binding, dict) and "path" in binding and "sha256" in binding:
            path = _verify(binding, name)
            verified[name] = {
                "path": str(path),
                "sha256": binding["sha256"],
                "bytes": path.stat().st_size,
            }
    if config.get("status") == "ready":
        if not config.get("requests") or not config.get("all_accepted_masks_retained"):
            raise M2SceneRunError("ready scene 未保留完整 request set")
        for request in config["requests"]:
            _verify(request["target_mask"], f"{request['request_id']}:mask")
            _verify(
                request["groundtruth_with_actor"],
                f"{request['request_id']}:groundtruth",
            )
            frame = int(request["frame"])
            if frame % 5 != 2:
                raise M2SceneRunError("M2 target 不是 development remainder=2")
            for support_frame, _ in request["support_views"]:
                if int(support_frame) % 5 in {2, 4}:
                    raise M2SceneRunError("M2 support 混入 development/heldout")
    return verified


def _runtime_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "inputs": {
            "checkpoint": config["inputs"]["checkpoint"]["path"],
            "source_config": config["inputs"]["drivestudio_source_config"]["path"],
        },
        "runtimes": {
            "drivestudio_checkout": config["runtime"]["drivestudio_checkout"]
        },
    }


def _load_background(config: Mapping[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    checkpoint = torch.load(
        config["inputs"]["checkpoint"]["path"],
        map_location="cpu",
        weights_only=False,
    )
    payload = checkpoint["models"]["Background"]
    names = (
        "_means",
        "_scales",
        "_quats",
        "_features_dc",
        "_features_rest",
        "_opacities",
    )
    state = {name: payload[name].detach().cpu().numpy() for name in names}
    count = int(state["_means"].shape[0])
    with np.load(config["inputs"]["evidence_state"]["path"], allow_pickle=False) as evidence:
        model = evidence["base_model"].astype(np.int8)
        indices = evidence["base_index"].astype(np.int64)
        gaussian_ids_all = evidence["gaussian_id"].astype(np.int64)
    selected = model == 0
    order = np.argsort(indices[selected], kind="stable")
    if not np.array_equal(indices[selected][order], np.arange(count, dtype=np.int64)):
        raise M2SceneRunError("evidence state 的 Background base_index 漂移")
    gaussian_ids = gaussian_ids_all[selected][order]
    if np.unique(gaussian_ids).size != count:
        raise M2SceneRunError("Background Gaussian IDs 非唯一")
    return state, gaussian_ids


def _native_donor_index(
    config: Mapping[str, Any], state: Mapping[str, np.ndarray]
) -> tuple[Any, dict[str, Any]]:
    sidecar_binding = config["actor"]["semantic_sidecar"]
    sidecar = _verify(sidecar_binding, "semantic sidecar")
    count = int(state["_means"].shape[0])
    with np.load(sidecar, allow_pickle=False) as payload:
        if int(payload["background_count"].item()) != count:
            raise M2SceneRunError("semantic sidecar Background count 漂移")
        actor_score = payload["semantic_score"][:count].astype(np.float32)
        view_count = payload["visible_view_count"][:count].astype(np.float32)
        visible_mass = payload["visible_mass"][:count].astype(np.float32)
    # sidecar 来自 frozen train-only masks；这里的 camera support 仅作保守可见代理，
    # 不声称每行有独立的多相机观测计数。
    camera_support_proxy = (view_count > 0).astype(np.uint8)
    donor = config["asset_build"]["donor"]
    index = build_patch_index(
        means=state["_means"],
        raw_scales=state["_scales"],
        raw_opacities=state["_opacities"],
        features_dc=state["_features_dc"],
        actor_semantic_score=actor_score,
        train_view_observation_count=view_count,
        visibility_mass=visible_mass,
        multi_camera_count=camera_support_proxy,
        native_donor_mask=np.ones(count, dtype=bool),
        patch_sizes_m=donor["patch_sizes_m"],
        thresholds=donor["thresholds"],
    )
    return index, {
        "background_count": count,
        "patch_count": int(index.patch_ids.size),
        "valid_patch_count": int(np.sum(index.exclusion_flags == 0)),
        "flat_membership_count": int(index.flat_indices.size),
        "support_source": "frozen_train_only_semantic_sidecar",
        "multi_camera_count_claim": "binary_visibility_proxy_only",
        "heldout_used": False,
    }


def _mean_l1(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return float("inf")
    return float(
        np.abs(left.astype(np.float32) - right.astype(np.float32))[selected].mean()
    )


def _depth_mae(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    selected = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(left)
        & np.isfinite(right)
        & (left > 1e-4)
        & (right > 1e-4)
    )
    if not selected.any():
        return float("inf")
    return float(np.mean(np.abs(left[selected] - right[selected])))


def _static_lidar_mae(rendered: Mapping[str, Any], excluded: np.ndarray) -> float:
    measured = np.asarray(rendered["measured_lidar_depth"], np.float32).squeeze()
    predicted = np.asarray(rendered["depth"], np.float32).squeeze()
    dynamic = np.asarray(rendered["dynamic_mask"], bool).squeeze()
    egocar = np.asarray(rendered["egocar_mask"], bool).squeeze()
    valid = (
        np.isfinite(measured)
        & np.isfinite(predicted)
        & (measured > 1e-4)
        & (predicted > 1e-4)
        & ~dynamic
        & ~egocar
        & ~np.asarray(excluded, bool)
    )
    return float(np.mean(np.abs(predicted[valid] - measured[valid]))) if valid.any() else float("nan")


def _evaluate_render(
    *,
    rendered: Mapping[str, Any],
    base: Mapping[str, Any],
    erase: Mapping[str, Any],
    mask: np.ndarray,
    observed: np.ndarray,
    cross_rgb: np.ndarray,
    background_depth_reference: np.ndarray,
    trainer: Any,
    device: torch.device,
) -> dict[str, Any]:
    rgb = np.asarray(rendered["rgb"], np.uint8)
    groundtruth = np.asarray(base["groundtruth"], np.uint8)
    # 真实 remove hole 没有同视角背景 GT；用 GT compositing 排除该 ROI，
    # global/SSIM/LPIPS 只度量有真值的非目标区域。
    valid_global = rgb.copy()
    valid_global[mask] = groundtruth[mask]
    full = np.ones(mask.shape, dtype=bool)
    static = (
        ~np.asarray(base["dynamic_mask"], bool).squeeze()
        & ~np.asarray(base["egocar_mask"], bool).squeeze()
        & ~mask
    )
    effect = np.max(
        np.abs(rgb.astype(np.int16) - np.asarray(erase["rgb"], np.uint8).astype(np.int16)),
        axis=2,
    ) >= 2
    hole_coverage = float(np.sum(effect & mask) / max(int(mask.sum()), 1))
    result = {
        "global_valid_psnr_db": psnr_uint8(valid_global, groundtruth, full),
        "global_valid_ssim": ssim_uint8(valid_global, groundtruth),
        "global_valid_lpips_alex": lpips_uint8(trainer, valid_global, groundtruth, device),
        "static_psnr_db": psnr_uint8(rgb, groundtruth, static),
        "static_lidar_depth_mae_m": _static_lidar_mae(rendered, mask),
        "hole_cross_view_l1_uint8": _mean_l1(rgb, cross_rgb, observed),
        "hole_cross_view_psnr_db": (
            psnr_uint8(rgb, cross_rgb, observed) if observed.any() else float("nan")
        ),
        "hole_geometry_mae_m": _depth_mae(
            np.asarray(rendered["depth"], np.float32).squeeze(),
            np.asarray(background_depth_reference, np.float32),
            mask,
        ),
        "hole_effect_pixels": int(np.sum(effect & mask)),
        "hole_coverage": hole_coverage,
        "truth_scope": {
            "global_valid": "observed_groundtruth_excluding_undefined_remove_hole",
            "hole_rgb": "cross_view_consistency_not_same_view_GT",
            "hole_same_view_GT_metrics": "undefined",
        },
    }
    result["edit_error"] = float(
        np.mean(
            [
                min(result["hole_cross_view_l1_uint8"] / 25.0, 1.0),
                min(result["hole_geometry_mae_m"] / 0.5, 1.0),
                1.0 - result["hole_coverage"],
            ]
        )
    )
    return result


def _atomic_abstain_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Record R0 as the exact base/no-operation arm, never as a repair success."""

    result = dict(metrics)
    result.update(
        {
            "hole_effect_pixels": 0,
            "hole_coverage": 0.0,
            "operation_success": False,
            "atomic_noop": True,
        }
    )
    result["edit_error"] = float(
        np.mean(
            [
                min(float(result["hole_cross_view_l1_uint8"]) / 25.0, 1.0),
                min(float(result["hole_geometry_mae_m"]) / 0.5, 1.0),
                1.0,
            ]
        )
    )
    return result


def _matched_arm_records(
    *,
    expected_arms: list[str],
    candidate_records: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    abstain_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Make every frozen matched arm explicit, including unavailable/failed arms."""

    if len(expected_arms) != len(set(expected_arms)):
        raise M2SceneRunError("matched repair arms must be unique")
    candidates = {str(row["arm"]): row for row in candidate_records}
    failed: dict[str, list[str]] = {}
    for row in failures:
        failed.setdefault(str(row["arm"]), []).append(str(row["error"]))
    output: list[dict[str, Any]] = []
    for arm in expected_arms:
        if arm == "ABSTAIN":
            output.append(
                {"arm": arm, "status": "atomic_noop", "metrics": dict(abstain_metrics)}
            )
        elif arm == "RISK_ROUTER":
            output.append({"arm": arm, "status": "pending_development_selection"})
        elif arm in candidates:
            output.append({"arm": arm, "status": "executed", **candidates[arm]})
        else:
            output.append(
                {
                    "arm": arm,
                    "status": "abstain",
                    "reasons": failed.get(arm, ["ABSTAIN_CANDIDATE_UNAVAILABLE"]),
                }
            )
    return output


def _candidate_from_asset(
    *,
    candidate_id: str,
    method: str,
    provenance: str,
    binding: Any,
    metrics: Mapping[str, Any],
    temporal_std_uint8: float | None,
    uncertainty: float,
    normalization: Mapping[str, float],
    evidence: Mapping[str, Any],
) -> RepairCandidate:
    risks = normalized_repair_risks(
        photo_l1_uint8=float(metrics["hole_cross_view_l1_uint8"]),
        geometry_mae_m=float(metrics["hole_geometry_mae_m"]),
        temporal_std_uint8=temporal_std_uint8,
        uncertainty=float(uncertainty),
        gaussian_count=int(binding.gaussian_count),
        normalization=normalization,
    )
    return RepairCandidate(
        candidate_id=candidate_id,
        method=method,
        gaussians=binding,
        provenance=provenance,
        evidence={**dict(evidence), "metrics": _json_safe(metrics)},
        **risks,
    )


def _render_asset(
    *,
    trainer: Any,
    dataset: Any,
    device: torch.device,
    erase_delta: Mapping[str, np.ndarray],
    asset: Mapping[str, np.ndarray],
    frame: int,
    camera_id: int,
    base_rgb: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    with temporary_repair_composition(
        trainer.models, erase_delta=erase_delta, asset=asset
    ) as audit:
        rendered = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=frame,
            camera_id=camera_id,
            device=device,
        )
    rollback = render_snapshot(
        trainer=trainer,
        dataset=dataset,
        frame=frame,
        camera_id=camera_id,
        device=device,
    )
    if not np.array_equal(rollback["rgb"], base_rgb):
        raise M2SceneRunError("repair composition rollback render 非 exact")
    return rendered, audit


def _cap_delta(delta: Mapping[str, np.ndarray], limit: int) -> dict[str, np.ndarray]:
    count = int(np.asarray(delta["means"]).shape[0])
    if count <= limit:
        return {name: np.asarray(value).copy() for name, value in delta.items()}
    source_ids = np.asarray(delta["source_gaussian_ids"], np.int64)
    feather = np.asarray(delta["feather_weight"], np.float32)
    chosen = np.lexsort((source_ids, -feather))[: int(limit)]
    chosen.sort()
    return {name: np.asarray(value)[chosen].copy() for name, value in delta.items()}


def _process_request(
    *,
    config: Mapping[str, Any],
    request: Mapping[str, Any],
    trainer: Any,
    dataset: Any,
    device: torch.device,
    erase_delta: Mapping[str, np.ndarray],
    background_state: Mapping[str, np.ndarray],
    gaussian_ids: np.ndarray,
    patch_index: Any,
    request_dir: Path,
) -> dict[str, Any]:
    request_dir.mkdir(parents=True)
    frame, camera_id = int(request["frame"]), int(request["camera_id"])
    mask = load_binary_mask(Path(request["target_mask"]["path"]))
    base = render_snapshot(
        trainer=trainer, dataset=dataset, frame=frame, camera_id=camera_id, device=device
    )
    supports = []
    for support_frame, support_camera in request["support_views"]:
        snapshot = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=int(support_frame),
            camera_id=int(support_camera),
            device=device,
        )
        snapshot["frame"] = int(support_frame)
        snapshot["camera_id"] = int(support_camera)
        supports.append(snapshot)
    cross_rgb, observed, cross_audit = combine_cross_view_splats(
        supports=supports,
        target=base,
        target_mask=mask,
        projection=config["asset_build"]["projection"],
    )
    completed, _, unseen = make_completion(
        target=base,
        target_mask=mask,
        cross_rgb=cross_rgb,
        observed=observed,
        persistence_mask=mask,
        config=config["asset_build"],
    )
    with temporary_spatial_composition(trainer.models, erase_delta=erase_delta):
        erase_render = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=frame,
            camera_id=camera_id,
            device=device,
        )
    rollback = render_snapshot(
        trainer=trainer, dataset=dataset, frame=frame, camera_id=camera_id, device=device
    )
    if not np.array_equal(rollback["rgb"], base["rgb"]):
        raise M2SceneRunError("erase-only rollback render 非 exact")
    imageio.imwrite(request_dir / "base_with_actor.png", base["rgb"])
    imageio.imwrite(request_dir / "erase_only.png", erase_render["rgb"])
    imageio.imwrite(request_dir / "mask.png", mask.astype(np.uint8) * 255)
    imageio.imwrite(request_dir / "cross_view_observed.png", observed.astype(np.uint8) * 255)
    imageio.imwrite(request_dir / "completion_diagnostic.png", completed)

    gaussian_cfg = config["asset_build"]["gaussians"]
    rest_shape = tuple(trainer.models["Background"]._features_rest.shape[1:])
    candidate_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    candidate_objects: list[RepairCandidate] = []
    assets: dict[str, Mapping[str, np.ndarray]] = {}
    normalization = config["asset_build"]["risk_normalization"]
    temporal_std = cross_audit.get("temporal_color_std_uint8_mean_multi_support")

    def register(
        *,
        arm: str,
        method: str,
        provenance: str,
        asset: Mapping[str, np.ndarray],
        uncertainty: float,
        evidence: Mapping[str, Any],
        temporal_std_uint8: float | None = None,
    ) -> dict[str, Any]:
        candidate_id = str(np.asarray(asset["candidate_id"]).item())
        asset_path = request_dir / "assets" / f"{arm.lower()}.npz"
        binding = atomic_save_repair_asset(asset_path, asset)
        rendered, composition_audit = _render_asset(
            trainer=trainer,
            dataset=dataset,
            device=device,
            erase_delta=erase_delta,
            asset=asset,
            frame=frame,
            camera_id=camera_id,
            base_rgb=base["rgb"],
        )
        metrics = _evaluate_render(
            rendered=rendered,
            base=base,
            erase=erase_render,
            mask=mask,
            observed=observed,
            cross_rgb=cross_rgb,
            background_depth_reference=base["background_depth"],
            trainer=trainer,
            device=device,
        )
        metrics["operation_success"] = True
        metrics["atomic_noop"] = False
        candidate = _candidate_from_asset(
            candidate_id=candidate_id,
            method=method,
            provenance=provenance,
            binding=binding,
            metrics=metrics,
            temporal_std_uint8=temporal_std_uint8,
            uncertainty=uncertainty,
            normalization=normalization,
            evidence=evidence,
        )
        weights = RepairRiskWeights(**config["risk"]["weights"])
        score = score_repair_candidate(candidate, weights)
        imageio.imwrite(request_dir / f"{arm.lower()}.png", rendered["rgb"])
        row = {
            "arm": arm,
            "candidate": candidate.to_dict(),
            "risk_score_default_weights": score.to_dict(),
            "metrics": metrics,
            "composition_audit": composition_audit,
            "render_sha256": sha256_file(request_dir / f"{arm.lower()}.png"),
        }
        candidate_records.append(row)
        candidate_objects.append(candidate)
        assets[candidate_id] = asset
        return row

    observed_fraction = float(observed.sum() / max(mask.sum(), 1))
    if observed.any():
        try:
            points = completion_points_from_view(
                rgb=cross_rgb,
                depth=base["background_depth"],
                mask=observed,
                observed_cross_view=observed,
                intrinsics=base["intrinsics"],
                camera_to_world=base["camera_to_world"],
                stride=int(gaussian_cfg["target_stride"]),
                scale_multiplier=float(gaussian_cfg["scale_multiplier"]),
                minimum_scale_m=float(gaussian_cfg["minimum_scale_m"]),
                maximum_scale_m=float(gaussian_cfg["maximum_scale_m"]),
            )
            asset = completion_points_to_repair_asset(
                points,
                candidate_id=f"{request['request_id']}__observed",
                method="OBSERVED",
                provenance="observed_cross_view",
                features_rest_shape=rest_shape,
                opacity=float(gaussian_cfg["opacity"]),
                target_frame=frame,
                target_camera_id=camera_id,
            )
            register(
                arm="OBSERVED",
                method="OBSERVED",
                provenance="observed_cross_view",
                asset=asset,
                uncertainty=1.0 - observed_fraction,
                evidence={"cross_view": cross_audit, "observed_fraction": observed_fraction},
                temporal_std_uint8=temporal_std,
            )
        except Exception as exc:
            failures.append({"arm": "OBSERVED", "error": f"{type(exc).__name__}: {exc}"})
    else:
        failures.append({"arm": "OBSERVED", "error": "ABSTAIN_NO_CROSS_VIEW_SUPPORT"})

    try:
        full_inpaint = cv2.inpaint(
            cv2.cvtColor(erase_render["rgb"], cv2.COLOR_RGB2BGR),
            mask.astype(np.uint8) * 255,
            float(config["asset_build"]["projection"]["inpaint_radius_pixels"]),
            cv2.INPAINT_TELEA,
        )
        full_inpaint = cv2.cvtColor(full_inpaint, cv2.COLOR_BGR2RGB)
        imageio.imwrite(request_dir / "telea_full_same_hole_diagnostic.png", full_inpaint)
        points = completion_points_from_view(
            rgb=full_inpaint,
            depth=base["background_depth"],
            mask=mask,
            observed_cross_view=np.zeros_like(mask),
            intrinsics=base["intrinsics"],
            camera_to_world=base["camera_to_world"],
            stride=int(gaussian_cfg["target_stride"]),
            scale_multiplier=float(gaussian_cfg["scale_multiplier"]),
            minimum_scale_m=float(gaussian_cfg["minimum_scale_m"]),
            maximum_scale_m=float(gaussian_cfg["maximum_scale_m"]),
        )
        asset = completion_points_to_repair_asset(
            points,
            candidate_id=f"{request['request_id']}__telea",
            method="GENERATED",
            provenance="generated_telea",
            features_rest_shape=rest_shape,
            opacity=float(gaussian_cfg["opacity"]),
            target_frame=frame,
            target_camera_id=camera_id,
        )
        register(
            arm="TELEA",
            method="GENERATED",
            provenance="generated_telea",
            asset=asset,
            uncertainty=0.8,
            evidence={
                "inpaint": "opencv_telea_deterministic_full_same_hole",
                "same_hole_pixels": int(mask.sum()),
                "uses_cross_view_rgb": False,
            },
            temporal_std_uint8=None,
        )
    except Exception as exc:
        failures.append({"arm": "TELEA", "error": f"{type(exc).__name__}: {exc}"})

    try:
        donor_cfg = config["asset_build"]["donor"]
        anchor = build_hole_anchor(
            delete_mask=mask,
            first_hit_depth=base["depth"],
            rgb=base["rgb"],
            intrinsics=base["intrinsics"],
            camera_to_world=base["camera_to_world"],
            cross_view_observed_pixels=int(observed.sum()),
            patch_sizes_m=donor_cfg["patch_sizes_m"],
            bottom_quantile=float(donor_cfg["anchor"]["bottom_quantile"]),
            robust_quantiles=tuple(donor_cfg["anchor"]["robust_quantiles"]),
            minimum_anchor_pixels=int(donor_cfg["anchor"]["minimum_anchor_pixels"]),
            minimum_cross_view_observed_pixels=int(
                donor_cfg["anchor"]["minimum_cross_view_observed_pixels"]
            ),
            ring_pixels=int(donor_cfg["anchor"]["context_ring_pixels"]),
        )
        donor_candidates = search_donors(
            index=patch_index,
            anchor=anchor,
            top_k=int(donor_cfg["search"]["top_k"]),
            weights=donor_cfg["search"]["weights"],
            minimum_spatial_separation_m=float(
                donor_cfg["search"]["minimum_spatial_separation_m"]
            ),
            minimum_tangent_confidence=float(
                donor_cfg["search"]["minimum_tangent_confidence"]
            ),
            maximum_abs_yaw_radians=float(
                donor_cfg["search"]["maximum_abs_yaw_radians"]
            ),
            maximum_abs_vertical_offset_m=float(
                donor_cfg["search"]["maximum_abs_vertical_offset_m"]
            ),
        )
        # matched RoadPatch 使用冻结 index distance rank-1；cohort router 不参与 donor 内部选择。
        donor = donor_candidates[0]
        delta, donor_manifest = materialize_patch_delta(
            index=patch_index,
            candidate=donor,
            anchor=anchor,
            background_state=background_state,
            source_gaussian_ids=gaussian_ids,
            target_role=str(request["role"]),
            opacity_feather_width_m=float(
                donor_cfg["delta"]["opacity_feather_width_m"]
            ),
            maximum_rgb_affine=float(donor_cfg["delta"]["maximum_rgb_affine"]),
            minimum_scale_m=float(donor_cfg["delta"]["minimum_scale_m"]),
            maximum_scale_m=float(donor_cfg["delta"]["maximum_scale_m"]),
            duplicate_radius_m=float(donor_cfg["delta"]["duplicate_radius_m"]),
            base_tree=cKDTree(np.asarray(background_state["_means"], np.float64)),
        )
        delta = _cap_delta(delta, int(donor_cfg["delta"]["maximum_rows_per_target"]))
        asset = roadpatch_delta_to_repair_asset(
            delta,
            candidate_id=f"{request['request_id']}__roadpatch",
            confidence=float(1.0 / (1.0 + donor.distance)),
            target_frame=frame,
            target_camera_id=camera_id,
        )
        register(
            arm="ROADPATCH",
            method="DONOR",
            provenance="native_scene_donor",
            asset=asset,
            uncertainty=float(np.clip(donor.distance / 2.0, 0.0, 1.0)),
            evidence={
                "anchor": {
                    "center_xyz": anchor.center_xyz.tolist(),
                    "patch_size_m": anchor.patch_size_m,
                    "valid_point_count": anchor.valid_point_count,
                },
                "rank1": {
                    "patch_id": donor.patch_id,
                    "distance": donor.distance,
                    "geometry_distance": donor.geometry_distance,
                    "appearance_distance": donor.appearance_distance,
                    "semantic_distance": donor.semantic_distance,
                    "visibility_distance": donor.visibility_distance,
                },
                "top5_patch_ids": [row.patch_id for row in donor_candidates],
                "manifest": donor_manifest,
            },
            temporal_std_uint8=None,
        )
    except Exception as exc:
        failures.append({"arm": "ROADPATCH", "error": f"{type(exc).__name__}: {exc}"})

    failures.append(
        {
            "arm": "GENERATED",
            "error": config["asset_build"]["generated_unavailable_action"],
        }
    )
    candidate_records.sort(key=lambda row: row["arm"])
    abstain_metrics = _atomic_abstain_metrics(
        _evaluate_render(
            rendered=base,
            base=base,
            erase=erase_render,
            mask=mask,
            observed=observed,
            cross_rgb=cross_rgb,
            background_depth_reference=base["background_depth"],
            trainer=trainer,
            device=device,
        )
    )
    matched_arms = _matched_arm_records(
        expected_arms=list(config["ablations"]["matched_repair_arms"]),
        candidate_records=candidate_records,
        failures=failures,
        abstain_metrics=abstain_metrics,
    )
    return {
        "request_id": request["request_id"],
        "frame": frame,
        "camera_id": camera_id,
        "mask_pixels": int(mask.sum()),
        "observed_pixels": int(observed.sum()),
        "unseen_pixels": int(unseen.sum()),
        "cross_view_audit": cross_audit,
        "candidate_count": len(candidate_records),
        "candidates": candidate_records,
        "candidate_failures": failures,
        "matched_arms": matched_arms,
        "generated_model_executable": False,
        "development_content_read": True,
        "development_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
    }


def run(
    config: Mapping[str, Any],
    run_dir: Path,
    *,
    max_requests: int | None,
    input_config_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    verified = preflight(config, phase="smoke" if max_requests else "formal")
    run_dir.mkdir(parents=True)
    input_scene_config = _snapshot_input_scene_config(input_config_path, run_dir)
    _write_json(
        run_dir / "status.json",
        {"task_id": TASK_ID, "status": "running", "scene": config["scene"]},
    )
    if config["status"] == "abstain":
        summary = {
            "schema_version": "worldsim_v4_m2_scene_summary_v1",
            "task_id": TASK_ID,
            "scene": config["scene"],
            "status": "abstain",
            "reason": config["reason"],
            "retained_in_denominator": True,
            "request_count": 0,
            "input_scene_config": input_scene_config,
            "heldout_content_read": False,
            "test_quality_read": False,
            "project_git_head": _git_head(),
        }
        _write_json(run_dir / "summary.json", summary)
        _write_json(
            run_dir / "status.json",
            {"task_id": TASK_ID, "status": "done", "scene": config["scene"], "summary_sha256": sha256_file(run_dir / "summary.json")},
        )
        return summary
    if not torch.cuda.is_available():
        raise M2SceneRunError("M2 real renderer 需要 CUDA")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    checkpoint = Path(config["inputs"]["checkpoint"]["path"])
    checkpoint_before = sha256_file(checkpoint)
    dataset, trainer = build_runtime(_runtime_config(config), device)
    trainer.set_eval()
    if hasattr(trainer, "optimizer"):
        raise M2SceneRunError("M2 evaluator 禁止 optimizer")
    erase_delta = load_erase_delta(config["inputs"]["erase_delta"]["path"])
    state, gaussian_ids = _load_background(config)
    patch_index, patch_audit = _native_donor_index(config, state)
    requests = list(config["requests"])
    if max_requests is not None:
        requests = requests[: int(max_requests)]
    rows = []
    for index, request in enumerate(requests, start=1):
        print(
            f"M2 {config['scene']} request {index}/{len(requests)} {request['request_id']}",
            flush=True,
        )
        rows.append(
            _process_request(
                config=config,
                request=request,
                trainer=trainer,
                dataset=dataset,
                device=device,
                erase_delta=erase_delta,
                background_state=state,
                gaussian_ids=gaussian_ids,
                patch_index=patch_index,
                request_dir=run_dir / "artifacts/requests" / request["request_id"],
            )
        )
    checkpoint_after = sha256_file(checkpoint)
    if checkpoint_after != checkpoint_before:
        raise M2SceneRunError("M2 source checkpoint 被修改")
    source_snapshot = run_dir / "source_snapshot"
    source_snapshot.mkdir(exist_ok=True)
    for source in (
        Path(config["source_config"]["path"]),
        Path(__file__),
        PROJECT_ROOT / "motion_proj/worldsim_v4/repair_assets.py",
        PROJECT_ROOT / "motion_proj/worldsim_v4/repair_builders.py",
    ):
        shutil.copy2(source, source_snapshot / source.name)
    summary = {
        "schema_version": "worldsim_v4_m2_scene_summary_v1",
        "task_id": TASK_ID,
        "scene": config["scene"],
        "status": "done",
        "phase": "smoke" if max_requests is not None else "formal_development",
        "request_count": len(rows),
        "frozen_request_count": int(config["request_count"]),
        "candidate_count": sum(int(row["candidate_count"]) for row in rows),
        "requests": rows,
        "native_donor_index": patch_audit,
        "verified_inputs": verified,
        "input_scene_config": input_scene_config,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "checkpoint_immutable": True,
        "duration_seconds": time.monotonic() - started,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "development_content_read": True,
        "development_optimization_read": False,
        "heldout_content_read": False,
        "test_quality_read": False,
        "project_git_head": _git_head(),
        "project_git_dirty": _git_dirty(),
    }
    _write_json(run_dir / "summary.json", summary)
    _write_json(
        run_dir / "status.json",
        {
            "task_id": TASK_ID,
            "status": "done",
            "scene": config["scene"],
            "summary_sha256": sha256_file(run_dir / "summary.json"),
        },
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-requests", type=int)
    args = parser.parse_args()
    if args.run_dir.exists():
        raise FileExistsError(args.run_dir)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    try:
        summary = run(
            config,
            args.run_dir.resolve(),
            max_requests=args.max_requests,
            input_config_path=args.config.resolve(),
        )
        print(
            json.dumps(
                {
                    "status": summary["status"],
                    "scene": summary["scene"],
                    "request_count": summary["request_count"],
                    "candidate_count": summary.get("candidate_count", 0),
                },
                sort_keys=True,
            )
        )
        return 0
    except Exception as exc:
        args.run_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "task_id": TASK_ID,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "heldout_content_read": False,
            "test_quality_read": False,
        }
        _write_json(args.run_dir / "status.json", event)
        print(json.dumps(event, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
