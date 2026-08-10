#!/usr/bin/env python3
"""执行 V3.3 S2 RoadPatch-Lite 锚点预检、donor 选择与正式验收。"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Iterator, Mapping

import imageio.v2 as imageio
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import cKDTree
from skimage.metrics import structural_similarity
import torch
import yaml


PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from motion_proj.worldsim_v33.instance_field import load_instance_field  # noqa: E402
from motion_proj.worldsim_v33.roadpatch import (  # noqa: E402
    HoleAnchor,
    atomic_save_patch_delta,
    build_hole_anchor,
    conservative_delete_mask,
    load_patch_index,
    materialize_patch_delta,
    search_donors,
    sha256_arrays,
    validate_patch_delta,
)
from scripts.eval_worldsim_v3_a3_r1_heldout import (  # noqa: E402
    load_model_checkpoint_read_only,
)
from scripts.lift_worldsim_v32_semantics import build_runtime  # noqa: E402
from scripts.run_worldsim_v32_s2_3dgic import (  # noqa: E402
    lpips_uint8,
    psnr_uint8,
    render_snapshot,
    ssim_uint8,
)
from scripts.run_worldsim_v33_s1_instance_field import render_actor  # noqa: E402


BACKGROUND_ATTRS = {
    "_means": "means",
    "_scales": "raw_scales",
    "_quats": "quats",
    "_features_dc": "features_dc",
    "_features_rest": "features_rest",
    "_opacities": "raw_opacities",
}


def sha256_file(path: str | Path, chunk_bytes: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_bytes):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def verify_file(path: str | Path, expected: str, role: str) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"{role} 不存在: {source}")
    actual = sha256_file(source)
    if actual != expected:
        raise RuntimeError(f"{role} SHA 漂移: expected={expected} actual={actual}")
    return {"path": str(source), "sha256": actual, "bytes": source.stat().st_size}


def load_binary_npz(path: str | Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "binary" not in payload.files:
            raise RuntimeError(f"mask 缺 binary: {path}")
        return payload["binary"].astype(bool)


def load_binary_png(path: str | Path) -> np.ndarray:
    value = np.asarray(imageio.imread(path))
    if value.ndim == 3:
        value = np.any(value > 0, axis=2)
    return value.astype(bool)


def preflight(
    config: dict[str, Any],
    config_path: Path,
    patch_index: Path,
    patch_index_sha256: str,
    patch_index_manifest: Path,
    patch_index_manifest_sha256: str,
) -> dict[str, Any]:
    if config.get("schema_version") != "worldsim_v33_s2_roadpatch_v1":
        raise ValueError("S2 config schema 漂移")
    if tuple(config["scene"]["bev_axes"]) != (0, 2) or int(config["scene"]["vertical_axis"]) != 1:
        raise ValueError("RoadPatch 坐标契约必须是 x/z BEV、y 竖直")
    verified = {
        "checkpoint": verify_file(
            config["inputs"]["checkpoint"], config["inputs"]["checkpoint_sha256"], "checkpoint"
        ),
        "source_config": verify_file(
            config["inputs"]["source_config"], config["inputs"]["source_config_sha256"], "source_config"
        ),
        "s1_config": verify_file(
            config["inputs"]["s1_config"], config["inputs"]["s1_config_sha256"], "s1_config"
        ),
        "s1_instance_field": verify_file(
            config["inputs"]["s1_instance_field"], config["inputs"]["s1_instance_field_sha256"], "s1_instance_field"
        ),
        "s1_summary": verify_file(
            config["inputs"]["s1_summary"], config["inputs"]["s1_summary_sha256"], "s1_summary"
        ),
        "patch_index": verify_file(patch_index, patch_index_sha256, "patch_index"),
        "patch_index_manifest": verify_file(
            patch_index_manifest,
            patch_index_manifest_sha256,
            "patch_index_manifest",
        ),
    }
    index_manifest = json.loads(patch_index_manifest.read_text(encoding="utf-8"))
    if index_manifest["config"]["sha256"] != sha256_file(config_path):
        raise RuntimeError("patch index manifest 未绑定本次 config SHA")
    if index_manifest["patch_index"]["sha256"] != patch_index_sha256:
        raise RuntimeError("patch index manifest 与 index SHA 不一致")
    if index_manifest["source_checkpoint_sha256_before"] != config["inputs"]["checkpoint_sha256"]:
        raise RuntimeError("patch index manifest 未绑定 D2 checkpoint")
    if index_manifest["lineage_audit"]["v32_generated_rows"]["generated_rows_admitted_to_donor_source"] != 0:
        raise RuntimeError("patch index manifest 混入 V3.2 generated donor")
    for role, target in config["targets"].items():
        for name in ("semantic_mask", "cross_view_observed_mask", "completion_reference"):
            verified[f"target:{role}:{name}"] = verify_file(
                target[name], target[f"{name}_sha256"], f"target:{role}:{name}"
            )
    if int(config["search"]["top_k"]) != 5:
        raise ValueError("RoadPatch top-K 必须冻结为 5")
    return verified


def load_background_state(config: dict[str, Any]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    checkpoint = torch.load(
        config["inputs"]["checkpoint"], map_location="cpu", weights_only=False
    )
    background = checkpoint["models"]["Background"]
    state = {
        name: background[name].detach().cpu().numpy()
        for name in BACKGROUND_ATTRS
    }
    count = int(state["_means"].shape[0])
    if count != int(config["inputs"]["checkpoint_background_count"]):
        raise RuntimeError("D2 Background 行数漂移")
    ancestry = background["worldsim_a2_ancestry"]["fields"]
    gaussian_ids = ancestry["gaussian_id"].detach().cpu().numpy().astype(np.int64)
    if gaussian_ids.shape != (count,) or np.unique(gaussian_ids).size != count:
        raise RuntimeError("D2 Background Gaussian ID 漂移")
    return state, gaussian_ids


def anchor_json(anchor: HoleAnchor) -> dict[str, Any]:
    return {
        "center_xyz": anchor.center_xyz.astype(float).tolist(),
        "bounds_bev": anchor.bounds_bev.astype(float).tolist(),
        "patch_size_m": float(anchor.patch_size_m),
        "tangent_yaw": float(anchor.tangent_yaw),
        "tangent_confidence": float(anchor.tangent_confidence),
        "context_rgb_mean": anchor.context_rgb_mean.astype(float).tolist(),
        "context_rgb_std": anchor.context_rgb_std.astype(float).tolist(),
        "valid_point_count": int(anchor.valid_point_count),
        "cross_view_observed_pixels": int(anchor.cross_view_observed_pixels),
    }


def candidate_json(candidate: Any) -> dict[str, Any]:
    return {
        "patch_index": int(candidate.patch_index),
        "patch_id": candidate.patch_id,
        "distance": float(candidate.distance),
        "geometry_distance": float(candidate.geometry_distance),
        "appearance_distance": float(candidate.appearance_distance),
        "semantic_distance": float(candidate.semantic_distance),
        "visibility_distance": float(candidate.visibility_distance),
        "yaw_radians": float(candidate.yaw_radians),
        "vertical_offset_m": float(candidate.vertical_offset_m),
    }


def build_target_anchors(
    *,
    config: dict[str, Any],
    trainer: Any,
    dataset: Any,
    field: Mapping[str, np.ndarray],
    index: Any,
    device: torch.device,
    output_dir: Path,
) -> dict[str, dict[str, Any]]:
    hard = np.asarray(field["hard_instance_id"], dtype=np.int32)
    logits = torch.from_numpy(np.asarray(field["instance_opacity_logit"], dtype=np.float32)).to(device)
    records: dict[str, dict[str, Any]] = {}
    for role, target in config["targets"].items():
        global_ids_np = np.flatnonzero(hard == int(target["dataset_instance_id"])).astype(np.int64)
        if global_ids_np.size == 0:
            raise RuntimeError(f"{role} 的 S1 instance field 没有 Gaussian")
        global_ids = torch.from_numpy(global_ids_np).to(device)
        frame, camera_id = int(target["frame"]), int(target["camera_id"])
        with torch.inference_mode():
            instance_probability = render_actor(
                trainer=trainer,
                dataset=dataset,
                frame=frame,
                camera=camera_id,
                global_ids=global_ids,
                logits=logits,
                device=device,
            ).detach().float().cpu().numpy()
        if instance_probability.ndim == 3:
            instance_probability = np.squeeze(instance_probability)
        instance_mask = instance_probability >= float(config["anchor"]["instance_mask_threshold"])
        semantic_mask = load_binary_npz(target["semantic_mask"])
        delete_mask = conservative_delete_mask(instance_mask, semantic_mask)
        cross_view = load_binary_png(target["cross_view_observed_mask"])
        visible = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=frame,
            camera_id=camera_id,
            device=device,
        )
        hidden = render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=frame,
            camera_id=camera_id,
            device=device,
            hide_actor=int(target["rigid_model_index"]),
        )
        if not (delete_mask.shape == cross_view.shape == visible["depth"].shape):
            raise RuntimeError(f"{role} anchor 图像 shape 漂移")
        anchor = build_hole_anchor(
            delete_mask=delete_mask,
            first_hit_depth=visible["depth"],
            rgb=visible["rgb"],
            intrinsics=visible["intrinsics"],
            camera_to_world=visible["camera_to_world"],
            cross_view_observed_pixels=int(np.sum(cross_view & delete_mask)),
            patch_sizes_m=config["index"]["patch_sizes_m"],
            bottom_quantile=float(config["anchor"]["bottom_quantile"]),
            robust_quantiles=tuple(float(value) for value in config["anchor"]["robust_quantiles"]),
            minimum_anchor_pixels=int(config["anchor"]["minimum_anchor_pixels"]),
            minimum_cross_view_observed_pixels=int(config["anchor"]["minimum_cross_view_observed_pixels"]),
            ring_pixels=int(config["anchor"]["context_ring_pixels"]),
        )
        candidates = search_donors(
            index=index,
            anchor=anchor,
            top_k=int(config["search"]["top_k"]),
            weights=config["search"]["weights"],
            minimum_spatial_separation_m=float(config["search"]["minimum_spatial_separation_m"]),
            minimum_tangent_confidence=float(config["search"]["minimum_tangent_confidence"]),
            maximum_abs_yaw_radians=float(config["search"]["maximum_abs_yaw_radians"]),
            maximum_abs_vertical_offset_m=float(config["search"]["maximum_abs_vertical_offset_m"]),
        )
        role_dir = output_dir / "targets" / role
        role_dir.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(role_dir / "instance_probability.png", np.clip(instance_probability * 255.0, 0, 255).astype(np.uint8))
        imageio.imwrite(role_dir / "delete_mask.png", delete_mask.astype(np.uint8) * 255)
        imageio.imwrite(role_dir / "visible_rgb.png", visible["rgb"])
        imageio.imwrite(role_dir / "actor_hidden_base_rgb.png", hidden["rgb"])
        records[role] = {
            "config": target,
            "anchor": anchor,
            "candidates": candidates,
            "delete_mask": delete_mask,
            "cross_view": cross_view,
            "visible": visible,
            "hidden": hidden,
            "audit": {
                "assigned_gaussians": int(global_ids_np.size),
                "instance_mask_pixels": int(instance_mask.sum()),
                "semantic_mask_pixels": int(semantic_mask.sum()),
                "delete_mask_pixels": int(delete_mask.sum()),
                "semantic_recall_of_delete": float(delete_mask.sum() / max(semantic_mask.sum(), 1)),
                "cross_view_pixels_in_delete": int(np.sum(cross_view & delete_mask)),
                "anchor": anchor_json(anchor),
                "top5": [candidate_json(candidate) for candidate in candidates],
            },
        }
    return records


@contextmanager
def temporary_background_delta(background: Any, delta: Mapping[str, np.ndarray]) -> Iterator[None]:
    """临时追加 delta 参数，并在退出时恢复同一批 Parameter 对象。"""

    validate_patch_delta(delta)
    originals: dict[str, torch.nn.Parameter] = {}
    for attribute, delta_name in BACKGROUND_ATTRS.items():
        original = getattr(background, attribute)
        originals[attribute] = original
        tail = torch.from_numpy(np.asarray(delta[delta_name])).to(
            device=original.device, dtype=original.dtype
        )
        combined = torch.cat([original.detach(), tail], dim=0)
        setattr(
            background,
            attribute,
            torch.nn.Parameter(combined, requires_grad=False),
        )
    try:
        yield
    finally:
        for attribute, original in originals.items():
            setattr(background, attribute, original)
        if any(getattr(background, name) is not value for name, value in originals.items()):
            raise RuntimeError("temporary delta 未恢复原 Background Parameter 对象")


def combine_deltas(deltas: list[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    if not deltas:
        raise ValueError("没有可合并 delta")
    fields = set(deltas[0])
    if any(set(delta) != fields for delta in deltas):
        raise RuntimeError("delta schema 不一致")
    combined = {
        name: np.concatenate([np.asarray(delta[name]) for delta in deltas], axis=0)
        for name in sorted(fields)
    }
    validate_patch_delta(combined)
    return combined


def mean_l1_uint8(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        raise ValueError("L1 mask 为空")
    return float(
        np.mean(
            np.abs(left.astype(np.float32) - right.astype(np.float32))[selected]
        )
    )


def boundary_depth_jump(depth: np.ndarray, mask: np.ndarray) -> float:
    values: list[np.ndarray] = []
    depth = np.asarray(depth, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        shifted_mask = np.roll(mask, shift=(dy, dx), axis=(0, 1))
        shifted_depth = np.roll(depth, shift=(dy, dx), axis=(0, 1))
        boundary = mask & ~shifted_mask & np.isfinite(depth) & np.isfinite(shifted_depth)
        boundary &= (depth > 1e-4) & (shifted_depth > 1e-4)
        if boundary.any():
            values.append(np.abs(depth[boundary] - shifted_depth[boundary]))
    if not values:
        return float("inf")
    return float(np.mean(np.concatenate(values)))


def save_panel(path: Path, images: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, np.concatenate(images, axis=1))


def select_target_donors(
    *,
    config: dict[str, Any],
    trainer: Any,
    dataset: Any,
    index: Any,
    records: dict[str, dict[str, Any]],
    background_state: Mapping[str, np.ndarray],
    source_gaussian_ids: np.ndarray,
    device: torch.device,
    artifacts: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, np.ndarray]]:
    base_tree = cKDTree(np.asarray(background_state["_means"], dtype=np.float64))
    selected: dict[str, dict[str, Any]] = {}
    selected_deltas: list[dict[str, np.ndarray]] = []
    background = trainer.models["Background"]
    for role, record in records.items():
        target = record["config"]
        delete = record["delete_mask"]
        observed = record["cross_view"] & delete
        hidden = record["hidden"]
        completion = np.asarray(imageio.imread(target["completion_reference"]))
        ring = binary_dilation(
            delete, iterations=int(config["anchor"]["context_ring_pixels"])
        ) & ~delete
        protected_outside = ~binary_dilation(
            delete, iterations=int(config["anchor"]["context_ring_pixels"])
        )
        candidate_rows: list[dict[str, Any]] = []
        materialized: dict[str, tuple[dict[str, np.ndarray], dict[str, Any]]] = {}
        role_dir = artifacts / "targets" / role / "candidates"
        role_dir.mkdir(parents=True, exist_ok=True)
        for rank, candidate in enumerate(record["candidates"], start=1):
            delta, provenance = materialize_patch_delta(
                index=index,
                candidate=candidate,
                anchor=record["anchor"],
                background_state=background_state,
                source_gaussian_ids=source_gaussian_ids,
                target_role=role,
                opacity_feather_width_m=float(config["delta"]["opacity_feather_width_m"]),
                maximum_rgb_affine=float(config["delta"]["maximum_rgb_affine"]),
                minimum_scale_m=float(config["delta"]["minimum_scale_m"]),
                maximum_scale_m=float(config["delta"]["maximum_scale_m"]),
                duplicate_radius_m=float(config["delta"]["duplicate_radius_m"]),
                base_tree=base_tree,
            )
            with temporary_background_delta(background, delta):
                rendered = render_snapshot(
                    trainer=trainer,
                    dataset=dataset,
                    frame=int(target["frame"]),
                    camera_id=int(target["camera_id"]),
                    device=device,
                    hide_actor=int(target["rigid_model_index"]),
                )
            pixel_effect = np.mean(
                np.abs(rendered["rgb"].astype(np.float32) - hidden["rgb"].astype(np.float32)),
                axis=2,
            )
            effect = delete & (pixel_effect > 1.0)
            coverage = float(effect.sum() / max(delete.sum(), 1))
            observed_l1 = mean_l1_uint8(rendered["rgb"], completion, observed)
            seam_l1 = mean_l1_uint8(rendered["rgb"], hidden["rgb"], ring)
            outside_l1 = mean_l1_uint8(
                rendered["rgb"], hidden["rgb"], protected_outside
            )
            depth_jump = boundary_depth_jump(rendered["depth"], delete)
            normalized = {
                "missing_coverage": 1.0 - coverage,
                "observed_l1": observed_l1 / 255.0,
                "seam_l1": seam_l1 / 255.0,
                "depth_discontinuity": depth_jump / 10.0,
                "outside_l1": outside_l1 / 255.0,
                "index_distance": float(candidate.distance),
            }
            score = float(
                sum(
                    float(config["selection"]["score_weights"][name]) * value
                    for name, value in normalized.items()
                )
            )
            gates = {
                "maximum_delta_rows": int(delta["means"].shape[0])
                <= int(config["delta"]["maximum_rows_per_target"]),
                "minimum_effect_pixels": int(effect.sum())
                >= int(config["selection"]["minimum_effect_pixels"]),
                "minimum_effect_coverage": coverage
                >= float(config["selection"]["minimum_effect_coverage"]),
                "maximum_outside_l1": outside_l1
                <= float(config["selection"]["maximum_outside_l1_uint8"]),
                "finite_depth_discontinuity": math.isfinite(depth_jump),
            }
            row = {
                "rank": rank,
                **candidate_json(candidate),
                "delta_rows": int(delta["means"].shape[0]),
                "delta_arrays_sha256": sha256_arrays(delta),
                "effect_pixels": int(effect.sum()),
                "effect_coverage": coverage,
                "observed_l1_uint8": observed_l1,
                "seam_l1_uint8": seam_l1,
                "outside_l1_uint8": outside_l1,
                "depth_boundary_jump_m": depth_jump,
                "normalized_score_terms": normalized,
                "selection_score": score,
                "gates": gates,
                "eligible": all(gates.values()),
                "provenance": provenance,
            }
            candidate_rows.append(row)
            materialized[candidate.patch_id] = (delta, provenance)
            imageio.imwrite(role_dir / f"rank{rank}_{candidate.patch_id}.png", rendered["rgb"])
            save_panel(
                role_dir / f"rank{rank}_{candidate.patch_id}_panel.png",
                [record["visible"]["rgb"], hidden["rgb"], rendered["rgb"], completion],
            )
        eligible = [row for row in candidate_rows if row["eligible"]]
        eligible.sort(key=lambda row: (row["selection_score"], row["patch_id"]))
        if not eligible:
            atomic_json(role_dir / "candidate_metrics.json", candidate_rows)
            raise RuntimeError(f"{role} top-5 没有通过 train/dev 选择门的 donor，ABSTAIN")
        winner = eligible[0]
        delta, provenance = materialized[winner["patch_id"]]
        selected[role] = {
            "winner": winner,
            "all_candidates": candidate_rows,
            "provenance": provenance,
        }
        selected_deltas.append(delta)
        atomic_json(role_dir / "candidate_metrics.json", candidate_rows)
        atomic_json(role_dir / "selected.json", selected[role])
    return selected, combine_deltas(selected_deltas)


def lidar_depth_mae(rendered: Mapping[str, Any]) -> float:
    measured = np.asarray(rendered["measured_lidar_depth"], dtype=np.float32).squeeze()
    predicted = np.asarray(rendered["depth"], dtype=np.float32).squeeze()
    dynamic = np.asarray(rendered["dynamic_mask"], dtype=bool).squeeze()
    egocar = np.asarray(rendered["egocar_mask"], dtype=bool).squeeze()
    valid = (
        np.isfinite(measured)
        & np.isfinite(predicted)
        & (measured > 1e-4)
        & (predicted > 1e-4)
        & ~dynamic
        & ~egocar
    )
    return float(np.mean(np.abs(predicted[valid] - measured[valid]))) if valid.any() else float("nan")


def evaluate_heldout(
    *,
    config: dict[str, Any],
    trainer: Any,
    dataset: Any,
    combined_delta: Mapping[str, np.ndarray],
    device: torch.device,
    artifacts: Path,
) -> dict[str, Any]:
    views = [(int(frame), int(camera)) for frame, camera in config["evaluation"]["heldout_views"]]
    b0_checkpoint = Path(config["inputs"]["v32_telea"]["checkpoint"])
    if sha256_file(b0_checkpoint) != config["inputs"]["v32_telea"]["checkpoint_sha256"]:
        raise RuntimeError("B0 Telea checkpoint SHA 漂移")
    load_model_checkpoint_read_only(trainer, b0_checkpoint, device)
    trainer.set_eval()
    b0 = {
        view: render_snapshot(
            trainer=trainer,
            dataset=dataset,
            frame=view[0],
            camera_id=view[1],
            device=device,
        )
        for view in views
    }
    load_model_checkpoint_read_only(
        trainer, Path(config["inputs"]["checkpoint"]), device
    )
    trainer.set_eval()
    background = trainer.models["Background"]
    rows: list[dict[str, Any]] = []
    heldout_dir = artifacts / "heldout"
    heldout_dir.mkdir(parents=True, exist_ok=True)
    with temporary_background_delta(background, combined_delta):
        for frame, camera_id in views:
            b1 = render_snapshot(
                trainer=trainer,
                dataset=dataset,
                frame=frame,
                camera_id=camera_id,
                device=device,
            )
            base = b0[(frame, camera_id)]
            groundtruth = base["groundtruth"]
            full = np.ones(groundtruth.shape[:2], dtype=bool)
            static = ~np.asarray(base["dynamic_mask"], dtype=bool).squeeze()
            static &= ~np.asarray(base["egocar_mask"], dtype=bool).squeeze()
            row = {
                "frame": frame,
                "camera_id": camera_id,
                "b0_psnr": psnr_uint8(base["rgb"], groundtruth, full),
                "b1_psnr": psnr_uint8(b1["rgb"], groundtruth, full),
                "b0_static_psnr": psnr_uint8(base["rgb"], groundtruth, static),
                "b1_static_psnr": psnr_uint8(b1["rgb"], groundtruth, static),
                "b0_ssim": ssim_uint8(base["rgb"], groundtruth),
                "b1_ssim": ssim_uint8(b1["rgb"], groundtruth),
                "b0_lpips": lpips_uint8(trainer, base["rgb"], groundtruth, device),
                "b1_lpips": lpips_uint8(trainer, b1["rgb"], groundtruth, device),
                "b0_static_lidar_depth_mae_m": lidar_depth_mae(base),
                "b1_static_lidar_depth_mae_m": lidar_depth_mae(b1),
            }
            row.update(
                {
                    "psnr_delta": row["b1_psnr"] - row["b0_psnr"],
                    "static_psnr_delta": row["b1_static_psnr"] - row["b0_static_psnr"],
                    "ssim_delta": row["b1_ssim"] - row["b0_ssim"],
                    "lpips_delta": row["b1_lpips"] - row["b0_lpips"],
                    "static_lidar_depth_mae_delta_m": row["b1_static_lidar_depth_mae_m"]
                    - row["b0_static_lidar_depth_mae_m"],
                }
            )
            rows.append(row)
            save_panel(
                heldout_dir / f"frame{frame:03d}_cam{camera_id}_panel.png",
                [base["rgb"], b1["rgb"], groundtruth],
            )
    mean = {
        key: float(np.mean([row[key] for row in rows]))
        for key in (
            "b0_psnr", "b1_psnr", "psnr_delta",
            "b0_static_psnr", "b1_static_psnr", "static_psnr_delta",
            "b0_ssim", "b1_ssim", "ssim_delta",
            "b0_lpips", "b1_lpips", "lpips_delta",
            "b0_static_lidar_depth_mae_m", "b1_static_lidar_depth_mae_m",
            "static_lidar_depth_mae_delta_m",
        )
    }
    gates = {
        "maximum_psnr_degradation": mean["psnr_delta"]
        >= -float(config["evaluation"]["maximum_heldout_psnr_degradation_db"]),
        "maximum_ssim_degradation": mean["ssim_delta"]
        >= -float(config["evaluation"]["maximum_heldout_ssim_degradation"]),
        "maximum_lpips_increase": mean["lpips_delta"]
        <= float(config["evaluation"]["maximum_heldout_lpips_increase"]),
    }
    report = {
        "protocol": "heldout_confirmation_after_train_dev_donor_selection",
        "baseline": "B0 V3.2 Telea generated Background",
        "candidate": "B1 V3.3 RoadPatch immutable delta",
        "rows": rows,
        "mean": mean,
        "gates": gates,
        "accepted": all(gates.values()),
    }
    atomic_json(heldout_dir / "evaluation.json", report)
    return report


def snapshot_sources(run_dir: Path, config_path: Path) -> dict[str, Any]:
    output = run_dir / "source_snapshot"
    output.mkdir(parents=True, exist_ok=True)
    sources = {
        "config": config_path.resolve(),
        "runner": Path(__file__).resolve(),
        "roadpatch": PROJECT / "motion_proj" / "worldsim_v33" / "roadpatch.py",
    }
    result: dict[str, Any] = {}
    for role, source in sources.items():
        target = output / source.name
        shutil.copy2(source, target)
        result[role] = {
            "path": str(target),
            "sha256": sha256_file(target),
            "bytes": target.stat().st_size,
        }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--patch-index", required=True, type=Path)
    parser.add_argument("--patch-index-sha256", required=True)
    parser.add_argument("--patch-index-manifest", required=True, type=Path)
    parser.add_argument("--patch-index-manifest-sha256", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--phase", choices=("anchor", "formal"), default="anchor")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.run_dir.exists() and any(
        (args.run_dir / name).exists() for name in ("status.json", "summary.json", "artifacts")
    ):
        raise FileExistsError(f"run-dir 已含结构化输出: {args.run_dir}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = args.run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.time()
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    verified = preflight(
        config,
        args.config,
        args.patch_index,
        args.patch_index_sha256,
        args.patch_index_manifest,
        args.patch_index_manifest_sha256,
    )
    checkpoint_before = sha256_file(config["inputs"]["checkpoint"])
    atomic_json(args.run_dir / "status.json", {"state": "running", "phase": args.phase, "started_unix": started})
    if not torch.cuda.is_available():
        raise RuntimeError("RoadPatch GPU anchor 需要 CUDA")
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    index = load_patch_index(args.patch_index)
    index.validate(background_count=int(config["inputs"]["checkpoint_background_count"]))
    field = load_instance_field(config["inputs"]["s1_instance_field"])
    dataset, trainer = build_runtime(config, device)
    trainer.set_eval()
    records = build_target_anchors(
        config=config,
        trainer=trainer,
        dataset=dataset,
        field=field,
        index=index,
        device=device,
        output_dir=artifacts,
    )
    anchor_report = {
        "protocol": "S1_instance_mask_intersection_SAM2_then_bottom_first_hit_depth",
        "heldout_used": False,
        "targets": {role: record["audit"] for role, record in records.items()},
    }
    atomic_json(artifacts / "anchor_preflight.json", anchor_report)
    formal_report: dict[str, Any] | None = None
    if args.phase == "formal":
        background_state, source_gaussian_ids = load_background_state(config)
        selected, combined_delta = select_target_donors(
            config=config,
            trainer=trainer,
            dataset=dataset,
            index=index,
            records=records,
            background_state=background_state,
            source_gaussian_ids=source_gaussian_ids,
            device=device,
            artifacts=artifacts,
        )
        delta_path = artifacts / "roadpatch_delta.npz"
        atomic_save_patch_delta(delta_path, combined_delta)
        repeat_path = artifacts / "roadpatch_delta.repeat.npz"
        atomic_save_patch_delta(repeat_path, combined_delta)
        deterministic_delta = sha256_file(delta_path) == sha256_file(repeat_path)
        repeat_path.unlink()
        if not deterministic_delta:
            raise RuntimeError("RoadPatch delta 重序列化非字节确定")
        selection_report = {
            "protocol": "train_dev_only_top5_selection_before_heldout_confirmation",
            "heldout_used_for_selection": False,
            "targets": selected,
            "combined_delta": {
                "path": str(delta_path),
                "sha256": sha256_file(delta_path),
                "bytes": delta_path.stat().st_size,
                "rows": int(combined_delta["means"].shape[0]),
                "arrays_sha256": sha256_arrays(combined_delta),
                "deterministic_reserialization": True,
                "provenance": "GENERATED_BY_PATCH_REUSE",
                "source_checkpoint_mutated": False,
            },
        }
        atomic_json(artifacts / "selection.json", selection_report)
        heldout = evaluate_heldout(
            config=config,
            trainer=trainer,
            dataset=dataset,
            combined_delta=combined_delta,
            device=device,
            artifacts=artifacts,
        )
        selection_gates = {
            role: bool(record["winner"]["eligible"])
            for role, record in selected.items()
        }
        acceptance = {
            "coordinate_contract_xz_bev_y_vertical": True,
            "native_d2_donors_only": bool(
                np.all(combined_delta["provenance_code"] == 2)
            ),
            "both_targets_have_top5": all(
                len(record["candidates"]) == 5 for record in records.values()
            ),
            "train_dev_selection_passed": all(selection_gates.values()),
            "heldout_not_used_for_selection": True,
            "heldout_confirmation_passed": bool(heldout["accepted"]),
            "delta_deterministic": deterministic_delta,
            "delta_schema_valid": True,
        }
        formal_report = {
            "selection": selection_report,
            "heldout": heldout,
            "selection_gates": selection_gates,
            "acceptance": acceptance,
            "accepted": all(acceptance.values()),
        }
        atomic_json(artifacts / "acceptance.json", formal_report)
    checkpoint_after = sha256_file(config["inputs"]["checkpoint"])
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("anchor preflight 修改了 D2 checkpoint")
    elapsed = time.time() - started
    snapshot = snapshot_sources(args.run_dir, args.config)
    summary = {
        "task_id": config["task_id"],
        "state": "completed",
        "phase": args.phase,
        "elapsed_seconds": elapsed,
        "config_sha256": sha256_file(args.config),
        "patch_index_sha256": sha256_file(args.patch_index),
        "checkpoint_immutable": True,
        "heldout_used": False,
        "target_count": len(records),
        "all_targets_top5": all(len(record["candidates"]) == 5 for record in records.values()),
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
        "verified_inputs": verified,
        "source_snapshot": snapshot,
    }
    if formal_report is not None:
        summary.update(
            {
                "accepted": bool(formal_report["accepted"]),
                "delta_sha256": formal_report["selection"]["combined_delta"]["sha256"],
                "delta_rows": formal_report["selection"]["combined_delta"]["rows"],
                "selected_donors": {
                    role: record["winner"]["patch_id"]
                    for role, record in formal_report["selection"]["targets"].items()
                },
                "heldout_mean": formal_report["heldout"]["mean"],
                "acceptance": formal_report["acceptance"],
            }
        )
    atomic_json(args.run_dir / "summary.json", summary)
    atomic_json(
        args.run_dir / "status.json",
        {
            "state": "completed",
            "phase": args.phase,
            "elapsed_seconds": elapsed,
            "summary_sha256": sha256_file(args.run_dir / "summary.json"),
            "anchor_preflight_sha256": sha256_file(artifacts / "anchor_preflight.json"),
            **(
                {
                    "acceptance_sha256": sha256_file(artifacts / "acceptance.json"),
                    "selection_sha256": sha256_file(artifacts / "selection.json"),
                }
                if formal_report is not None
                else {}
            ),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
