#!/usr/bin/env python
"""将 train-only SAM2 masks 深度一致地提升为只读 D2 Gaussian sidecar。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
import torch
import yaml

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from motion_proj.resim.drivestudio_adapter import gsplat_first_hit_from_info
from motion_proj.worldsim_v32.semantic_schema import (
    binary_inner_boundary, classify_gaussians, label_counts, semantic_posterior,
    sha256_file, validate_disjoint_split,
)
from scripts.eval_worldsim_v3_a3_r1_heldout import (
    get_view_data, load_model_checkpoint_read_only, release_trainer_render_info,
)
from scripts.materialize_worldsim_v3_a3_s_b_sidecar import render_variant


def atomic_json(path: Path, payload: object) -> None:
    tmp = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + f".partial.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def build_runtime(config: dict[str, Any], device: torch.device):
    root = Path(config["runtimes"]["drivestudio_checkout"])
    sys.path.insert(0, str(root))
    from datasets.driving_dataset import DrivingDataset
    from utils.misc import import_str

    source = OmegaConf.load(config["inputs"]["source_config"])
    dataset = DrivingDataset(data_cfg=source.data)
    trainer = import_str(source.trainer.type)(
        **source.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=source.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    if hasattr(trainer, "optimizer"):
        raise RuntimeError("S1 必须构造无 optimizer 的只读 trainer")
    load_model_checkpoint_read_only(trainer, Path(config["inputs"]["checkpoint"]), device)
    trainer.set_eval()
    return dataset, trainer


def intersections(info: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """返回按 ray 排序的 Gaussian、pixel、T_before*alpha 与投影深度。"""
    from gsplat.cuda._wrapper import rasterize_to_indices_in_range

    h, w = int(info["height"]), int(info["width"])
    means2d = info["means2d"]
    trans = torch.ones((int(info["n_cameras"]), h, w), device=means2d.device, dtype=means2d.dtype)
    gids, pids, cids = rasterize_to_indices_in_range(
        0, 2**31 - 1, trans, means2d, info["conics"], info["opacities"],
        w, h, int(info["tile_size"]), info["isect_offsets"], info["flatten_ids"],
    )
    if gids.numel() == 0:
        empty_i = np.empty(0, np.int64)
        return empty_i, empty_i, np.empty(0, np.float64), np.empty(0, np.float32)
    rays = cids * (h * w) + pids
    order = torch.argsort(rays, stable=True)
    gids, pids, cids, rays = gids[order], pids[order], cids[order], rays[order]
    projected = means2d[cids, gids]
    conics, opacity = info["conics"][cids, gids], info["opacities"][cids, gids]
    x = pids.remainder(w).to(projected.dtype) + 0.5
    y = torch.div(pids, w, rounding_mode="floor").to(projected.dtype) + 0.5
    dx, dy = projected[:, 0] - x, projected[:, 1] - y
    sigma = 0.5 * (conics[:, 0] * dx.square() + conics[:, 2] * dy.square()) + conics[:, 1] * dx * dy
    alpha = torch.minimum(opacity * torch.exp(-sigma), opacity.new_tensor(0.999)).double()
    log_survival = torch.log1p(-alpha)
    prefix = torch.cumsum(log_survival, 0)
    starts = torch.ones_like(rays, dtype=torch.bool)
    starts[1:] = rays[1:] != rays[:-1]
    starts_at = torch.nonzero(starts, as_tuple=False).flatten()
    bases = torch.zeros(starts_at.numel(), device=means2d.device, dtype=torch.float64)
    if starts_at.numel() > 1:
        bases[1:] = prefix[starts_at[1:] - 1]
    segments = torch.cumsum(starts.long(), 0) - 1
    weights = torch.exp(prefix - log_survival - bases[segments]) * alpha
    depths = info["depths"]
    if depths.ndim == 3:
        depths = depths[..., 0]
    if depths.ndim == 2:
        values = depths[cids, gids]
    else:
        values = depths[gids]
    return (
        gids.cpu().numpy().astype(np.int64), pids.cpu().numpy().astype(np.int64),
        weights.cpu().numpy().astype(np.float64), values.detach().cpu().numpy().astype(np.float32),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    masks = json.loads(args.mask_manifest.read_text(encoding="utf-8"))
    heldout = [int(v) for v in masks["heldout_frames"]]
    rows = masks["masks"]
    validate_disjoint_split(sorted({int(r["frame"]) for r in rows}), heldout)
    if not masks.get("heldout_excluded") or any(int(r["frame"]) in heldout for r in rows):
        raise RuntimeError("mask manifest 存在 heldout 泄漏")
    source = Path(config["inputs"]["checkpoint"])
    checkpoint_before = sha256_file(source)
    if checkpoint_before != config["inputs"]["checkpoint_sha256"]:
        raise RuntimeError("D2 checkpoint SHA 漂移")
    if not torch.cuda.is_available():
        raise RuntimeError("S1 semantic lift 需要可见 CUDA GPU")
    device = torch.device(args.device)
    started = time.monotonic()
    torch.cuda.set_device(device)
    torch.cuda.reset_peak_memory_stats(device)
    dataset, trainer = build_runtime(config, device)
    counts = {name: int(trainer.models[name]._means.shape[0]) for name in ("Background", "RigidNodes")}
    bg_count, total = counts["Background"], sum(counts.values())
    point_ids = trainer.models["RigidNodes"].point_ids[:, 0].detach().cpu().numpy().astype(np.int64)
    accum = {
        role: {
            "projected": np.zeros(total),
            "visible": np.zeros(total),
            "semantic": np.zeros(total),
            "boundary": np.zeros(total),
            "visible_views": np.zeros(total, np.int32),
            "positive_views": np.zeros(total, np.int32),
        }
        for role in config["actors"]
    }
    grouped: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        if sha256_file(Path(row["mask"])) != row["mask_sha256"]:
            raise RuntimeError(f"mask SHA 漂移: {row['mask']}")
        grouped.setdefault((int(row["frame"]), int(row["camera_id"])), []).append(row)
    lift_cfg = config["lift"]
    ordered_views = sorted(grouped.items())
    for view_index, ((frame, camera), view_rows) in enumerate(ordered_views, start=1):
        image_infos, camera_infos, *_ = get_view_data(dataset, frame, camera, device)
        try:
            with torch.inference_mode():
                trainer(image_infos, camera_infos)
            gids, pids, weights, depths = intersections(trainer.info)
            first, valid = gsplat_first_hit_from_info(trainer.info, alpha_threshold=float(lift_cfg["first_hit_alpha_threshold"]))
            first_flat, valid_flat = np.asarray(first).reshape(-1), np.asarray(valid).reshape(-1).astype(bool)
            tol = np.maximum(float(lift_cfg["depth_absolute_tolerance_m"]), float(lift_cfg["depth_relative_tolerance"]) * np.abs(first_flat[pids]))
            consistent = valid_flat[pids] & np.isfinite(depths) & (np.abs(depths - first_flat[pids]) <= tol)
            good_gids, good_pids, good_weights = gids[consistent], pids[consistent], weights[consistent]
            # bincount 对大量重复 Gaussian 索引远快于逐元素 np.add.at。
            projected = np.bincount(
                gids, weights=weights, minlength=total
            ).astype(np.float64, copy=False)
            visible = np.bincount(
                good_gids, weights=good_weights, minlength=total
            ).astype(np.float64, copy=False)
            roles_in_view = {row["role"] for row in view_rows}
            for role in roles_in_view:
                accum[role]["projected"] += projected
                accum[role]["visible"] += visible
                accum[role]["visible_views"] += (visible > 0).astype(np.int32)
            for row in view_rows:
                binary_image = np.load(
                    row["mask"], allow_pickle=False
                )["binary"].astype(bool)
                binary = binary_image.reshape(-1)
                boundary = binary_inner_boundary(binary_image).reshape(-1)
                selected = binary[good_pids]
                boundary_selected = boundary[good_pids]
                semantic = np.bincount(
                    good_gids[selected],
                    weights=good_weights[selected],
                    minlength=total,
                ).astype(np.float64, copy=False)
                boundary_mass = np.bincount(
                    good_gids[boundary_selected],
                    weights=good_weights[boundary_selected],
                    minlength=total,
                ).astype(np.float64, copy=False)
                role = row["role"]
                accum[role]["semantic"] += semantic
                accum[role]["boundary"] += boundary_mass
                accum[role]["positive_views"] += (semantic > 0).astype(np.int32)
        finally:
            release_trainer_render_info(trainer)
        if view_index == 1 or view_index % 10 == 0 or view_index == len(ordered_views):
            print(
                f"S1 lift progress: {view_index}/{len(ordered_views)} "
                f"frame={frame} camera={camera}",
                flush=True,
            )

    actor_outputs = {}
    for role, actor in config["actors"].items():
        values = accum[role]
        posterior = semantic_posterior(values["semantic"], values["visible"])
        depth_consistency_rate = semantic_posterior(
            values["visible"], values["projected"]
        )
        boundary_score = semantic_posterior(
            values["boundary"], values["semantic"]
        )
        negative_views = np.maximum(
            values["visible_views"] - values["positive_views"], 0
        ).astype(np.int32)
        core = np.zeros(total, dtype=bool)
        core[bg_count:] = point_ids == int(actor["rigid_model_index"])
        labels = classify_gaussians(
            posterior=posterior, semantic_mass=values["semantic"], positive_view_count=values["positive_views"], core_mask=core,
            semantic_threshold=float(lift_cfg["semantic_positive_posterior"]),
            ambiguous_threshold=float(lift_cfg["ambiguous_posterior_low"]),
            minimum_semantic_mass=float(lift_cfg["minimum_visible_mass"]),
            minimum_positive_views=int(lift_cfg["minimum_positive_views"]),
        )
        output = args.output_dir / f"{role}.npz"
        atomic_npz(
            output,
            labels=labels,
            semantic_score=posterior,
            posterior=posterior,
            weighted_positive=values["semantic"],
            semantic_mass=values["semantic"],
            weighted_total=values["visible"],
            visible_mass=values["visible"],
            projected_mass=values["projected"],
            num_positive_views=values["positive_views"],
            positive_view_count=values["positive_views"],
            num_negative_views=negative_views,
            visible_view_count=values["visible_views"],
            depth_consistency_rate=depth_consistency_rate,
            boundary_score=boundary_score,
            background_count=np.asarray(bg_count),
            rigid_point_ids=point_ids,
        )
        actor_outputs[role] = {"path": str(output), "sha256": sha256_file(output), "label_counts": label_counts(labels), "core_count": int(core.sum())}

    smoke_dir = args.output_dir / "smoke"
    smoke_dir.mkdir()
    smoke = []
    for role, actor in config["actors"].items():
        candidate = next(row for row in rows if row["role"] == role and int(row["positive_pixels"]) > 0)
        for variant in ("original", "delete", "lateral"):
            rendered = render_variant(trainer=trainer, dataset=dataset, checkpoint=source, frame=int(candidate["frame"]), camera=int(candidate["camera_id"]), model_index=int(actor["rigid_model_index"]), variant=variant, device=device)
            path = smoke_dir / f"{role}__f{int(candidate['frame']):03d}__c{int(candidate['camera_id'])}__{variant}.png"
            imageio.imwrite(path, rendered["rgb"])
            smoke.append({"role": role, "frame": int(candidate["frame"]), "camera_id": int(candidate["camera_id"]), "variant": variant, "path": str(path), "sha256": sha256_file(path)})
            release_trainer_render_info(trainer)
    checkpoint_after = sha256_file(source)
    if checkpoint_after != checkpoint_before:
        raise RuntimeError("D2 checkpoint 在 S1 后发生 mutation")
    manifest = {
        "schema_version": "worldsim_v32_s1_gaussian_sidecar_v1",
        "task_id": config["task_id"],
        "sidecar_only": True,
        "heldout_excluded": True,
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "gaussian_counts": counts,
        "posterior_fields": {
            "semantic_score": "weighted_positive / weighted_total",
            "depth_consistency_rate": "depth-consistent contribution / projected contribution",
            "boundary_score": "inner-boundary contribution / weighted_positive",
        },
        "runtime": {
            "wall_seconds": time.monotonic() - started,
            "cuda_device": torch.cuda.get_device_name(device),
            "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
            "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        },
        "actors": actor_outputs,
        "smoke": smoke,
    }
    atomic_json(args.output_dir / "semantic_manifest.json", manifest)
    print(json.dumps({"status": "done", "manifest": str(args.output_dir / "semantic_manifest.json"), "actors": actor_outputs}, ensure_ascii=False))


if __name__ == "__main__":
    main()
