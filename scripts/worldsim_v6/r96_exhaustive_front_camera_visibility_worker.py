#!/usr/bin/env python3
"""Render every lifecycle-active scene0048 actor/front-camera frame pair."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--camera-offset", required=True, type=int)
    parser.add_argument("--opacity-threshold", required=True, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",")]
    candidates = [int(value) for value in args.candidates.split(",")]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    run_root = checkpoint.parent
    sys.path.insert(0, str(args.repo_root.resolve()))
    sys.path.insert(0, str(run_root / "backup"))
    sys.path.append(str(args.upstream_root.resolve()))

    import torch
    from datasets.driving_dataset import DrivingDataset
    from models.gaussians.basics import dataclass_gs
    from omegaconf import OmegaConf
    from utils.misc import import_str

    checkpoint_before = sha256(checkpoint)
    cfg = OmegaConf.load(run_root / "config.yaml")
    cfg.data.preload_device = "cpu"
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=torch.device("cuda"),
    )
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint), load_only_model=True)
    trainer.set_eval()
    rigid = trainer.models["RigidNodes"]
    lifecycle = rigid.instances_fv.detach().cpu().to(torch.bool).numpy()
    point_ids = rigid.point_ids.detach().reshape(-1).to(torch.int64)
    rows = []
    camera_downscale = trainer._get_downscale_factor()
    with torch.inference_mode():
        for frame in frames:
            image_infos, camera_infos = dataset.full_image_set.get_image(
                frame * 3 + args.camera_offset, camera_downscale
            )
            for values in (image_infos, camera_infos):
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.cuda(non_blocking=True)
            trainer(image_infos, camera_infos)
            cam = trainer.process_camera(
                camera_infos=camera_infos,
                image_ids=image_infos["img_idx"].flatten()[0],
                novel_view=False,
            )
            native = rigid.get_gaussians(cam)
            for actor_index in candidates:
                if not bool(lifecycle[frame, actor_index]):
                    continue
                mask = point_ids == actor_index
                actor = dataclass_gs(
                    _means=native["_means"][mask], _scales=native["_scales"][mask],
                    _quats=native["_quats"][mask], _rgbs=native["_rgbs"][mask],
                    _opacities=native["_opacities"][mask], detach_keys=[], extras=None,
                )
                rendered, _ = trainer.render_gaussians(
                    gs=actor, cam=cam, near_plane=trainer.render_cfg.near_plane,
                    far_plane=trainer.render_cfg.far_plane, render_mode="RGB+ED",
                    radius_clip=trainer.render_cfg.get("radius_clip", 0.0),
                )
                opacity = rendered["opacity"].detach().cpu().numpy().squeeze(-1)
                native_opacity = native["_opacities"][mask].detach().cpu().numpy().squeeze(-1)
                rows.append(
                    {
                        "frame_index": frame, "camera_offset": args.camera_offset,
                        "actor_model_index": actor_index, "primitive_count": int(mask.sum().item()),
                        "native_frame_valid": True,
                        "nonzero_opacity_primitives": int((native_opacity > 0).sum()),
                        "actor_effect_pixels": int((opacity > args.opacity_threshold).sum()),
                        "render_opacity_sum": float(opacity.astype(np.float64).sum()),
                        "render_opacity_max": float(opacity.max()),
                    }
                )
    write_json(
        output / "EXHAUSTIVE_VISIBILITY.json",
        {
            "schema_version": "worldsim_v6.r96_exhaustive_front_visibility.v1",
            "frames": frames, "candidates": candidates, "camera_offset": args.camera_offset,
            "opacity_threshold": args.opacity_threshold, "rows": rows,
        },
    )
    write_json(
        output / "WORKER_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r96_worker_audit.v1",
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": sha256(checkpoint),
            "upstream_commit": subprocess.check_output(
                ["git", "-C", str(args.upstream_root.resolve()), "rev-parse", "HEAD"], text=True
            ).strip(),
            "row_count": len(rows),
            "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
            "wall_seconds": time.monotonic() - started,
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
