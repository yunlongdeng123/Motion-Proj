#!/usr/bin/env python3
"""从冻结 StreetGS RigidNodes 枚举全部 actor 与 lifecycle。"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    run_root = checkpoint.parent
    sys.path.insert(0, str(args.repo_root.resolve()))
    sys.path.insert(0, str(run_root / "backup"))
    sys.path.append(str(args.upstream_root.resolve()))

    import torch
    from datasets.driving_dataset import DrivingDataset
    from omegaconf import OmegaConf
    from utils.misc import import_str

    checkpoint_before = _sha256(checkpoint)
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
    point_ids = rigid.point_ids.detach().reshape(-1).to(torch.int64).cpu().numpy()
    if lifecycle.ndim != 2:
        raise RuntimeError(f"instances_fv 不是二维矩阵：{lifecycle.shape}")
    np.save(output / "ACTOR_LIFECYCLE_MATRIX.npy", lifecycle, allow_pickle=False)
    actors = []
    for actor_index in range(int(lifecycle.shape[1])):
        frame_valid = lifecycle[:, actor_index]
        lifecycle_path = output / f"ACTOR_{actor_index:04d}_FRAME_VALID.npy"
        np.save(lifecycle_path, frame_valid, allow_pickle=False)
        transitions = np.nonzero(frame_valid[1:] != frame_valid[:-1])[0] + 1
        actors.append({
            "actor_model_index": actor_index,
            "primitive_count": int((point_ids == actor_index).sum()),
            "active_frame_count": int(frame_valid.sum()),
            "inactive_frame_count": int((~frame_valid).sum()),
            "transition_indices": transitions.astype(int).tolist(),
            "first_active_frame": int(np.flatnonzero(frame_valid)[0]) if frame_valid.any() else None,
            "last_active_frame": int(np.flatnonzero(frame_valid)[-1]) if frame_valid.any() else None,
            "lifecycle_path": lifecycle_path.name,
            "lifecycle_sha256": _sha256(lifecycle_path),
        })
    valid_point_ids = (point_ids >= 0) & (point_ids < lifecycle.shape[1])
    _write_json(output / "ACTOR_INVENTORY.json", {
        "schema_version": "worldsim_v6.r54_native_rigid_actor_inventory.v1",
        "frame_count": int(lifecycle.shape[0]),
        "actor_model_count": int(lifecycle.shape[1]),
        "primitive_count": int(point_ids.size),
        "assigned_primitive_count": int(valid_point_ids.sum()),
        "out_of_range_point_id_count": int((~valid_point_ids).sum()),
        "unique_point_ids": np.unique(point_ids).astype(int).tolist(),
        "lifecycle_matrix_path": "ACTOR_LIFECYCLE_MATRIX.npy",
        "lifecycle_matrix_sha256": _sha256(output / "ACTOR_LIFECYCLE_MATRIX.npy"),
        "actors": actors,
    })
    _write_json(output / "WORKER_AUDIT.json", {
        "schema_version": "worldsim_v6.r54_inventory_worker_audit.v1",
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": _sha256(checkpoint),
        "upstream_commit": subprocess.check_output(["git", "-C", str(args.upstream_root.resolve()), "rev-parse", "HEAD"], text=True).strip(),
        "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "wall_seconds": time.monotonic() - started,
        "training_started": False,
        "confirmation_content_read": False,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
