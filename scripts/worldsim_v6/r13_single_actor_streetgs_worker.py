#!/usr/bin/env python3
"""在冻结 StreetGS checkpoint 上渲染一个预注册 actor 的删除结果。"""

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


def _numpy(value):
    return value.detach().cpu().numpy()


def _save_render(path: Path, outputs: dict) -> None:
    np.savez_compressed(
        path,
        rgb=_numpy(outputs["rgb"]).astype(np.float16),
        depth=_numpy(outputs["depth"]).astype(np.float32),
        opacity=_numpy(outputs["opacity"]).astype(np.float16),
        dynamic_opacity=_numpy(outputs["Dynamic_opacity"]).astype(np.float16),
        dynamic_depth=_numpy(outputs["Dynamic_depth"]).astype(np.float32),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--upstream-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", required=True)
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",")]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.checkpoint.resolve()
    run_root = checkpoint.parent
    backup = run_root / "backup"
    sys.path.insert(0, str(args.repo_root.resolve()))
    sys.path.insert(0, str(backup))
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
    camera_downscale = trainer._get_downscale_factor()
    rigid = trainer.models["RigidNodes"]
    point_ids = rigid.point_ids.detach().reshape(-1).to(torch.int64)
    unique_ids, counts = torch.unique(point_ids, sorted=True, return_counts=True)
    visibility = rigid.instances_fv.detach().bool()
    eligible = []
    for model_index, count in zip(unique_ids.tolist(), counts.tolist()):
        model_index = int(model_index)
        if model_index < 0 or model_index >= visibility.shape[1]:
            continue
        if all(bool(visibility[frame, model_index].item()) for frame in frames):
            eligible.append((int(count), model_index))
    if not eligible:
        raise RuntimeError("没有在两帧均可见的 StreetGS actor model index")
    selected_count, selected_index = max(eligible, key=lambda row: (row[0], -row[1]))
    selected_mask = point_ids == selected_index
    selection = {
        "schema_version": "worldsim_v6.r13_single_actor_selection.v1",
        "selected_model_index": selected_index,
        "selected_gaussian_count": selected_count,
        "eligible_actor_count": len(eligible),
        "frame_indices": frames,
        "selection_rule": "visible_in_both_frames_maximum_gaussian_count_then_smallest_index",
    }
    (output / "ACTOR_SELECTION.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    rows = []
    with torch.inference_mode():
        for frame in frames:
            image_infos, camera_infos = dataset.full_image_set.get_image(
                frame * 3, camera_downscale
            )
            for values in (image_infos, camera_infos):
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.cuda(non_blocking=True)
            logged = trainer(image_infos, camera_infos)
            logged_name = f"frame{frame:03d}_logged.npz"
            _save_render(output / logged_name, logged)
            rows.append(
                {
                    "frame_index": frame,
                    "state": "logged",
                    "path": logged_name,
                    "sha256": _sha256(output / logged_name),
                }
            )
            opacity_before = rigid._opacities.detach().clone()
            rigid._opacities[selected_mask] = -100.0
            edited = trainer(image_infos, camera_infos)
            rigid._opacities.copy_(opacity_before)
            edited_name = f"frame{frame:03d}_single_actor_removed.npz"
            _save_render(output / edited_name, edited)
            rows.append(
                {
                    "frame_index": frame,
                    "state": "edited",
                    "path": edited_name,
                    "sha256": _sha256(output / edited_name),
                }
            )
    checkpoint_after = _sha256(checkpoint)
    if checkpoint_before != checkpoint_after:
        raise RuntimeError("StreetGS checkpoint before/after SHA 漂移")
    (output / "RENDER_MAP.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "worldsim_v6.r13_single_actor_streetgs_worker.v1",
        "checkpoint_sha256_before": checkpoint_before,
        "checkpoint_sha256_after": checkpoint_after,
        "source_backup": str(backup),
        "source_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream_root.resolve()), "rev-parse", "HEAD"], text=True
        ).strip(),
        "render_count": len(rows),
        "training_started": False,
        "confirmation_content_read": False,
        "wall_seconds": time.monotonic() - started,
        "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    (output / "AUDIT.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
