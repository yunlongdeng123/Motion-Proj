#!/usr/bin/env python3
"""用 checkpoint 自带 source backup 串行重渲染 StreetGS Discovery views。"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--shadow-root", required=True, type=Path)
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--partition", choices=("discovery", "confirmation"), default="discovery")
    parser.add_argument(
        "--upstream-root",
        type=Path,
        default=Path("/root/autodl-tmp/third_party/drivestudio-worldsim-v4-b0"),
    )
    args = parser.parse_args()
    checkpoint = args.checkpoint.resolve()
    log_dir = checkpoint.parent
    backup = log_dir / "backup"
    if not backup.is_dir():
        raise RuntimeError(f"checkpoint source backup 缺失：{backup}")
    sys.path.insert(0, str(backup))
    # checkpoint backup 不包含 large third_party/；只把训练时同 commit checkout
    # 作为缺失模块 fallback，checkpoint 自带源码仍保持最高 import 优先级。
    sys.path.append(str(args.upstream_root.resolve()))

    import torch
    from datasets.driving_dataset import DrivingDataset
    from omegaconf import OmegaConf
    from utils.misc import import_str

    rows = [json.loads(line) for line in args.records.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows or any(row.get("partition") != args.partition for row in rows):
        raise RuntimeError(f"records 必须为非空 {args.partition}-only")
    args.output.mkdir(parents=True, exist_ok=False)
    before = sha256_file(checkpoint)
    cfg = OmegaConf.load(log_dir / "config.yaml")
    cfg.data.data_root = str(args.shadow_root.resolve())
    cfg.data.preload_device = "cpu"
    torch.manual_seed(int(cfg.seed))
    torch.cuda.manual_seed_all(int(cfg.seed))
    device = torch.device("cuda")
    load_start = time.monotonic()
    dataset = DrivingDataset(data_cfg=cfg.data)
    trainer = import_str(cfg.trainer.type)(
        **cfg.trainer,
        num_timesteps=dataset.num_img_timesteps,
        model_config=cfg.model,
        num_train_images=len(dataset.train_image_set),
        num_full_images=len(dataset.full_image_set),
        test_set_indices=dataset.test_timesteps,
        scene_aabb=dataset.get_aabb().reshape(2, 3),
        device=device,
    )
    trainer.resume_from_checkpoint(ckpt_path=str(checkpoint), load_only_model=True)
    trainer.set_eval()
    load_seconds = time.monotonic() - load_start
    render_rows = []
    camera_downscale = trainer._get_downscale_factor()
    with torch.inference_mode():
        for sequence, row in enumerate(rows):
            start = time.monotonic()
            full_index = int(row["frame"]) * 3 + int(row["camera"])
            image_infos, cam_infos = dataset.full_image_set.get_image(full_index, camera_downscale)
            for values in (image_infos, cam_infos):
                for key, value in values.items():
                    if isinstance(value, torch.Tensor):
                        values[key] = value.cuda(non_blocking=True)
            result = trainer(image_infos, cam_infos)
            rgb = result["rgb"].detach().clamp(0.0, 1.0).cpu().numpy()
            if rgb.shape[0] == 3:
                rgb = np.transpose(rgb, (1, 2, 0))
            encoded = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)
            output = args.output / f"{sequence:05d}.png"
            Image.fromarray(encoded, mode="RGB").save(output)
            render_rows.append(
                {
                    "sequence": sequence,
                    "scene": row["scene"],
                    "frame": int(row["frame"]),
                    "camera": int(row["camera"]),
                    "partition": args.partition,
                    "prediction_path": str(output.resolve()),
                    "prediction_sha256": sha256_file(output),
                    "render_seconds": time.monotonic() - start,
                }
            )
    after = sha256_file(checkpoint)
    if before != after:
        raise RuntimeError("checkpoint before/after SHA 漂移")
    (args.output / "RENDER_MAP.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in render_rows), encoding="utf-8"
    )
    audit = {
        "schema": "worldsim_v521_streetgs_render_v1",
        "checkpoint": str(checkpoint),
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after,
        "source_backup": str(backup),
        "third_party_fallback": str(args.upstream_root.resolve()),
        "third_party_commit": subprocess.check_output(
            ["git", "-C", str(args.upstream_root.resolve()), "rev-parse", "HEAD"], text=True
        ).strip(),
        "third_party_diff_sha256": __import__("hashlib").sha256(
            subprocess.check_output(
                ["git", "-C", str(args.upstream_root.resolve()), "diff", "--binary"]
            )
        ).hexdigest(),
        "config": str((log_dir / "config.yaml").resolve()),
        "config_sha256": sha256_file(log_dir / "config.yaml"),
        "shadow_root": str(args.shadow_root.resolve()),
        "quality_partition": args.partition,
        "confirmation_original_pixels_decoded": len(render_rows) if args.partition == "confirmation" else 0,
        "views": len(render_rows),
        "load_seconds": load_seconds,
        "peak_torch_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_torch_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }
    (args.output / "RENDER_AUDIT.json").write_text(json.dumps(audit, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
