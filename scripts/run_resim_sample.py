#!/usr/bin/env python
"""在不修改 ReSim 源码的前提下启用确定性设置并启动官方采样入口。"""
from __future__ import annotations

import argparse
import importlib
import json
import os
import runpy
import sys
from pathlib import Path


def _configure_paths(resim_root: Path) -> Path:
    sat_dir = resim_root / "sat"
    sys.path.insert(0, str(resim_root / "SwissArmyTransformer"))
    sys.path.insert(0, str(sat_dir))
    return sat_dir


def _dataset_preflight(config_path: Path, output_path: Path) -> None:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(str(config_path))
    target = str(cfg.data.target)
    module_name, class_name = target.rsplit(".", 1)
    dataset_class = getattr(importlib.import_module(module_name), class_name)
    params = OmegaConf.to_container(cfg.data.params, resolve=True)
    if "n_subset" in params or "ind_subset" in params:
        raise RuntimeError("精确 manifest 禁止再使用 n_subset/ind_subset")
    data_path = str(cfg.args.valid_data[0])
    dataset = dataset_class(data_dir=data_path, **params)
    if len(dataset) != 1:
        raise RuntimeError(f"精确 manifest 应只有 1 个样本，实际 {len(dataset)}")
    item = dataset[0]
    mp4_shape = list(item["mp4"].shape)
    traj_shape = list(item["fut_traj"].shape)
    expected_frames = 4 * (int(cfg.args.sampling_num_frames) - 1) + 1
    checks = {
        "dataset_length_one": len(dataset) == 1,
        "rgb_frame_count": mp4_shape[0] == expected_frames,
        "video_size": mp4_shape[-2:] == list(cfg.args.sampling_video_size),
        "trajectory_shape": traj_shape == [8, 3],
        "apply_traj": bool(cfg.args.apply_traj),
        "use_ema": bool(cfg.args.use_ema),
        "save_gt_disabled": not bool(cfg.args.save_gt),
        "concat_gt_disabled": not bool(cfg.args.concat_gt_for_demo),
    }
    result = {
        "config": str(config_path.resolve()),
        "dataset_target": target,
        "dataset_length": len(dataset),
        "mp4_shape": mp4_shape,
        "fut_traj_shape": traj_shape,
        "text": str(item["txt"]),
        "lidar_pc_token": str(item["lidar_pc_token"]),
        "checks": checks,
        "passed": all(checks.values()),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not result["passed"]:
        raise RuntimeError(f"dataset preflight 检查失败: {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resim-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--preflight-output", default=None)
    args = parser.parse_args()

    resim_root = Path(args.resim_root).resolve()
    config_path = Path(args.config).resolve()
    sat_dir = _configure_paths(resim_root)
    os.chdir(sat_dir)

    if args.preflight_output:
        _dataset_preflight(config_path, Path(args.preflight_output).resolve())
        return

    import torch

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")

    sample_path = sat_dir / "sample_video.py"
    sys.argv = [str(sample_path), f"--base={config_path}"]
    runpy.run_path(str(sample_path), run_name="__main__")


if __name__ == "__main__":
    main()
