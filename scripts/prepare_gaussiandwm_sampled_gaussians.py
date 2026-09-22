#!/usr/bin/env python3
"""把 GaussianDWM sampled 裸 Tensor 包装成 CVPR loader 可读的 payload。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def _load_tensor(path: Path) -> torch.Tensor:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and "packed" in payload:
        payload = payload["packed"]
    if not isinstance(payload, torch.Tensor):
        raise TypeError(f"expected Tensor or packed Tensor dict: {path}")
    if payload.ndim != 2 or payload.shape[1] != 14:
        raise ValueError(f"expected [N,14], got {tuple(payload.shape)}: {path}")
    return payload.detach().cpu().contiguous()


def _view_name(directory: Path) -> str:
    marker = "_CAM_"
    if marker not in directory.name:
        raise ValueError(f"unexpected sampled view directory: {directory}")
    return "CAM_" + directory.name.split(marker, 1)[1]


def convert(source_root: Path, target_root: Path) -> dict[str, Any]:
    view_rows = []
    for view_root in sorted(path for path in source_root.iterdir() if path.is_dir()):
        source_files = sorted((view_root / "per_frame").glob("*.pt"))
        if not source_files:
            continue
        target_view = target_root / view_root.name / "per_frame"
        target_view.mkdir(parents=True, exist_ok=True)
        row_count = None
        for source in source_files:
            target = target_view / source.name
            if target.is_file():
                tensor = _load_tensor(target)
            else:
                tensor = _load_tensor(source)
                torch.save({"packed": tensor}, target)
            if row_count is None:
                row_count = int(tensor.shape[0])
        view_rows.append(
            {
                "source_directory": str(view_root.resolve()),
                "compatible_directory": str(target_view.parent.resolve()),
                "view": _view_name(view_root),
                "frame_count": len(source_files),
                "gaussians_per_frame": row_count,
                "feature_width": 14,
            }
        )
    if not view_rows:
        raise RuntimeError(f"no sampled Gaussian files found under {source_root}")
    return {
        "schema_version": "worldsim_v75_gaussiandwm_sampled_adapter_v1",
        "gpu_operations_run": False,
        "reason": "official sampled files are bare torch.Tensor payloads while the CVPR loader expects ndarray or mapping payloads",
        "source_root": str(source_root.resolve()),
        "compatible_root": str(target_root.resolve()),
        "view_count": len(view_rows),
        "frame_file_count": sum(row["frame_count"] for row in view_rows),
        "views": view_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    args = parser.parse_args()
    artifact = convert(args.source_root.resolve(), args.target_root.resolve())
    args.inventory.parent.mkdir(parents=True, exist_ok=True)
    args.inventory.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "views": artifact["view_count"],
                "frame_files": artifact["frame_file_count"],
                "compatible_root": artifact["compatible_root"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
