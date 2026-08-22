#!/usr/bin/env python3
"""Render logged and aggregate-actor-translated AD-GS sensor pairs for R134."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint_hashes(model_root: Path) -> dict[str, str]:
    root = model_root / "point_cloud/iteration_60000"
    return {path.name: _sha256(path) for path in sorted(root.glob("*")) if path.is_file()}


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _rgb_uint8(outputs: dict[str, Any]) -> np.ndarray:
    image = outputs["render"].detach().cpu().numpy()
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    return np.rint(np.clip(image, 0.0, 1.0) * 255.0).astype(np.uint8)


def _dynamic_pixels(outputs: dict[str, Any]) -> int:
    semantic = outputs.get("img_semantic")
    if semantic is None:
        raise RuntimeError("AD-GS worker did not return object semantic raster")
    return int((np.squeeze(semantic.detach().cpu().numpy()) > 0.1).sum())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--translation-world", required=True)
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",")]
    translation = np.asarray(
        [float(value) for value in args.translation_world.split(",")], dtype=np.float32
    )
    if translation.shape != (3,):
        raise RuntimeError("translation-world must contain exactly three values")
    source_root = args.source_root.resolve()
    model_root = args.model_root.resolve()
    adapter = args.adapter.resolve()
    output = args.output.resolve()
    sensors = output / "sensors"
    sensors.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source_root))

    import torch
    from gaussian_renderer import render
    from scene import Scene
    from scene.env import EnvironmentMap
    from scene.gaussian_model import GaussianModel
    from utils.general_utils import safe_state

    checkpoint_before = _checkpoint_hashes(model_root)
    model_args = eval(
        (model_root / "cfg_args").read_text(encoding="utf-8"),
        {"Namespace": Namespace},
    )
    model_args.source_path = str(adapter)
    model_args.model_path = str(model_root)
    model_args.data_device = "cuda:0"
    safe_state(True)
    torch.cuda.set_device(0)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    gaussians = GaussianModel(model_args.sh_degree, model_args.order_args)
    environment = EnvironmentMap(**model_args.env_args)
    scene = Scene(model_args, gaussians, environment, load_iteration=60000, shuffle=False)
    pipeline = SimpleNamespace(inv_depth=True, debug=False)
    views = {
        (int(round(float(view.fid))), int(view.cam_id)): view for view in scene.getTestCameras()
    }
    delta = torch.tensor(translation, dtype=gaussians._obj_xyz.dtype, device="cuda")[None, :]
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for frame in frames:
            view = views.get((frame, 0))
            if view is None:
                raise RuntimeError(f"AD-GS adapter missing frame={frame}, camera=0")
            view.cuda()
            view.time = frame / 195.0
            logged = render(view, gaussians, environment, pipeline, render_objmask=True)
            xyz_before = gaussians._obj_xyz.detach().clone()
            gaussians._obj_xyz.add_(delta)
            edited = render(view, gaussians, environment, pipeline, render_objmask=True)
            gaussians._obj_xyz.copy_(xyz_before)
            state_restored = bool(torch.equal(gaussians._obj_xyz, xyz_before))
            logged_rgb = _rgb_uint8(logged)
            edited_rgb = _rgb_uint8(edited)
            if logged_rgb.shape != edited_rgb.shape or logged_rgb.ndim != 3:
                raise RuntimeError("AD-GS logged/edited RGB shape drift")
            changed = int(np.any(logged_rgb != edited_rgb, axis=2).sum())
            relative = f"sensors/frame{frame:03d}.npz"
            path = output / relative
            np.savez_compressed(
                path,
                logged_rgb=logged_rgb,
                compiled_rgb=edited_rgb,
            )
            rows.append(
                {
                    "frame_index": frame,
                    "sensor_path": relative,
                    "sensor_sha256": _sha256(path),
                    "image_shape": list(logged_rgb.shape),
                    "edited_vs_logged_rgb_changed_pixels": changed,
                    "logged_dynamic_pixels": _dynamic_pixels(logged),
                    "edited_dynamic_pixels": _dynamic_pixels(edited),
                    "translation_world_m": translation.astype(float).tolist(),
                    "aggregate_actor_state_restored_exact": state_restored,
                }
            )
    checkpoint_after = _checkpoint_hashes(model_root)
    _write_jsonl(output / "FRAME_METRICS.jsonl", rows)
    _write_json(
        output / "WORKER_AUDIT.json",
        {
            "schema_version": "worldsim_v6.r134_adgs_sensor_worker.v1",
            "frame_count": len(rows),
            "translation_world_m": translation.astype(float).tolist(),
            "checkpoint_sha256_before": checkpoint_before,
            "checkpoint_sha256_after": checkpoint_after,
            "checkpoint_immutable": checkpoint_before == checkpoint_after,
            "all_actor_state_restored_exact": all(
                row["aggregate_actor_state_restored_exact"] for row in rows
            ),
            "wall_seconds": time.monotonic() - started,
            "peak_gpu_memory_mib": float(torch.cuda.max_memory_allocated() / (1024**2)),
            "training_started": False,
            "confirmation_content_read": False,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
