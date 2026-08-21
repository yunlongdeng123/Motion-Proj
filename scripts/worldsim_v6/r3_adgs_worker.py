#!/usr/bin/env python3
"""在冻结 AD-GS checkpoint 上串行渲染 R3 偏移与 actor 编辑。"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_hashes(model_root: Path) -> dict[str, str]:
    root = model_root / "point_cloud/iteration_60000"
    return {path.name: sha256_file(path) for path in sorted(root.glob("*")) if path.is_file()}


def tensor_numpy(value):
    return value.detach().cpu().numpy()


def save_render(path: Path, outputs: dict) -> None:
    semantic = outputs.get("img_semantic")
    if semantic is None:
        raise RuntimeError("AD-GS worker 未得到 object semantic raster")
    np.savez_compressed(
        path,
        rgb=np.transpose(tensor_numpy(outputs["render"]), (1, 2, 0)).astype(np.float16),
        depth=tensor_numpy(outputs["depth"]).astype(np.float32),
        opacity=tensor_numpy(outputs["img_opacity"]).astype(np.float16),
        dynamic_opacity=np.squeeze(tensor_numpy(semantic)).astype(np.float16),
    )


def shifted_camera(view, lateral_offset: float, forward_offset: float = 0.0):
    import torch
    from utils.graphics_utils import getWorld2View2

    novel = copy.deepcopy(view)
    world_to_camera = np.eye(4, dtype=np.float32)
    world_to_camera[:3, :3] = np.asarray(view.R)
    world_to_camera[:3, 3] = np.asarray(view.T)
    camera_to_world = np.linalg.inv(world_to_camera)
    camera_to_world[:3, 3] += camera_to_world[:3, 0] * lateral_offset
    camera_to_world[:3, 3] += camera_to_world[:3, 2] * forward_offset
    new_world_to_camera = np.linalg.inv(camera_to_world)
    novel.R = new_world_to_camera[:3, :3].astype(np.float32)
    novel.T = new_world_to_camera[:3, 3].astype(np.float32)
    novel.world_view_transform = torch.tensor(
        getWorld2View2(novel.R, novel.T, novel.trans, novel.scale),
        dtype=torch.float32,
        device="cuda",
    ).transpose(0, 1)
    novel.full_proj_transform = novel.world_view_transform.unsqueeze(0).bmm(
        novel.projection_matrix.unsqueeze(0)
    ).squeeze(0)
    novel.camera_center = novel.world_view_transform.inverse()[3, :3]
    return novel, camera_to_world[:3, 0].astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--adapter", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--frames", required=True)
    parser.add_argument("--offsets", required=True)
    parser.add_argument("--forward-extension", required=True, type=float)
    args = parser.parse_args()
    frames = [int(value) for value in args.frames.split(",")]
    offsets = [float(value) for value in args.offsets.split(",")]
    source_root = args.source_root.resolve()
    model_root = args.model_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(source_root))

    import torch
    from gaussian_renderer import render
    from scene import Scene
    from scene.env import EnvironmentMap
    from scene.gaussian_model import GaussianModel
    from utils.func_utils import get_func_result
    from utils.general_utils import safe_state

    before = checkpoint_hashes(model_root)
    model_args = eval(
        (model_root / "cfg_args").read_text(encoding="utf-8"),
        {"Namespace": Namespace},
    )
    model_args.source_path = str(args.adapter.resolve())
    model_args.model_path = str(model_root)
    model_args.data_device = "cuda:0"
    safe_state(True)
    torch.cuda.set_device(0)
    started = time.monotonic()
    gaussians = GaussianModel(model_args.sh_degree, model_args.order_args)
    environment = EnvironmentMap(**model_args.env_args)
    scene = Scene(model_args, gaussians, environment, load_iteration=60000, shuffle=False)
    pipeline = SimpleNamespace(inv_depth=True, debug=False)
    views = {
        (int(round(float(view.fid))), int(view.cam_id)): view
        for view in scene.getTestCameras()
    }
    rows = []
    with torch.inference_mode():
        for frame_index in frames:
            base = views.get((frame_index, 0))
            if base is None:
                raise RuntimeError(f"AD-GS development adapter 缺少 frame={frame_index}, camera=0")
            # development-only adapter 的本地归一化不能代替训练时 0..195 时间轴。
            base.time = frame_index / 195.0
            for lateral_offset in offsets:
                novel, _ = shifted_camera(base, lateral_offset)
                result = render(novel, gaussians, environment, pipeline, render_objmask=True)
                name = f"frame{frame_index:03d}_lat{lateral_offset:g}m.npz"
                save_render(output / name, result)
                rows.append(
                    {
                        "scene": args.scene,
                        "frontend": "ad_gs",
                        "frame_index": frame_index,
                        "camera_id": 0,
                        "variant": "camera_lateral",
                        "lateral_offset_m": lateral_offset,
                        "path": name,
                        "sha256": sha256_file(output / name),
                    }
                )

            novel, _ = shifted_camera(base, 0.0, args.forward_extension)
            result = render(novel, gaussians, environment, pipeline, render_objmask=True)
            name = f"frame{frame_index:03d}_fwd{args.forward_extension:g}m.npz"
            save_render(output / name, result)
            rows.append(
                {
                    "scene": args.scene,
                    "frontend": "ad_gs",
                    "frame_index": frame_index,
                    "camera_id": 0,
                    "variant": "camera_forward_extension",
                    "lateral_offset_m": 0.0,
                    "forward_offset_m": args.forward_extension,
                    "path": name,
                    "sha256": sha256_file(output / name),
                }
            )

            logged, right_world = shifted_camera(base, 0.0)
            edits = []
            opacity_before = gaussians._obj_opacity.detach().clone()
            gaussians._obj_opacity.fill_(-100.0)
            edits.append(("actor_remove_all", render(logged, gaussians, environment, pipeline, render_objmask=True)))
            gaussians._obj_opacity.copy_(opacity_before)

            xyz_before = gaussians._obj_xyz.detach().clone()
            gaussians._obj_xyz.add_(torch.tensor(right_world, device="cuda")[None, :])
            edits.append(("actor_translate_all_local_x_1m", render(logged, gaussians, environment, pipeline, render_objmask=True)))
            gaussians._obj_xyz.copy_(xyz_before)

            shifted_time = min(1.0, (frame_index + 2) / 195.0)
            current_delta = get_func_result(logged.time, gaussians.xyz_deform_param, gaussians.order_args["xyz"])
            shifted_delta = get_func_result(shifted_time, gaussians.xyz_deform_param, gaussians.order_args["xyz"])
            gaussians._obj_xyz.add_(shifted_delta - current_delta)
            edits.append(("actor_trajectory_time_shift_plus_2_frames", render(logged, gaussians, environment, pipeline, render_objmask=True)))
            gaussians._obj_xyz.copy_(xyz_before)
            for variant, result in edits:
                name = f"frame{frame_index:03d}_{variant}.npz"
                save_render(output / name, result)
                rows.append(
                    {
                        "scene": args.scene,
                        "frontend": "ad_gs",
                        "frame_index": frame_index,
                        "camera_id": 0,
                        "variant": variant,
                        "lateral_offset_m": 0.0,
                        "path": name,
                        "sha256": sha256_file(output / name),
                    }
                )

    after = checkpoint_hashes(model_root)
    if before != after:
        raise RuntimeError("AD-GS checkpoint bundle before/after SHA 漂移")
    (output / "RENDER_MAP.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    audit = {
        "schema_version": "worldsim_v6.r3_adgs_worker.v1",
        "scene": args.scene,
        "checkpoint_sha256_before": before,
        "checkpoint_sha256_after": after,
        "frames": frames,
        "offsets_m": offsets,
        "forward_extension_m": args.forward_extension,
        "development_content_read": True,
        "confirmation_content_read": False,
        "training_started": False,
        "render_count": len(rows),
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
