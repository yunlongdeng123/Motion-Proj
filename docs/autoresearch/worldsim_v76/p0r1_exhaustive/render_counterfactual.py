"""Render VAD-GS with an edited ego camera and a factual actor clock.

Example:
  python script/v76/render_counterfactual.py \
    --config configs/v76/nuscenes_000.yaml --intervention sweep \
    --camera 0 --frame 20 --offsets 0,0.5,1,2,3.5

The VAD-GS config parser consumes sys.argv at import time, so this script
parses its own options first and then passes only the config to VAD-GS.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ego_trajectory import camera_c2w, lane_change, lateral_sweep_pose, pose_delta, road_frame, speed_edit

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def _save_tensor_rgb(tensor: torch.Tensor, path: Path) -> None:
    rgb = (tensor.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).round().astype(np.uint8)
    Image.fromarray(rgb).save(path)


def _actor_graph_snapshot(gaussians):
    """从渲染器实际scene graph读取每个actor的世界变换，避免只检查时间戳。"""
    actors = {}
    offset = 0
    for name in gaussians.graph_obj_list:
        model = getattr(gaussians, name)
        count = model.get_xyz.shape[0]
        if count:
            actors[str(model.track_id)] = {
                'world_translation': gaussians.obj_trans[offset].detach().cpu().numpy().copy(),
                'world_quaternion': gaussians.obj_rots[offset].detach().cpu().numpy().copy(),
                'gaussian_count': int(count),
            }
        offset += count
    return actors


def _render_one(camera, factual_ego, edited_ego, renderer, gaussians, output_dir, stem, normal, tangent):
    factual_c2w = camera.get_extrinsic()
    factual_actor_ego = camera.ego_pose.detach().cpu().numpy().copy()
    factual_timestamp = camera.meta["timestamp"]
    gaussians.set_visibility(list(gaussians.model_name_id.keys()))
    gaussians.parse_camera(camera)
    factual_actors = _actor_graph_snapshot(gaussians)
    edited_c2w = camera_c2w(factual_ego, edited_ego, factual_c2w)
    camera.set_extrinsic(edited_c2w)
    try:
        result = renderer.render(camera, gaussians)
        edited_actors = _actor_graph_snapshot(gaussians)
        if factual_actors.keys() != edited_actors.keys():
            raise AssertionError('camera edit changed actor membership')
        actor_checks = []
        for key, factual in factual_actors.items():
            edited = edited_actors[key]
            trans_delta = float(np.max(np.abs(factual['world_translation']-edited['world_translation'])))
            quat_delta = float(np.max(np.abs(factual['world_quaternion']-edited['world_quaternion'])))
            if not np.isfinite([trans_delta, quat_delta]).all() or trans_delta > 1e-5 or quat_delta > 1e-6:
                raise AssertionError(f'actor {key} world transform changed during camera edit')
            actor_checks.append({'track_id':key, 'translation_delta_max_m':trans_delta,
                                 'quaternion_delta_max':quat_delta,
                                 **{k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in factual.items()}})
        for key in ('rgb', 'depth', 'acc'):
            if key in result and not torch.isfinite(result[key]).all():
                raise AssertionError(f'non-finite rendered {key}')
        _save_tensor_rgb(result["rgb"], output_dir / f"{stem}_rgb.png")
        for key in ("depth", "acc"):
            if key in result:
                np.save(output_dir / f"{stem}_{key}.npy", result[key].detach().cpu().numpy())
        actual = camera.get_extrinsic()
        if not np.allclose(actual, edited_c2w, atol=1e-5):
            raise AssertionError("Camera.set_extrinsic did not apply the requested c2w")
        if not np.allclose(camera.ego_pose.detach().cpu().numpy(), factual_actor_ego):
            raise AssertionError("actor ego pose changed during camera edit")
        if camera.meta["timestamp"] != factual_timestamp:
            raise AssertionError("actor world clock changed during camera edit")
    finally:
        camera.set_extrinsic(factual_c2w)
    return {
        "frame": int(camera.meta["frame"]),
        "camera": int(camera.meta["cam"]),
        "actor_timestamp": float(factual_timestamp),
        "actor_world_transform_checks": actor_checks,
        "render_finite": True,
        **pose_delta(factual_ego, edited_ego, tangent, normal),
        "factual_c2w": factual_c2w.tolist(),
        "counterfactual_c2w": edited_c2w.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--intervention", required=True, choices=("sweep", "lane_change", "speed"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--frame", type=int, help="logged frame for a lateral sweep")
    parser.add_argument("--offsets", default="0,0.5,1,2,3.5")
    parser.add_argument("--offset-m", type=float, default=3.5)
    parser.add_argument("--start-s", type=float, default=0.0)
    parser.add_argument("--duration-s", type=float, default=1.8)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--speed-factor", type=float, default=0.8)
    parser.add_argument("--output-dir")
    parser.add_argument("--iteration", type=int, help="checkpoint iteration to load, including iterations without a saved PLY")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")

    sys.argv = [sys.argv[0], "--config", args.config, "mode", "evaluate"]
    from lib.config import cfg
    if args.iteration is not None:
        cfg.loaded_iter = args.iteration
    from lib.datasets.dataset import Dataset
    from lib.models.scene import Scene
    from lib.models.street_gaussian_model import StreetGaussianModel
    from lib.models.street_gaussian_renderer import StreetGaussianRenderer

    output_dir = Path(args.output_dir or (Path(cfg.model_path) / "v76_counterfactual" / args.intervention))
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = Dataset()
    gaussians = StreetGaussianModel(dataset.scene_info.metadata)
    scene = Scene(gaussians=gaussians, dataset=dataset)
    renderer = StreetGaussianRenderer()
    cameras = [c for c in scene.getTrainCameras() + scene.getTestCameras() if c.meta["cam"] == args.camera]
    cameras.sort(key=lambda c: c.meta["frame"])
    if len(cameras) < 2:
        raise ValueError("need at least two logged frames for a road tangent")
    frames = np.array([c.meta["frame"] for c in cameras])
    if len(np.unique(frames)) != len(frames):
        raise ValueError("duplicate frame for selected camera")
    ego_poses = np.stack([c.ego_pose.detach().cpu().numpy() for c in cameras])
    _, tangents, normals = road_frame(ego_poses)
    seconds = (frames - frames[0]) / args.fps
    manifest = {
        "intervention": args.intervention,
        "config": args.config,
        "checkpoint_iteration": int(scene.loaded_iter),
        "actor_clock": "logged world timestamp, unchanged by ego edit",
        "samples": [],
    }

    with torch.no_grad():
        if args.intervention == "sweep":
            if args.frame is None:
                parser.error("--frame is required for a sweep")
            matching = np.flatnonzero(frames == args.frame)
            if len(matching) != 1:
                raise ValueError(f"frame {args.frame} is unavailable for camera {args.camera}")
            i = int(matching[0])
            offsets = [float(x) for x in args.offsets.split(",")]
            for offset in offsets:
                edited = lateral_sweep_pose(ego_poses, i, offset)
                row = _render_one(cameras[i], ego_poses[i], edited, renderer, gaussians,
                                  output_dir, f"frame{frames[i]:03d}_d{offset:+.2f}", normals[i], tangents[i])
                if abs(row["delta_d_m"] - offset) > 1e-4 or abs(row["delta_s_m"]) > 1e-4:
                    raise AssertionError("sweep moved outside the requested lateral direction")
                manifest["samples"].append(row)
        else:
            edited_poses = (lane_change(ego_poses, seconds, args.offset_m, args.start_s, args.duration_s)
                            if args.intervention == "lane_change" else speed_edit(ego_poses, args.speed_factor))
            for i, camera in enumerate(cameras):
                row = _render_one(camera, ego_poses[i], edited_poses[i], renderer, gaussians,
                                  output_dir, f"frame{frames[i]:03d}", normals[i], tangents[i])
                manifest["samples"].append(row)

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"output_dir": str(output_dir), "samples": len(manifest["samples"]),
                      "checkpoint_iteration": scene.loaded_iter}))


if __name__ == "__main__":
    main()
