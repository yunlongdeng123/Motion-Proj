"""确定性非平面道路/移动 Actor，用于 CPU 集成与恢复，不是真实场景结果。"""
from dataclasses import replace
from pathlib import Path
import json
import numpy as np
import torch
from PIL import Image
from shadow_rebuild.config import Config
from shadow_rebuild.model import SceneModel
from shadow_rebuild.data import Frame
from shadow_rebuild.raster import Camera
from shadow_rebuild.math3d import linear_to_srgb, quaternion_matrix


def create(root, steps=4):
    root = Path(root)
    root.mkdir(parents=True)
    (root/"frames").mkdir()
    grid = np.linspace(-1.2, 1.2, 5)
    road = np.array([[x, y, .03*x*x] for x in grid for y in grid], np.float32)
    body = np.array([[x, y, z] for x in (-.18, .18) for y in (-.16, .16) for z in (-.1, .1)], np.float32)
    xyz = np.concatenate((road, body))
    n = len(xyz)
    owner = np.r_[np.full(len(road), -1), np.zeros(len(body))].astype(np.int64)
    is_road = owner == -1
    albedo = np.tile([.35, .4, .45], (n, 1)).astype(np.float32)
    albedo[~is_road] = [.5, .18, .13]
    scales = np.tile([.34, .34, .02], (n, 1)).astype(np.float32)
    scales[~is_road] = [.17, .14, .08]
    seed = dict(xyz=xyz, rgb=albedo.copy(), scales=scales, quaternion=np.tile([1., 0, 0, 0], (n, 1)).astype(np.float32),
                owner=owner, road=is_road, reference_rgb=np.zeros((n, 3), np.float32),
                reference_confidence=is_road.astype(np.float32), neighbor=np.arange(n, dtype=np.int64))
    config = replace(Config(), geometry_steps=steps, material_steps=3, joint_steps=3, sun_samples=1,
                     shading_patch=8, uniform_rays=64, log_interval=2, checkpoint_interval=2,
                     densify_interval=2, densify_until=2, densify_fraction=.05, max_gaussians=100)
    truth = SceneModel(seed, config)
    with torch.no_grad():
        truth.field.albedo_logit.copy_(torch.logit(torch.tensor(albedo)))
    poses = np.broadcast_to(np.eye(4), (5, 1, 4, 4)).copy().astype(np.float32)
    poses[:, 0, 0, 3] = np.linspace(-.45, .45, 5)
    poses[:, 0, 1, 3] = -.25
    poses[:, 0, 2, 3] = .5
    valid = np.ones((5, 1), bool)
    camera_pose = np.diag([1., -1., -1., 1.]).astype(np.float32)
    camera_pose[:3, 3] = [0, 0, 4]
    intrinsic = [[24., 0, 11.5], [0, 24., 11.5], [0, 0, 1]]
    camera = Camera(intrinsic, camera_pose, 24, 24)
    frames = []
    with torch.no_grad():
        ambient = truth.light.ambient_logit.sigmoid()
        normal = quaternion_matrix(truth.field.quaternion)[..., 2]
        unshaded = truth.field.albedo_logit.sigmoid()*(ambient+(1-ambient)*(normal*truth.light.direction()).sum(-1).clamp_min(0)[:, None])
        seed["reference_rgb"] = linear_to_srgb(unshaded).numpy()
        seed["reference_rgb"][~is_road] = 0
        for i in range(5):
            fid = f"f{i:03d}"
            frame = Frame(fid, i, "test" if i == 2 else "train", camera, torch.zeros(24, 24, 3),
                          torch.zeros(24, 24, 1, dtype=torch.bool), torch.ones(1, dtype=torch.bool),
                          torch.zeros(24, 24, dtype=torch.bool), torch.zeros(24, 24), torch.tensor(poses[i]), torch.tensor(valid[i]))
            g, package = truth.buffers(frame, 1, "reference")
            rgb, _ = truth.full_image(package, g)
            png = (linear_to_srgb(rgb).clamp(0, 1)*255).round().byte().numpy()
            Image.fromarray(png).save(root/"frames"/(fid+".png"))
            masks = package["visible_labels"].numpy() > .25
            observed_road = (package["road"].numpy() > .8) & ~masks.any(-1)
            depth = package["depth"].numpy()*(package["alpha"].numpy() > .5)
            np.savez_compressed(root/"frames"/(fid+".npz"), masks=masks, mask_known=np.ones(1, bool), road=observed_road, depth=depth)
            frames.append(dict(frame_id=fid, source_index=i, split=frame.split, intrinsic=intrinsic,
                               world_from_camera=camera_pose.tolist(), rgb=f"frames/{fid}.png", supervision=f"frames/{fid}.npz"))
    seed["rgb"][is_road] = np.clip(seed["reference_rgb"][is_road]*.9, .01, .99)
    seed["rgb"][~is_road] = np.clip(seed["rgb"][~is_road]*.9, .01, .99)
    seed["observed_rgb"] = np.clip(albedo*.7, .01, .99)
    np.savez_compressed(root/"seed.npz", **seed)
    np.savez_compressed(root/"trajectory.npz", poses=poses, valid=valid)
    (root/"manifest.json").write_text(json.dumps(dict(schema="instance_surface_measurements_v3", units="meters", actor_ids=["synthetic_car"],
                                                     frames=frames, learned_checkpoint_used=False, reference_train_only=True, synthetic_fixture=True)), encoding="utf-8")
    return config
