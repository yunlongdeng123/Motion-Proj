#!/usr/bin/env python
"""C0 counterfactual render (v2) — DriveStudio StreetGS checkpoint + S0 edits."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, "/root/autodl-tmp/third_party/drivestudio")
from datasets.driving_dataset import DrivingDataset
from utils.misc import import_str


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    q = np.empty(4, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        q[0] = 0.25 * s
        q[1] = (R[2, 1] - R[1, 2]) / s
        q[2] = (R[0, 2] - R[2, 0]) / s
        q[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[0] = (R[2, 1] - R[1, 2]) / s
            q[1] = 0.25 * s
            q[2] = (R[0, 1] + R[1, 0]) / s
            q[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[0] = (R[0, 2] - R[2, 0]) / s
            q[1] = (R[0, 1] + R[1, 0]) / s
            q[2] = 0.25 * s
            q[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[0] = (R[1, 0] - R[0, 1]) / s
            q[1] = (R[0, 2] + R[2, 0]) / s
            q[2] = (R[1, 2] + R[2, 1]) / s
            q[3] = 0.25 * s
    return q / (np.linalg.norm(q) + 1e-12)


def to_device(obj, device):
    if isinstance(obj, dict):
        return {k: to_device(v, device) for k, v in obj.items()}
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj


def resolve_model_index(dataset, trainer, json_actor_id: int) -> int:
    """Map instances_info.json key → RigidNodes model index."""
    init = dataset.get_init_objects(
        cur_node_type="RigidNodes",
        instance_max_pts=int(trainer.model_config.RigidNodes.init.instance_max_pts),
        only_moving=bool(trainer.model_config.RigidNodes.init.only_moving),
        traj_length_thres=float(trainer.model_config.RigidNodes.init.traj_length_thres),
    )
    # init keys are dataset instance column indices (after visibility filter)
    true_ids = dataset.pixel_source.instances_true_id.cpu().numpy()
    ordered = list(init.keys())
    for model_i, col in enumerate(ordered):
        if int(true_ids[col]) == int(json_actor_id):
            return model_i
    raise KeyError(f"actor {json_actor_id} not in RigidNodes init set; ordered_true={[int(true_ids[c]) for c in ordered]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--edit_json", required=True)
    ap.add_argument("--variants", nargs="+", default=["V0", "V1", "V2"])
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--n_frames", type=int, default=8)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    cfg = OmegaConf.load(ckpt.parent / "config.yaml")
    edits = json.load(open(args.edit_json))
    json_actor = int(edits["actor_id"])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

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
    trainer.resume_from_checkpoint(ckpt_path=str(ckpt), load_only_model=True)
    trainer.set_eval()

    rigid = trainer.models["RigidNodes"]
    model_idx = resolve_model_index(dataset, trainer, json_actor)
    print(f"json_actor={json_actor} → model_idx={model_idx}; trans shape={tuple(rigid.instances_trans.shape)}")

    # camera_front_start for pose alignment (nuScenes)
    data_path = cfg.data.data_root.rstrip("/") + f"/{int(cfg.data.scene_idx):03d}"
    if not Path(data_path).exists():
        # absolute path already includes scene
        data_path = str(Path(cfg.data.data_root) / f"{int(cfg.data.scene_idx):03d}")
    start = int(cfg.data.start_timestep)
    cam0 = np.loadtxt(os.path.join(data_path, "extrinsics", f"{start:03d}_0.txt"))

    trans0 = rigid.instances_trans.detach().clone()
    quats0 = rigid.instances_quats.detach().clone()

    # sample frames inside model window
    Tm = rigid.instances_trans.shape[0]
    frame_locals = np.linspace(0, Tm - 1, args.n_frames, dtype=int).tolist()
    num_cams = dataset.pixel_source.num_cams

    report = dict(ckpt=str(ckpt), edit_json=args.edit_json, json_actor=json_actor,
                  model_idx=model_idx, variants={})

    for variant in args.variants:
        v = edits["variants"][variant]
        if variant != "V0" and not v.get("accepted", False):
            print("skip rejected", variant)
            continue
        rigid.instances_trans.data.copy_(trans0)
        rigid.instances_quats.data.copy_(quats0)
        # disable test-frame pose interpolation so edits stick
        for m in trainer.models.values():
            if hasattr(m, "in_test_set"):
                m.in_test_set = False

        with torch.no_grad():
            for fr in v["frames"]:
                t_abs = int(fr["frame"])
                local = t_abs - start
                if local < 0 or local >= Tm:
                    continue
                T_world = np.array(fr["obj_to_world"], float).reshape(4, 4)
                T_model = np.linalg.inv(cam0) @ T_world
                tvec = T_model[:3, 3]
                quat = mat_to_quat_wxyz(T_model[:3, :3])
                rigid.instances_trans.data[local, model_idx] = torch.tensor(
                    tvec, device=device, dtype=rigid.instances_trans.dtype
                )
                rigid.instances_quats.data[local, model_idx] = torch.tensor(
                    quat, device=device, dtype=rigid.instances_quats.dtype
                )

        vdir = out_dir / variant
        vdir.mkdir(exist_ok=True)
        psnrs = []
        for local in frame_locals:
            img_idx = int(local) * num_cams + 0  # CAM_FRONT
            image_infos, cam_infos = dataset.full_image_set.get_image(img_idx, camera_downscale=1.0)
            image_infos = to_device(image_infos, device)
            cam_infos = to_device(cam_infos, device)
            with torch.no_grad():
                out = trainer(image_infos, cam_infos)
            rgb = out["rgb"].detach().float().cpu().numpy()
            depth = out["depth"].detach().float().cpu().numpy().squeeze()
            rgb_u8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
            imageio.imwrite(vdir / f"rgb_t{local:03d}.png", rgb_u8)
            d_norm = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
            imageio.imwrite(vdir / f"depth_t{local:03d}.png", (d_norm * 255).astype(np.uint8))
            if "RigidNodes_rgb" in out:
                rr = out["RigidNodes_rgb"].detach().float().cpu().numpy()
                imageio.imwrite(vdir / f"rigid_t{local:03d}.png", (np.clip(rr, 0, 1) * 255).astype(np.uint8))
            gt = image_infos.get("pixels")
            if gt is not None and variant == "V0":
                gt_np = gt.detach().float().cpu().numpy()
                mse = float(np.mean((rgb - gt_np) ** 2))
                psnrs.append(float(-10 * np.log10(mse + 1e-10)))
        # locality: compare to V0 if available
        report["variants"][variant] = dict(
            n_rgb=len(list(vdir.glob("rgb_*.png"))),
            v0_psnr_mean=float(np.mean(psnrs)) if psnrs else None,
            accepted=bool(v.get("accepted", True)),
            peak_abs_dy=v.get("peak_abs_dy"),
        )
        print(variant, report["variants"][variant])

    # locality / effect metrics vs V0
    if "V0" in report["variants"]:
        for variant in args.variants:
            if variant == "V0" or variant not in report["variants"]:
                continue
            diffs_all, diffs_peak = [], []
            for local in frame_locals:
                a = imageio.imread(out_dir / "V0" / f"rgb_t{local:03d}.png").astype(np.float32)
                b = imageio.imread(out_dir / variant / f"rgb_t{local:03d}.png").astype(np.float32)
                if a.max() > 1.5:
                    a, b = a / 255.0, b / 255.0
                d = float(np.mean(np.abs(a - b)))
                diffs_all.append(d)
                diffs_peak.append(d)
            report[f"mean_abs_rgb_diff_{variant}_vs_V0"] = float(np.mean(diffs_all))
            report[f"max_abs_rgb_diff_{variant}_vs_V0"] = float(np.max(diffs_peak))
            print(
                f"mean/max_abs_rgb_diff_{variant}_vs_V0",
                report[f"mean_abs_rgb_diff_{variant}_vs_V0"],
                report[f"max_abs_rgb_diff_{variant}_vs_V0"],
            )

    json.dump(report, open(out_dir / "c0_render_report.json", "w"), indent=2)


if __name__ == "__main__":
    main()
