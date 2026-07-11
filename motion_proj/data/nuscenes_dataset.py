"""用于 Motion-Proj 的 nuScenes 未来视频数据集。

每个样本是来自单个相机（V1：CAM_FRONT）的 ``K`` 个关键帧组成的短片段，
并附带运动审计器（motion auditor）所需的几何信息：
内参、相机外参（cam2ego）、逐帧的 ego->global 位姿，以及带有实例 token、
投影到 2D 的真值 3D 检测框（这样在 V1 中投影器无需已训练的检测器即可
构建目标轨迹）。

返回的字典（单相机，``K`` = num_frames）：
    frames:      float 张量 [K, 3, H, W]，取值范围 [-1, 1]
    cond_frame:  float 张量 [3, H, W]            (= frames[0]，SVD 条件输入)
    intrinsics:  float 张量 [3, 3]               (已重新缩放到 H, W)
    cam2ego:     float 张量 [4, 4]
    ego2global:  float 张量 [K, 4, 4]
    boxes:       list[K]，其元素为 list[dict]      逐帧的真值检测框
    timestamps:  long 张量 [K]                   (微秒)
    lidar_depth: float 张量 [K,H,W]              (稀疏；无点像素为 0)
    sample_id:   str                               (稳定的缓存键)
"""
from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ..utils.geometry import make_transform, quaternion_to_matrix
from ..utils.logging import get_logger

log = get_logger(__name__)


def official_scene_names(version: str, split: str | None) -> set[str] | None:
    """解析与数据版本匹配的 nuScenes 官方 scene-level split。"""
    if split is None or split in {"", "all"}:
        return None

    from nuscenes.utils.splits import create_splits_scenes

    expected = {
        "v1.0-mini": {"mini_train", "mini_val"},
        "v1.0-trainval": {"train", "val"},
        "v1.0-test": {"test"},
    }
    if version not in expected:
        raise ValueError(f"不支持 split={split!r} 的 nuScenes 版本: {version}")
    if split not in expected[version]:
        allowed = ", ".join(sorted(expected[version]))
        raise ValueError(f"{version} 只允许 split={allowed}，实际为 {split!r}")

    splits = create_splits_scenes(verbose=False)
    if split not in splits:
        raise ValueError(f"nuScenes devkit 未提供 split={split!r}")
    return set(splits[split])


def _img_to_tensor(img: np.ndarray) -> torch.Tensor:
    """[H,W,3] uint8 -> [3,H,W] float，取值范围 [-1,1]。"""
    t = torch.from_numpy(np.asarray(img).copy()).float().permute(2, 0, 1) / 255.0
    return t * 2.0 - 1.0


class NuScenesFutureVideoDataset(Dataset):
    def __init__(self, cfg: Any):
        """``cfg`` 是 ``data`` 子配置（参见 configs/data/nuscenes_mini.yaml）。"""
        self.cfg = cfg
        self.dataroot = cfg.dataroot
        self.version = cfg.version
        self.cameras = list(cfg.cameras)
        assert len(self.cameras) == 1, "V1 supports a single camera; multi-view is future work"
        self.camera = self.cameras[0]
        self.K = int(cfg.num_frames)
        self.stride = int(cfg.frame_stride)
        self.H = int(cfg.height)
        self.W = int(cfg.width)
        self.min_vis = int(cfg.min_box_visibility)
        self.use_lidar_depth = bool(cfg.get("use_lidar_depth", True))
        self.split = str(cfg.get("split", "all"))

        meta_dir = os.path.join(self.dataroot, self.version)
        if not os.path.isdir(meta_dir):
            raise FileNotFoundError(
                f"nuScenes metadata not found at {meta_dir}. "
                f"Run scripts/extract_nuscenes_mini.sh first."
            )

        from nuscenes.nuscenes import NuScenes

        self.nusc = NuScenes(version=self.version, dataroot=self.dataroot, verbose=False)
        self.clips = self._build_clips()
        log.info("Built %d clips (%s, K=%d) from %s", len(self.clips), self.camera, self.K, self.version)

    # ------------------------------------------------------------------ 索引构建
    def _build_clips(self) -> list[list[str]]:
        """返回片段列表；每个片段是由 K 个 sample（关键帧）token 组成的列表。"""
        clips: list[list[str]] = []
        self.clip_records: list[dict[str, Any]] = []
        span = self.K * self.stride
        scenes = list(self.nusc.scene)
        selected_names = official_scene_names(self.version, self.split)
        if selected_names is not None:
            available_names = {str(scene["name"]) for scene in scenes}
            missing = selected_names - available_names
            if missing:
                preview = ", ".join(sorted(missing)[:5])
                raise RuntimeError(
                    f"{self.version}/{self.split} 缺少 {len(missing)} 个官方 scene，"
                    f"示例: {preview}"
                )
            scenes = sorted(
                (scene for scene in scenes if str(scene["name"]) in selected_names),
                key=lambda item: str(item["name"]),
            )
        self.scene_names = [str(scene["name"]) for scene in scenes]

        for scene in scenes:
            tokens: list[str] = []
            tok = scene["first_sample_token"]
            while tok:
                tokens.append(tok)
                tok = self.nusc.get("sample", tok)["next"]
            # 以 `stride` 为间隔、每 K 帧一组的非重叠窗口
            start = 0
            while start + span <= len(tokens):
                clip = [tokens[start + j * self.stride] for j in range(self.K)]
                clips.append(clip)
                self.clip_records.append(
                    {
                        "scene_name": str(scene["name"]),
                        "scene_token": str(scene["token"]),
                        "start_index": start,
                        "sample_tokens": clip,
                        "sample_id": f"{clip[0]}_{self.camera}",
                    }
                )
                start += span
        return clips

    def __len__(self) -> int:
        return len(self.clips)

    # ------------------------------------------------------------------ 数据加载
    def _load_frame(self, sample_token: str):
        from PIL import Image

        sample = self.nusc.get("sample", sample_token)
        cam_token = sample["data"][self.camera]
        # 检测框以相机坐标系返回
        from nuscenes.utils.geometry_utils import BoxVisibility

        data_path, boxes, cam_K = self.nusc.get_sample_data(
            cam_token, box_vis_level=BoxVisibility.ANY
        )
        sd = self.nusc.get("sample_data", cam_token)
        cs = self.nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
        ego = self.nusc.get("ego_pose", sd["ego_pose_token"])

        img = Image.open(data_path).convert("RGB")
        ow, oh = img.size
        img = img.resize((self.W, self.H), Image.BILINEAR)
        frame = _img_to_tensor(np.asarray(img))

        sx, sy = self.W / ow, self.H / oh
        K = torch.tensor(cam_K, dtype=torch.float32).clone()
        K[0, :] *= sx
        K[1, :] *= sy

        cam2ego = make_transform(
            quaternion_to_matrix(torch.tensor(cs["rotation"], dtype=torch.float32)),
            torch.tensor(cs["translation"], dtype=torch.float32),
        )
        ego2global = make_transform(
            quaternion_to_matrix(torch.tensor(ego["rotation"], dtype=torch.float32)),
            torch.tensor(ego["translation"], dtype=torch.float32),
        )

        box_list = self._boxes_to_2d(boxes, K, sx, sy, ow, oh)
        lidar_depth = self._lidar_depth(sample, cam_token, sx, sy) if self.use_lidar_depth else None
        return frame, K, cam2ego, ego2global, box_list, sd["timestamp"], lidar_depth

    def _lidar_depth(self, sample: dict, cam_token: str, sx: float, sy: float) -> torch.Tensor:
        """使用 nuScenes 官方变换链投影 LIDAR_TOP，并以最近点构造稀疏深度。"""
        lidar_token = sample["data"]["LIDAR_TOP"]
        points, depths, _ = self.nusc.explorer.map_pointcloud_to_image(
            lidar_token, cam_token, min_dist=1.0,
        )
        u = np.rint(points[0] * sx).astype(np.int64)
        v = np.rint(points[1] * sy).astype(np.int64)
        valid = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H) & np.isfinite(depths) & (depths > 0)
        flat = np.full(self.H * self.W, np.inf, dtype=np.float32)
        np.minimum.at(flat, v[valid] * self.W + u[valid], depths[valid].astype(np.float32))
        flat[~np.isfinite(flat)] = 0.0
        return torch.from_numpy(flat.reshape(self.H, self.W))

    def _boxes_to_2d(self, boxes, K: torch.Tensor, sx, sy, ow, oh) -> list[dict]:
        """将相机坐标系下的 3D 检测框投影为 2D xyxy，并附加实例 token。"""
        from nuscenes.utils.geometry_utils import view_points

        out: list[dict] = []
        Knp = K.numpy()
        for box in boxes:
            ann = self.nusc.get("sample_annotation", box.token)
            vis = int(ann.get("visibility_token", "0") or 0)
            corners = box.corners()  # [3, 8]，相机坐标系
            if (corners[2] <= 0.1).all():
                continue  # 完全位于相机后方
            pts = view_points(corners, Knp, normalize=True)[:2]  # [2,8] 像素
            u0, v0 = pts[0].min(), pts[1].min()
            u1, v1 = pts[0].max(), pts[1].max()
            # 裁剪到图像范围内并跳过退化情形
            u0, u1 = np.clip([u0, u1], 0, self.W - 1)
            v0, v1 = np.clip([v0, v1], 0, self.H - 1)
            if (u1 - u0) < 2 or (v1 - v0) < 2:
                continue
            out.append(
                {
                    "instance_token": ann["instance_token"],
                    "category": ann["category_name"],
                    "xyxy": np.array([u0, v0, u1, v1], dtype=np.float32),
                    "center_depth": float(box.center[2]),
                    "size3d": np.asarray(box.wlh, dtype=np.float32),
                    "visibility": vis,
                }
            )
        return out

    def __getitem__(self, idx: int) -> dict:
        clip = self.clips[idx]
        frames, Ks, c2e, e2g, boxes, ts, lidar_depths = [], [], [], [], [], [], []
        for tok in clip:
            f, K, cam2ego, ego2global, box_list, t, lidar_depth = self._load_frame(tok)
            frames.append(f)
            Ks.append(K)
            c2e.append(cam2ego)
            e2g.append(ego2global)
            boxes.append(box_list)
            ts.append(t)
            if lidar_depth is not None:
                lidar_depths.append(lidar_depth)
        frames_t = torch.stack(frames, 0)  # [K,3,H,W]
        item = {
            "frames": frames_t,
            "cond_frame": frames_t[0].clone(),
            "intrinsics": Ks[0],            # 对同一相机而言内参是恒定的
            "cam2ego": c2e[0],
            "ego2global": torch.stack(e2g, 0),
            "boxes": boxes,
            "timestamps": torch.tensor(ts, dtype=torch.long),
            "sample_id": f"{clip[0]}_{self.camera}",
        }
        if lidar_depths:
            item["lidar_depth"] = torch.stack(lidar_depths, 0)
        return item


def collate_fn(batch: list[dict]) -> dict:
    """整理函数：堆叠张量，但将变长的 ``boxes`` 保留为列表形式。"""
    out: dict = {}
    for key in batch[0]:
        if key in ("boxes", "sample_id"):
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch], 0)
    return out


def build_dataset(cfg: Any) -> NuScenesFutureVideoDataset:
    return NuScenesFutureVideoDataset(cfg.data if hasattr(cfg, "data") else cfg)
