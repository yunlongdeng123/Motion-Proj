"""从数据集逐帧的 GT 框中组装出逐实例的轨迹（tracks）。

V1 使用 nuScenes 的真值框（数据集中已投影到 2D），因此无需
学习式的检测器/跟踪器。关联依据是 ``instance_token``。
"""
from __future__ import annotations

import math

import torch

from .state import Track


def build_tracks(boxes_per_frame: list, num_frames: int) -> list[Track]:
    """为片段中任意一帧出现过的每个实例返回一个 ``Track``。"""
    inst_to_idx: dict[str, dict] = {}
    for blist in boxes_per_frame:
        for b in blist:
            inst_to_idx.setdefault(
                b["instance_token"], {"category": b["category"]}
            )

    tracks: list[Track] = []
    for inst, info in inst_to_idx.items():
        xyxy = torch.full((num_frames, 4), float("nan"))
        depth = torch.full((num_frames,), float("nan"))
        present = torch.zeros(num_frames, dtype=torch.bool)
        for t, blist in enumerate(boxes_per_frame):
            for b in blist:
                if b["instance_token"] == inst:
                    xyxy[t] = torch.as_tensor(b["xyxy"], dtype=torch.float32)
                    depth[t] = float(b["center_depth"])
                    present[t] = True
                    break
        tracks.append(
            Track(
                instance_token=inst,
                category=info["category"],
                xyxy=xyxy,
                depth=depth,
                present=present,
            )
        )
    return tracks
