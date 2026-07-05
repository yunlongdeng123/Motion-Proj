"""由 auditor 提取的低维运动状态（方案第 5 节）。"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class Track:
    """片段上逐实例的 2D 轨迹（目标缺席处为 NaN）。"""

    instance_token: str
    category: str
    xyxy: torch.Tensor          # [K, 4] (u0,v0,u1,v1)，缺席处为 NaN
    depth: torch.Tensor         # [K] 相机坐标系 z 值，缺席处为 NaN
    present: torch.Tensor       # [K] bool

    @property
    def center(self) -> torch.Tensor:
        u = (self.xyxy[:, 0] + self.xyxy[:, 2]) * 0.5
        v = (self.xyxy[:, 1] + self.xyxy[:, 3]) * 0.5
        return torch.stack([u, v], dim=-1)  # [K,2]

    @property
    def scale(self) -> torch.Tensor:
        w = (self.xyxy[:, 2] - self.xyxy[:, 0]).clamp_min(1e-3)
        h = (self.xyxy[:, 3] - self.xyxy[:, 1]).clamp_min(1e-3)
        return torch.stack([w, h], dim=-1)  # [K,2]


@dataclass
class MotionState:
    """一个片段的运动观测量集合。

    形状约定：``K`` 帧，``F = K-1`` 个相邻帧对，图像尺寸 ``H x W``。
    """

    u_static: torch.Tensor      # [F, H, W, 2] 观测到的（RAFT）光流 t->t+1
    u_ego: torch.Tensor         # [F, H, W, 2] 由自车运动诱导的静态光流
    static_mask: torch.Tensor   # [F, H, W]，取值 {0,1}：可靠静态像素
    flow_conf: torch.Tensor     # [F, H, W]，取值 [0,1]：前后向一致性
    depth: torch.Tensor         # [K, H, W]
    tracks: list[Track] = field(default_factory=list)
    meta: dict = field(default_factory=dict)  # intrinsics、cam2ego、ego2global、hw

    @property
    def static_residual(self) -> torch.Tensor:
        """经自车运动补偿的静态漂移场 ``u_static - u_ego``（[F,H,W,2]）。"""
        return self.u_static - self.u_ego

    def to(self, device) -> "MotionState":
        self.u_static = self.u_static.to(device)
        self.u_ego = self.u_ego.to(device)
        self.static_mask = self.static_mask.to(device)
        self.flow_conf = self.flow_conf.to(device)
        self.depth = self.depth.to(device)
        for tr in self.tracks:
            tr.xyxy = tr.xyxy.to(device)
            tr.depth = tr.depth.to(device)
            tr.present = tr.present.to(device)
        return self
