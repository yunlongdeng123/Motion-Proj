"""相机几何、SE3 工具，以及由自车运动（ego-motion）诱导的静态光流。

约定
-----------
- 所有位姿均为 4x4 齐次矩阵，采用列向量约定：``X' = T @ X``。
- ``cam2ego``：相机 -> 自车（传感器外参），4x4。
- ``ego2global[k]``：第 k 帧的自车 -> 全局坐标（来自 nuScenes ego_pose），4x4。
- 内参 ``K``：3x3 针孔模型。
- 图像使用像素坐标 (u, v)，其中 u 沿宽度方向，v 沿高度方向。
- 光流为 ``(du, dv)``，表示第 t 帧的某像素映射到 t+1 帧时的位置。
"""
from __future__ import annotations

import torch


# ----------------------------------------------------------------------------- SE3
def se3_inverse(T: torch.Tensor) -> torch.Tensor:
    """求 4x4（可批量 ``[...,4,4]``）刚体变换的逆。"""
    R = T[..., :3, :3]
    t = T[..., :3, 3:4]
    Rt = R.transpose(-1, -2)
    top = torch.cat([Rt, -Rt @ t], dim=-1)
    bottom = torch.zeros_like(T[..., 3:4, :])
    bottom[..., 0, 3] = 1.0
    return torch.cat([top, bottom], dim=-2)


def make_transform(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """由旋转 ``R [3,3]`` 和平移 ``t [3]`` 构建 4x4 变换矩阵。"""
    T = torch.eye(4, dtype=R.dtype, device=R.device)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def quaternion_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """nuScenes 风格的四元数 (w, x, y, z) -> 3x3 旋转矩阵。"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    w, x, y, z = q.unbind(-1)
    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(*q.shape[:-1], 3, 3)
    return R


# ------------------------------------------------------------------------- projection
def project_points(X_cam: torch.Tensor, K: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """将相机坐标系下的点 ``X_cam [...,3]`` 投影到像素 ``[...,2]``。"""
    z = X_cam[..., 2:3].clamp_min(eps)
    uv = (K @ (X_cam / z).unsqueeze(-1)).squeeze(-1)[..., :2]
    return uv


def pixel_grid(h: int, w: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """返回 ``[H, W, 2]`` 的 (u, v) 像素中心坐标网格。"""
    vs = torch.arange(h, device=device, dtype=dtype)
    us = torch.arange(w, device=device, dtype=dtype)
    vv, uu = torch.meshgrid(vs, us, indexing="ij")
    return torch.stack([uu, vv], dim=-1)


def backproject(uv: torch.Tensor, depth: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
    """将像素 ``uv [...,2]`` 结合 ``depth [...]`` 反投影为相机坐标点 ``[...,3]``。"""
    ones = torch.ones_like(uv[..., :1])
    pix_h = torch.cat([uv, ones], dim=-1)            # [...,3]
    K_inv = torch.inverse(K)
    rays = (K_inv @ pix_h.unsqueeze(-1)).squeeze(-1)  # [...,3]
    return rays * depth.unsqueeze(-1)


# ----------------------------------------------------------- 自车运动诱导的静态光流
def relative_cam_transform(
    cam2ego: torch.Tensor,
    ego2global_t: torch.Tensor,
    ego2global_tp1: torch.Tensor,
) -> torch.Tensor:
    """在假设点在世界中静止的前提下，将 cam@t 中的点映射到 cam@(t+1) 的 4x4 变换。"""
    cam2global_t = ego2global_t @ cam2ego
    cam2global_tp1 = ego2global_tp1 @ cam2ego
    return se3_inverse(cam2global_tp1) @ cam2global_t


def ego_induced_flow(
    depth: torch.Tensor,
    K: torch.Tensor,
    cam2ego: torch.Tensor,
    ego2global_t: torch.Tensor,
    ego2global_tp1: torch.Tensor,
) -> torch.Tensor:
    """*静态*场景从 t -> t+1 会呈现的光流 ``[H,W,2]``。

    在静态区域上，观测光流（例如 RAFT）与该光流场之间的任何偏差，
    即为 ``E_static`` 所使用的、经自车运动补偿后的静态漂移。

    Args:
        depth: 第 t 帧的度量深度 ``[H,W]``（相机坐标系下的 z）。
        K: ``[3,3]`` 内参。
        cam2ego, ego2global_t, ego2global_tp1: 4x4 变换矩阵。
    """
    h, w = depth.shape
    uv = pixel_grid(h, w, device=depth.device, dtype=depth.dtype)        # [H,W,2]
    X_t = backproject(uv, depth, K)                                       # [H,W,3]

    M = relative_cam_transform(cam2ego, ego2global_t, ego2global_tp1)     # [4,4]
    ones = torch.ones_like(X_t[..., :1])
    Xh = torch.cat([X_t, ones], dim=-1)                                   # [H,W,4]
    X_tp1 = (M @ Xh.reshape(-1, 4, 1)).reshape(h, w, 4)[..., :3]

    uv_tp1 = project_points(X_tp1, K)                                     # [H,W,2]
    flow = uv_tp1 - uv
    return flow


def flow_to_grid(flow: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """将前向光流 ``[H,W,2]`` (du,dv) 转换为供 ``F.grid_sample`` 使用的归一化采样网格，
    使得给定 t+1 帧图像时可重建出 t 帧图像。"""
    uv = pixel_grid(h, w, device=flow.device, dtype=flow.dtype)
    src = uv + flow
    gx = 2.0 * src[..., 0] / max(w - 1, 1) - 1.0
    gy = 2.0 * src[..., 1] / max(h - 1, 1) - 1.0
    return torch.stack([gx, gy], dim=-1)  # [H,W,2]
