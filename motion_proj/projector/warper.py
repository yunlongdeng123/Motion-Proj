"""渲染器/形变器（renderer/warper）Gamma(a_y, s)：将低维状态映射回视频空间。

V1 合成两层：
  - 静态层：通过自车诱导光流对锚定帧（anchor frame）进行反向形变来渲染每一帧
    （得到与自车运动一致的无漂移静态背景）；
  - 动态层：将每个目标的外观（从 y 中裁剪）粘贴到其经时间平滑后的框位置上。

这样可将修正限制在几何渲染器的像域之内（方案第 8.2 节）：不使用原始 RGB
梯度，只使用形变（warps）与轨迹调整。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..auditor.state import Track
from ..utils.geometry import ego_induced_flow, flow_to_grid


@torch.no_grad()
def render_static(
    frames: torch.Tensor,       # [K,3,H,W]
    depth: torch.Tensor,        # [K,H,W]
    intrinsics: torch.Tensor,   # [3,3]
    cam2ego: torch.Tensor,      # [4,4]
    ego2global: torch.Tensor,   # [K,4,4]
    anchor: int = 0,
) -> torch.Tensor:
    """通过对锚定帧进行形变得到无漂移静态背景 ``[K,3,H,W]``。"""
    k, _, h, w = frames.shape
    anchor_img = frames[anchor : anchor + 1]
    out = []
    for t in range(k):
        if t == anchor:
            out.append(frames[t])
            continue
        flow_t2a = ego_induced_flow(
            depth[t], intrinsics, cam2ego, ego2global[t], ego2global[anchor]
        )
        grid = flow_to_grid(flow_t2a, h, w).unsqueeze(0)
        warped = F.grid_sample(
            anchor_img, grid, mode="bilinear", padding_mode="border", align_corners=True
        )[0]
        out.append(warped)
    return torch.stack(out, 0)


def _soft_rect_mask(h: int, w: int, box, feather: float = 0.15, device=None) -> torch.Tensor:
    """为整数框生成带羽化边缘的软 [H,W] 掩码，取值 [0,1]。"""
    u0, v0, u1, v1 = [float(x) for x in box]
    ys = torch.arange(h, device=device).float()
    xs = torch.arange(w, device=device).float()
    fx = max((u1 - u0) * feather, 1.0)
    fy = max((v1 - v0) * feather, 1.0)
    mx = (torch.sigmoid((xs - u0) / fx) * torch.sigmoid((u1 - xs) / fx)).clamp(0, 1)
    my = (torch.sigmoid((ys - v0) / fy) * torch.sigmoid((v1 - ys) / fy)).clamp(0, 1)
    return my[:, None] * mx[None, :]


@torch.no_grad()
def composite_objects(
    base: torch.Tensor,         # [K,3,H,W] 静态背景
    y: torch.Tensor,            # [K,3,H,W] 原始帧（外观来源）
    orig_tracks: list[Track],
    smooth_tracks: list[Track],
    support_flags: dict,
):
    """将平滑后的目标裁剪块粘贴到静态底图上。返回 (x_dagger, obj_mask)。"""
    k, _, h, w = base.shape
    x = base.clone()
    obj_mask = torch.zeros(k, h, w, device=base.device)
    orig_by_inst = {tr.instance_token: tr for tr in orig_tracks}

    for sm in smooth_tracks:
        orig = orig_by_inst.get(sm.instance_token)
        if orig is None:
            continue
        supported = support_flags.get(sm.instance_token, torch.ones(k, dtype=torch.bool))
        for t in range(k):
            if not bool(sm.present[t]) or not bool(supported[t]) or not bool(orig.present[t]):
                continue
            ob = orig.xyxy[t]
            sb = sm.xyxy[t]
            ou0, ov0, ou1, ov1 = [int(round(float(c))) for c in ob]
            su0, sv0, su1, sv1 = [int(round(float(c))) for c in sb]
            ou0, ou1 = max(0, ou0), min(w, ou1)
            ov0, ov1 = max(0, ov0), min(h, ov1)
            su0, su1 = max(0, su0), min(w, su1)
            sv0, sv1 = max(0, sv0), min(h, sv1)
            if ou1 - ou0 < 2 or ov1 - ov0 < 2 or su1 - su0 < 2 or sv1 - sv0 < 2:
                continue
            crop = y[t : t + 1, :, ov0:ov1, ou0:ou1]
            resized = F.interpolate(
                crop, size=(sv1 - sv0, su1 - su0), mode="bilinear", align_corners=False
            )[0]
            m = _soft_rect_mask(sv1 - sv0, su1 - su0, [0, 0, su1 - su0, sv1 - sv0], device=base.device)
            region = x[t, :, sv0:sv1, su0:su1]
            x[t, :, sv0:sv1, su0:su1] = m[None] * resized + (1 - m[None]) * region
            obj_mask[t, sv0:sv1, su0:su1] = torch.maximum(obj_mask[t, sv0:sv1, su0:su1], m)
    return x.clamp(-1, 1), obj_mask
