"""冻结（frozen）的 RAFT 光流，并带有前后向一致性置信度。"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from ..utils.geometry import pixel_grid
from ..utils.logging import get_logger

log = get_logger(__name__)


class RAFTFlow:
    """封装 torchvision 的 ``raft_large``（冻结、no-grad）。"""

    def __init__(self, device: str = "cuda", dtype: torch.dtype = torch.float32):
        from torchvision.models.optical_flow import Raft_Large_Weights, raft_large

        self.device = device
        self.dtype = dtype
        self.weights = Raft_Large_Weights.DEFAULT
        self.model = raft_large(weights=self.weights, progress=False).to(device, dtype).eval()
        self.model.requires_grad_(False)

    @staticmethod
    def _pad_to_multiple(x: torch.Tensor, m: int = 8):
        _, _, h, w = x.shape
        ph, pw = (m - h % m) % m, (m - w % m) % m
        return F.pad(x, (0, pw, 0, ph)), (h, w)

    @torch.no_grad()
    def flow(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """对图像对 ``[N,3,H,W]``（取值 [-1,1]）计算 a->b 的光流。返回 ``[N,H,W,2]``。"""
        a = a.to(self.device, self.dtype)
        b = b.to(self.device, self.dtype)
        ap, (h, w) = self._pad_to_multiple(a)
        bp, _ = self._pad_to_multiple(b)
        out = self.model(ap, bp)[-1]              # [N,2,H',W']
        out = out[..., :h, :w]
        return out.permute(0, 2, 3, 1).contiguous()  # [N,H,W,2]

    @torch.no_grad()
    def flow_with_confidence(self, frames: torch.Tensor):
        """对片段 ``[K,3,H,W]`` 计算前向光流以及前后向一致性（fb-consistency）。

        返回 ``(fwd [F,H,W,2], conf [F,H,W]，取值 [0,1])``，其中 ``F = K-1``。
        """
        a, b = frames[:-1], frames[1:]
        fwd = self.flow(a, b)
        bwd = self.flow(b, a)
        conf = self._fb_consistency(fwd, bwd)
        return fwd, conf

    @staticmethod
    def _fb_consistency(fwd: torch.Tensor, bwd: torch.Tensor, alpha: float = 0.05, beta: float = 0.5):
        """置信度，取值 [0,1]：前向/后向光流相互抵消处置信度高。"""
        f, h, w, _ = fwd.shape
        grid = pixel_grid(h, w, device=fwd.device, dtype=fwd.dtype)        # [H,W,2]
        src = grid + fwd
        gx = 2.0 * src[..., 0] / max(w - 1, 1) - 1.0
        gy = 2.0 * src[..., 1] / max(h - 1, 1) - 1.0
        samp = torch.stack([gx, gy], dim=-1).unsqueeze(0).expand(f, -1, -1, -1)
        bwd_warp = F.grid_sample(
            bwd.permute(0, 3, 1, 2), samp, align_corners=True, mode="bilinear", padding_mode="border"
        ).permute(0, 2, 3, 1)
        err = (fwd + bwd_warp).pow(2).sum(-1)                              # |fwd + bwd(warp)|^2
        mag = fwd.pow(2).sum(-1) + bwd_warp.pow(2).sum(-1)
        thresh = alpha * mag + beta
        conf = (err <= thresh).float()
        return conf
