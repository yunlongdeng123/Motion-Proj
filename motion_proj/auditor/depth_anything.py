"""用于自车光流（ego-flow）的单目深度。若可用则使用 Depth-Anything，
否则退化为常数平面（constant plane），以便流水线无需额外权重即可运行。"""
from __future__ import annotations

import torch

from ..utils.logging import get_logger

log = get_logger(__name__)


def relative_inverse_to_depth(
    prediction: torch.Tensor,
    default_depth: float,
    min_relative_disparity: float = 0.05,
) -> torch.Tensor:
    """将可能带符号的相对逆深度 logits 转成有限、正的深度代理。"""
    finite = prediction[torch.isfinite(prediction)]
    if finite.numel() == 0:
        return torch.full_like(prediction, default_depth)
    lo = torch.quantile(finite, 0.01)
    hi = torch.quantile(finite, 0.99)
    disparity = torch.nan_to_num(
        prediction, nan=float(lo), posinf=float(hi), neginf=float(lo),
    ).clamp(lo, hi)
    disparity = disparity - lo + 1e-3
    disparity = disparity / disparity.median().clamp_min(1e-3)
    return default_depth / disparity.clamp_min(min_relative_disparity)


class DepthEstimator:
    def __init__(self, device: str = "cuda", model_id: str = "depth-anything/Depth-Anything-V2-Small-hf",
                 default_depth: float = 20.0, enable: bool = True):
        self.device = device
        self.default_depth = float(default_depth)
        self.pipe = None
        if enable:
            try:
                from transformers import pipeline

                self.pipe = pipeline("depth-estimation", model=model_id, device=0 if device == "cuda" else -1)
                log.info("Loaded depth model %s", model_id)
            except Exception as e:  # pragma: no cover - 取决于权重/网络
                log.warning("Depth model unavailable (%s); using constant-plane fallback", e)

    @torch.no_grad()
    def depth(self, frames: torch.Tensor) -> torch.Tensor:
        """``[K,3,H,W]``（取值 [-1,1]）-> 近似度量深度 ``[K,H,W]``（相机 z 值）。"""
        k, _, h, w = frames.shape
        if self.pipe is None:
            return torch.full((k, h, w), self.default_depth, device=frames.device)
        from PIL import Image

        out = []
        for i in range(k):
            img = ((frames[i] + 1) / 2).clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
            pred = self.pipe(Image.fromarray(img))["predicted_depth"]
            d = pred.to(frames.device).float()
            if d.dim() == 3:
                d = d[0]
            d = torch.nn.functional.interpolate(
                d[None, None], size=(h, w), mode="bilinear", align_corners=False
            )[0, 0]
            # 模型输出是相对逆深度（值越大表示越近），不是可直接反投影的 z 深度。
            d = relative_inverse_to_depth(d, self.default_depth)
            out.append(d)
        return torch.stack(out, 0)
