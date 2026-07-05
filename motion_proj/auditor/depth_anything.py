"""用于自车光流（ego-flow）的单目深度。若可用则使用 Depth-Anything，
否则退化为常数平面（constant plane），以便流水线无需额外权重即可运行。"""
from __future__ import annotations

import torch

from ..utils.logging import get_logger

log = get_logger(__name__)


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
            # Depth-Anything 输出的是相对深度（类似逆深度）；将其映射为一个正的度量代理值。
            d = d / d.mean().clamp_min(1e-6) * self.default_depth
            out.append(d)
        return torch.stack(out, 0)
