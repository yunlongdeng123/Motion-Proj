"""v8.1 Seen-to-Scene 复现的 CPU 张量原语。

这里只定义遮挡与传播契约；参考帧选择、流估计、细化及扩散训练由上层实现。
"""

from .masks import apply_hole_mask, make_border_outpaint_mask
from .propagation import backward_warp, fuse_reference_latents, resize_flow

__all__ = [
    "apply_hole_mask",
    "backward_warp",
    "fuse_reference_latents",
    "make_border_outpaint_mask",
    "resize_flow",
]
