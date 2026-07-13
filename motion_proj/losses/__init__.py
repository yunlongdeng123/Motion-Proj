"""Motion-Proj 损失子包。"""
from .anchor import anchor_loss
from .diffusion import real_loss
from .flow import flow_warp_charbonnier_loss
from .projection import projection_loss
from .v2 import correction_v_loss, outside_mask_preserve_v_loss, teacher_relative_v_target
from .tube import bound_gate, edm_weight, sample_tube_sigma

__all__ = [
    "projection_loss",
    "teacher_relative_v_target",
    "correction_v_loss",
    "outside_mask_preserve_v_loss",
    "real_loss",
    "flow_warp_charbonnier_loss",
    "anchor_loss",
    "sample_tube_sigma",
    "bound_gate",
    "edm_weight",
]
