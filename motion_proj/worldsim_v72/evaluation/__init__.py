"""WorldSim V7.2 的统一评测算子。"""

from motion_proj.worldsim_v72.evaluation.surface_metrics import (
    deterministic_farthest_point_sample,
    evaluate_point_surface,
)

__all__ = ["deterministic_farthest_point_sample", "evaluate_point_surface"]
