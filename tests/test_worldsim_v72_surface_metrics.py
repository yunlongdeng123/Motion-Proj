from __future__ import annotations

import numpy as np

from motion_proj.worldsim_v72.evaluation.surface_metrics import (
    deterministic_farthest_point_sample,
    evaluate_point_surface,
)


def test_surface_generation_is_target_independent_and_return_partition_is_literal() -> None:
    inputs = np.asarray([[3.0, 0.0, 0.0], [5.0, 0.0, 0.0], [5.0, 1.0, 0.0]], dtype=np.float32)
    surface = deterministic_farthest_point_sample(inputs, 2)
    target = np.asarray([[5.0, 0.0, 0.0], [5.0, 1.0, 0.0]], dtype=np.float32)
    origins = np.zeros((2, 3), dtype=np.float32)
    result = evaluate_point_surface(
        surface,
        target,
        origins,
        lateral_tolerance_m=0.05,
        depth_tolerance_m=0.05,
        distance_chunk_size=2,
        ray_chunk_size=2,
        point_chunk_size=2,
    )
    counts = result["conditional_return"]
    assert counts["early_count"] == 1
    assert counts["hit_count"] == 1
    assert counts["miss_count"] == 0
    assert counts["full_return_metrics_supported"] is False
