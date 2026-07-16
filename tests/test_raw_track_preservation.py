from __future__ import annotations

import torch

from motion_proj.preference.paired_tracks import PairedQuerySet, propagate_common_queries


def test_diagnostic_smoothing_does_not_modify_raw_points() -> None:
    query_set = PairedQuerySet(
        points=torch.tensor([[1.0, 1.0]]),
        query_ids=("dyn-000",),
        strata=("dynamic",),
        selection_scores=torch.ones(1),
        valid=True,
    )
    frames = torch.zeros(4, 3, 8, 8)
    flow = torch.zeros(3, 8, 8, 2)
    flow[0, ..., 0] = 1.0
    flow[1, ..., 0] = 4.0
    flow[2, ..., 0] = -1.0
    backward = -flow
    confidence = torch.ones(3, 8, 8)

    result = propagate_common_queries(
        candidate_id="candidate",
        frames=frames,
        query_set=query_set,
        observed_flow=flow,
        backward_flow=backward,
        flow_confidence=confidence,
        fb_alpha=0.05,
        fb_beta=0.5,
    )

    assert result.optional_smoothed_points is not None
    assert result.diagnostics["raw_preserved"] is True
    assert result.raw_points.data_ptr() != result.optional_smoothed_points.data_ptr()
    assert not torch.allclose(
        result.raw_points, result.optional_smoothed_points, rtol=0.0, atol=0.0, equal_nan=True
    )
    assert torch.equal(result.raw_points[:, 0], query_set.points)
