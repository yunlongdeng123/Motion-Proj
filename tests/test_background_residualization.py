from __future__ import annotations

import torch

from motion_proj.preference.paired_tracks import PairedQuerySet, RawTrackObservation
from motion_proj.preference.residual_motion import fit_sparse_affine_background


def test_sparse_affine_removes_common_translation_without_future_gt() -> None:
    points0 = torch.tensor([[4.0, 4.0], [12.0, 4.0], [4.0, 12.0], [12.0, 12.0]])
    points = points0[:, None].repeat(1, 4, 1)
    points[:, :, 0] += torch.arange(4).float()[None] * 2.0
    query_set = PairedQuerySet(
        points=points0,
        query_ids=tuple(f"bg-{i}" for i in range(4)),
        strata=("background",) * 4,
        selection_scores=torch.ones(4),
        valid=True,
    )
    observation = RawTrackObservation(
        candidate_id="candidate",
        query_set_hash=query_set.query_set_hash,
        raw_points=points,
        raw_visibility=torch.ones(4, 4, dtype=torch.bool),
        raw_confidence=torch.ones(4, 4),
        forward_backward_error=torch.zeros(4, 4),
    )

    field = fit_sparse_affine_background(
        observation,
        query_set,
        torch.ones(4, 4, dtype=torch.bool),
        image_hw=(20, 20),
        min_points=4,
        irls_iterations=3,
    )

    predicted = field.predict(points[:, 0], 0)
    assert field.valid.all()
    assert torch.allclose(predicted, torch.tensor([[2.0, 0.0]]).repeat(4, 1), atol=1.0e-4)
