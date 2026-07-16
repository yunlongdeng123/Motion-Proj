from __future__ import annotations

import torch

from motion_proj.preference.paired_tracks import (
    PairedQuerySet,
    propagate_common_queries,
    select_common_queries,
)


SETTINGS = {
    "min_confidence": 0.5,
    "background_queries": 4,
    "dynamic_queries": 4,
    "minimum_background_queries": 2,
    "minimum_dynamic_queries": 2,
    "min_distance": 1.0,
    "background_residual_quantile": 0.50,
    "dynamic_residual_quantile": 0.75,
    "texture_quantile": 0.0,
}


def _flow_fields() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    flow = torch.zeros(2, 6, 6, 2)
    flow[:, :, 3:, 0] = 3.0
    confidence = torch.ones(2, 6, 6)
    background = torch.zeros_like(flow)
    frame = torch.linspace(-1.0, 1.0, 108).reshape(3, 6, 6)
    return frame, flow, confidence, background


def test_same_query_ids_and_start_points_are_used_across_siblings() -> None:
    frame, flow, confidence, background = _flow_fields()
    query_set = select_common_queries(frame, flow, confidence, background, SETTINGS)
    frames = torch.stack([frame, frame, frame])
    backward = -flow
    first = propagate_common_queries(
        candidate_id="a",
        frames=frames,
        query_set=query_set,
        observed_flow=flow,
        backward_flow=backward,
        flow_confidence=confidence,
        fb_alpha=0.05,
        fb_beta=0.5,
    )
    second = propagate_common_queries(
        candidate_id="b",
        frames=frames,
        query_set=query_set,
        observed_flow=flow,
        backward_flow=backward,
        flow_confidence=confidence,
        fb_alpha=0.05,
        fb_beta=0.5,
    )

    assert query_set.valid
    assert len(set(query_set.query_ids)) == len(query_set.query_ids)
    assert set(query_set.strata) == {"background", "dynamic"}
    assert first.query_set_hash == second.query_set_hash == query_set.query_set_hash
    assert torch.equal(first.raw_points[:, 0], query_set.points)
    assert torch.equal(second.raw_points[:, 0], query_set.points)


def test_query_set_rejects_duplicate_ids_across_strata() -> None:
    try:
        PairedQuerySet(
            points=torch.tensor([[0.0, 0.0], [1.0, 1.0]]),
            query_ids=("q", "q"),
            strata=("background", "dynamic"),
            selection_scores=torch.ones(2),
            valid=True,
        )
    except ValueError as exc:
        assert "唯一" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("重复 query_id 必须 fail closed")
