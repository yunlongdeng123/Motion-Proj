from __future__ import annotations

import torch

from motion_proj.preference.common_support import build_common_support, paired_masked_means
from motion_proj.preference.paired_tracks import PairedQuerySet, RawTrackObservation


def _query_set() -> PairedQuerySet:
    points = torch.tensor([[float(index * 10), float((index % 3) * 12)] for index in range(8)])
    return PairedQuerySet(
        points=points,
        query_ids=tuple([f"bg-{i}" for i in range(4)] + [f"dyn-{i}" for i in range(4)]),
        strata=tuple(["background"] * 4 + ["dynamic"] * 4),
        selection_scores=torch.ones(8),
        valid=True,
    )


def _observation(candidate: str, query_set: PairedQuerySet, visibility: torch.Tensor) -> RawTrackObservation:
    points = query_set.points[:, None].repeat(1, 4, 1)
    points[:, :, 0] += torch.arange(4).float()[None]
    points[~visibility] = float("nan")
    return RawTrackObservation(
        candidate_id=candidate,
        query_set_hash=query_set.query_set_hash,
        raw_points=points,
        raw_visibility=visibility,
        raw_confidence=visibility.float(),
        forward_backward_error=torch.where(visibility, torch.zeros_like(visibility, dtype=torch.float32), torch.nan),
    )


def test_common_support_uses_same_denominator_and_dropout_reduces_coverage() -> None:
    query_set = _query_set()
    full = torch.ones(8, 4, dtype=torch.bool)
    dropped = full.clone()
    dropped[4:, 2:] = False
    a = _observation("a", query_set, full)
    b_full = _observation("b-full", query_set, full)
    b_drop = _observation("b-drop", query_set, dropped)
    settings = {
        "minimum_common_dynamic_queries": 4,
        "minimum_common_background_queries": 4,
        "minimum_common_dynamic_observations": 12,
        "minimum_spatial_clusters": 1,
        "minimum_frames_per_query": 3,
        "min_confidence": 0.5,
        "max_fb_error_px": 1.0,
        "cluster_cell_size_px": 16.0,
    }

    complete = build_common_support(query_set, a, b_full, settings, window_starts=(0,))[0]
    attacked = build_common_support(query_set, a, b_drop, settings, window_starts=(0,))[0]

    assert complete.valid
    assert not attacked.valid
    assert attacked.diagnostics["common_dynamic_denominator"] < complete.diagnostics["common_dynamic_denominator"]
    assert attacked.diagnostics["expected_dynamic_denominator"] == complete.diagnostics["expected_dynamic_denominator"]
    values_a = torch.ones(8, 4)
    values_b = torch.full((8, 4), 2.0)
    mean_a, mean_b, count = paired_masked_means(values_a, values_b, complete.window_mask)
    assert (mean_a, mean_b, count) == (1.0, 2.0, 32)
