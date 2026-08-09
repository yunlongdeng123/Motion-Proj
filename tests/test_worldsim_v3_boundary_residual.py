from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml

from motion_proj.worldsim_v3.boundary_residual import (
    apply_boundary_scale_cap,
    binary_boundary_band,
    boundary_residual_order,
    photometric_residual_map,
    sample_projected_centers,
    validate_a2_d2_contract,
)


ROOT = Path(__file__).resolve().parents[1]


def contract() -> dict:
    return yaml.safe_load(
        (ROOT / "configs/worldsim_v3/a2_d2_protocol_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_contract_is_frozen_to_d1_and_excludes_d3_d4() -> None:
    payload = contract()
    validate_a2_d2_contract(payload)
    assert payload["paired_intervention"]["order"] == [
        "d1-actor-quota",
        "d2-boundary-residual",
    ]
    assert "depth_residual_ordering" in payload["scope_boundary"][
        "forbidden_in_d2"
    ]
    assert "provenance_aware_pruning" in payload["scope_boundary"][
        "forbidden_in_d2"
    ]


def test_contract_rejects_ordering_drift() -> None:
    payload = contract()
    payload["ordering"]["keys"][0] = "screen_grad_desc"
    with pytest.raises(ValueError, match="ordering key drift"):
        validate_a2_d2_contract(payload)


def test_binary_boundary_band_is_two_sided_and_three_pixels() -> None:
    mask = torch.zeros(15, 15)
    mask[5:10, 5:10] = 1
    band = binary_boundary_band(mask, radius_pixels=3)
    assert band.dtype is torch.bool
    assert band[2, 7]
    assert band[7, 7]
    assert band[12, 7]
    assert not band[0, 0]


def test_binary_boundary_band_treats_image_exterior_as_background() -> None:
    mask = torch.ones(9, 9)
    band = binary_boundary_band(mask, radius_pixels=2)
    assert band[0].all()
    assert band[:, 0].all()
    assert not band[4, 4]


def test_photometric_residual_is_detached_channel_mean_l1() -> None:
    prediction = torch.tensor([[[0.0, 0.5, 1.0]]], requires_grad=True)
    target = torch.tensor([[[1.0, 0.0, 0.5]]])
    residual = photometric_residual_map(prediction, target)
    torch.testing.assert_close(residual, torch.tensor([[2.0 / 3.0]]))
    assert residual.requires_grad is False


def test_projected_center_sampling_skips_invisible_and_outside() -> None:
    boundary = torch.zeros(4, 5)
    boundary[2, 3] = 1
    residual = torch.arange(20, dtype=torch.float32).reshape(4, 5)
    sampled = sample_projected_centers(
        means2d=torch.tensor(
            [[2.6, 1.6], [1.0, 1.0], [-0.1, 1.0], [float("nan"), 0.0]]
        ),
        radii=torch.tensor([2.0, 0.0, 1.0, 1.0]),
        boundary_map=boundary,
        residual_map=residual,
    )
    assert sampled["indices"].tolist() == [0]
    assert sampled["boundary"].tolist() == [1.0]
    assert sampled["photometric_residual"].tolist() == [13.0]


def test_order_is_stable_lexicographic_with_index_tiebreak() -> None:
    candidates = torch.tensor([4, 0, 3, 2, 1], dtype=torch.long)
    boundary_mean = torch.tensor([0.0, 0.5, 0.5, 0.0, float("nan")])
    boundary_count = torch.tensor([1, 1, 1, 1, 0])
    residual_mean = torch.tensor([0.9, 0.2, 0.3, 1.0, 5.0])
    residual_count = torch.tensor([1, 1, 1, 1, 1])
    screen_grad = torch.tensor([0.9, 0.1, 0.1, 0.5, 10.0])
    ranked = boundary_residual_order(
        candidate_indices=candidates,
        boundary_mean=boundary_mean,
        boundary_count=boundary_count,
        residual_mean=residual_mean,
        residual_count=residual_count,
        screen_grad=screen_grad,
    )
    assert ranked.tolist() == [2, 1, 3, 0, 4]


def test_scale_cap_preserves_axis_ratios_and_ignores_non_boundary() -> None:
    scales = torch.tensor([[4.0, 2.0, 1.0], [5.0, 1.0, 1.0], [2.0, 1.0, 0.5]])
    updated, capped = apply_boundary_scale_cap(
        log_scales=torch.log(scales),
        boundary_mean=torch.tensor([0.5, 0.0, float("nan")]),
        boundary_count=torch.tensor([2, 2, 0]),
        maximum_scale=2.0,
    )
    activated = torch.exp(updated)
    assert capped.tolist() == [True, False, False]
    torch.testing.assert_close(activated[0], torch.tensor([2.0, 1.0, 0.5]))
    torch.testing.assert_close(activated[1:], scales[1:])
