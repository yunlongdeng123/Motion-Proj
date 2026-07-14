from __future__ import annotations

import torch

from motion_proj.auditor.state import Track
from motion_proj.diagnostics.target_validity import (
    dilate_latent_mask,
    make_hybrid_latent,
    source_duplication_rows,
)


def _track(token: str, box: list[float]) -> Track:
    xyxy = torch.tensor([box, box], dtype=torch.float32)
    return Track(token, "generated_point/dynamic_residual", xyxy, torch.ones(2), torch.ones(2, dtype=torch.bool))


def test_hybrid_is_exact_outside_mask_and_frame0_is_hard_frozen():
    base = torch.zeros(1, 3, 4, 2, 2)
    full = torch.ones_like(base)
    mask = torch.zeros(1, 3, 1, 2, 2)
    mask[:, 1, :, 0, 0] = 1
    hybrid = make_hybrid_latent(base, full, mask)
    assert torch.equal(hybrid[:, 0], base[:, 0])
    assert hybrid[0, 1, 0, 0, 0] == 1
    assert hybrid[0, 1, 1, 1, 1] == 0


def test_dilation_keeps_frame0_mask_zero():
    mask = torch.zeros(3, 1, 5, 5)
    mask[1, 0, 2, 2] = 1
    dilated = dilate_latent_mask(mask, 1)
    assert torch.count_nonzero(dilated[0]) == 0
    assert int(torch.count_nonzero(dilated[1])) == 9


def test_source_duplication_proxy_flags_retained_source_and_new_destination():
    base = torch.zeros(2, 3, 12, 12)
    target = base.clone()
    # source box [1:4,1:4] remains zero; destination [7:10,1:4] gains content.
    target[:, :, 1:4, 7:10] = 1
    source = _track("generated_dynamic_residual_000", [1, 1, 4, 4])
    projected = _track("generated_dynamic_residual_000", [7, 1, 10, 4])
    rows = source_duplication_rows(
        base, target, [source], [projected],
        minimum_destination_change_l1=1.0e-4,
        maximum_source_change_l1=1.0e-6,
        maximum_overlap_iou=0.5,
    )
    assert len(rows) == 2
    assert all(row["source_retained_duplication_proxy"] for row in rows)
