from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from motion_proj.worldsim_v3.local_refinement import (
    affected_pixel_mask,
    auditable_update_rows,
    audit_frozen_rows,
    classify_support,
    mutable_parameter_fields,
    validate_a3_protocol,
)


PROJECT = Path(__file__).resolve().parents[1]


def contract() -> dict:
    return yaml.safe_load(
        (PROJECT / "configs/worldsim_v3/a3_local_refine_protocol_v1.yaml").read_text(
            encoding="utf-8"
        )
    )


def test_frozen_a3_protocol_is_valid() -> None:
    validate_a3_protocol(contract())


def test_protocol_cannot_authorize_formal_training() -> None:
    value = contract()
    value["formal_training_authorized"] = True
    with pytest.raises(ValueError, match="not authorized"):
        validate_a3_protocol(value)


def test_protocol_rejects_untracked_v2_dependency_drift() -> None:
    value = contract()
    value["committed_inputs"]["forbidden_dependencies"].pop()
    with pytest.raises(ValueError, match="untracked V2"):
        validate_a3_protocol(value)


def test_protocol_keeps_s_b_rgb_loss_forbidden() -> None:
    value = contract()
    value["support_strata"]["S-B-geometric"]["rgb_loss_allowed"] = True
    with pytest.raises(ValueError, match="must not use RGB"):
        validate_a3_protocol(value)


def test_observed_support_requires_excluded_target_view() -> None:
    with pytest.raises(ValueError, match="exclude the target"):
        classify_support(
            alternate_observed_rgb=True,
            target_excluded=False,
            calibrated_reprojection=True,
            lidar_measured_valid=False,
            multiview_geometry_count=0,
        )


def test_support_precedence_and_geometric_fallback() -> None:
    assert (
        classify_support(
            alternate_observed_rgb=True,
            target_excluded=True,
            calibrated_reprojection=True,
            lidar_measured_valid=True,
            multiview_geometry_count=3,
        )
        == "S-A-observed"
    )
    assert (
        classify_support(
            alternate_observed_rgb=False,
            target_excluded=True,
            calibrated_reprojection=False,
            lidar_measured_valid=True,
            multiview_geometry_count=0,
        )
        == "S-B-geometric"
    )
    assert (
        classify_support(
            alternate_observed_rgb=False,
            target_excluded=True,
            calibrated_reprojection=False,
            lidar_measured_valid=False,
            multiview_geometry_count=2,
        )
        == "S-B-geometric"
    )


def test_unsupported_support_stays_s_c() -> None:
    assert (
        classify_support(
            alternate_observed_rgb=False,
            target_excluded=True,
            calibrated_reprojection=False,
            lidar_measured_valid=False,
            multiview_geometry_count=1,
        )
        == "S-C-unsupported"
    )


def test_affected_mask_unions_inputs_then_dilates() -> None:
    source = np.zeros((9, 9), dtype=bool)
    edited = np.zeros_like(source)
    hole = np.zeros_like(source)
    source[1, 1] = True
    edited[4, 4] = True
    hole[7, 7] = True
    result = affected_pixel_mask(
        source, edited, supported_hole=hole, dilation_radius=1
    )
    assert result.sum() == 27
    assert result[4, 4] and result[0, 0] and result[8, 8]


def test_auditable_rows_exclude_s_c() -> None:
    rows = auditable_update_rows(
        torch.tensor([True, True, True, False]),
        ["S-A-observed", "S-B-geometric", "S-C-unsupported", "S-A-observed"],
    )
    assert rows.tolist() == [True, True, False, False]


def test_r1_mutable_fields_are_only_background_opacity_and_scale() -> None:
    assert mutable_parameter_fields("r1-reactivate") == {
        "Background._opacities",
        "Background._scales",
    }
    with pytest.raises(ValueError, match="cardinality"):
        mutable_parameter_fields("r3-hole-seed")


def test_frozen_row_audit_allows_only_authorized_inside_change() -> None:
    before = {
        "Background._opacities": torch.zeros(3, 1),
        "Background._means": torch.zeros(3, 3),
        "RigidNodes._opacities": torch.zeros(2, 1),
    }
    after = deepcopy(before)
    after["Background._opacities"] = before["Background._opacities"].clone()
    after["Background._opacities"][1] = 1
    audit = audit_frozen_rows(
        before,
        after,
        mutable_rows=torch.tensor([False, True, False]),
        mutable_fields={"Background._opacities"},
    )
    assert audit["pass"]


def test_frozen_row_audit_detects_outside_or_actor_drift() -> None:
    before = {
        "Background._scales": torch.zeros(3, 3),
        "RigidNodes._means": torch.zeros(2, 3),
    }
    outside = {name: value.clone() for name, value in before.items()}
    outside["Background._scales"][0] = 1
    audit = audit_frozen_rows(
        before,
        outside,
        mutable_rows=torch.tensor([False, True, False]),
        mutable_fields={"Background._scales"},
    )
    assert not audit["pass"]

    actor = {name: value.clone() for name, value in before.items()}
    actor["RigidNodes._means"][0] = 1
    audit = audit_frozen_rows(
        before,
        actor,
        mutable_rows=torch.tensor([False, True, False]),
        mutable_fields={"Background._scales"},
    )
    assert not audit["pass"]
