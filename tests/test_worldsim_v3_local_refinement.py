from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from motion_proj.worldsim_v3.local_refinement import (
    affected_pixel_mask,
    audit_optimizer_frozen_rows,
    auditable_update_rows,
    audit_frozen_rows,
    classify_support,
    load_a3_refinement_sidecar,
    LocalRefinementGuard,
    measured_background_support_mask,
    merge_s_b_row_observations,
    mutable_parameter_fields,
    projected_background_rows,
    snapshot_optimizer_state,
    validate_a3_protocol,
    validate_a3_sidecar_manifest,
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


def _sidecar_manifest(arrays_sha256: str) -> dict:
    return {
        "schema_version": 1,
        "task_id": "WS-V3-A3-LOCAL-REFINE-01",
        "audit_version": "A3-R1-SIDECAR-v1",
        "variant": "r1-reactivate",
        "formal_training_authorized": False,
        "protocol_sha256": "1" * 64,
        "checkpoint_sha256": "2" * 64,
        "background_point_count": 4,
        "arrays": {
            "path": "rows.npz",
            "sha256": arrays_sha256,
            "affected_rows_key": "affected_background_rows",
            "support_strata_key": "support_strata_codes",
        },
        "evidence": {
            "support_provenance_complete": True,
            "heldout_frames": list(range(10, 200, 10)),
            "heldout_excluded_from_support": True,
            "typed_depth_truth_tiers": {
                "depth_render_expected": "diagnostic",
                "depth_surface_first_hit": "T1",
                "depth_lidar_measured": "T0",
            },
        },
    }


def test_sidecar_load_excludes_affected_s_c_and_checks_sha(tmp_path: Path) -> None:
    arrays_path = tmp_path / "rows.npz"
    np.savez(
        arrays_path,
        affected_background_rows=np.array([True, True, True, False]),
        support_strata_codes=np.array([0, 1, 2, 0], dtype=np.uint8),
    )
    arrays_sha256 = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    manifest = _sidecar_manifest(arrays_sha256)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_a3_refinement_sidecar(
        manifest_path,
        protocol_sha256="1" * 64,
        checkpoint_sha256="2" * 64,
    )
    assert loaded.mutable_rows.tolist() == [True, True, False, False]

    manifest["arrays"]["sha256"] = "3" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="arrays SHA drift"):
        load_a3_refinement_sidecar(
            manifest_path,
            protocol_sha256="1" * 64,
            checkpoint_sha256="2" * 64,
        )


def test_sidecar_manifest_rejects_incomplete_provenance() -> None:
    manifest = _sidecar_manifest("3" * 64)
    manifest["evidence"]["support_provenance_complete"] = False
    with pytest.raises(ValueError, match="provenance is incomplete"):
        validate_a3_sidecar_manifest(
            manifest,
            protocol_sha256="1" * 64,
            checkpoint_sha256="2" * 64,
        )


def test_guard_masks_gradients_and_audits_adam_outside_exact() -> None:
    parameters = {
        "Background._opacities": torch.nn.Parameter(torch.ones(4, 1)),
        "Background._scales": torch.nn.Parameter(torch.ones(4, 3)),
        "Background._means": torch.nn.Parameter(torch.ones(4, 3)),
        "RigidNodes._means": torch.nn.Parameter(torch.ones(2, 3)),
    }
    optimizer = torch.optim.Adam(
        [
            {"params": [parameter], "name": name, "lr": 0.01}
            for name, parameter in parameters.items()
        ],
        lr=0.0,
    )
    mutable_rows = torch.tensor([True, True, False, False])
    guard = LocalRefinementGuard(
        parameters=parameters,
        optimizer=optimizer,
        mutable_rows=mutable_rows,
    )
    before = {name: value.detach().clone() for name, value in parameters.items()}
    sum(value.square().sum() for value in parameters.values()).backward()
    gradient_audit = guard.before_optimizer_step()
    assert gradient_audit["checks"]["Background._opacities"]["inside_nonzero"] == 2
    assert gradient_audit["checks"]["Background._opacities"]["inside_l2"] > 0
    assert parameters["Background._means"].grad is None
    assert torch.count_nonzero(parameters["Background._scales"].grad[~mutable_rows]) == 0
    optimizer.step()
    audit = guard.after_optimizer_step()
    assert audit["pass"]
    assert torch.equal(
        before["Background._opacities"][~mutable_rows],
        parameters["Background._opacities"].detach()[~mutable_rows],
    )
    assert torch.equal(
        before["RigidNodes._means"], parameters["RigidNodes._means"].detach()
    )


def test_optimizer_audit_detects_outside_momentum_drift() -> None:
    parameter = torch.nn.Parameter(torch.zeros(3, 1))
    optimizer = torch.optim.Adam([parameter], lr=0.1)
    optimizer.zero_grad()
    parameter.grad = torch.zeros_like(parameter)
    optimizer.step()
    parameters = {"Background._opacities": parameter}
    before = snapshot_optimizer_state(optimizer, parameters)
    optimizer.state[parameter]["exp_avg"][2] = 1
    after = snapshot_optimizer_state(optimizer, parameters)
    audit = audit_optimizer_frozen_rows(
        before,
        after,
        parameter_shapes={"Background._opacities": parameter.shape},
        mutable_rows=torch.tensor([True, False, False]),
        mutable_fields={"Background._opacities"},
    )
    assert not audit["pass"]


def test_measured_support_excludes_both_actor_footprints() -> None:
    affected = np.ones((3, 4), dtype=bool)
    source = np.zeros_like(affected)
    edited = np.zeros_like(affected)
    source[0, 0] = True
    edited[0, 1] = True
    depth = np.ones((3, 4), dtype=np.float32)
    depth[0, 2] = 0
    result = measured_background_support_mask(
        affected_mask=affected,
        source_actor_footprint=source,
        edited_actor_footprint=edited,
        depth_lidar_measured=depth,
    )
    assert not result[0, 0] and not result[0, 1] and not result[0, 2]
    assert result[1:, :].all()


def test_projected_rows_keep_checkpoint_order_and_visibility() -> None:
    means = torch.tensor(
        [[[0.49, 0.49], [2.0, 1.0], [3.0, 2.0], [1.0, 1.0]]]
    )
    radii = torch.tensor([[1, 1, 0, 1]], dtype=torch.int32)
    mask = np.zeros((3, 4), dtype=bool)
    mask[0, 0] = True
    mask[1, 2] = True
    rows = projected_background_rows(
        means2d=means,
        radii=radii,
        pixel_mask=mask,
        background_point_count=3,
    )
    assert rows.tolist() == [True, True, False]


def test_s_b_merge_keeps_unsupported_affected_rows_s_c() -> None:
    affected, strata = merge_s_b_row_observations(
        [
            (
                torch.tensor([True, True, False, False]),
                torch.tensor([False, True, False, False]),
            ),
            (
                torch.tensor([False, False, True, False]),
                torch.tensor([False, False, True, False]),
            ),
        ],
        background_point_count=4,
    )
    assert affected.tolist() == [True, True, True, False]
    assert strata.tolist() == [2, 1, 1, 2]
