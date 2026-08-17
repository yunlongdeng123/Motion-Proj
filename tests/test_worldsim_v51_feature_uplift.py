from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as functional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v51.feature_uplift import (
    sample_patch_grid_bilinear,
    uplift_b0_b1,
)
from motion_proj.worldsim_v51.protocol import load_yaml, sha256_file
from scripts.audit_worldsim_v51_stage_b_operator_parity import (
    _independent_dense_reference,
    _synthetic_payload,
    execute_parity,
    validate_config,
)


CONFIG = ROOT / "configs/worldsim_v51/stage_b_operator_parity_v1.yaml"


def test_operator_config_binds_ludvig_resource_and_locks() -> None:
    config, resource_freeze = validate_config(CONFIG)
    assert resource_freeze["status"] == "done"
    assert config["ludvig_source"]["license"] == "non-commercial"
    assert config["ludvig_source"]["vendored_into_project"] is False
    assert config["operator"]["optional_pruning"] is False
    assert config["locks"]["method_quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"


def test_b0_b1_match_independent_dense_reference() -> None:
    config, _ = validate_config(CONFIG)
    payload = _synthetic_payload(config)
    result = uplift_b0_b1(
        gaussian_id=payload["gaussian_id"],
        view_id=payload["view_id"],
        contribution_weight=payload["contribution_weight"],
        pixel_features=payload["pixel_features"],
        gaussian_count=config["synthetic"]["gaussian_count"],
        minimum_intersection_contribution=config["operator"][
            "minimum_intersection_contribution"
        ],
        minimum_gaussian_view_mass=config["operator"][
            "minimum_gaussian_view_mass"
        ],
        epsilon=config["operator"]["epsilon"],
    )
    reference_b0, reference_b1 = _independent_dense_reference(payload, config)
    assert np.allclose(result["b0_feature"], reference_b0, atol=1e-6, rtol=0)
    assert np.allclose(result["b1_feature"], reference_b1, atol=1e-6, rtol=0)
    assert np.linalg.norm(result["b1_feature"] - result["b0_feature"]) > 1e-4
    assert result["report"]["dropped_gaussian_view_count"] >= 1


def test_duplicate_rows_and_permuted_chunks_are_bit_exact() -> None:
    gaussian = np.asarray([0, 0, 1, 0, 1, 1], dtype=np.int64)
    view = np.asarray([2, 2, 2, 3, 3, 3], dtype=np.int64)
    weight = np.asarray([0.3, 0.2, 0.4, 0.1, 0.25, 0.15], dtype=np.float64)
    feature = np.asarray(
        [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12]],
        dtype=np.float32,
    )
    first = uplift_b0_b1(
        gaussian_id=gaussian,
        view_id=view,
        contribution_weight=weight,
        pixel_features=feature,
        gaussian_count=3,
    )
    order = np.asarray([5, 2, 0, 4, 1, 3])
    second = uplift_b0_b1(
        gaussian_id=gaussian[order],
        view_id=view[order],
        contribution_weight=weight[order],
        pixel_features=feature[order],
        gaussian_count=3,
    )
    for name in (
        "b0_feature",
        "b1_feature",
        "b0_denominator",
        "b1_denominator",
        "supported_view_count",
    ):
        assert np.array_equal(first[name], second[name])
    assert np.array_equal(first["b0_feature"][2], np.zeros(2, dtype=np.float32))
    assert np.array_equal(first["b1_feature"][2], np.zeros(2, dtype=np.float32))


def test_constant_feature_is_conserved_on_covered_gaussians() -> None:
    gaussian = np.asarray([0, 0, 1, 1], dtype=np.int64)
    view = np.asarray([0, 1, 0, 1], dtype=np.int64)
    weight = np.asarray([2.0, 0.5, 0.25, 3.0], dtype=np.float64)
    constant = np.asarray([1.5, -2.0, 0.25], dtype=np.float32)
    result = uplift_b0_b1(
        gaussian_id=gaussian,
        view_id=view,
        contribution_weight=weight,
        pixel_features=np.broadcast_to(constant, (4, 3)).copy(),
        gaussian_count=3,
    )
    assert np.allclose(result["b0_feature"][:2], constant, atol=1e-6, rtol=0)
    assert np.allclose(result["b1_feature"][:2], constant, atol=1e-6, rtol=0)
    assert result["supported_view_count"].tolist() == [2, 2, 0]


def test_lazy_bilinear_matches_dense_align_corners_false() -> None:
    grid = np.asarray(
        [
            [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]],
            [[-1.0, 0.5, 2.0], [3.5, 5.0, 6.5]],
        ],
        dtype=np.float32,
    )
    pixels = np.arange(4 * 6, dtype=np.int64)
    lazy = sample_patch_grid_bilinear(
        grid, pixels, image_height=4, image_width=6
    )
    dense = (
        functional.interpolate(
            torch.from_numpy(grid).unsqueeze(0),
            size=(4, 6),
            mode="bilinear",
            align_corners=False,
        )[0]
        .permute(1, 2, 0)
        .reshape(-1, 2)
        .numpy()
    )
    assert np.allclose(lazy, dense, atol=2e-6, rtol=0)
    assert lazy.dtype == np.float32


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("gaussian_id", np.asarray([3]), "越界"),
        ("view_id", np.asarray([-1]), "非负"),
        ("contribution_weight", np.asarray([-0.1]), "非负"),
        ("pixel_features", np.asarray([[np.nan]], dtype=np.float32), "finite"),
    ],
)
def test_uplift_rejects_invalid_payload(field: str, value: np.ndarray, message: str) -> None:
    payload = {
        "gaussian_id": np.asarray([0]),
        "view_id": np.asarray([0]),
        "contribution_weight": np.asarray([0.5]),
        "pixel_features": np.asarray([[1.0]], dtype=np.float32),
        "gaussian_count": 2,
    }
    payload[field] = value
    with pytest.raises(ValueError, match=message):
        uplift_b0_b1(**payload)


def test_full_synthetic_parity_suite_passes() -> None:
    config, _ = validate_config(CONFIG)
    report = execute_parity(config)
    assert all(report["checks"].values())
    assert report["b0_max_absolute_error"] <= 1e-6
    assert report["b1_max_absolute_error"] <= 1e-6
    assert report["lazy_bilinear_max_absolute_error"] <= 2e-6


def test_operator_parity_freeze_binds_terminal_and_quality_locks() -> None:
    freeze = load_yaml(
        ROOT / "configs/worldsim_v51/stage_b_operator_parity_freeze_v1.yaml"
    )
    run = Path(freeze["canonical_run"]["path"])
    for relative, expected in freeze["canonical_run"]["hashes"].items():
        assert (run / relative).is_file()
        assert sha256_file(run / relative) == expected
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert all(summary["report"]["checks"].values())
    assert summary["checkpoint_immutable"] is True
    assert freeze["parity"]["b0_max_absolute_error"] == 0.0
    assert freeze["parity"]["b1_max_absolute_error"] == 0.0
    assert summary["real_image_feature_read"] is False
    assert summary["method_quality_read"] is False
    assert summary["validation_quality_read"] is False
    assert summary["test_quality_read"] is False
    assert freeze["locks"]["m2_status"] == "pending"
    assert freeze["locks"]["m3_status"] == "pending"
