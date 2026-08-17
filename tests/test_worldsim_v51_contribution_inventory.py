from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.smoke_worldsim_v51_one_view_contribution import (
    summarize_contributions,
    validate_config,
)


CONFIG = ROOT / "configs/worldsim_v51/stage_b_one_view_contribution_v1.yaml"


def test_one_view_config_binds_operator_checkpoint_and_locks() -> None:
    config, operator_freeze = validate_config(CONFIG)
    assert operator_freeze["status"] == "done"
    assert config["scene"]["expected_total_gaussians"] == 859613
    assert config["contribution"]["consume_pixel_rgb_values"] is False
    assert config["contribution"]["consume_membership_proxy"] is False
    assert config["locks"]["method_quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"


def test_contribution_inventory_applies_both_floors_and_keeps_zero() -> None:
    report = summarize_contributions(
        gaussian_id=np.asarray([0, 0, 1, 1, 1, 2, 2]),
        pixel_id=np.asarray([0, 1, 0, 1, 2, 2, 3]),
        contribution_weight=np.asarray(
            [0.0001, 0.0002, 0.4, 0.3, 0.00001, 0.0001, 0.0001]
        ),
        gaussian_count=4,
        pixel_count=5,
        minimum_intersection_contribution=1e-4,
        minimum_gaussian_view_mass=1e-3,
        quantiles=[0.0, 0.5, 1.0],
    )
    assert report["raw_intersection_count"] == 7
    assert report["supported_intersection_count"] == 6
    assert report["gaussian_with_intersection_support"] == 3
    assert report["gaussian_after_view_mass_floor"] == 1
    assert report["gaussian_dropped_by_view_mass_floor"] == 2
    assert report["gaussian_coverage_after_mass_floor"] == 0.25
    assert report["intersection_rows_persisted"] is False


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("gaussian_id", np.asarray([4]), "gaussian_id"),
        ("pixel_id", np.asarray([5]), "pixel_id"),
        ("contribution_weight", np.asarray([np.nan]), "finite"),
    ],
)
def test_contribution_inventory_rejects_invalid_rows(
    name: str, value: np.ndarray, message: str
) -> None:
    payload = {
        "gaussian_id": np.asarray([0]),
        "pixel_id": np.asarray([0]),
        "contribution_weight": np.asarray([0.5]),
        "gaussian_count": 4,
        "pixel_count": 5,
        "minimum_intersection_contribution": 1e-4,
        "minimum_gaussian_view_mass": 1e-3,
        "quantiles": [0.0, 0.5, 1.0],
    }
    payload[name] = value
    with pytest.raises(ValueError, match=message):
        summarize_contributions(**payload)
