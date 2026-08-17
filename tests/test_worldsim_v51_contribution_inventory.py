from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
import pytest
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.smoke_worldsim_v51_one_view_contribution import (
    summarize_contributions,
    validate_config,
)
from motion_proj.worldsim_v51.protocol import sha256_file


CONFIG = ROOT / "configs/worldsim_v51/stage_b_one_view_contribution_v4.yaml"


def test_one_view_config_binds_operator_checkpoint_and_locks() -> None:
    config, operator_freeze = validate_config(CONFIG)
    assert operator_freeze["status"] == "done"
    assert config["scene"]["expected_total_gaussians"] == 859613
    assert config["scene"]["sensor_image_size_wh"] == [1600, 900]
    assert config["scene"]["source_downscale_when_loading"] == [2, 2, 2]
    assert config["scene"]["model_native_renderer_size_wh"] == [800, 450]
    assert config["runtime"]["python"].endswith("/envs/drivestudio/bin/python")
    assert config["runtime"]["required_imports"] == ["pytorch3d", "gsplat"]
    assert config["contribution"]["consume_pixel_rgb_values"] is False
    assert config["contribution"][
        "dataset_infrastructure_materializes_image_mask_lidar"
    ] is True
    assert config["contribution"]["consume_membership_proxy"] is False
    assert config["locks"]["method_quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"
    assert config["resources"]["maximum_nvidia_peak_mib"] == 16384
    assert config["resources"]["maximum_torch_reserved_peak_mib"] == 16384
    assert config["recovery_from"]["observed_nvidia_peak_mib"] == 14234
    assert config["recovery_from"]["status_sha256"] == (
        "8b8ebe17c71b74bbb91c1e468e1bb4f1e8594b2f7ac344df2f61dac32b2118bf"
    )


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


def test_one_view_contribution_freeze_binds_terminal_and_locks() -> None:
    freeze = yaml.safe_load(
        (
            ROOT
            / "configs/worldsim_v51/stage_b_one_view_contribution_freeze_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    run = Path(freeze["canonical_run"]["path"])
    for relative, expected in freeze["canonical_run"]["hashes"].items():
        assert (run / relative).is_file()
        assert sha256_file(run / relative) == expected

    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["conclusion"] == "one_h_view_renderer_contribution_denominator_ready"
    assert summary["source_commit"] == freeze["canonical_run"]["source_commit"]
    assert summary["checkpoint_immutable"] is True
    assert summary["contribution_inventory"]["gaussian_after_view_mass_floor"] == 313764
    assert summary["contribution_inventory"]["intersection_rows_persisted"] is False
    assert summary["resource"]["nvidia_smi_peak_used_mib"] == 14234
    assert summary["pixel_rgb_values_consumed"] is False
    assert summary["lidar_values_consumed"] is False
    assert summary["membership_proxy_read"] is False
    assert summary["method_quality_read"] is False
    assert summary["validation_quality_read"] is False
    assert summary["test_quality_read"] is False
    assert freeze["locks"]["m2_status"] == "pending"
    assert freeze["locks"]["m3_status"] == "pending"
