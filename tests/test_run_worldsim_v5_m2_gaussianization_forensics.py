from __future__ import annotations

from pathlib import Path

import numpy as np

from scripts.run_worldsim_v5_m2_gaussianization_forensics import (
    effective_opacity_gain,
    load_config,
    mechanism_decision,
)


def _row(base: float, opaque: float, dense: float, combined: float) -> dict:
    return {
        "status": "done",
        "baseline_replay_exact": True,
        "arms": {
            "BASE": {"post_geometry_error": {"mae_m": base}},
            "OPAQUE": {"post_geometry_error": {"mae_m": opaque}},
            "DENSE": {"post_geometry_error": {"mae_m": dense}},
            "DENSE_OPAQUE": {"post_geometry_error": {"mae_m": combined}},
        },
    }


def _gate() -> dict:
    return {
        "minimum_evaluable_request_count": 18,
        "minimum_post_mae_improvement_m": 0.1,
        "minimum_improvement_request_count": 14,
        "require_mean_delta_below_m": 0.0,
        "require_median_delta_below_m": 0.0,
    }


def test_opacity_gain_removes_existing_background_occupancy() -> None:
    base = np.asarray([0.0, 0.5, 0.9])
    composed = np.asarray([0.5, 0.75, 0.95])
    assert np.allclose(effective_opacity_gain(base, composed), 0.5)


def test_mechanism_requires_broad_support_and_does_not_select_arm() -> None:
    rows = [_row(3.0, 2.5, 3.1, 2.4) for _ in range(18)]
    result = mechanism_decision(rows, _gate())
    assert result["candidate_deltas"]["OPAQUE"]["mechanism_gate_passed"] is True
    assert result["candidate_deltas"]["DENSE"]["mechanism_gate_passed"] is False
    assert result["method_arm_selected"] is False
    assert result["baseline_replay_exact_count"] == 18


def test_formal_config_freezes_factor_arms_without_search() -> None:
    project = Path(__file__).resolve().parents[1]
    config = load_config(
        project / "configs/worldsim_v5/m2_gaussianization_forensics_scene0471_v1.yaml"
    )
    assert [arm["id"] for arm in config["gaussianization_arms"]] == [
        "BASE",
        "OPAQUE",
        "DENSE",
        "DENSE_OPAQUE",
    ]
    assert config["scope"]["parameter_search_performed"] is False
    assert config["scope"]["method_arm_selection_performed"] is False
