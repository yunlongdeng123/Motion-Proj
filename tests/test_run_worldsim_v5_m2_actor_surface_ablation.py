from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from scripts.run_worldsim_v5_m2_actor_surface_ablation import (
    _raw_depth_payload,
    load_config,
    selection,
)


def _row(g0: float, g1: float) -> dict:
    return {
        "status": "done",
        "arms": {
            "G0_ROBUST_PLANE": {"raw_geometry_error": {"mae_m": g0}},
            "G1_PIECEWISE_PLANE": {"raw_geometry_error": {"mae_m": g1}},
        },
    }


def _gate() -> dict:
    return {
        "minimum_evaluable_request_count": 18,
        "minimum_g1_raw_improvement_m": 0.5,
        "minimum_g1_raw_improvement_request_count": 14,
        "require_mean_g1_raw_delta_below_m": 0.0,
        "require_median_g1_raw_delta_below_m": 0.0,
    }


def test_g1_requires_broad_not_outlier_only_support() -> None:
    rows = [_row(4.0, 3.0) for _ in range(14)] + [_row(2.0, 2.1) for _ in range(8)]
    result = selection(rows, _gate())
    assert result["gate_passed"] is True
    assert result["g1_raw_improvement_request_count"] == 14


def test_g1_rejects_mean_only_outlier_win() -> None:
    rows = [_row(4.0, 4.1) for _ in range(21)] + [_row(100.0, 0.0)]
    result = selection(rows, _gate())
    assert result["gate_passed"] is False
    assert result["conclusion"] == "g1_piecewise_surface_rejected_on_model_proxy"


def test_g2_uses_generic_frozen_gate() -> None:
    arms = ("G0_ROBUST_PLANE", "G2_MOVING_LEAST_SQUARES")
    rows = [
        {
            "status": "done",
            "arms": {
                arms[0]: {"raw_geometry_error": {"mae_m": 4.0}},
                arms[1]: {"raw_geometry_error": {"mae_m": 3.0}},
            },
        }
        for _ in range(18)
    ]
    gate = {
        "minimum_evaluable_request_count": 18,
        "minimum_candidate_raw_improvement_m": 0.5,
        "minimum_candidate_raw_improvement_request_count": 14,
        "require_mean_candidate_raw_delta_below_m": 0.0,
        "require_median_candidate_raw_delta_below_m": 0.0,
    }
    result = selection(rows, gate, arms)
    assert result["gate_passed"] is True
    assert result["g2_raw_improvement_request_count"] == 18
    assert result["conclusion"] == "g2_moving_least_squares_surface_supported_on_model_proxy"


def test_g2_formal_config_freezes_next_surface_arm() -> None:
    project = Path(__file__).resolve().parents[1]
    config = load_config(
        project / "configs/worldsim_v5/m2_actor_g0_g2_surface_scene0471_v1.yaml"
    )
    assert config["phase"] == "per_actor_g0_g2_raw_surface_ablation"
    assert config["surface"]["matched_models"] == [
        "G0_ROBUST_PLANE",
        "G2_MOVING_LEAST_SQUARES",
    ]
    assert config["scope"]["gaussianization_started"] is False


def test_artifact_payload_uses_model_names_not_metric_mapping_indices() -> None:
    arms = ("G0_ROBUST_PLANE", "G2_MOVING_LEAST_SQUARES")
    states = {
        arms[0]: {"fit": SimpleNamespace(depth=np.asarray([[1.0]]))},
        arms[1]: {"fit": SimpleNamespace(depth=np.asarray([[2.0]]))},
    }
    payload = _raw_depth_payload(states, arms)
    assert sorted(payload) == ["g0_raw_depth", "g2_raw_depth"]
    assert payload["g2_raw_depth"].dtype == np.float16
