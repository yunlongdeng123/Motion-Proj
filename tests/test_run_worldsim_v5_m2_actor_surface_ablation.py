from __future__ import annotations

from scripts.run_worldsim_v5_m2_actor_surface_ablation import selection


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
