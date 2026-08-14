from __future__ import annotations

from scripts.run_worldsim_v5_m2_surface_ablation import surface_selection


def _row(raw0: float, raw3: float, post0: float, post3: float) -> dict:
    def metrics(raw: float, post: float) -> dict:
        return {
            "raw_geometry_error": {"mae_m": raw},
            "post_gaussianization_render_error": {"mae_m": post},
        }

    return {
        "status": "done",
        "frame": 2,
        "camera_id": 0,
        "arms": {
            "G0_ROBUST_PLANE": {"staged_metrics": metrics(raw0, post0)},
            "G3_ROBUST_QUADRATIC": {"staged_metrics": metrics(raw3, post3)},
        },
    }


def _gate() -> dict:
    return {
        "minimum_evaluable_views": 4,
        "minimum_g3_raw_improvement_m": 0.5,
        "minimum_g3_raw_improvement_view_count": 3,
        "require_mean_g3_raw_delta_below_m": 0.0,
        "maximum_mean_g3_post_render_regression_m": 0.1,
    }


def test_g3_requires_matched_raw_and_post_gate() -> None:
    rows = [_row(3.0, 2.0, 2.0, 2.05) for _ in range(4)]
    result = surface_selection(rows, _gate())
    assert result["gate_passed"] is True
    assert result["conclusion"] == "g3_surface_supported_on_model_proxy"


def test_post_render_regression_rejects_g3_even_if_raw_improves() -> None:
    rows = [_row(3.0, 1.0, 2.0, 2.2) for _ in range(4)]
    result = surface_selection(rows, _gate())
    assert result["gate_passed"] is False
    assert result["conclusion"] == "g3_surface_rejected_on_model_proxy"
