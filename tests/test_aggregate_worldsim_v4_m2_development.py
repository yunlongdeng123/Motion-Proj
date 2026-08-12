from __future__ import annotations

from scripts.aggregate_worldsim_v4_m2_development import (
    _fixed_arm_rows,
    evaluate_acceptance_gates,
    select_development_operating_point,
)


def _candidate(identifier: str, arm: str, risk: float, error: float) -> dict:
    method = "OBSERVED" if arm == "OBSERVED" else "GENERATED"
    return {
        "arm": arm,
        "candidate": {
            "candidate_id": identifier,
            "method": method,
            "photo_risk": risk,
            "geometry_risk": risk,
            "temporal_risk": risk,
            "uncertainty": risk,
            "compute_cost": risk,
        },
        "metrics": {"edit_error": error},
    }


def _request(scene: str, identifier: str, candidates: list[dict], r0: float) -> dict:
    return {
        "scene": scene,
        "request_id": identifier,
        "candidates": candidates,
        "matched_arms": [
            {
                "arm": "ABSTAIN",
                "status": "atomic_noop",
                "metrics": {
                    "edit_error": r0,
                    "atomic_noop": True,
                },
            }
        ],
    }


def test_selection_is_scene_balanced_and_requires_error_separation() -> None:
    requests = [
        _request("a", "easy", [_candidate("o1", "OBSERVED", 0.1, 0.1)], 1.0),
        _request("b", "hard", [_candidate("t1", "TELEA", 0.9, 0.9)], 1.0),
    ]
    selected, grid = select_development_operating_point(
        requests,
        weight_grid=[
            {
                "name": "balanced",
                "photo": 0.2,
                "geometry": 0.2,
                "temporal": 0.2,
                "uncertainty": 0.2,
                "compute_cost": 0.2,
            }
        ],
        threshold_grid=[0.5],
        tie_priority=["OBSERVED", "DONOR", "GENERATED"],
        require_meaningful_abstention=True,
    )
    assert len(grid) == 1
    assert selected["statistics"]["request_coverage"] == 0.5
    assert selected["statistics"]["meaningful_abstention_gate"] is True
    assert selected["statistics"]["scene_balanced_policy_edit_error"] == 0.55


def test_unavailable_fixed_arm_falls_back_to_atomic_abstain() -> None:
    request = _request("a", "x", [_candidate("t", "TELEA", 0.2, 0.3)], 0.8)
    rows = _fixed_arm_rows([request], "OBSERVED")
    assert rows[0]["available"] is False
    assert rows[0]["policy_arm"] == "ABSTAIN"
    assert rows[0]["metrics"]["edit_error"] == 0.8


def _quality(psnr: float, ssim: float, lpips: float, hole_psnr: float, geom: float, lidar: float) -> dict:
    values = {
        "global_valid_psnr_db": psnr,
        "global_valid_ssim": ssim,
        "global_valid_lpips_alex": lpips,
        "hole_cross_view_psnr_db": hole_psnr,
        "hole_geometry_mae_m": geom,
        "static_lidar_depth_mae_m": lidar,
    }
    return {key: {"scene_balanced_mean": value} for key, value in values.items()}


def test_acceptance_gate_allows_one_improved_hole_endpoint() -> None:
    router = {"quality": _quality(29.95, 0.899, 0.205, 22.0, 1.1, 0.51)}
    baseline = {"arm": "TELEA", "quality": _quality(30.0, 0.9, 0.2, 20.0, 1.0, 0.5)}
    gates = {
        "global_valid": {
            "psnr_delta_db_min": -0.1,
            "ssim_delta_min": -0.002,
            "lpips_delta_max": 0.01,
        },
        "hole_endpoints": {
            "metrics": {
                "hole_cross_view_psnr_db": "maximize",
                "hole_geometry_mae_m": "minimize",
            },
            "same_view_background_gt_metrics": "undefined_not_scored",
        },
        "static_lidar_depth_mae_degradation_m_max": 0.02,
    }
    selective = {
        "abstain_minus_accepted_counterfactual_error": 0.1,
        "meaningful_abstention_gate": True,
    }
    report = evaluate_acceptance_gates(
        router=router, baseline=baseline, selective=selective, gates=gates
    )
    assert report["checks"]["hole_any_endpoint"]["passed"] is True
    assert report["all_gates_passed"] is True
