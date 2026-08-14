from __future__ import annotations

from scripts.run_worldsim_v5_m2_actor_geometry import mechanism


def _row(raw: float, delta: float) -> dict:
    return {
        "status": "done",
        "staged_metrics": {
            "raw_geometry_error": {"mae_m": raw},
            "gaussianization_delta_mae_m": delta,
        },
    }


def _gate() -> dict:
    return {
        "minimum_evaluable_request_count": 18,
        "raw_failure_mae_m": 0.5,
        "raw_failure_fraction_to_unlock_g1": 0.6,
        "median_raw_failure_mae_m_to_unlock_g1": 0.5,
        "gaussianization_primary_delta_mae_m": 0.1,
        "gaussianization_primary_fraction": 0.6,
    }


def test_per_actor_gate_unlocks_g1_without_union_mask_bias() -> None:
    rows = [_row(1.0, 0.0) for _ in range(14)] + [_row(0.1, 0.0) for _ in range(8)]
    result = mechanism(rows, _gate())
    assert result["g1_unlocked_for_next_development_run"] is True
    assert result["gaussianization_primary"] is False
    assert result["conclusion"] == "per_actor_g0_builder_primary_g1_unlocked"


def test_per_actor_gate_requires_evaluable_denominator() -> None:
    result = mechanism([_row(2.0, 1.0) for _ in range(17)], _gate())
    assert result["conclusion"] == "insufficient_per_actor_requests"
    assert result["g1_unlocked_for_next_development_run"] is False
