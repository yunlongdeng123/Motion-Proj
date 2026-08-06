from copy import deepcopy

from scripts.finalize_worldsim_v3_a1_dev import (
    assess_alternative,
    select_candidate,
)


def sample_row(variant: str = "c0-off") -> dict:
    return {
        "variant": variant,
        "global_lpips": 0.18,
        "total_gaussians": 100,
        "train_seconds": 10.0,
        "e1": {"median": 0.06, "p90": 0.15, "coverage": 0.10},
        "e2": {
            "boundary-support": {"mean": 0.0035, "p90": 0.0063, "coverage": 0.35},
            "high-support": {"mean": 0.0048, "p90": 0.0109, "coverage": 0.26},
        },
        "actors": {
            "boundary-support": {"actor_lpips": 0.07, "boundary_lpips": 0.05},
            "high-support": {"actor_lpips": 0.09, "boundary_lpips": 0.05},
        },
    }


def test_boundary_only_e2_gain_is_not_endpoint_improvement() -> None:
    baseline = sample_row()
    candidate = deepcopy(baseline)
    candidate["variant"] = "c2-factorized-isp"
    candidate["e2"]["boundary-support"]["mean"] = 0.0030
    candidate["e2"]["boundary-support"]["p90"] = 0.0060
    candidate["e2"]["high-support"]["p90"] = 0.0110
    result = assess_alternative(candidate, baseline)
    assert result["e2"]["roles"]["boundary-support"]["improved"] is True
    assert result["e2"]["roles"]["high-support"]["non_degraded"] is False
    assert result["e2"]["improved"] is False
    assert result["eligible"] is False


def test_primary_gain_requires_other_endpoint_and_lpips_non_degradation() -> None:
    baseline = sample_row()
    candidate = deepcopy(baseline)
    candidate["variant"] = "c3-bounded-pose"
    candidate["e1"]["median"] = 0.05
    candidate["e1"]["p90"] = 0.14
    result = assess_alternative(candidate, baseline)
    assert result["e1"]["improved"] is True
    assert result["e2"]["non_degraded"] is True
    assert result["appearance"]["acceptable"] is True
    assert result["eligible"] is True

    candidate["actors"]["high-support"]["actor_lpips"] = 0.091
    assert assess_alternative(candidate, baseline)["eligible"] is False


def test_selection_falls_back_to_c0_and_registers_exact_alias() -> None:
    rows = {"c0-off": sample_row()}
    for variant in ("c1-native", "c2-factorized-isp", "c3-bounded-pose"):
        row = sample_row(variant)
        row["e1"]["median"] += 0.01
        row["e1"]["p90"] += 0.01
        rows[variant] = row
    decision = select_candidate(rows)
    assert decision["selected_variant"] == "c0-off"
    assert decision["decision_status"] == "done_off"
    assert decision["fallback_used"] is True
    assert decision["confirmation"]["selected_c_star_is_exact_alias"] is True
    assert decision["confirmation"]["unique_training_runs"] == 8


def test_tie_break_prefers_lower_global_lpips_after_primary_count() -> None:
    rows = {"c0-off": sample_row()}
    for variant in ("c1-native", "c2-factorized-isp", "c3-bounded-pose"):
        row = sample_row(variant)
        row["e1"]["median"] = 0.05
        row["e1"]["p90"] = 0.14
        rows[variant] = row
    rows["c2-factorized-isp"]["global_lpips"] = 0.16
    rows["c3-bounded-pose"]["global_lpips"] = 0.17
    decision = select_candidate(rows)
    assert decision["selected_variant"] == "c2-factorized-isp"
    assert decision["decision_status"] == "done_enhanced"
