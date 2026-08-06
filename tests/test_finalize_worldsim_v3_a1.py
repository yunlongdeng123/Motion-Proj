from copy import deepcopy

from scripts.finalize_worldsim_v3_a1 import assess_confirmation


def row(boundary_status: str = "done") -> dict:
    return {
        "global_lpips": 0.2,
        "e1": {"median": 0.05, "p90": 0.14, "coverage": 0.07},
        "e2": {
            "boundary-support": {"status": boundary_status, "coverage": 0.4 if boundary_status == "done" else None, "mean": 0.004 if boundary_status == "done" else None, "p90": 0.009 if boundary_status == "done" else None},
            "high-support": {"status": "done", "coverage": 0.25, "mean": 0.005, "p90": 0.010},
        },
        "actors": {
            "boundary-support": {"status": boundary_status, "actor_lpips": 0.13 if boundary_status == "done" else None, "boundary_lpips": 0.04 if boundary_status == "done" else None},
            "high-support": {"status": "done", "actor_lpips": 0.05, "boundary_lpips": 0.03},
        },
    }


def test_abstain_role_is_excluded_without_replacement() -> None:
    baseline = row("ABSTAIN")
    candidate = deepcopy(baseline)
    candidate["e1"]["median"] = 0.04
    candidate["e1"]["p90"] = 0.13
    result = assess_confirmation(candidate, baseline)
    assert result["e2"]["roles"]["boundary-support"]["status"] == "ABSTAIN"
    assert result["eligible"] is True


def test_lower_e2_errors_with_lower_coverage_fail_contract() -> None:
    baseline = row()
    candidate = deepcopy(baseline)
    candidate["e1"]["median"] = 0.04
    candidate["e1"]["p90"] = 0.13
    candidate["e2"]["high-support"]["mean"] = 0.004
    candidate["e2"]["high-support"]["p90"] = 0.009
    candidate["e2"]["high-support"]["coverage"] = 0.24
    result = assess_confirmation(candidate, baseline)
    assert result["e1"]["improved"] is True
    assert result["e2"]["non_degraded"] is False
    assert result["eligible"] is False


def test_lpips_degradation_rejects_otherwise_eligible_candidate() -> None:
    baseline = row()
    candidate = deepcopy(baseline)
    candidate["e1"]["median"] = 0.04
    candidate["e1"]["p90"] = 0.13
    candidate["actors"]["high-support"]["actor_lpips"] = 0.06
    result = assess_confirmation(candidate, baseline)
    assert result["appearance"]["acceptable"] is False
    assert result["eligible"] is False
