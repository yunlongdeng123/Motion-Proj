from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_worldsim_v51_stage_a_screening.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v51_stage_a_screening", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


METRICS = (
    "iou_at_frozen_threshold",
    "boundary_f1",
    "false_positive_semantic_mass",
    "false_negative_semantic_mass",
    "brier",
    "ece",
    "nll",
)


def _scene(boundary_delta, coverage=0.7, accepted=0.1, abstained=0.2):
    delta = {name: 0.0 for name in METRICS}
    delta["boundary_f1"] = boundary_delta
    return {
        "evaluation_delta_a1_minus_b3": delta,
        "a2_selective_metrics": {
            "coverage": coverage,
            "accepted_subset_error": accepted,
            "abstained_subset_error": abstained,
        },
    }


GATE = {
    "required_scene_count": 2,
    "boundary_f1_nonnegative_scene_count_minimum": 2,
    "boundary_f1_clearly_positive_scene_count_minimum": 1,
    "boundary_f1_clear_delta_minimum_inclusive": 0.001,
    "mean_boundary_f1_delta_minimum_exclusive": 0.0,
    "mean_false_negative_semantic_mass_delta_maximum": 0.02,
    "calibration": {
        "maximum_mean_brier_delta": 0.005,
        "maximum_mean_ece_delta": 0.005,
    },
    "unknown_candidate": {
        "mean_coverage_minimum": 0.60,
        "abstained_subset_error_must_exceed_accepted": True,
    },
}
POLICY = {"maximum_survivors_after_s": 1}


def test_a2_is_selected_when_conditional_and_selective_gates_pass():
    result = RUNNER.evaluate_screening_gate(
        [_scene(0.001), _scene(0.0)], GATE, POLICY
    )
    assert result["passed_conditional_gate"] is True
    assert result["passed_a2_selective_gate"] is True
    assert result["selected_survivor"] == "A2"


def test_a1_is_fallback_when_unknown_coverage_fails():
    result = RUNNER.evaluate_screening_gate(
        [_scene(0.001, coverage=0.5), _scene(0.0, coverage=0.5)],
        GATE,
        POLICY,
    )
    assert result["passed_conditional_gate"] is True
    assert result["passed_a2_selective_gate"] is False
    assert result["selected_survivor"] == "A1"


def test_u2_is_fallback_when_conditional_gate_fails():
    result = RUNNER.evaluate_screening_gate(
        [_scene(-0.001), _scene(0.002)], GATE, POLICY
    )
    assert result["passed_conditional_gate"] is False
    assert result["selected_survivor"] == "U2_B3"
