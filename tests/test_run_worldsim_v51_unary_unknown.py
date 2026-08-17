from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_worldsim_v51_unary_unknown.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v51_a2_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _scene(boundary: float, coverage: float, accepted: float, abstained: float):
    return {
        "evaluation_delta_a2_minus_b3": {
            "iou_at_frozen_threshold": 0.01,
            "boundary_f1": boundary,
            "false_positive_semantic_mass": -0.01,
            "false_negative_semantic_mass": 0.01,
            "brier": -0.01,
            "ece": 0.0,
            "nll": -0.01,
        },
        "selective_metrics": {
            "coverage": coverage,
            "accepted_subset_error": accepted,
            "abstained_subset_error": abstained,
        },
    }


def _gate():
    return {
        "boundary_f1_positive_scene_count_minimum_vs_b3": 2,
        "mean_boundary_f1_delta_minimum_exclusive_vs_b3": 0.0,
        "mean_iou_delta_minimum_vs_b3": 0.0,
        "mean_fn_semantic_mass_delta_maximum_vs_b3": 0.02,
        "brier_or_ece_must_improve_vs_b3": True,
        "mean_coverage_minimum": 0.60,
        "abstained_subset_error_must_exceed_accepted": True,
    }


def test_unknown_gate_requires_error_concentration_and_coverage() -> None:
    result = RUNNER._unknown_gate(
        [
            _scene(0.02, 0.8, 0.1, 0.3),
            _scene(0.01, 0.7, 0.1, 0.2),
            _scene(-0.005, 0.6, 0.2, 0.3),
        ],
        _gate(),
    )
    assert result["passed"] is True
    assert result["scene_balanced_mean_coverage"] == pytest.approx(0.7)
    assert result["scene_balanced_error_separation"] > 0.0


def test_unknown_gate_fails_closed_on_empty_abstained_subset() -> None:
    rows = [
        _scene(0.02, 1.0, 0.1, 0.3),
        _scene(0.01, 0.8, 0.1, 0.2),
        _scene(-0.005, 0.7, 0.2, 0.3),
    ]
    rows[0]["selective_metrics"]["abstained_subset_error"] = None
    result = RUNNER._unknown_gate(rows, _gate())
    assert result["passed"] is False
    assert (
        result["selective_checks"][
            "all_scenes_have_accepted_and_abstained_pixels"
        ]
        is False
    )
