from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_worldsim_v51_unary_visibility.py"
SPEC = importlib.util.spec_from_file_location("worldsim_v51_a1_runner", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def _delta(boundary: float, iou: float, fn: float, brier: float, ece: float):
    return {
        "iou_at_frozen_threshold": iou,
        "boundary_f1": boundary,
        "false_positive_semantic_mass": -0.01,
        "false_negative_semantic_mass": fn,
        "brier": brier,
        "ece": ece,
        "nll": -0.01,
    }


def test_stage_a_gate_requires_scene_replication_and_calibration() -> None:
    gate = {
        "boundary_f1_positive_scene_count_minimum": 2,
        "mean_boundary_f1_delta_minimum_exclusive": 0.0,
        "mean_iou_delta_minimum": 0.0,
        "mean_fn_semantic_mass_delta_maximum": 0.02,
        "brier_or_ece_must_improve": True,
    }
    result = RUNNER._stage_a_gate(
        [
            _delta(0.02, 0.01, 0.01, -0.01, 0.0),
            _delta(0.01, 0.00, 0.01, 0.00, -0.01),
            _delta(-0.005, 0.00, 0.01, 0.00, 0.0),
        ],
        gate,
    )
    assert result["passed"] is True
    assert result["positive_boundary_f1_scene_count"] == 2

    failed = RUNNER._stage_a_gate(
        [
            _delta(0.02, 0.01, 0.01, 0.01, 0.01),
            _delta(-0.01, 0.00, 0.01, 0.01, 0.01),
            _delta(-0.01, 0.00, 0.01, 0.01, 0.01),
        ],
        gate,
    )
    assert failed["passed"] is False
    assert failed["checks"]["boundary_f1_positive_scene_count"] is False
    assert failed["checks"]["brier_or_ece_improves"] is False
