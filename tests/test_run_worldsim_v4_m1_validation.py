from __future__ import annotations

import pytest

from scripts.run_worldsim_v4_m1 import M1RunError
from scripts.run_worldsim_v4_m1_validation import (
    validation_confirmation_gate,
    verify_runtime_python,
)


def method(boundary: float, fn: float, ece: float, brier: float) -> dict:
    return {
        "aggregate": {
            "boundary_f1": boundary,
            "false_negative_semantic_mass": fn,
            "ece": ece,
            "brier": brier,
            "iou": boundary,
            "false_positive_semantic_mass": 0.1,
        }
    }


def gates() -> dict:
    return {
        "boundary_f1_scene_mean_delta_min": 0.03,
        "false_negative_semantic_mass_delta_max": 0.01,
        "require_ece_or_brier_improvement": True,
        "other_calibration_metric_max_degradation": 0.01,
        "base_rgb_exact": True,
    }


def test_validation_gate_requires_four_of_all_six_scenes() -> None:
    per_scene = {}
    for index in range(4):
        per_scene[f"scene-good-{index}"] = {
            "v33_o1": method(0.4, 0.2, 0.2, 0.2),
            "frozen_m1": method(0.5, 0.205, 0.18, 0.19),
        }
    per_scene["scene-bad"] = {
        "v33_o1": method(0.4, 0.2, 0.2, 0.2),
        "frozen_m1": method(0.39, 0.2, 0.21, 0.21),
    }
    result = validation_confirmation_gate(
        per_scene=per_scene,
        scene_mean={
            "v33_o1": {
                "boundary_f1": 0.4,
                "false_negative_semantic_mass": 0.2,
                "ece": 0.2,
                "brier": 0.2,
            },
            "frozen_m1": {
                "boundary_f1": 0.47,
                "false_negative_semantic_mass": 0.205,
                "ece": 0.18,
                "brier": 0.19,
            },
        },
        gates=gates(),
        required_scene_count=6,
        base_exact=True,
    )
    assert result["status"] == "pass"
    assert result["required_directional_support_scene_count"] == 4
    assert result["directional_support_scene_count"] == 4
    assert result["arm_search_performed"] is False
    assert result["calibration_fit_performed"] is False


def test_validation_gate_counts_abstain_as_no_directional_support() -> None:
    per_scene = {}
    for index in range(3):
        per_scene[f"scene-good-{index}"] = {
            "v33_o1": method(0.4, 0.2, 0.2, 0.2),
            "frozen_m1": method(0.5, 0.2, 0.19, 0.2),
        }
    result = validation_confirmation_gate(
        per_scene=per_scene,
        scene_mean={
            "v33_o1": {
                "boundary_f1": 0.4,
                "false_negative_semantic_mass": 0.2,
                "ece": 0.2,
                "brier": 0.2,
            },
            "frozen_m1": {
                "boundary_f1": 0.5,
                "false_negative_semantic_mass": 0.2,
                "ece": 0.19,
                "brier": 0.2,
            },
        },
        gates=gates(),
        required_scene_count=6,
        base_exact=True,
    )
    assert result["status"] == "reject"
    assert result["checks"]["strict_majority_of_all_required_scenes"] is False


def test_validation_gate_rejects_base_mutation_even_with_quality_gain() -> None:
    per_scene = {
        f"scene-{index}": {
            "v33_o1": method(0.4, 0.2, 0.2, 0.2),
            "frozen_m1": method(0.5, 0.2, 0.19, 0.2),
        }
        for index in range(6)
    }
    result = validation_confirmation_gate(
        per_scene=per_scene,
        scene_mean={
            "v33_o1": {
                "boundary_f1": 0.4,
                "false_negative_semantic_mass": 0.2,
                "ece": 0.2,
                "brier": 0.2,
            },
            "frozen_m1": {
                "boundary_f1": 0.5,
                "false_negative_semantic_mass": 0.2,
                "ece": 0.19,
                "brier": 0.2,
            },
        },
        gates=gates(),
        required_scene_count=6,
        base_exact=False,
    )
    assert result["status"] == "reject"
    assert result["checks"]["base_rgb_and_checkpoint_exact"] is False


def test_validation_runtime_requires_one_frozen_drivestudio_python(tmp_path) -> None:
    runtime = tmp_path / "env" / "bin" / "python"
    configs = [
        {"status": "ready", "runtime": {"drivestudio_python": str(runtime)}},
        {"status": "abstain"},
        {"status": "ready", "runtime": {"drivestudio_python": str(runtime)}},
    ]
    assert verify_runtime_python(configs, executable=runtime) == str(runtime.resolve())
    with pytest.raises(M1RunError, match="frozen DriveStudio Python"):
        verify_runtime_python(configs, executable=tmp_path / "other-python")


def test_validation_runtime_rejects_scene_drift(tmp_path) -> None:
    configs = [
        {
            "status": "ready",
            "runtime": {"drivestudio_python": str(tmp_path / "python-a")},
        },
        {
            "status": "ready",
            "runtime": {"drivestudio_python": str(tmp_path / "python-b")},
        },
    ]
    with pytest.raises(M1RunError, match="runtime Python drift"):
        verify_runtime_python(configs, executable=tmp_path / "python-a")
