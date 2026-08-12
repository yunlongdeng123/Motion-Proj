from __future__ import annotations

from copy import deepcopy

from scripts.run_worldsim_v4_m1 import choose_calibration, gate_preview


def test_calibration_selection_uses_brier_then_ece_then_boundary() -> None:
    metrics = {
        "raw": {"brier": 0.2, "ece": 0.1, "boundary_f1": 0.4, "false_negative_semantic_mass": 0.1},
        "temperature": {"brier": 0.1, "ece": 0.09, "boundary_f1": 0.395, "false_negative_semantic_mass": 0.105},
        "beta": {"brier": 0.1, "ece": 0.08, "boundary_f1": 0.2, "false_negative_semantic_mass": 0.4},
    }
    assert choose_calibration(metrics, ["raw", "temperature", "beta"]) == "temperature"
    tied = deepcopy(metrics)
    tied["temperature"]["boundary_f1"] = 0.2
    tied["temperature"]["false_negative_semantic_mass"] = 0.4
    assert choose_calibration(tied, ["raw", "temperature", "beta"]) == "raw"


def test_gate_preview_enforces_all_preregistered_m1_checks() -> None:
    reference = {
        "boundary_f1": 0.20,
        "false_negative_semantic_mass": 0.10,
        "ece": 0.20,
        "brier": 0.20,
    }
    candidate = {
        "boundary_f1": 0.24,
        "false_negative_semantic_mass": 0.105,
        "ece": 0.19,
        "brier": 0.205,
    }
    gates = {
        "boundary_f1_scene_mean_delta_min": 0.03,
        "false_negative_semantic_mass_delta_max": 0.01,
        "other_calibration_metric_max_degradation": 0.01,
    }
    assert gate_preview(
        reference=reference, candidate=candidate, gates=gates, base_rgb_exact=True
    )["status"] == "pass"
    assert gate_preview(
        reference=reference, candidate=candidate, gates=gates, base_rgb_exact=False
    )["status"] == "fail"
