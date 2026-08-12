from __future__ import annotations

from copy import deepcopy

import numpy as np

from scripts.run_worldsim_v4_m1 import (
    candidate_probability_vectors,
    choose_calibration,
    choose_evidence_arm,
    choose_mask_threshold,
    gate_preview,
)


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


def test_candidate_arm_vectors_share_support_and_zero_pad() -> None:
    field = {
        "hard_instance_id": np.asarray([7, -1, -1]),
        "instance_opacity": np.asarray([0.8, 0.2, 0.3]),
    }
    state = {"posterior": np.asarray([0.9, 0.85, 0.4])}
    ids, vectors = candidate_probability_vectors(
        field=field,
        state=state,
        actor_instance_id=7,
        candidate_arms={"owned_only": None, "posterior_080": 0.8},
    )
    np.testing.assert_array_equal(ids, [0, 1])
    np.testing.assert_allclose(vectors["v33_o1"], [0.8, 0.0])
    np.testing.assert_allclose(vectors["raw__owned_only"], [0.9, 0.0])
    np.testing.assert_allclose(vectors["raw__posterior_080"], [0.9, 0.85])


def test_evidence_arm_selection_preserves_fn_then_maximizes_boundary() -> None:
    metrics = {
        "v33_o1": {"false_negative_semantic_mass": 0.1},
        "raw__wide": {
            "false_negative_semantic_mass": 0.05,
            "boundary_f1": 0.2,
            "iou": 0.2,
            "brier": 0.1,
            "ece": 0.1,
        },
        "raw__narrow": {
            "false_negative_semantic_mass": 0.105,
            "boundary_f1": 0.3,
            "iou": 0.3,
            "brier": 0.1,
            "ece": 0.1,
        },
    }
    assert choose_evidence_arm(
        metrics,
        ["raw__wide", "raw__narrow"],
        false_negative_mass_max_degradation=0.01,
    ) == "raw__narrow"


def test_mask_threshold_selection_respects_fn_gate_then_boundary() -> None:
    search = {
        0.1: {
            "false_negative_semantic_mass": 0.05,
            "boundary_f1": 0.25,
            "iou": 0.2,
            "false_positive_semantic_mass": 0.3,
        },
        0.5: {
            "false_negative_semantic_mass": 0.5,
            "boundary_f1": 0.8,
            "iou": 0.7,
            "false_positive_semantic_mass": 0.1,
        },
    }
    assert choose_mask_threshold(
        search,
        reference={"false_negative_semantic_mass": 0.1},
        false_negative_mass_max_degradation=0.01,
    ) == 0.1
