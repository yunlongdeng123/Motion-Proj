from __future__ import annotations

from motion_proj.worldsim_v51.superprimitive_evaluation import evaluate_e0b_h_gate


def _metrics(boundary: float, iou: float, fn: float) -> dict[str, float]:
    return {
        "boundary_f1": boundary,
        "iou_at_frozen_threshold": iou,
        "false_negative_semantic_mass": fn,
        "false_positive_semantic_mass": 0.1,
        "brier": 0.1,
        "ece": 0.1,
        "nll": 0.1,
    }


def _gates() -> tuple[dict[str, float | int], dict[str, float | int]]:
    primary = {
        "scene_count": 3,
        "minimum_positive_boundary_f1_scenes": 2,
        "minimum_scene_balanced_boundary_f1_delta_exclusive": 0.0,
        "minimum_scene_balanced_iou_delta": 0.0,
        "maximum_scene_balanced_false_negative_semantic_mass_delta": 0.02,
    }
    mechanism = {
        "scene_count": 3,
        "minimum_nonnegative_boundary_f1_scenes": 2,
        "minimum_scene_balanced_boundary_f1_delta_exclusive": 0.0,
        "minimum_scene_balanced_iou_delta": 0.0,
        "maximum_scene_balanced_false_negative_semantic_mass_delta": 0.0,
    }
    return primary, mechanism


def test_e0b_h_gate_requires_primary_and_mechanism_pass() -> None:
    scenes = []
    for name, gain in zip(("a", "b", "c"), (0.03, 0.02, -0.01)):
        baseline = _metrics(0.3, 0.4, 0.2)
        d0 = _metrics(0.31, 0.405, 0.21)
        e0b = _metrics(0.31 + gain, 0.41, 0.205)
        scenes.append(
            {
                "scene": name,
                "evaluation_aggregate": {"U2_B3_G0": baseline, "D0": d0, "E0B": e0b},
            }
        )
    primary, mechanism = _gates()
    report = evaluate_e0b_h_gate(scenes, primary, mechanism)
    assert report["pass"] is True
    assert report["primary_gate"]["pass"] is True
    assert report["mechanism_gate"]["pass"] is True

    scenes[2]["evaluation_aggregate"]["E0B"]["false_negative_semantic_mass"] = 0.5
    report = evaluate_e0b_h_gate(scenes, primary, mechanism)
    assert report["pass"] is False
    assert report["mechanism_gate"]["pass"] is False
