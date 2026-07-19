import numpy as np

from motion_proj.diagnostics.resim_c1_screen import action_error, evaluate_gates, mean_abs_uint8


def test_action_error_prefers_matching_class_and_turn():
    good = action_error(
        {"predicted_class": "left", "predicted_displacement_m": 10.0, "predicted_lateral_m": 2.0},
        request_class="left",
        target_displacement_m=10.0,
        target_lateral_m=2.0,
    )
    bad = action_error(
        {"predicted_class": "right", "predicted_displacement_m": 10.0, "predicted_lateral_m": -2.0},
        request_class="left",
        target_displacement_m=10.0,
        target_lateral_m=2.0,
    )
    assert good["action_error"] < bad["action_error"]
    assert good["class_err"] == 0.0
    assert bad["turn_err"] == 1.0


def test_mean_abs_uint8_zero_for_identical():
    frames = np.zeros((4, 8, 8, 3), dtype=np.uint8)
    assert mean_abs_uint8(frames, frames) == 0.0


def test_evaluate_gates_requires_seven_of_eight_e_wins():
    def fake_arm(error, sharp=1.0, flicker=0.1):
        return {
            "action_error": {"action_error": error},
            "quality": {
                "finite": True,
                "sharpness_median": sharp,
                "temporal_l1_max": flicker,
            },
            "proxy_prediction": {"predicted_displacement_m": 0.1},
        }

    moving = []
    for index in range(8):
        e_err = 0.2 if index < 7 else 1.5
        f_err = 1.0
        moving.append({
            "action_class": "forward",
            "paired_improvement": f_err - e_err,
            "future_effect_mae": 5.0,
            "history_effect_mae": 0.5,
            "E": fake_arm(e_err),
            "F": fake_arm(f_err),
        })
    stationary = [{
        "action_class": "stationary",
        "paired_improvement": 0.0,
        "future_effect_mae": 0.0,
        "history_effect_mae": 0.0,
        "E": fake_arm(0.1),
        "F": fake_arm(0.1),
    } for _ in range(2)]
    gates = {
        "minimum_moving_e_wins": 7,
        "minimum_moving_contexts": 8,
        "require_median_improvement_positive": True,
        "future_effect_min_abs": 1.0,
        "history_effect_null_multiplier": 2.0,
        "sharpness_ratio_min": 0.5,
        "flicker_ratio_max": 2.0,
    }
    null = {"future_mean_abs": 0.0, "full_mean_abs": 0.0}
    result = evaluate_gates(moving + stationary, null=null, gates=gates, stationary_p95=1.0)
    assert result["passed"] is True
    assert result["e_wins"] == 7
