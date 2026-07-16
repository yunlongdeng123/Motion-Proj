from __future__ import annotations

from motion_proj.preference.calibration import measurement_ropes, split_conformal_threshold


def test_measurement_rope_uses_benign_noise_not_human_direction() -> None:
    values = {
        "punc": [0.1, 0.2, 0.3],
        "acceleration": [0.01, 0.02, 0.04],
        "curvature": [0.02, 0.03, 0.05],
        "coherence": [0.1, 0.15, 0.2],
    }
    floors = {name: 0.01 for name in values}
    result = measurement_ropes(values, quantile=0.9, minimums=floors)
    assert set(result) == set(values)
    assert all(row["rope"] >= floors[name] for name, row in result.items())

    conformal = split_conformal_threshold([1.0, 1.5, 2.0], alpha=0.1)
    assert conformal["threshold"] == 2.0
    assert conformal["rank"] == 3
