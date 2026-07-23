import numpy as np
import pytest

from motion_proj.resim.label_regeneration import (
    TypedDepth,
    cumulative_first_hit,
    lidar_measured_depth,
    render_expected_depth,
)


def test_expected_first_hit_and_measured_depth_are_typed_distinctly():
    expected = render_expected_depth(np.array([[4.2]]), np.array([[0.8]]))
    first = cumulative_first_hit(
        np.array([[[2.0, 4.0, 7.0]]]),
        np.array([[[0.2, 0.5, 0.8]]]),
        threshold=0.6,
    )
    measured = lidar_measured_depth(np.array([[3.1, 0.0]]))
    assert expected.name == "depth_render_expected" and expected.truth_tier == "diagnostic"
    assert first.value[0, 0] == 4.0 and first.truth_tier == "T1"
    assert measured.valid.tolist() == [[True, False]] and measured.truth_tier == "T0"
    with pytest.raises(ValueError):
        TypedDepth(np.ones((1, 1)), np.ones((1, 1), bool), "depth_render_expected", "T0", "bad")
