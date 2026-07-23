import numpy as np

from motion_proj.resim.certificates import (
    ComponentResult,
    Verdict,
    aggregate_components,
    kinematic_certificate,
    unavailable_component,
)


def test_unknown_is_never_coerced_to_pass():
    values = [
        ComponentResult("kinematic", Verdict.PASS, {}, "ok"),
        unavailable_component("road_support", "map_unavailable"),
    ]
    assert aggregate_components(values) is Verdict.UNKNOWN
    assert aggregate_components(values, required=["kinematic"]) is Verdict.PASS


def test_fail_dominates_unknown_and_teleport_is_detected():
    result = kinematic_certificate(
        np.asarray([[0, 0, 0], [1, 0, 0], [30, 0, 0]]),
        np.asarray([0.0, 0.1, 0.2]),
        max_speed_mps=60,
        max_acceleration_mps2=20,
        max_step_m=5,
    )
    assert result.verdict is Verdict.FAIL
    assert aggregate_components(
        [result, unavailable_component("road_support", "map_unavailable")]
    ) is Verdict.FAIL
