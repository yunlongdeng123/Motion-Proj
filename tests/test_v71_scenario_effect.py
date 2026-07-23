from motion_proj.resim.safety_geometry import OrientedBox
from motion_proj.resim.scenario_effect import ScenarioThresholds, evaluate_scenario_effect


def _boxes(y_values):
    return [OrientedBox((12 + i * 0.1, y, 0), (4, 1.8, 1.5), 0) for i, y in enumerate(y_values)]


def test_positive_and_matched_negative_are_machine_determinate():
    thresholds = ScenarioThresholds()
    source = _boxes([4.0] * 10)
    positive = _boxes([4.0, 3.5, 2.8, 2.0, 1.2, 0.8, 0.4, 0.2, 0.0, 0.0])
    negative = _boxes([4.0, 3.9, 3.8, 3.7, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1])
    pos = evaluate_scenario_effect(
        source, positive, ego_speed_mps=6.0, corridor_half_width_m=1.75,
        dt_s=0.1, thresholds=thresholds, corridor_source="proxy",
    )
    neg = evaluate_scenario_effect(
        source, negative, ego_speed_mps=6.0, corridor_half_width_m=1.75,
        dt_s=0.1, thresholds=thresholds, corridor_source="proxy",
    )
    assert pos["positive"] and pos["label_transition"] == "0->1"
    assert neg["negative"] and neg["label_transition"] == "0->0"
