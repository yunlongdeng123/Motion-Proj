import pytest

from scripts.run_dr_v2_m5_perception import match_detections


def detection(box, score=0.9, label="car"):
    return {"box_xyxy": box, "score": score, "label": label}


def test_match_detections_reports_false_disappearance_and_class_change() -> None:
    result = match_detections(
        [detection([0, 0, 10, 10]), detection([20, 20, 30, 30])],
        [detection([0, 0, 10, 10], score=0.8, label="truck")],
    )
    assert result["match_count"] == 1
    assert result["matched_iou_mean"] == pytest.approx(1.0)
    assert result["false_disappearance_rate"] == pytest.approx(0.5)
    assert result["class_change_rate"] == pytest.approx(1.0)
