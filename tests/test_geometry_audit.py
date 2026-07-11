from motion_proj.eval.geometry_audit import summarize


def _settings():
    return {
        "maximum_static_drift": 1000.0,
        "minimum_depth_lidar_pearson": 0.5,
        "minimum_corruption_detection_rate": 1.0,
        "minimum_mean_eligible_fraction": 0.70,
    }


def _row(eligible=0.8):
    return {
        "static_drift": 4.0,
        "depth_lidar_pearson_mean": 0.85,
        "corruption_detected": True,
        "ego_valid_fraction": 0.9,
        "static_mask_fraction": 0.75,
        "eligible_fraction": eligible,
        "finite": True,
        "track_energy_decreased": True,
    }


def test_geometry_audit_accepts_healthy_rows():
    summary = summarize([_row(), _row()], _settings())
    assert summary["acceptance"]["accepted"]
    assert summary["acceptance"]["failed_checks"] == []


def test_geometry_audit_reports_eligible_gate_failure():
    summary = summarize([_row(0.58), _row(0.62)], _settings())
    assert not summary["acceptance"]["accepted"]
    assert summary["acceptance"]["failed_checks"] == ["eligible_gate"]
