from motion_proj.eval.multicamera_eval import P3_CAMERAS, aggregate_camera_summaries


def test_p3_has_five_non_front_cameras_and_macro_aggregation():
    assert len(P3_CAMERAS) == 5
    assert "CAM_FRONT" not in P3_CAMERAS
    summaries = {
        camera: {"adapters": {"base": {"aggregate": {"fvd8_mean": float(index)}}}}
        for index, camera in enumerate(P3_CAMERAS)
    }
    result = aggregate_camera_summaries(summaries)
    assert result["macro"]["base"]["fvd8_mean"] == 2.0
    assert result["multi_camera_sync"] is False
