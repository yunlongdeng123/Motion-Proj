from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_h_uplift import validate_config


CONFIG = ROOT / "configs/worldsim_v51/stage_b_h_uplift_v1.yaml"


def test_h_uplift_config_binds_three_scenes_45_views_and_quality_locks() -> None:
    config, records, record_by_view = validate_config(CONFIG)
    assert [scene["name"] for scene in config["scenes"]] == [
        "scene-0471",
        "scene-1087",
        "scene-0379",
    ]
    assert len(records) == 45
    assert len(record_by_view) == 45
    assert config["view_contract"]["views_per_scene"] == 15
    assert config["view_contract"]["model_native_renderer_size_wh"] == [800, 450]
    assert config["operator"]["sparse_transpose"] == "scipy_csr_float64"
    assert config["operator"]["dense_intersection_feature_materialization"] is False
    assert config["operator"]["minimum_intersection_contribution"] == 1e-4
    assert config["operator"]["minimum_gaussian_view_mass"] == 1e-3
    assert config["locks"]["h_feature_sidecar_read"] is True
    assert config["locks"]["h_renderer_start"] is True
    assert config["locks"]["uplift_feature_compute"] is True
    assert config["locks"]["pixel_rgb_values_consumed"] is False
    assert config["locks"]["membership_proxy_read"] is False
    assert config["locks"]["method_quality_read"] is False
    assert config["locks"]["validation_quality_read"] is False
    assert config["locks"]["test_quality_read"] is False
    assert config["locks"]["m2_status"] == "pending"
    assert config["locks"]["m3_status"] == "pending"
