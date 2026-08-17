from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_worldsim_v51_h_uplift import validate_config
from motion_proj.worldsim_v51.feature_sidecar import array_sha256
from motion_proj.worldsim_v51.protocol import sha256_file


CONFIG = ROOT / "configs/worldsim_v51/stage_b_h_uplift_v2.yaml"


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
    assert config["resources"]["maximum_nvidia_peak_mib"] == 22528
    assert config["resources"]["maximum_torch_reserved_peak_mib"] == 22528
    assert config["recovery"]["observed_nvidia_peak_mib"] == 20554
    assert config["recovery"]["allowed_change"] == "raise_nvidia_and_torch_resource_ceiling_only"
    assert config["failure_ledger_delta"] == ["V51-F23"]
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


def test_h_uplift_freeze_binds_terminal_manifest_and_sidecars() -> None:
    import yaml

    freeze = yaml.safe_load(
        (
            ROOT / "configs/worldsim_v51/stage_b_h_uplift_freeze_v1.yaml"
        ).read_text(encoding="utf-8")
    )
    run = Path(freeze["canonical_run"]["path"])
    for relative, expected in freeze["canonical_run"]["hashes"].items():
        assert (run / relative).is_file()
        assert sha256_file(run / relative) == expected
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    assert summary["status"] == "done"
    assert summary["report"]["processed_scene_count"] == 3
    assert summary["report"]["processed_view_count"] == 45
    assert all(item["immutable"] for item in summary["report"]["checkpoint_records"])
    feature_manifest = json.loads(
        (run / "artifacts/gaussian_feature_manifest.json").read_text(encoding="utf-8")
    )
    assert len(feature_manifest["records"]) == 6
    for record in feature_manifest["records"]:
        path = run / record["path"]
        assert sha256_file(path) == record["file_sha256"]
        with np.load(path, allow_pickle=False) as archive:
            assert array_sha256(archive["feature"]) == record["feature_content_sha256"]
            assert array_sha256(archive["weight"]) == record["weight_content_sha256"]
            assert (
                array_sha256(archive["supported_view_count"])
                == record["supported_view_count_content_sha256"]
            )
    assert not list(run.rglob("*.partial*"))
    assert summary["membership_proxy_read"] is False
    assert summary["method_quality_read"] is False
    assert summary["validation_quality_read"] is False
    assert summary["test_quality_read"] is False
    assert freeze["locks"]["m2_status"] == "pending"
    assert freeze["locks"]["m3_status"] == "pending"
