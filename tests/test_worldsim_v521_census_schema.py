from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from motion_proj.worldsim_v521.census import (
    CensusError,
    CensusProtocol,
    assert_unique_keys,
    evaluate_discovery_view,
    temporal_proxy_row,
)


ROOT = Path(__file__).resolve().parents[1]


def frozen_record(target: Path) -> dict:
    import hashlib

    return {
        "dataset": "nuscenes",
        "scene": "scene-0001",
        "scene_index": 1,
        "frame": 2,
        "sample_token": None,
        "canonical_sample_index": 2,
        "partition": "discovery",
        "camera": 0,
        "target_path": str(target),
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "eligible_bases": ["adgs", "streetgs"],
        "quality_decoded": False,
    }


def test_contract_freezes_ranking_before_quality() -> None:
    config = yaml.safe_load((ROOT / "configs/worldsim_v521/census_protocol_v1.yaml").read_text(encoding="utf-8"))
    assert config["status"] == "frozen_before_discovery_quality_read"
    assert config["partition"]["allowed_quality_partition"] == "discovery"
    assert config["partition"]["confirmation_pixel_decode"] is False
    assert config["ranking_contract"]["scalar_composite_score"] == "forbidden"
    assert config["ranking_contract"]["tie_break"] == ["scene", "canonical_sample_index", "camera", "base"]
    assert config["metrics"]["geometry"]["status"] == "undefined_no_comparable_base_depth"


def test_discovery_view_schema_and_undefined_regions(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    prediction = tmp_path / "prediction.png"
    dynamic = tmp_path / "dynamic.png"
    Image.fromarray(np.full((8, 12, 3), 80, dtype=np.uint8)).save(target)
    Image.fromarray(np.full((8, 12, 3), 70, dtype=np.uint8)).save(prediction)
    Image.fromarray(np.zeros((8, 12), dtype=np.uint8)).save(dynamic)
    protocol = CensusProtocol(metric_width=12, metric_height=8)
    base_row, actor_row = evaluate_discovery_view(
        base="adgs",
        record=frozen_record(target),
        prediction_path=prediction,
        dynamic_mask_path=dynamic,
        lpips_model=None,
        renderer_provenance={"source": "fixture"},
        resource={},
        protocol=protocol,
    )
    assert base_row["partition"] == "discovery"
    assert base_row["metrics"]["actor"]["status"] == "undefined_empty_region"
    assert base_row["metrics"]["geometry"]["status"] == "undefined_no_comparable_base_depth"
    assert actor_row["undefined_reason"] == "undefined_no_instance_region"
    assert actor_row["region_provenance"]["is_ground_truth"] is False


def test_confirmation_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    dynamic = tmp_path / "dynamic.png"
    Image.new("RGB", (12, 8)).save(target)
    Image.new("L", (12, 8)).save(dynamic)
    record = frozen_record(target)
    record["partition"] = "confirmation"
    with pytest.raises(CensusError, match="Discovery"):
        evaluate_discovery_view(
            base="streetgs", record=record, prediction_path=target, dynamic_mask_path=dynamic,
            lpips_model=None, renderer_provenance={}, resource={},
            protocol=CensusProtocol(metric_width=12, metric_height=8),
        )


def test_temporal_proxy_cannot_name_temporal_failure(tmp_path: Path) -> None:
    target = tmp_path / "target.png"
    Image.new("RGB", (12, 8)).save(target)
    earlier = frozen_record(target)
    later = dict(earlier, frame=7, canonical_sample_index=7)
    shape = (8, 12, 3)
    zeros = np.zeros(shape, dtype=np.float64)
    ones = np.ones(shape, dtype=np.float64) * 0.2
    mask = np.zeros(shape[:2], dtype=bool)
    row = temporal_proxy_row(
        earlier=earlier, later=later,
        earlier_prediction=zeros, earlier_target=zeros,
        later_prediction=ones, later_target=zeros,
        earlier_dynamic=mask, later_dynamic=mask,
        protocol=CensusProtocol(metric_width=12, metric_height=8),
    )
    assert row["status"] == "unwarped_temporal_proxy"
    assert row["may_trigger_b_temporal"] is False
    assert row["may_trigger_b_occ"] is False
    assert row["metrics"]["global_residual_change_l1"] == pytest.approx(0.2)


def test_duplicate_primary_key_rejected() -> None:
    rows = [{"base": "adgs", "scene": "s", "frame": 2, "camera": 0}] * 2
    with pytest.raises(CensusError, match="重复主键"):
        assert_unique_keys(rows, ("base", "scene", "frame", "camera"))
