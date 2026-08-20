from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_m123_overlap_contract_is_frozen_and_fail_closed() -> None:
    config = yaml.safe_load((ROOT / "configs/worldsim_v521/m123_reaudit_v1.yaml").read_text(encoding="utf-8"))
    assert config["status"] == "frozen_before_exact_overlap_quality_read"
    assert config["exact_overlap"]["historical_aggregate_mapping_forbidden"] is True
    assert config["exact_overlap"]["minimum_independent_scenes"] == 2
    assert config["m2"]["router_refit"] is False
    assert config["m3"]["parameter_search"] is False
    assert config["quality_locks"] == {
        "validation": False, "test": False, "kitti": False, "confirmation_before_p9": False,
    }
