from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v521.protocol import (
    ProtocolError,
    canonical_unit_key,
    partition_for_unit,
    temporal_window_partition,
    validate_partition_invariance,
    validate_quality_read,
)


@pytest.mark.parametrize("role", ["validation", "test", "fresh_validation", "fresh_test", "kitti"])
def test_fresh_or_external_quality_role_is_locked(role: str) -> None:
    with pytest.raises(ProtocolError):
        validate_quality_read(split_role=role)


def test_forbidden_quality_path_is_locked_even_under_discovery_role() -> None:
    with pytest.raises(ProtocolError):
        validate_quality_read(split_role="discovery", paths=["/data/fresh_test/render.png"])


def test_partition_hash_consumes_only_canonical_sample() -> None:
    unit_key = canonical_unit_key("nuscenes", "scene-0230", sample_token="abc")
    expected = partition_for_unit(unit_key)
    rows = [
        {**expected, "base": "streetgs", "camera": 0, "actor": "a"},
        {**expected, "base": "adgs", "camera": 2, "actor": "b"},
    ]
    validate_partition_invariance(rows)
    assert rows[0]["partition"] == rows[1]["partition"]
    assert rows[0]["split_hash"] == rows[1]["split_hash"]


def test_cross_partition_temporal_window_is_undefined() -> None:
    result = temporal_window_partition(
        [{"partition": "discovery"}, {"partition": "confirmation"}]
    )
    assert result == {"status": "undefined_cross_partition", "partition": None}
