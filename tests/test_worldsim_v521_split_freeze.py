from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from motion_proj.worldsim_v521.asset_audit import enumerate_quality_blind_targets


def test_shared_sample_partition_is_identical_for_all_cameras(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    for camera in (0, 1, 2):
        (images / f"002_{camera}.jpg").write_bytes(f"camera-{camera}".encode())
    samples, views = enumerate_quality_blind_targets(
        dataset="nuscenes",
        scene="scene-0001",
        scene_index=0,
        scene_root=tmp_path,
        expected_frames=3,
        cameras=[0, 1, 2],
        eligible_bases=["streetgs", "adgs"],
    )
    assert len(samples) == 1
    assert len(views) == 3
    assert {row["partition"] for row in views} == {samples[0]["partition"]}
    assert {row["split_hash"] for row in views} == {samples[0]["split_hash"]}
    assert all(row["quality_decoded"] is False for row in views)
