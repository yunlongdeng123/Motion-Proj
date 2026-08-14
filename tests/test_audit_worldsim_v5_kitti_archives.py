from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

from scripts.audit_worldsim_v5_kitti_archives import (
    ARCHIVE_SPECS,
    build_metadata,
    inspect_archive,
    render_markdown,
    sha256_file,
)


CALIBRATION = (
    "P2: 100 0 50 0 0 100 50 0 0 0 1 0\n"
    "P3: 100 0 50 -54 0 100 50 0 0 0 1 0\n"
    "R_rect 1 0 0 0 1 0 0 0 1\n"
    "Tr_velo_cam 1 0 0 0 0 1 0 0 0 0 1 0\n"
)


def _write_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(name, payload)


def _build_archives(root: Path, *, drop_right_frame: bool = False) -> dict[str, str]:
    training = ("0000", "0001")
    testing = ("0000",)
    frames = ("000000", "000001")
    for component, filename, suffix in (
        ("image_02", "data_tracking_image_2.zip", ".png"),
        ("image_03", "data_tracking_image_3.zip", ".png"),
        ("velodyne", "data_tracking_velodyne.zip", ".bin"),
    ):
        members = {}
        for split, sequences in (("training", training), ("testing", testing)):
            for sequence in sequences:
                for frame in frames:
                    if drop_right_frame and component == "image_03" and split == "training" and sequence == "0001" and frame == "000001":
                        continue
                    payload = b"png" if suffix == ".png" else b"\x00" * 16
                    members[f"{split}/{component}/{sequence}/{frame}{suffix}"] = payload
        _write_zip(root / filename, members)
    labels = {
        f"training/label_02/{sequence}.txt": (
            "0 0 Car 0 0 0 10 10 20 20 1.5 1.6 4.0 0 1.5 15 0\n"
            "1 0 Car 0 0 0 10 10 20 20 1.5 1.6 4.0 0 1.5 15 0\n"
        ).encode()
        for sequence in training
    }
    _write_zip(root / "data_tracking_label_2.zip", labels)
    oxts = {}
    calib = {}
    for split, sequences in (("training", training), ("testing", testing)):
        for sequence in sequences:
            oxts[f"{split}/oxts/{sequence}.txt"] = (
                "1 0 0 0 0 1 0 0 0 0 1 0\n" * len(frames)
            ).encode()
            calib[f"{split}/calib/{sequence}.txt"] = CALIBRATION.encode()
    _write_zip(root / "data_tracking_oxts.zip", oxts)
    _write_zip(root / "data_tracking_calib.zip", calib)
    _write_zip(
        root / "devkit_tracking.zip",
        {
            "devkit/readme.txt": b"fixture",
            "devkit/python/data/tracking/evaluate_tracking.seqmap.training": (
                b"0000 empty 000000 000001\n0001 empty 000000 000001\n"
            ),
            "devkit/python/data/tracking/evaluate_tracking.seqmap.test": (
                b"0000 empty 000000 000001\n"
            ),
        },
    )
    return {spec.filename: sha256_file(root / spec.filename) for spec in ARCHIVE_SPECS}


def test_complete_archive_set_builds_ready_metadata(tmp_path: Path) -> None:
    hashes = _build_archives(tmp_path)
    manifest = build_metadata(
        tmp_path,
        project_root=tmp_path,
        audit_date="2026-08-14",
        sha256_by_filename=hashes,
        expected_training_ids=("0000", "0001"),
        expected_testing_ids=("0000",),
        safety_margin_bytes=0,
    )
    assert manifest["status"] == "ready_for_staging"
    assert all(manifest["gates"].values())
    assert manifest["dataset_summary"]["training_frame_count"] == 4
    assert manifest["dataset_summary"]["testing_frame_count"] == 2
    assert manifest["sequence_records"][0]["labels"]["track_id_count"] == 1
    assert "不构造第三相机" in render_markdown(manifest)


def test_sensor_frame_mismatch_blocks_adapter(tmp_path: Path) -> None:
    hashes = _build_archives(tmp_path, drop_right_frame=True)
    manifest = build_metadata(
        tmp_path,
        project_root=tmp_path,
        audit_date="2026-08-14",
        sha256_by_filename=hashes,
        expected_training_ids=("0000", "0001"),
        expected_testing_ids=("0000",),
        safety_margin_bytes=0,
    )
    assert manifest["status"] == "blocked_dataset_adapter"
    assert manifest["gates"]["sensor_frame_alignment"] is False


def test_unsafe_member_is_reported(tmp_path: Path) -> None:
    spec = ARCHIVE_SPECS[0]
    path = tmp_path / spec.filename
    _write_zip(
        path,
        {
            "training/velodyne/0000/000000.bin": b"\x00" * 16,
            "../escape.bin": b"bad",
        },
    )
    index = inspect_archive(path, spec, hashlib.sha256(path.read_bytes()).hexdigest())
    assert index.public["unsafe_member_count"] == 1
    assert index.public["unexpected_file_member_count"] == 1
