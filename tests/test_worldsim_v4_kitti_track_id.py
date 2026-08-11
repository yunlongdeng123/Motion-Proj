from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from motion_proj.worldsim_v4.datasets.kitti import (
    ADAPTER_GATES,
    build_frame_partitions,
    build_tracking_manifest,
    canonical_json_sha256,
    detect_kitti_layout,
    parse_raw_tracklet_ids,
    parse_tracking_calibration,
    parse_tracking_labels,
    project_camera_points,
    transform_lidar_to_rectified_camera,
)


def _tracking_root(root: Path, sequences: tuple[str, ...] = ("0000", "0001")) -> Path:
    training = root / "training"
    for component in ("image_02", "image_03", "velodyne"):
        for sequence in sequences:
            (training / component / sequence).mkdir(parents=True, exist_ok=True)
    for component in ("label_02", "calib", "poses"):
        (training / component).mkdir(parents=True, exist_ok=True)
    calibration = (
        "P2: 100 0 50 0 0 100 50 0 0 0 1 0\n"
        "P3: 100 0 50 -54 0 100 50 0 0 0 1 0\n"
        "R0_rect: 1 0 0 0 1 0 0 0 1\n"
        "Tr_velo_to_cam: 1 0 0 0 0 1 0 0 0 0 1 0\n"
    )
    label = "0 0 Car 0 0 0 10 10 20 20 1.5 1.6 4.0 0 1.5 15 0\n"
    for sequence in sequences:
        (training / "calib" / f"{sequence}.txt").write_text(calibration, encoding="utf-8")
        (training / "label_02" / f"{sequence}.txt").write_text(label, encoding="utf-8")
        pose_lines = []
        for frame in range(5):
            stem = f"{frame:06d}"
            (training / "image_02" / sequence / f"{stem}.png").write_bytes(b"image02")
            (training / "image_03" / sequence / f"{stem}.png").write_bytes(b"image03")
            np.asarray([[0.0, 0.0, 10.0, 1.0]], dtype=np.float32).tofile(
                training / "velodyne" / sequence / f"{stem}.bin"
            )
            pose_lines.append(f"1 0 0 {frame * 0.1} 0 1 0 0 0 0 1 0")
        (training / "poses" / f"{sequence}.txt").write_text(
            "\n".join(pose_lines) + "\n", encoding="utf-8"
        )
    return root


def test_missing_root_is_explicit_and_download_free(tmp_path: Path) -> None:
    requested = tmp_path / "KITTI"
    audit = detect_kitti_layout(requested)
    assert audit["status"] == "blocked_local_dataset_missing"
    assert audit["layout"] == "missing"
    assert audit["requested_root"] == str(requested)
    assert audit["download_attempted"] is False


def test_tracking_layout_and_two_camera_contract(tmp_path: Path) -> None:
    audit = detect_kitti_layout(_tracking_root(tmp_path / "KITTI"))
    assert audit["status"] == "ready"
    assert audit["layout"] == "tracking_training"
    assert audit["camera_contract"] == ["image_02", "image_03"]
    assert audit["sequence_ids"] == ["0000", "0001"]


def test_tracking_manifest_passes_all_twelve_adapter_gates(tmp_path: Path) -> None:
    layout = detect_kitti_layout(_tracking_root(tmp_path / "KITTI"))
    first = build_tracking_manifest(layout, smoke_count=2, formal_count=10)
    second = build_tracking_manifest(layout, smoke_count=2, formal_count=10)
    assert first == second
    assert first["status"] == "done"
    assert set(first["gates"]) == set(ADAPTER_GATES)
    assert all(first["gates"].values())
    assert first["manifest_sha256"] == canonical_json_sha256(
        {key: value for key, value in first.items() if key != "manifest_sha256"}
    )
    assert [row["role"] for row in first["sequences"]] == ["adapter_smoke", "adapter_smoke"]


def test_track_id_and_calibration_projection_contract(tmp_path: Path) -> None:
    root = _tracking_root(tmp_path / "KITTI", ("0000",)) / "training"
    labels = parse_tracking_labels(root / "label_02" / "0000.txt")
    calibration = parse_tracking_calibration(root / "calib" / "0000.txt")
    assert labels[0]["track_id"] == 0
    camera = transform_lidar_to_rectified_camera(
        np.asarray([[0.0, 0.0, 10.0, 1.0]]), calibration
    )
    pixel = project_camera_points(camera, calibration["P2"])
    np.testing.assert_allclose(pixel, [[50.0, 50.0]])


def test_frame_partitions_are_complete_disjoint() -> None:
    partitions = build_frame_partitions(range(10))
    flattened = [frame for rows in partitions.values() for frame in rows]
    assert sorted(flattened) == list(range(10))
    assert len(flattened) == len(set(flattened))


def test_raw_layout_is_detected_only_when_contract_is_complete(tmp_path: Path) -> None:
    date = tmp_path / "KITTI" / "2011_09_26"
    drive = date / "2011_09_26_drive_0001_sync"
    for path in (
        drive / "image_02" / "data",
        drive / "image_03" / "data",
        drive / "velodyne_points" / "data",
        drive / "oxts" / "data",
    ):
        path.mkdir(parents=True, exist_ok=True)
    (drive / "tracklet_labels.xml").write_text(
        "<boost_serialization><tracklets><item/></tracklets></boost_serialization>",
        encoding="utf-8",
    )
    (date / "calib_cam_to_cam.txt").write_text("P_rect_02: 1\n", encoding="utf-8")
    (date / "calib_velo_to_cam.txt").write_text("R: 1\n", encoding="utf-8")
    audit = detect_kitti_layout(tmp_path / "KITTI")
    assert audit["layout"] == "raw_sync"
    assert audit["status"] == "ready"
    assert parse_raw_tracklet_ids(drive / "tracklet_labels.xml") == [0]


def test_missing_root_formal_audit_records_blocked_terminal(tmp_path: Path, monkeypatch) -> None:
    from scripts import audit_worldsim_v4_kitti as auditor

    monkeypatch.setattr(auditor, "SNAPSHOT_RELPATHS", ())
    monkeypatch.setattr(auditor, "_git", lambda *_args: "synthetic")
    config = tmp_path / "config.yaml"
    config.write_text(
        "task_id: WS-V4-D1-KITTI-ADAPTER-01\n"
        "dataset:\n  root: " + str(tmp_path / "missing") + "\n"
        "protocol:\n  adapter_smoke_sequences: 2\n  cross_domain_target_sequences: 10\n"
        "freeze:\n  expected_status: blocked_local_dataset_missing\n",
        encoding="utf-8",
    )
    run_dir = tmp_path / "run"
    summary = auditor.run(config, run_dir, tmp_path)
    assert summary["status"] == "blocked"
    assert summary["reason"] == "blocked_local_dataset_missing"
    assert json.loads((run_dir / "status.json").read_text(encoding="utf-8"))["status"] == "blocked"
    assert (run_dir / "manifest.json").is_file()
