#!/usr/bin/env python3
"""Validate a prepared scene and save a real RGB/LiDAR/sky visual audit."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def read_ply(path):
    types = {"float": "<f4", "uchar": "u1"}
    fields = []
    with path.open("rb") as handle:
        assert handle.readline() == b"ply\n"
        assert handle.readline() == b"format binary_little_endian 1.0\n"
        count = int(handle.readline().decode().split()[-1])
        while True:
            parts = handle.readline().decode("ascii").split()
            if parts == ["end_header"]:
                break
            assert parts[0] == "property"
            fields.append((parts[2], types[parts[1]]))
        data = np.fromfile(handle, dtype=fields, count=count)
        assert len(data) == count and not handle.read(1)
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", type=Path, required=True)
    args = parser.parse_args()
    root = args.scene
    manifest = json.loads((root / "preparation_manifest.json").read_text())
    ids = [entry["id"] for entry in manifest["frames"]]
    expected = set(ids)
    for subdir, suffix in (("calib", ".txt"), ("pose", ".txt"), ("velodyne", ".bin"),
                           ("image/image_0", ".png"), ("sky/sky_0", ".png")):
        assert {p.stem for p in (root / subdir).glob("*" + suffix)} == expected, subdir
    stats = []
    for index, name in enumerate(ids):
        image = Image.open(root / "image/image_0" / (name + ".png")).convert("RGB")
        mask = np.array(Image.open(root / "sky/sky_0" / (name + ".png")))
        assert mask.shape == (image.height, image.width)
        assert set(np.unique(mask)).issubset({0, 255})
        calibration = {}
        for line in (root / "calib" / (name + ".txt")).read_text().splitlines():
            label, values = line.split(":", 1)
            calibration[label] = np.fromstring(values, sep=" ")
        assert len(calibration) == 11
        K = calibration["P0"].reshape(3, 4)[:, :3]
        extrinsic = calibration["Tr_velo_to_cam_0"].reshape(3, 4)
        pose = np.loadtxt(root / "pose" / (name + ".txt"))
        assert pose.shape == (4, 4) and np.allclose(pose[3], [0, 0, 0, 1])
        assert np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-5)
        assert np.isclose(np.linalg.det(extrinsic[:, :3]), 1, atol=1e-5)
        assert K[0, 0] > 0 and K[1, 1] > 0
        points = np.fromfile(root / "velodyne" / (name + ".bin"), dtype=np.float32).reshape(-1, 6)
        assert len(points) and np.isfinite(points).all()
        camera_points = np.pad(points[:, :3], ((0, 0), (0, 1)), constant_values=1) @ extrinsic.T
        front = camera_points[camera_points[:, 2] > 0]
        projected = front @ K.T
        uv = projected[:, :2] / projected[:, 2:3]
        visible = (uv[:, 0] >= 0) & (uv[:, 0] < image.width) & (uv[:, 1] >= 0) & (uv[:, 1] < image.height)
        assert np.count_nonzero(visible) > 0
        stat = {"frame": name, "lidar_points": len(points), "front_visible_points": int(np.count_nonzero(visible)),
                "sky_fraction": float(np.mean(mask > 0))}
        stats.append(stat)
        if index == len(ids) // 2:
            base = image.resize((960, round(image.height * 960 / image.width)))
            width, height = base.size
            projected_image = base.copy()
            draw = ImageDraw.Draw(projected_image)
            pixel_points = uv[visible] * (width / image.width)
            depth = front[visible, 2]
            selected = np.linspace(0, len(pixel_points) - 1, min(len(pixel_points), 10000)).astype(int)
            for at in selected:
                x, y = pixel_points[at]
                t = min(float(depth[at]) / 70, 1)
                color = (int(255 * (1 - t)), int(255 * t), 80)
                draw.ellipse((x - 1, y - 1, x + 1, y + 1), fill=color)
            sky_image = image.copy()
            tint = np.array(sky_image).astype(np.float32)
            tint[mask > 0] = tint[mask > 0] * 0.45 + np.array([30, 130, 255]) * 0.55
            sky_image = Image.fromarray(tint.astype(np.uint8)).resize(base.size)
            canvas = Image.new("RGB", (width * 3, height + 30), "white")
            for panel, title, column in ((base, "RGB " + name, 0), (projected_image, "Official calibration + LiDAR", 1), (sky_image, "SegFormer sky mask", 2)):
                canvas.paste(panel, (column * width, 30))
                ImageDraw.Draw(canvas).text((column * width + 8, 8), title, fill="black")
            canvas.save(root / "data_quality_overview.png")
    ply = read_ply(root / "points3d.ply")
    assert len(ply) == manifest["num_points"]
    for key in ("x", "y", "z", "nx", "ny", "nz", "time"):
        assert np.isfinite(ply[key]).all()
    assert np.array_equal(ply["nx"], ply["time"])
    expected_times = np.array([f["model_time"] for f in manifest["frames"]])
    assert np.min(np.abs(ply["time"][:, None] - expected_times[None, :]), axis=1).max() < 1e-6
    summary = {"status": "PASS", "frames": len(ids), "frame_ids_first_last": [ids[0], ids[-1]],
               "initialization_points": len(ply), "model_time_range": [float(ply["time"].min()), float(ply["time"].max())],
               "ply_sha256": hashlib.sha256((root / "points3d.ply").read_bytes()).hexdigest(),
               "dynamic_prepared": False, "frame_checks": stats,
               "preview": str(root / "data_quality_overview.png")}
    (root / "validation.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: value for key, value in summary.items() if key != "frame_checks"}, indent=2))


if __name__ == "__main__":
    main()
