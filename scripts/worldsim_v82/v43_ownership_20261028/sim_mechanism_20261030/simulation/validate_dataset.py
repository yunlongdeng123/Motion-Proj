#!/usr/bin/env python3
"""离线检查仿真标定、深度语义、观测隔离和参考配对；不生成训练输入。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_mask(path):
    values = np.asarray(Image.open(path).convert("L"))
    if not set(np.unique(values)).issubset({0, 255}):
        raise AssertionError(f"mask不是二值0/255：{path}")
    return values > 0


def validate(root):
    root = root.resolve()
    obs = root / "observations"
    privileged = root / "privileged"
    manifest = read_json(obs / "manifest.json")
    assert manifest["status"] == "complete", "生成器尚未完整收口"
    assert manifest["declared_input_mode"] == "ideal_sensor_and_tracks_interface_validation"
    tracks = {track["frame_id"]: track for track in read_json(obs / "ideal_tracks.json")["tracks"]}
    forbidden = {"surface_id", "object_id", "semantic_id", "owner", "owner_id", "mesh", "triangle_id"}
    results, warnings = [], []
    for frame in manifest["frames"]:
        frame_id = frame["id"]
        rgb_path = obs / frame["rgb"]
        for field in ("rgb", "body_mask", "sky_mask", "lidar"):
            path = (obs / frame[field]).resolve()
            assert obs.resolve() in path.parents, f"训练输入路径越出observations：{path}"
            assert path.is_file(), str(path)
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        body = read_mask(obs / frame["body_mask"])
        sky = read_mask(obs / frame["sky_mask"])
        assert rgb.shape == (frame["height"], frame["width"], 3)
        assert body.shape == rgb.shape[:2] and sky.shape == body.shape
        assert not np.any(body & sky), "车体和sky mask重叠"
        assert int(body.sum()) > 100, "车体投影过小或未渲染"
        assert float(rgb.std()) > 4.0, "RGB接近空白图"
        k = np.asarray(frame["intrinsics"], dtype=np.float64)
        c2w = np.asarray(frame["camera_to_world"], dtype=np.float64)
        w2c = np.asarray(frame["world_to_camera"], dtype=np.float64)
        assert np.allclose(c2w @ w2c, np.eye(4), atol=1e-6)
        assert np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-6)
        assert np.isclose(np.linalg.det(c2w[:3, :3]), 1.0, atol=1e-6)
        lidar = np.load(obs / frame["lidar"], allow_pickle=False)
        assert not forbidden.intersection(lidar.files), "LiDAR输入泄漏表面或owner真值"
        xyz = lidar["xyz"]
        assert xyz.ndim == 2 and xyz.shape[1] == 3 and len(xyz) > 100
        assert np.isfinite(xyz).all()
        assert np.array_equal(xyz, lidar["xyz_world"])
        xyz_camera = xyz @ w2c[:3, :3].T + w2c[:3, 3]
        uv = xyz_camera[:, :2] / xyz_camera[:, 2:] * (k[0, 0], k[1, 1]) + (k[0, 2], k[1, 2])
        uv_error = float(np.max(np.abs(uv - lidar["uv"])))
        assert uv_error < 5e-4, f"LiDAR uv标定不一致：{uv_error}"
        lidar_z_error = float(np.max(np.abs(xyz_camera[:, 2] - lidar["depth_z_m"])))
        assert lidar_z_error < 5e-5
        lidar_range_error = float(np.max(np.abs(np.linalg.norm(xyz - c2w[:3, 3], axis=1) - lidar["range_m"])))
        assert lidar_range_error < 5e-5
        dense = np.load(privileged / "original" / f"{frame_id:06d}.npz", allow_pickle=False)
        valid = dense["valid"]
        assert np.array_equal(body, dense["semantic_id"] == 10), "PNG车体mask和中心射线mesh mask不一致"
        assert np.array_equal(sky, ~valid), "PNG sky mask方向或保存不一致"
        assert np.array_equal(dense["surface_id"] > 0, valid)
        assert np.all(dense["depth_z_m"][valid] > 0) and np.all(dense["range_m"][valid] > 0)
        points = dense["xyz_world"][valid]
        dense_z = points @ w2c[2, :3] + w2c[2, 3]
        dense_z_error = float(np.max(np.abs(dense_z - dense["depth_z_m"][valid])))
        assert dense_z_error < 5e-5
        dense_range_error = float(np.max(np.abs(np.linalg.norm(points - c2w[:3, 3], axis=1) - dense["range_m"][valid])))
        assert dense_range_error < 5e-5
        # 仅检查传感器覆盖，z=0来自仿真坐标定义；此诊断不写入任何方法输入。
        center = np.asarray(tracks[frame_id]["box_center_world"])
        near_ground = (np.linalg.norm(xyz[:, :2] - center[:2], axis=1) <= 3.0) & (np.abs(xyz[:, 2]) < 0.15)
        ground = xyz[near_ground]
        cells = np.unique(np.floor(ground[:, :2] / 0.5).astype(np.int32), axis=0)
        span = np.ptp(ground[:, :2], axis=0).tolist() if len(ground) else [0.0, 0.0]
        if len(cells) < 30 or min(span) < 2.0:
            warnings.append(f"frame {frame_id}: 车周3m内路面稀疏覆盖仅{len(cells)}个0.5m cell，XY跨度={span}")
        results.append({"frame_id": frame_id, "vehicle_pixels": int(body.sum()), "sky_pixels": int(sky.sum()),
                        "lidar_points": len(xyz), "near_vehicle_ground_points": len(ground),
                        "near_vehicle_ground_cells_0_5m": len(cells), "near_vehicle_ground_xy_span_m": span,
                        "max_lidar_reprojection_error_px": uv_error, "max_lidar_z_error_m": lidar_z_error,
                        "max_lidar_range_error_m": lidar_range_error,
                        "max_dense_z_error_m": dense_z_error, "max_dense_range_error_m": dense_range_error})

    references = read_json(privileged / "reference_manifest.json")
    by_key = {(item["frame_id"], item["edit"], item["vehicle_shadow"]): item for item in references["references"]}
    paired = []
    for frame_id in references["keyframes"]:
        for edit in ("none", "delete", "shift", "rotate"):
            on, off = by_key[(frame_id, edit, True)], by_key[(frame_id, edit, False)]
            rgb_on = np.asarray(Image.open(privileged / on["rgb"]).convert("RGB"), dtype=np.float32)
            rgb_off = np.asarray(Image.open(privileged / off["rgb"]).convert("RGB"), dtype=np.float32)
            dense = np.load(privileged / on["geometry"], allow_pickle=False)
            road = np.isin(dense["semantic_id"], (1, 2))
            diff = np.abs(rgb_on - rgb_off).mean(axis=-1)
            road_changed = int(np.sum((diff > 2.0) & road))
            mean_road_delta = float(diff[road].mean()) if road.any() else 0.0
            if edit == "delete":
                assert float(diff.max()) == 0.0, "删除车辆后shadow开关仍改变图像，配对状态不干净"
                assert not np.any(dense["semantic_id"] == 10)
            elif road_changed < 20:
                warnings.append(f"frame {frame_id}/{edit}: 车影开关只改变{road_changed}个路面像素")
            paired.append({"frame_id": frame_id, "edit": edit, "changed_road_pixels_gt2_255": road_changed,
                           "mean_road_rgb_abs_delta_255": mean_road_delta})
    heldout = references.get("heldout", [])
    for item in heldout:
        assert not item["train_input"]
        assert (privileged / item["rgb"]).is_file() and (privileged / item["geometry"]).is_file()
        original = next(frame for frame in manifest["frames"] if frame["id"] == item["id"])
        displacement = np.linalg.norm(np.asarray(item["camera_to_world"])[:3, 3] - np.asarray(original["camera_to_world"])[:3, 3])
        assert abs(displacement - 1.0) < 1e-5
    mesh = np.load(privileged / "geometry" / "mesh_reference.npz", allow_pickle=False)
    surfaces = np.load(privileged / "geometry" / "fixed_surface_samples.npz", allow_pickle=False)
    assert len(mesh["surface_id"]) == len(mesh["triangles"])
    assert np.array_equal(mesh["surface_id"], np.arange(1, len(mesh["triangles"]) + 1))
    assert np.all(np.isin(surfaces["surface_id"], mesh["surface_id"]))
    assert np.isfinite(surfaces["xyz_object"]).all()
    # r1上传时玻璃舱face winding尚未修正。此检查只读登记法线边界，不改写既有数据。
    cabin_faces = mesh["triangles"][mesh["triangle_object_id"] == 104]
    cabin_vertices = mesh["vertices_local"][cabin_faces]
    cabin_center = mesh["vertices_local"][np.unique(cabin_faces)].mean(axis=0)
    cabin_normals = np.cross(cabin_vertices[:, 1] - cabin_vertices[:, 0], cabin_vertices[:, 2] - cabin_vertices[:, 0])
    cabin_signed = np.sum(cabin_normals * (cabin_vertices.mean(axis=1) - cabin_center), axis=1)
    inward = int((cabin_signed < -1e-8).sum())
    orientation_status = "passed" if inward == 0 else "requires_object104_sign_correction_for_oriented_normals"
    if inward:
        warnings.append(f"object104玻璃舱{inward}/{len(cabin_faces)}面朝内；深度/ID/交点仍自洽，有向法线评估须显式翻转该对象法线；未改写原始数据")
    return {"status": "passed", "scope": "simulator consistency and input-boundary audit; not reconstruction quality or real-world validity",
            "observation_input_isolation": "passed", "coordinate_and_depth_checks": "passed", "frame_checks": results,
            "reference_shadow_pairs": paired, "heldout_views": len(heldout),
            "oriented_normals_status": orientation_status, "glass_cabin_inward_triangles": inward,
            "mesh_triangles": len(mesh["triangles"]), "fixed_surface_samples": len(surfaces["surface_id"]), "warnings": warnings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    report = validate(args.dataset)
    destination = args.report or args.dataset / "validation.json"
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "frames": len(report["frame_checks"]), "warnings": report["warnings"], "report": str(destination)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
