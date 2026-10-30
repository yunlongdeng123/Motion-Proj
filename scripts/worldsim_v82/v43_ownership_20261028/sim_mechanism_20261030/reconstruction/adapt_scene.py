"""把独立仿真的受限传感器观测转换为现有 DrivingEditor Waymo 输入。

只消费 observations/；不读取独立网格、表面标签、参考编辑图或 owner ID。
动态初始化按simulator角点像素坐标查mask；近深度/主连通分量规则继承actor_cpu。
"""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def observation_path(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or "privileged" in path.parts:
        raise ValueError("训练适配器只允许读取 observations 内的显式观测文件")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def save_ply(path, xyz, timestamps):
    """与原 actor_cpu.save_ply 相同：normals 和 time 均保存模型时间。"""
    dtype = np.dtype([(key, "<f4") for key in ("x", "y", "z", "nx", "ny", "nz")]
                     + [(key, "u1") for key in ("red", "green", "blue")] + [("time", "<f4")])
    vertices = np.zeros(len(xyz), dtype=dtype)
    for axis, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, axis]
    for key in ("nx", "ny", "nz", "time"):
        vertices[key] = timestamps
    header = "ply\nformat binary_little_endian 1.0\nelement vertex %d\n" % len(xyz)
    header += "".join("property %s %s\n" % ("uchar" if key in ("red", "green", "blue") else "float", key)
                      for key in dtype.names) + "end_header\n"
    with Path(path).open("xb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def actor_indices(xyz_ego, camera_xyz, intrinsics, body):
    """角点坐标floor查mask；原生2x2最近深度+0.5m、0.5m主分量不变。"""
    positive = np.flatnonzero(np.isfinite(camera_xyz).all(axis=1) & (camera_xyz[:, 2] > .01))
    uvw = camera_xyz[positive] @ intrinsics.T
    # sim K的(u,v)以左上角点为(0,0)，像素中心为(col+.5,row+.5)。
    # 所属像素cell为floor(uv)；旧actor_cpu的floor(uv+.5)会把中心射线偏移一格。
    # native Camera射线同用(u-cx+.5)/fx，rasterizer的ndc2Pix隐含-.5；K无需改动。
    uv = np.floor(uvw[:, :2] / uvw[:, 2:3]).astype(np.int64)
    height, width = body.shape
    inside = ((uv[:, 0] >= 0) & (uv[:, 0] < width)
              & (uv[:, 1] >= 0) & (uv[:, 1] < height))
    source, uv = positive[inside], uv[inside]
    depth = camera_xyz[source, 2]
    bins_width = (width + 1) // 2
    bins = (uv[:, 1] // 2) * bins_width + uv[:, 0] // 2
    nearest = np.full(((height + 1) // 2) * bins_width, np.inf)
    np.minimum.at(nearest, bins, depth)
    hit = body[uv[:, 1], uv[:, 0]]
    retained = source[hit & (depth <= nearest[bins] + .5)]
    before = len(retained)
    if before:
        edges = cKDTree(xyz_ego[retained]).query_pairs(.5, output_type="ndarray")
        graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])), shape=(before, before))
        _, labels = connected_components(graph, directed=False)
        counts = np.bincount(labels)
        dominant = labels == counts.argmax()
        if dominant.sum() < before * .5:
            raise ValueError("观测动态 LiDAR 没有占比至少 0.5 的物理主分量；不自动清洗或造点")
        retained = retained[dominant]
    return retained.astype(np.int32), int(hit.sum()), before


def build(args):
    root = args.observations.resolve()
    if "privileged" in root.parts:
        raise ValueError("输入必须是受限 observations 目录")
    manifest_path = observation_path(root, args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    frames = manifest["frames"]
    if len(frames) < 2 or len({str(frame["id"]) for frame in frames}) != len(frames):
        raise ValueError("至少需要两个唯一观测视图")
    if args.output.exists():
        raise FileExistsError("不覆盖已存在的适配数据集；使用新的 attempt 目录")
    args.output.mkdir(parents=True)
    for directory in ("image/image_0", "mask/mask_0/Annotations", "sky/sky_0", "calib", "pose", "velodyne",
                      "dynamic_image/dynamic_image_0", "dynamic_masks/dynamic_mask_0",
                      "dynamic_actor_assets/actor_0/body_masks"):
        (args.output / directory).mkdir(parents=True)
    all_world, all_times, dynamic_world, dynamic_times = [], [], [], []
    source_rows, reports, names, size = {}, [], [], None
    interval = float(args.frame_interval)
    if interval <= 0:
        raise ValueError("模型 frame_interval 必须为正")
    times = (np.arange(len(frames), dtype=np.float64) - (len(frames) - 1) / 2) * interval
    for index, row in enumerate(frames):
        raw_id = str(row["id"])
        name = "%06d" % int(raw_id) if raw_id.isdigit() else raw_id
        if not name.isalnum() or Path(name).name != name:
            raise ValueError("frame id 必须是安全的字母数字文件名")
        names.append(name)
        rgb_path = observation_path(root, row["rgb"])
        body_path = observation_path(root, row["body_mask"])
        sky_path = observation_path(root, row["sky_mask"])
        lidar_path = observation_path(root, row["lidar"])
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"))
        body_raw = np.asarray(Image.open(body_path).convert("L"))
        sky_raw = np.asarray(Image.open(sky_path).convert("L"))
        body, sky = body_raw > 0, sky_raw > 0
        if (body.shape != rgb.shape[:2] or sky.shape != body.shape
                or not set(np.unique(body_raw)).issubset({0, 1, 255})
                or not set(np.unique(sky_raw)).issubset({0, 1, 255})):
            raise ValueError("RGB/body/sky 必须尺寸一致；mask 必须是二值")
        current_size = [int(rgb.shape[1]), int(rgb.shape[0])]
        if size is not None and size != current_size:
            raise ValueError("观测分辨率在序列内发生变化")
        size = current_size
        intrinsic = np.asarray(row.get("intrinsics", row.get("K")), dtype=np.float64).reshape(3, 3)
        c2w = np.asarray(row["camera_to_world"], dtype=np.float64).reshape(4, 4)
        if (not np.isfinite(intrinsic).all() or intrinsic[0, 0] <= 0 or intrinsic[1, 1] <= 0
                or not np.isfinite(c2w).all() or not np.allclose(c2w[3], [0, 0, 0, 1])
                or not np.allclose(c2w[:3, :3].T @ c2w[:3, :3], np.eye(3), atol=1e-5)
                or np.linalg.det(c2w[:3, :3]) < .999):
            raise ValueError("相机必须采用米制刚体 OpenCV camera_to_world 和有效 K")
        with np.load(lidar_path, allow_pickle=False) as returns:
            if "xyz_world" not in returns:
                raise ValueError("LiDAR NPZ 必须显式给 xyz_world，不能猜测 xyz 的坐标系")
            world = np.asarray(returns["xyz_world"], dtype=np.float64)
            intensity = np.asarray(returns["intensity"], dtype=np.float32).reshape(-1) if "intensity" in returns else np.ones(len(world), dtype=np.float32)
        if world.ndim != 2 or world.shape[1] != 3 or not len(world) or not np.isfinite(world).all() or len(intensity) != len(world):
            raise ValueError("需要非空、有限的真实仿真 raycast 返回点；不允许稠密网格替代")
        # world-aligned ego 保留 Z-up；相机姿态仅放进 LiDAR→camera 标定。
        pose = np.eye(4)
        pose[:3, 3] = c2w[:3, 3]
        xyz_ego = world - pose[:3, 3]
        lidar_to_cam = np.eye(4)
        lidar_to_cam[:3, :3] = c2w[:3, :3].T
        points = np.column_stack((xyz_ego, intensity, np.zeros((len(world), 2)))).astype(np.float32)
        camera_xyz = points[:, :3].astype(np.float64) @ lidar_to_cam[:3, :3].T
        points[:, 5] = times[index]
        points.tofile(args.output / "velodyne" / (name + ".bin"))
        np.savetxt(args.output / "pose" / (name + ".txt"), pose, fmt="%.17g")
        projection = np.column_stack((intrinsic, np.zeros(3)))
        payload = "P0: " + " ".join("%.17g" % value for value in projection.ravel()) + "\n"
        payload += "Tr_velo_to_cam_0: " + " ".join("%.17g" % value for value in lidar_to_cam[:3].ravel()) + "\n"
        (args.output / "calib" / (name + ".txt")).write_text(payload, encoding="ascii")
        shutil.copy2(rgb_path, args.output / "image/image_0" / (name + ".png"))
        Image.fromarray(body.astype(np.uint8)).save(args.output / "mask/mask_0/Annotations" / (name + ".png"))
        Image.fromarray(sky.astype(np.uint8) * 255).save(args.output / "sky/sky_0" / (name + ".png"))
        dynamic_rgb = np.zeros_like(rgb)
        dynamic_rgb[body] = rgb[body]
        Image.fromarray(dynamic_rgb).save(args.output / "dynamic_image/dynamic_image_0" / (name + ".png"))
        for folder in ("dynamic_masks/dynamic_mask_0", "dynamic_actor_assets/actor_0/body_masks"):
            Image.fromarray(body.astype(np.uint8) * 255).save(args.output / folder / (name + ".png"))
        selected, projected, before = actor_indices(points[:, :3].astype(np.float64), camera_xyz, intrinsic, body)
        # 保存的 world PLY 必须正好等于 loader 从 float32 ego 返回恢复的坐标。
        loader_world = (points[:, :3].astype(np.float64) + pose[:3, 3]).astype(np.float32)
        source_rows[name] = selected
        all_world.append(loader_world)
        all_times.append(np.full(len(world), times[index], dtype=np.float32))
        dynamic_world.append(loader_world[selected])
        dynamic_times.append(np.full(len(selected), times[index], dtype=np.float32))
        reports.append(dict(id=name, source_frame_id=row["id"], source_index=index, model_time=float(times[index]), lidar_returns=len(world),
                            body_pixels=int(body.sum()), sky_pixels=int(sky.sum()), projected_body_points=projected,
                            before_component=before, dynamic_seed_points=len(selected)))
    if names != sorted(names) or len(set(names)) != len(names):
        raise ValueError("原生 loader 按文件名排序；frame ids 必须唯一且与时间顺序一致")
    whole = np.concatenate(all_world)
    whole_times = np.concatenate(all_times)
    actor = np.concatenate(dynamic_world)
    if len(actor) < 3:
        raise ValueError("整个序列动态 LiDAR 少于 3 点；不使用 placeholder 或特权网格补点")
    selected = np.random.RandomState(0).choice(len(whole), args.static_seed_count, replace=True)
    save_ply(args.output / "points3d.ply", whole[selected], whole_times[selected])
    save_ply(args.output / "points3d_dynamic_0.ply", actor, np.concatenate(dynamic_times))
    np.savez_compressed(args.output / "dynamic_actor_assets/actor_0/source_lidar_indices.npz", **source_rows)
    if (root / "ideal_tracks.json").is_file():
        shutil.copy2(root / "ideal_tracks.json", args.output / "ideal_tracks.json")
    anchor = min(10, len(frames) // 2)
    scale = args.render_scale
    render_wh = [int(value / scale) for value in size]
    prepared = dict(schema_version=1, scene_id=manifest.get("scene_id", root.parent.name), observations=str(root),
                    frames=reports, model_frame_interval=interval, original_wh=size, render_wh=render_wh,
                    render_scale=scale, training_ready=True, mask_input=args.mask_provenance,
                    independent_reference_used_for_training=False, simulator_owner_id_used_for_training=False,
                    static_seed_points=args.static_seed_count, unique_source_lidar_returns=len(whole), dynamic_seed_points=len(actor),
                    static_seed_sampling="inherited RandomState(0).choice(all sparse world LiDAR, count, replace=True)",
                    dynamic_seed_recipe="simulator corner-coordinate floor(uv) body lookup; inherited actor_cpu 2x2 nearest depth +0.5m, 0.5m dominant component",
                    pixel_coordinate_contract=dict(simulator_origin="upper-left image corner (0,0)",
                        simulator_pixel_centers="(column+0.5,row+0.5)", mask_lookup="floor(projected_uv)",
                        intrinsics="unchanged simulator K; no principal-point shift",
                        engineering_delta="legacy actor_cpu floor(uv+0.5) corrected only for simulator mask lookup; depth/component thresholds unchanged"),
                    static_exclusion="unchanged loader strict nearest dynamic distance <0.1m",
                    shadow_supervision="absent; body mask only", ideal_track_input="explicit simulator sensor oracle; copied for downstream, not used for seed point filtering",
                    extra_conditions=["single simulated camera", "simulator sensor oracle body/sky masks", "non-photoreal renderer", "finite sparse LiDAR rays"],
                    followup_scene_spec=dict(scene_id=manifest.get("scene_id", root.parent.name), source_path=str(args.output.resolve()),
                        frame_ids=names, camera_directory="image_0", frame_ids_with_camera=["image_0/" + name for name in names],
                        raw_frame_range=[0, len(frames) - 1], model_timestamps=times.tolist(), original_wh=size,
                        render_wh=render_wh, anchor_frame=names[anchor], anchor_index=anchor, anchor_timestamp=float(times[anchor]),
                        actor_ply_paths=[str((args.output / "points3d_dynamic_0.ply").resolve())]))
    write(args.output / "preparation_manifest.json", prepared)
    write(args.output / "adapter_receipt.json", dict(status="PREPARED_NATIVE_WAYMO_INPUT", frames=len(frames),
          privileged_files_read=False, observation_manifest=str(manifest_path), output=str(args.output.resolve()),
          pixel_lookup_engineering_delta="simulator corner coordinates use floor(uv); K/poses unchanged",
          mask_input=args.mask_provenance, training_executed=False, scientific_contamination_claim=False))
    print(json.dumps(dict(output=str(args.output.resolve()), frames=len(frames), static_seeds=args.static_seed_count,
                          dynamic_seeds=len(actor), render_wh=render_wh), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-interval", type=float, default=.02)
    parser.add_argument("--render-scale", type=float, default=1.)
    parser.add_argument("--static-seed-count", type=int, default=600000)
    parser.add_argument("--mask-provenance", default="simulator sensor oracle body/sky, authorized mechanism condition")
    args = parser.parse_args()
    if args.static_seed_count < 1 or args.render_scale <= 0:
        parser.error("seed count 和 render scale 必须为正")
    build(args)


if __name__ == "__main__":
    main()
