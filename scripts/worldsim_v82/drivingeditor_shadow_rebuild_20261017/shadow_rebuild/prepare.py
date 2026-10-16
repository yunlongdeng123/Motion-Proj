"""CPU：从已有 Waymo 原始测量导出全视域新场景，不调用 Scene/GN11 loader。

输入 source_scene 是原始导出的 image/calib/pose/velodyne，annotations 是
完整真实 laser_labels audit；reviewed_instances 只复用真实 mask 身份映射，
绝不读取其中的已训练高斯/SH。全场景不使用历史修复 ROI 或原渲染混合。
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_name, "2")
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from scipy.ndimage import binary_dilation
from .data import clean_reference, read_json, validate_seed
from .train import write


def transform(xyz, pose):
    return xyz@pose[:3, :3].T+pose[:3, 3]


def calibration(path, shrink):
    rows = {line.split(":", 1)[0]: np.fromstring(line.split(":", 1)[1], sep=" ")
            for line in path.read_text().splitlines() if ":" in line}
    p = rows["P0"].reshape(3, 4)
    if not np.allclose(p[:, 3], 0, atol=1e-7):
        raise ValueError("当前 CPU adapter 需要 P0 第四列为0；不能静默忽略 stereo baseline")
    tr = np.eye(4)
    tr[:3] = rows["Tr_velo_to_cam_0"].reshape(3, 4)
    k = p[:, :3].copy()
    k[:2] *= shrink
    # PIL resize 使用像素中心；实际输出尺寸改变时保持中心对齐。
    k[0, 2] += (shrink-1)/2
    k[1, 2] += (shrink-1)/2
    return k, tr


def project(xyz, k, world_from_camera, shape):
    camera = transform(xyz, np.linalg.inv(world_from_camera))
    uvw = camera@k.T
    uv = uvw[:, :2]/np.maximum(uvw[:, 2:3], 1e-6)
    ij = np.floor(uv+.5).astype(np.int64)
    h, w = shape
    good = (camera[:, 2] > .1) & (ij[:, 0] >= 0) & (ij[:, 0] < w) & (ij[:, 1] >= 0) & (ij[:, 1] < h)
    return ij, camera[:, 2], good


def zbuffer(xyz, k, pose, shape):
    ij, z, good = project(xyz, k, pose, shape)
    depth = np.full(shape[0]*shape[1], np.inf)
    bins = ij[good, 1]*shape[1]+ij[good, 0]
    np.minimum.at(depth, bins, z[good])
    return depth.reshape(shape)


def voxel_rows(xyz, cell):
    _, ids = np.unique(np.floor(xyz/cell).astype(np.int64), axis=0, return_index=True)
    return ids


def local_geometry(xyz, cell, ground=False):
    count = len(xyz)
    tree = cKDTree(xyz)
    distance, neighbors = tree.query(xyz, k=min(count, 12), workers=1)
    if count == 1:
        return np.array([[cell, cell, cell/5]], np.float32), np.array([[1, 0, 0, 0]], np.float32), np.array([0])
    if neighbors.ndim == 1:
        neighbors = neighbors[:, None]
    sampled = xyz[neighbors]
    centered = sampled-sampled.mean(1, keepdims=True)
    cov = np.einsum("nki,nkj->nij", centered, centered)/neighbors.shape[1]
    _, vectors = np.linalg.eigh(cov)
    normal = vectors[..., 0]
    if ground:
        normal *= np.where(normal[:, 2:3] >= 0, 1, -1)
    else:
        # LiDAR 外表面法线朝向各资产局部中心之外；不构造盒体表面。
        normal *= np.where((normal*(xyz-np.median(xyz, 0))).sum(-1, keepdims=True) >= 0, 1, -1)
    u = vectors[..., 2]
    v = np.cross(normal, u)
    basis = np.stack((u, v, normal), -1)
    qxyzw = Rotation.from_matrix(basis).as_quat()
    q = qxyzw[:, [3, 0, 1, 2]].astype(np.float32)
    tangent = np.clip(np.median(distance[:, 1:], axis=1)*.7, cell*.5, cell*1.5)
    scales = np.stack((tangent, tangent, np.maximum(.004, tangent*.15)), -1).astype(np.float32)
    return scales, q, neighbors[:, 1]


def ground_candidates(xyz, tile=2., tolerance=.2):
    """LiDAR 局部低表面候选，不是假称语义道路。每格独立高度，允许坡度/起伏。"""
    cells = np.floor(xyz[:, :2]/tile).astype(np.int64)
    _, inverse = np.unique(cells, axis=0, return_inverse=True)
    bottom = np.full(inverse.max()+1, np.inf)
    np.minimum.at(bottom, inverse, xyz[:, 2])
    low = xyz[:, 2] < bottom[inverse]+tolerance
    _, q, _ = local_geometry(xyz, .12, ground=True)
    normal = Rotation.from_quat(q[:, [1, 2, 3, 0]]).as_matrix()[..., 2]
    return low & (normal[:, 2] > .85)


def box_pose(label, world_from_vehicle):
    b = label["box"]
    pose = np.eye(4)
    pose[:3, :3] = Rotation.from_euler("z", b["heading"]).as_matrix()
    pose[:3, 3] = [b["center_x"], b["center_y"], b["center_z"]]
    return world_from_vehicle@pose, np.array([b["length"], b["width"], b["height"]])


def build(args):
    scene, output = Path(args.scene), Path(args.output)
    if output.exists():
        raise FileExistsError("只写全新目录；不覆盖原始数据或历史实验")
    prepared = Path(args.reviewed_instances)
    original = dict(np.load(prepared/"instances.npz", allow_pickle=False))
    selection_ids = original["instance_ids"].tolist()
    manifest = read_json(scene/"preparation_manifest.json")
    rows = manifest["frames"]
    source = Path(args.annotations)
    if source.is_dir():
        audits = {r["id"]: read_json(source/(r["id"]+".json")) for r in rows}
    else:
        annotation = read_json(source)
        audits = {r["frame_id"]: r for r in (annotation["frames"] if isinstance(annotation, dict) else annotation)}
    if set(audits) != {r["id"] for r in rows}:
        raise ValueError("原始 annotation 与 RGB 窗口不匹配")
    if list(original["frame_ids"]) != [r["id"] for r in rows]:
        raise ValueError("真实 mask bundle 帧顺序不匹配")
    actors = sorted({l["id"] for a in audits.values() for l in a["laser_labels"] if l["type"] in (1, "TYPE_VEHICLE")})
    actor_index = {name: i for i, name in enumerate(actors)}
    poses = np.broadcast_to(np.eye(4), (len(rows), len(actors), 4, 4)).copy()
    valid = np.zeros((len(rows), len(actors)), bool)
    train = [r["source_index"] not in set(args.test_raw) for r in rows]
    if sum(train) < 3:
        raise ValueError("至少三个原始训练时刻")
    gathered = {i: [] for i in range(-1, len(actors))}
    rgb_gathered = {i: [] for i in gathered}
    frames = []
    unknown = []
    world_origin = np.loadtxt(scene/"pose"/(rows[0]["id"]+".txt"))[:3, 3].copy()
    # 所有输入身份核对后创建输出；异常保留 partial 以便定位，不自动删除。
    output.mkdir(parents=True)
    (output/"frames").mkdir()
    for frame_index, row in enumerate(rows):
        fid = row["id"]
        audit = audits[fid]
        if audit.get("source_index", row["source_index"]) != row["source_index"]:
            raise ValueError("真实 raw index 不匹配")
        world_from_vehicle = np.loadtxt(scene/"pose"/(fid+".txt"))
        if "frame_pose_world_from_vehicle" in audit and not np.allclose(np.asarray(audit["frame_pose_world_from_vehicle"]).reshape(4, 4), world_from_vehicle, atol=1e-6):
            raise ValueError("annotation/测量 ego pose 不匹配")
        world_from_vehicle[:3, 3] -= world_origin
        with Image.open(scene/"image/image_0"/(fid+".png")) as image:
            w, h = image.width//args.downsample, image.height//args.downsample
            if w*args.downsample != image.width or h*args.downsample != image.height:
                raise ValueError("图像尺寸必须整除 downsample")
            rgb = np.asarray(image.convert("RGB").resize((w, h), Image.Resampling.LANCZOS))
        k, camera_from_vehicle = calibration(scene/"calib"/(fid+".txt"), 1/args.downsample)
        camera_pose = world_from_vehicle@np.linalg.inv(camera_from_vehicle)
        lidar = np.fromfile(scene/"velodyne"/(fid+".bin"), np.float32).reshape(-1, 6)[:, :3]
        if not len(lidar) or not np.isfinite(lidar).all():
            raise ValueError("原始 LiDAR 无效")
        world = transform(lidar, world_from_vehicle)
        owner = np.full(len(lidar), -1, np.int32)
        covered = np.zeros(len(lidar), np.int32)
        all_boxes = []
        for label in audit["laser_labels"]:
            pose, size = box_pose(label, world_from_vehicle)
            local = transform(world, np.linalg.inv(pose))
            inside = (np.abs(local) <= size/2+.03).all(-1)
            covered[inside] += 1
            if label["id"] in actor_index:
                actor = actor_index[label["id"]]
                poses[frame_index, actor], valid[frame_index, actor] = pose, True
                owner[inside] = actor
            else:
                owner[inside] = -2
            all_boxes.append((pose, size))
        owner[covered > 1] = -2
        ij, z, in_image = project(world, k, camera_pose, (h, w))
        depth = zbuffer(world, k, camera_pose, (h, w))
        colors = np.full((len(world), 3), .5, np.float32)
        visible = np.flatnonzero(in_image)
        visible = visible[z[visible] <= depth[ij[visible, 1], ij[visible, 0]]+.15]
        colors[visible] = rgb[ij[visible, 1], ij[visible, 0]]/255
        labels = np.asarray(Image.open(prepared/"instance_masks"/(fid+".png")).resize((w, h), Image.Resampling.NEAREST))
        masks = np.zeros((h, w, len(actors)), bool)
        known = ~valid[frame_index].copy()  # 只有真实完整 laser_labels 中不存在的 Actor 可判已知不在场。
        for a, name in enumerate(actors):
            if name in selection_ids:
                j = selection_ids.index(name)
                if bool(original["mask_known"][frame_index, j]):
                    masks[..., a] = labels == j+1
                    known[a] = True
        if not known.all():
            unknown.append(dict(frame_id=fid, actors=[actors[i] for i in np.flatnonzero(~known)]))
        # 所有真实框用于去除点归属歧义和 road reference 的遮挡保护，不生成 body mask。
        guard = masks.any(-1).copy()
        for pose, size in all_boxes:
            corners = np.array([[x, y, z] for x in (-.5, .5) for y in (-.5, .5) for z in (-.5, .5)])*size
            uv, _, ok = project(transform(corners, pose), k, camera_pose, (h, w))
            if ok.any():
                lo = np.maximum(uv[ok].min(0)-2, 0)
                hi = np.minimum(uv[ok].max(0)+2, [w-1, h-1])
                guard[lo[1]:hi[1]+1, lo[0]:hi[0]+1] = True
        if train[frame_index]:
            for a in gathered:
                take = (owner == a)
                if a == -1:
                    take &= in_image & (z < args.max_depth)
                    local = world[take]
                else:
                    local = transform(world[take], np.linalg.inv(poses[frame_index, a]))
                gathered[a].append(local)
                rgb_gathered[a].append(colors[take])
        frame = dict(frame_id=fid, source_index=row["source_index"], split="train" if train[frame_index] else "test",
                     rgb=f"frames/{fid}.png", supervision=f"frames/{fid}.npz", intrinsic=k.tolist(), world_from_camera=camera_pose.tolist())
        Image.fromarray(rgb).save(output/frame["rgb"])
        depth[~np.isfinite(depth)] = 0
        # road 像素标签在全局静态 surface 建成后赋值；临时字段只在 CPU 内存。
        frames.append((frame, rgb, masks, known, depth.astype(np.float32), guard))
    fields = []
    offset = 0
    for a in gathered:
        xyz = np.concatenate(gathered[a])
        color = np.concatenate(rgb_gathered[a])
        if not len(xyz):
            if a >= 0:
                raise ValueError(f"{actors[a]} 没有训练帧真实 LiDAR；不得造盒体补成资产")
            raise ValueError("没有真实静态 LiDAR")
        cell = args.static_voxel if a == -1 else args.actor_voxel
        selected = voxel_rows(xyz, cell)
        xyz, color = xyz[selected], color[selected]
        if a == -1:
            road = ground_candidates(xyz)
        else:
            road = np.zeros(len(xyz), bool)
        scales, q, neighbor = local_geometry(xyz, cell, ground=(a == -1))
        fields.append(dict(xyz=xyz.astype(np.float32), rgb=color.astype(np.float32), observed_rgb=color.astype(np.float32).copy(), scales=scales, quaternion=q,
                           owner=np.full(len(xyz), a, np.int64), road=road,
                           reference_rgb=np.zeros_like(color), reference_confidence=np.zeros(len(xyz), np.float32), neighbor=neighbor+offset))
        offset += len(xyz)
    seed = {key: np.concatenate([f[key] for f in fields]) for key in fields[0]}
    road_ids = np.flatnonzero(seed["road"])
    samples = np.zeros((len(road_ids), len(rows), 3), np.float32)
    observed = np.zeros((len(road_ids), len(rows)), bool)
    for i, (row, rgb, masks, known, depth, guard) in enumerate(frames):
        ij, z, good = project(seed["xyz"][road_ids], np.array(row["intrinsic"]), np.array(row["world_from_camera"]), depth.shape)
        ids = np.flatnonzero(good)
        yy, xx = ij[ids, 1], ij[ids, 0]
        # 必须有同一像素真实深度支持，不用模型深度、宽松平面遮挡或 test RGB。
        keep = (depth[yy, xx] > 0) & (np.abs(depth[yy, xx]-z[ids]) < .2) & ~guard[yy, xx]
        ids, yy, xx = ids[keep], yy[keep], xx[keep]
        if train[i]:
            samples[ids, i] = rgb[yy, xx]/255
            observed[ids, i] = True
        road_mask = np.zeros(depth.shape, bool)
        road_mask[yy, xx] = True
        # 只扩 2px 支撑邻域；不填整片被遮挡区域，不使用过去手选 ROI。
        road_mask = binary_dilation(road_mask, iterations=2) & ~guard
        np.savez_compressed(output/row["supervision"], masks=masks, mask_known=known, road=road_mask, depth=depth)
    reference, confidence = clean_reference(samples, observed)
    seed["reference_rgb"][road_ids] = reference
    seed["reference_confidence"][road_ids] = confidence
    trusted = road_ids[confidence > 0]
    seed["rgb"][trusted] = reference[confidence > 0]
    validate_seed(seed, len(actors))
    np.savez_compressed(output/"seed.npz", **seed)
    np.savez_compressed(output/"trajectory.npz", poses=poses.astype(np.float32), valid=valid)
    manifest_out = dict(schema="instance_surface_measurements_v3", units="meters", actor_ids=actors,
                        frames=[f[0] for f in frames], learned_checkpoint_used=False, reference_train_only=True,
                        source_scene=str(scene.resolve()), annotations=str(source.resolve()),
                        reviewed_instances=str(prepared.resolve()), reference_raw_frames=[r["source_index"] for i, r in enumerate(rows) if train[i]],
                        initialization="raw LiDAR world/canonical points + observed RGB; no learned scene/environment",
                        original_world_origin_m=world_origin.tolist(),
                        road_label_source="LiDAR local-low-surface + normal candidate; not semantic/manual ground truth",
                        road_geometry="nonplanar local PCA surface Gaussians; no fixed plane",
                        test_geometry_or_RGB_used_to_build_seed=False, original_files_modified=False,
                        partial_mask_pairs=unknown, reference_valid_points=int((confidence > 0).sum()),
                        full_mask_supervision_ready=not bool(unknown), human_verdict=None)
    write(output/"manifest.json", manifest_out)
    print(json.dumps(dict(status="CPU_PREPARED_INPUT_AUDIT_REQUIRED", output=str(output), gaussians=len(seed["xyz"]),
                          actors=len(actors), unknown_mask_frames=len(unknown), GPU_started=False), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--reviewed-instances", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--downsample", type=int, default=4)
    parser.add_argument("--static-voxel", type=float, default=.12)
    parser.add_argument("--actor-voxel", type=float, default=.04)
    parser.add_argument("--max-depth", type=float, default=80)
    parser.add_argument("--test-raw", type=int, nargs="+", default=[101, 113, 125])
    args = parser.parse_args()
    if args.downsample < 1 or min(args.static_voxel, args.actor_voxel, args.max_depth) <= 0:
        raise ValueError("预处理尺寸无效")
    build(args)


if __name__ == "__main__":
    main()
