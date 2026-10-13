#!/usr/bin/env python3
"""真实 DEVA body 的隔离 CPU 适配；默认只检查，不导入模型或发布训练 ready。"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_key] = "1"
import argparse
import json
from pathlib import Path
import sys
sys.dont_write_bytecode = True
import numpy as np
from PIL import Image
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_new(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def calibration(path):
    return {line.split(":", 1)[0]: np.fromstring(line.split(":", 1)[1], sep=" ")
            for line in Path(path).read_text().splitlines() if ":" in line}


def project(lidar, masks, calib):
    # 原 dual_actor_cpu.py build 的投影/可见深度运算，阈值及顺序保持。
    xyz = lidar[:, :3].astype(np.float64)
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    camera = homogeneous @ calib["Tr_velo_to_cam_0"].reshape(3, 4).T
    positive = np.flatnonzero(np.isfinite(camera).all(axis=1) & (camera[:, 2] > .01))
    uvw = np.column_stack((camera[positive], np.ones(len(positive)))) @ calib["P0"].reshape(3, 4).T
    uv = np.floor(uvw[:, :2] / uvw[:, 2:3] + .5).astype(np.int64)
    h, w = masks.shape
    inside = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    source_indices, uv = positive[inside], uv[inside]
    depth = camera[source_indices, 2]
    bins_w = (w + 1) // 2
    bins = (uv[:, 1] // 2) * bins_w + (uv[:, 0] // 2)
    nearest = np.full(((h + 1) // 2) * bins_w, np.inf)
    np.minimum.at(nearest, bins, depth)
    return xyz, homogeneous, source_indices, uv, depth, bins, nearest


def actor_subset(masks, ids, visibility, projection, pose, name, i, body_override=None):
    xyz, homogeneous, source_indices, uv, depth, bins, nearest = projection
    body = np.isin(masks, ids) if body_override is None else body_override
    assert body.any() == (visibility == "visible")
    assert all(np.any(masks == instance) for instance in ids)
    mask_hit = body[uv[:, 1], uv[:, 0]]
    keep = mask_hit & (depth <= nearest[bins] + .5)
    retained = source_indices[keep]
    before_component = len(retained)
    if before_component:
        edges = cKDTree(xyz[retained]).query_pairs(.5, output_type="ndarray")
        graph = coo_matrix((np.ones(len(edges), dtype=np.uint8), (edges[:, 0], edges[:, 1])),
                           shape=(before_component, before_component))
        _, labels = connected_components(graph, directed=False)
        counts = np.bincount(labels)
        dominant = labels == counts.argmax()
        assert dominant.sum() >= before_component * .5, f"No dominant physical component: actor{i}/{name}"
        retained = retained[dominant]
    world = (homogeneous[retained] @ pose.T)[:, :3].astype(np.float32)
    assert np.isfinite(world).all()
    return body, retained.astype(np.int32), world, int(mask_hit.sum()), before_component


def save_ply(path, xyz, timestamps):
    dtype = np.dtype([(k, "<f4") for k in ("x", "y", "z", "nx", "ny", "nz")] +
                     [(k, "u1") for k in ("red", "green", "blue")] + [("time", "<f4")])
    vertices = np.zeros(len(xyz), dtype=dtype)
    for axis, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, axis]
    for key in ("nx", "ny", "nz", "time"):
        vertices[key] = timestamps
    header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n"
    header += "".join("property " + ("uchar" if k in ("red", "green", "blue") else "float") + " " + k + "\n" for k in dtype.names) + "end_header\n"
    with path.open("xb") as stream:
        stream.write(header.encode("ascii"))
        vertices.tofile(stream)


def frame_input(scene, name):
    rgb = np.array(Image.open(scene / "image/image_0" / (name + ".png")).convert("RGB"))
    masks = np.array(Image.open(scene / "mask/mask_0/Annotations" / (name + ".png")))
    sky = np.array(Image.open(scene / "sky/sky_0" / (name + ".png")))
    require(rgb.shape == (1280, 1920, 3), "RGB must be real FRONT 1920x1280")
    require(masks.shape == sky.shape == rgb.shape[:2] and masks.dtype == np.uint8, "Mask/sky dimensions or short-ID dtype invalid")
    require(set(np.unique(sky)).issubset({0, 255}), "Sky must be real binary 0/255")
    lidar = np.fromfile(scene / "velodyne" / (name + ".bin"), dtype=np.float32).reshape(-1, 6)
    require(len(lidar) > 0 and np.isfinite(lidar).all(), "Missing/invalid real Nx6 LiDAR")
    pose = np.loadtxt(scene / "pose" / (name + ".txt")).reshape(4, 4)
    calib = calibration(scene / "calib" / (name + ".txt"))
    require(np.isfinite(pose).all() and calib["Tr_velo_to_cam_0"].size == 12 and calib["P0"].size == 12, "Pose/calibration invalid")
    require(all(np.isfinite(calib[k]).all() for k in ("P0", "Tr_velo_to_cam_0")), "Nonfinite calibration")
    return rgb, masks, lidar, pose, calib


def resolved_body(scene, name, masks, row):
    """原DEVA或已真实推理并逐帧审查的SAM候选；不填洞、不改DEVA、不用GT框。"""
    override = row.get("body_override")
    if override is None:
        return np.isin(masks, row["instance_ids"])
    require(row["visibility"] == "visible" and row.get("reviewed") is True,
            "SAM override only for actually reviewed visible body")
    require(override.get("source") == "cached_SAM_vit_h_real_inference" and
            override.get("root_engineering_reviewed") is True and
            str(override.get("review_note") or "").strip(), "SAM candidate review required")
    receipt = read(override["execution_receipt"])
    require(receipt.get("status") == "REAL_SAM_CANDIDATES_EXPORTED_REVIEW_PENDING" and
            receipt.get("scene") == str(scene) and receipt.get("returncode") == 0 and
            receipt.get("model") == "cached_SAM_vit_h" and receipt.get("GT_boxes_used") is False and
            receipt.get("source_DEVA_modified") is False and receipt.get("amp") is False,
            "Real cached SAM execution receipt required")
    require(all(Path(receipt[k]).is_file() for k in ("checkpoint", "source_script", "interpreter")),
            "Recorded SAM cache/source/interpreter missing")
    frames = [r for r in receipt["frames"] if r["frame_id"] == name]
    require(len(frames) == 1 and frames[0]["prompt"].get("GT_boxes_used") is False,
            "SAM real frame/prompt receipt mismatch")
    chosen = [r for r in frames[0]["model_output_candidates"]
              if r["candidate_index"] == override["candidate_index"] and r["path"] == override["path"]]
    require(len(chosen) == 1, "Body override must be exact recorded real model candidate")
    actual = np.array(Image.open(override["path"]))
    require(actual.dtype == np.uint8 and actual.shape == masks.shape and
            set(np.unique(actual)).issubset({0, 255}) and np.any(actual) and
            int((actual > 0).sum()) == chosen[0]["area"], "Exact binary SAM candidate invalid")
    return actual > 0


def preflight(scene, selection_path, receipt_path):
    require(__debug__, "Do not run with python -O; inherited geometry assertions must remain active")
    require(selection_path and receipt_path, "Reviewed selection and actual sky/DEVA execution receipt required; unknown IDs are blocked")
    prep, chosen, execution = read(scene / "preparation_manifest.json"), read(selection_path), read(receipt_path)
    rows = prep["frames"]
    names = [row["id"] for row in rows]
    require(len(names) == 49 and len(set(names)) == 49 and all(Path(n).name == n and n.isalnum() for n in names), "Exact49 unique frame IDs required")
    window = prep.get("raw_training_range")
    require(isinstance(window, list) and len(window) == 2 and window[1] - window[0] == 48 and [row["source_index"] for row in rows] == list(range(window[0], window[1] + 1)), "Exact49 declared raw window and source mapping required")
    require(prep.get("training_ready") is False, "Prepared source must remain pending independent training readiness")
    require(prep.get("model_frame_interval", .02) == .02 and np.allclose([r["model_time"] for r in rows], np.arange(49) * .02 - .48, rtol=0, atol=1e-12), "Original .02 model times required")
    for directory in ("image/image_0", "mask/mask_0/Annotations", "sky/sky_0"):
        require(sorted(p.stem for p in (scene / directory).glob("*.png")) == sorted(names), "Exact49 real files required: " + directory)
    require((scene / "points3d.ply").is_file(), "Existing real static seed required; adapter does not regenerate it")
    require(chosen.get("status") == "visually_reviewed" and chosen.get("scene") == str(scene), "Selection must refer to this real scene and actual visual review")
    actors = chosen.get("actors")
    require(isinstance(actors, list) and len(actors) > 0, "Actual actor count is unknown; no default actor/ID")
    require([a["actor_index"] for a in actors] == list(range(len(actors))), "Actor indices must be contiguous from0")
    for actor in actors:
        require(bool(actor.get("description")) and bool(actor.get("raw_track_id")), "Physical identity required for later GT diagnostic audit; never used to select points")
        require([r["frame_id"] for r in actor["frames"]] == names, "Each actor needs all49 reviewed frames")
        for row in actor["frames"]:
            ids = row.get("instance_ids")
            require(row.get("reviewed") is True and bool(str(row.get("review_note") or "").strip()), "Every actor/frame needs real visual review")
            require(row.get("visibility") in ("visible", "not_visible") and isinstance(ids, list), "Explicit visibility and instance IDs required")
            require(bool(ids) == (row["visibility"] == "visible") and len(ids) == len(set(ids)), "Empty mask only for reviewed nonvisible frame")
            require(all(type(i) is int and 1 <= i <= 255 for i in ids), "DEVA short IDs must be actual integers in1..255")
    require(execution.get("status") == "REAL_RED49_SKY_AND_DEVA_VALIDATED_ACTOR_SELECTION_PENDING" and execution.get("scene_id") == prep.get("scene_id"), "Actual root queue terminal receipt required")
    require(execution.get("sky_frames") == 49 and execution.get("training_ready") is False, "Queue must have real49 sky and pending training readiness")
    stages = execution.get("stages")
    require(isinstance(stages, list) and len(stages) == 3, "Actual sky/deva/mask_validation stage records required")
    by_stage = {row["stage"]: row for row in stages}
    require(set(by_stage) == {"sky", "deva", "mask_validation"}, "Queue stage names invalid")
    for stage, evidence in by_stage.items():
        command = evidence.get("command")
        require(evidence.get("returncode") == 0 and evidence.get("completed_utc") and isinstance(command, list) and len(command) >= 2, "Actual completed execution evidence required: " + stage)
        require((Path(receipt_path).parent / (stage + ".log")).is_file(), "Actual merged stdout/stderr log required: " + stage)
        require(Path(command[0]).is_file() and Path(command[1]).is_file(), "Recorded interpreter/source missing: " + stage)
        def argument(key):
            return command[command.index(key) + 1] if key in command else None
        if stage in ("sky", "mask_validation"):
            require(argument("--scene") == str(scene), "Queue command scene differs")
        else:
            require(argument("--img_path") == str(scene / "image/image_0") and argument("--output") == str(scene / "mask/mask_0"), "DEVA command input/output differs")
    maskroot = scene / "mask/mask_0"
    validation = read(maskroot / "mask_validation.json")
    require(validation.get("status") == "passed" and validation.get("frames") == 49 and validation.get("mask_dtype") == "uint8", "Real DEVA validator must pass49")
    announced = read(maskroot / "pred.json")["annotations"]
    require(len(announced) == 49, "pred.json exact49 required")
    by_name = {Path(row["file_name"]).stem: row for row in announced}
    require(set(by_name) == set(names), "pred.json frame mapping mismatch")
    for frame_index, name in enumerate(names):
        rgb, masks, lidar, pose, calib = frame_input(scene, name)
        ids, areas = np.unique(masks, return_counts=True)
        counts = {int(k): int(v) for k, v in zip(ids, areas) if k != 0}
        declared = {int(x["id"]): int(x["area"]) for x in by_name[name]["segments_info"]}
        require(counts == declared, "Actual PNG IDs/areas must match pred.json: " + name)
        used = set()
        resolved_union = np.zeros(masks.shape, dtype=bool)
        for actor in actors:
            selected = actor["frames"][frame_index]["instance_ids"]
            require(set(selected).issubset(counts) and not used.intersection(selected), "Unknown/overlapping selected IDs: " + name)
            used.update(selected)
            body = resolved_body(scene, name, masks, actor["frames"][frame_index])
            require(not np.any(resolved_union & body), "Actual resolved actor bodies overlap: " + name)
            resolved_union |= body
    return prep, chosen


def build(scene, selection_path, receipt_path):
    prep, chosen = preflight(scene, selection_path, receipt_path)
    roots = [scene / "dynamic_image/dynamic_image_0", scene / "dynamic_masks/dynamic_mask_0", scene / "dynamic_actor_assets"]
    reportpath = scene / "dynamic_multi_actor_preparation.json"
    require(not any(p.exists() for p in roots + [reportpath]) and not list(scene.glob("points3d_dynamic*.ply")), "No-clobber: existing/partial actor outputs require review")
    for root in roots:
        root.mkdir(parents=True)
    rgbroot, unionroot, actorroot = roots
    count = len(chosen["actors"])
    clouds, pointtimes, indices, reports = [[] for _ in range(count)], [[] for _ in range(count)], [{} for _ in range(count)], []
    for actor in chosen["actors"]:
        i = actor["actor_index"]
        (actorroot / f"actor_{i}/body_masks").mkdir(parents=True)
        reports.append({k: actor[k] for k in ("actor_index", "description", "raw_track_id")})
        reports[-1]["frames"] = []
    for frame_index, row in enumerate(prep["frames"]):
        name = row["id"]
        rgb, masks, lidar, pose, calib = frame_input(scene, name)
        projection = project(lidar, masks, calib)
        union = np.zeros(masks.shape, dtype=bool)
        for actor in chosen["actors"]:
            i = actor["actor_index"]
            selected = actor["frames"][frame_index]
            body_input = resolved_body(scene, name, masks, selected)
            body, retained, world, projected, before = actor_subset(masks, selected["instance_ids"], selected["visibility"], projection, pose, name, i, body_input)
            require(not np.any(union & body), "Actor categorical bodies overlap")
            union |= body
            clouds[i].append(world)
            pointtimes[i].append(np.full(len(world), row["model_time"], dtype=np.float32))
            indices[i][name] = retained
            Image.fromarray(body.astype(np.uint8) * 255).save(actorroot / f"actor_{i}/body_masks" / (name + ".png"))
            reports[i]["frames"].append({"frame_id": name, "model_time": row["model_time"], "visibility": selected["visibility"], "instance_ids": selected["instance_ids"], "body_override": selected.get("body_override"), "pixels": int(body.sum()), "projected_points": projected, "visible_before_component": before, "retained_real_points": len(world), "geometry_status": "real_lidar_points" if len(world) else ("visible_without_lidar" if body.any() else "not_visible")})
        dynamic_rgb = np.zeros_like(rgb)
        dynamic_rgb[union] = rgb[union]
        Image.fromarray(dynamic_rgb).save(rgbroot / (name + ".png"))
        Image.fromarray(union.astype(np.uint8) * 255).save(unionroot / (name + ".png"))
        print(json.dumps({"frame_id": name, "union_pixels": int(union.sum()), "real_points": [r["frames"][-1]["retained_real_points"] for r in reports]}), flush=True)
    for i in range(count):
        xyz, times = np.concatenate(clouds[i]), np.concatenate(pointtimes[i])
        require(len(xyz) > 0, f"Actor{i} has no real points; no placeholder/empty PLY")
        save_ply(scene / f"points3d_dynamic_{i}.ply", xyz, times)
        np.savez_compressed(actorroot / f"actor_{i}/source_lidar_indices.npz", **indices[i])
        reports[i]["total_real_points"] = len(xyz)
    result = {"status": "built_pending_validation", "scene": str(scene), "actors": reports, "frames": 49, "selection_path": str(selection_path), "execution_receipt": str(receipt_path), "gpu_used": False, "training_ready": False, "gt_3d_boxes_used_to_filter_training_points": False, "m_detector_used": False, "shadow_supervision": "absent", "static_seed_modified": False, "static_exclusion": "Original loader strict Euclidean0.1m remains pending actual loader", "geometry": "Original dual_actor_cpu projection/depth/component; raw frame pose world XYZ; nx/ny/nz/time all model_time .02"}
    write_new(reportpath, result)
    return result


def validate(scene, selection_path, receipt_path, raw_audit):
    prep, chosen = preflight(scene, selection_path, receipt_path)
    require(raw_audit and Path(raw_audit).is_dir(), "Red raw_actor_identity_frames required for diagnostic QA; not a point filter")
    from plyfile import PlyData
    count = len(chosen["actors"])
    clouds = [PlyData.read(str(scene / f"points3d_dynamic_{i}.ply"))["vertex"] for i in range(count)]
    indices = [np.load(scene / f"dynamic_actor_assets/actor_{i}/source_lidar_indices.npz", allow_pickle=False) for i in range(count)]
    offsets, qa = [0] * count, [[] for _ in range(count)]
    for frame_index, row in enumerate(prep["frames"]):
        name = row["id"]
        rgb, masks, lidar, pose, calib = frame_input(scene, name)
        union = np.zeros(masks.shape, dtype=bool)
        labeldata = read(Path(raw_audit) / (name + ".json"))
        for actor in chosen["actors"]:
            i = actor["actor_index"]
            body = resolved_body(scene, name, masks, actor["frames"][frame_index])
            actual = np.array(Image.open(scene / f"dynamic_actor_assets/actor_{i}/body_masks" / (name + ".png")))
            require(set(np.unique(actual)).issubset({0, 255}) and np.array_equal(actual > 0, body) and not np.any(union & body), "Exact full body union failed")
            union |= body
            source = indices[i][name]
            require(source.dtype == np.int32 and np.all(source >= 0) and np.all(source < len(lidar)) and len(np.unique(source)) == len(source), "Real source indices invalid")
            world = (np.column_stack((lidar[source, :3].astype(np.float64), np.ones(len(source)))) @ pose.T)[:, :3].astype(np.float32)
            vertices = clouds[i][offsets[i]:offsets[i] + len(source)]
            offsets[i] += len(source)
            require(np.array_equal(np.column_stack([vertices[k] for k in ("x", "y", "z")]), world), "PLY must be exact sensor subset in raw world pose")
            require(all(np.all(vertices[k] == np.float32(row["model_time"])) for k in ("nx", "ny", "nz", "time")), "Temporal PLY properties changed")
            # GT中心/尺寸仅诊断并失败；完全不删除点，也不造body。
            gt = [x for x in labeldata["laser_labels"] if x["raw_track_id"] == actor["raw_track_id"]]
            if len(world):
                require(len(gt) == 1, "Physical identity diagnostic GT observation missing/ambiguous")
                dimensions = np.array([gt[0]["source_label"]["box"][k] for k in ("length", "width", "height")])
                distances = np.linalg.norm(world.astype(np.float64) - np.array(gt[0]["world_center_xyz"]), axis=1)
                bound = np.linalg.norm(dimensions) / 2 + 2.0
                require(distances.max() <= bound, f"Physical geometry QA failed: actor{i}/{name}; review, no automatic cleaning")
                qa[i].append({"frame_id": name, "points": len(world), "max_distance_to_raw_track_center_m": float(distances.max()), "diagnostic_bound_m": float(bound)})
        binary = np.array(Image.open(scene / "dynamic_masks/dynamic_mask_0" / (name + ".png")))
        dynamic = np.array(Image.open(scene / "dynamic_image/dynamic_image_0" / (name + ".png")).convert("RGB"))
        expected = np.zeros_like(rgb)
        expected[union] = rgb[union]
        require(set(np.unique(binary)).issubset({0, 255}) and np.array_equal(binary > 0, union) and np.array_equal(dynamic, expected), "Canonical union / GT*union dynamicRGB differs")
    require(all(offsets[i] == len(clouds[i]) and offsets[i] > 0 for i in range(count)), "PLY offsets/counts invalid")
    result = {"status": "PASS_ACTOR_CPU_ONLY", "frames": 49, "actor_count": count, "real_points": offsets, "actor_geometry_qa": qa, "all_ply_vertices_exact_real_sensor_subset": True, "all_temporal_properties_exact": True, "binary_union_and_rgb_exact": True, "physical_gt_labels_used": "Diagnostics only; fail rather than filter", "gpu_used": False, "training_ready": False, "pending": ["Actual CPU loader + new transform/anchor", "Original strict0.1m static exclusion + retained static actors audit", "SGCM ground/regions", "GN11 gate/runtime and independent runner", "Shadow remains absent"]}
    write_new(scene / "dynamic_multi_actor_validation.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", nargs="?", default="check", choices=("check", "build", "validate"))
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--selection", type=Path)
    p.add_argument("--execution-receipt", type=Path)
    p.add_argument("--raw-audit", type=Path)
    args = p.parse_args()
    try:
        scene = args.scene.resolve()
        if args.stage == "check":
            prep, chosen = preflight(scene, args.selection, args.execution_receipt)
            result = {"status": "READY_ACTOR_CPU_PLAN", "frames": 49, "actor_count": len(chosen["actors"]), "output_written": False, "training_ready": False}
        elif args.stage == "build":
            result = build(scene, args.selection, args.execution_receipt)
        else:
            result = validate(scene, args.selection, args.execution_receipt, args.raw_audit)
        require(not any(n == "torch" or n.startswith("tensorflow") for n in sys.modules), "Forbidden framework imported")
        print(json.dumps(result, ensure_ascii=False, allow_nan=False))
    except (ValueError, KeyError, OSError, AssertionError) as error:
        print(json.dumps({"status": "BLOCKED_ACTOR_CPU", "reason": str(error), "training_ready": False}, ensure_ascii=False))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
