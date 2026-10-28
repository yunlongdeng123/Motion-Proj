"""第一场景原始观测阴影候选；CPU，禁止读取删除/补景/渲染 RGB。

这是可审计的候选证据，不是阴影分割真值。道路语义只用于选择实际
LiDAR 表面；源帧是否完全无阴影未知，因此道路反证只能标为 weak。
"""
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _name in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_name] = "2"

import argparse
import heapq
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import (binary_dilation, binary_erosion,
                           distance_transform_edt, label, map_coordinates)

CTX = "14705303724557273004_3105_000_3125_000"
TARGET = "gTriXsyDrbzzU6apS1WErQ"
DEFAULT_ROOT = Path("/root/autodl-tmp/drivingeditor_repro/campaign/v43_ownership_20261028")
H, W = 320, 480
REASONS = {
    "outside_candidate_domain": 1, "body_occludes_surface": 2,
    "query_not_confident_road": 4, "query_no_matching_real_lidar": 8,
    "other_actor_near_query_surface": 16, "no_exposure_matched_source": 32,
    "insufficient_target_away_source_observations": 64,
    "source_rgb_not_stable": 128, "no_query_darkening": 256,
    "query_rgb_not_multiplicative_attenuation": 512,
    "insufficient_target_motion_witness": 1024,
    "source_shadow_state_unknown": 2048, "local_plane_uncertain": 4096,
    "weak_road_counterevidence_only": 8192,
}
METHOD = {
    "schema": "raw-observation-shadow-evidence-v1",
    "context": CTX, "target_track": TARGET,
    "input_roles": {
        "rgb": "original fullclip camera RGB only; never deletion/inpainting/render",
        "lidar_pose_calibration": "original observed road geometry and visibility",
        "semantics": "frozen model prediction; road selection, not shadow absence",
        "body": "existing target body pseudo-mask; preserved separately, not human GT",
        "annotations": "observed target motion/contact geometry and actor exclusion",
    },
    "cityscapes_encoding": "saved prediction is Cityscapes+1; prediction-1 gives 0..18; labels=0 may be low-confidence unknown",
    "source_stride": 4,
    "source_minimum": "3 distinct target-away original observations spanning >=0.5s",
    "plane_fit_radius_m": 15.0, "plane_inlier_residual_m": 0.12,
    "plane_max_median_residual_m": 0.10, "plane_min_inliers": 100,
    "candidate_footprint_distance_m": 5.0,
    "source_target_footprint_clearance_m": 4.0,
    "source_other_actor_clearance_m": 2.5,
    "query_other_actor_clearance_m": 1.0,
    "lidar_pixel_support_radius": 2.5, "lidar_depth_tolerance_m": 0.65,
    "exposure_min_anchors": 60, "exposure_max_median_rgb_error": 0.045,
    "reference_max_median_rgb_deviation": 0.045,
    "query_max_luminance_ratio": 0.84, "query_min_luminance_drop": 0.035,
    "attenuation_max_rgb_residual": 0.035,
    "motion_witness_offsets_raw": [-6, -3, 3, 6],
    "motion_witness_min_target_displacement_m": 0.8,
    "motion_witness_min_support": 2, "candidate_min_component_pixels": 6,
    "road_counterevidence_certified": False,
    "source_shadow_state_unknown": True,
    "training_updates": 0, "GPU_calls": 0, "human_GT": None,
    "future_frame_access": "offline diagnostic uses entire original clip; not an online claim",
    "failure_ledger_refs": [], "failure_ledger_delta": "none",
}


def sample(array, uv, order=1):
    """只对已知投影采样，越界返回 NaN；mask 使用最近邻。"""
    if array.ndim == 2:
        return map_coordinates(array.astype(np.float32), [uv[:, 1], uv[:, 0]],
                               order=order, mode="constant", cval=np.nan,
                               prefilter=False)
    return np.stack([sample(array[..., k], uv, order) for k in range(array.shape[2])], 1)


def luminance(rgb):
    return rgb @ np.array([0.2126, 0.7152, 0.0722])


def footprint_distance(points_xy, actor):
    if actor is None:
        return np.full(len(points_xy), np.inf)
    local = (points_xy - actor["center"][:2]) @ actor["rotation"]
    outside = np.maximum(np.abs(local) - actor["half_extent"], 0)
    return np.linalg.norm(outside, axis=1)


def actor_distances(points_xy, frame, exclude_target=False):
    distance = np.full(len(points_xy), np.inf)
    for actor in frame["actors"]:
        if exclude_target and actor["id"] == TARGET:
            continue
        distance = np.minimum(distance, footprint_distance(points_xy, actor))
    return distance


class RawClip:
    def __init__(self, root):
        self.root = root
        self.source = root / "target_unknown_v2/fullclip_inputs"
        self.semantics = root / "target_unknown_v2/fullclip_semantics"
        self.dataset = root / "datasets" / CTX
        manifest = json.loads((self.source / "manifest.json").read_text())
        if manifest["context"] != CTX:
            raise ValueError("仅允许第一场景")
        self.frames = {int(f["raw"]): f for f in manifest["frames"]}
        self.queries = json.loads((self.dataset / "manifest.json").read_text())["frames"]
        self.cache = {}

    def get(self, raw):
        if raw in self.cache:
            return self.cache[raw]
        meta = self.frames[raw]
        fid = meta["frame_id"]
        pose = np.loadtxt(self.source / "pose" / (fid + ".txt"))
        cal = {line.split(":")[0]: np.fromstring(line.split(":")[1], sep=" ")
               for line in (self.source / "calib" / (fid + ".txt")).read_text().splitlines()}
        camera = cal["Tr_velo_to_cam_0"].reshape(3, 4)
        intrinsic = cal["P0"].reshape(3, 4)[:, :3].copy()
        intrinsic[:2] /= 4.0
        rgb = np.asarray(Image.open(self.source / "rgb" / ("%03d.png" % raw)).convert("RGB"))
        if rgb.shape != (H, W, 3):
            raise ValueError((raw, rgb.shape))
        with np.load(self.semantics / ("%03d.npz" % raw)) as sem:
            prediction = sem["prediction"].copy()
            confidence = sem["confidence"].astype(np.float32)
            if prediction.min() < 1 or prediction.max() > 19:
                raise ValueError("未核实的语义编码: %s" % raw)
            city = prediction.astype(np.int16) - 1
        body_path = self.dataset / ("%03d.npz" % raw)
        body = np.zeros((H, W), bool)
        body_rgb_max_difference = None
        if body_path.exists():
            with np.load(body_path) as old:
                body = old["body"].copy()
                body_rgb_max_difference = int(np.abs(old["rgb"].astype(int) - rgb.astype(int)).max())
        actors = []
        for obj in json.loads((self.source / "annotations" / (fid + ".json")).read_text()):
            if obj["type"] not in ["TYPE_VEHICLE", "TYPE_PEDESTRIAN", "TYPE_CYCLIST"]:
                continue
            box = obj.get("camera_synced_box") or obj["box"]
            c, s = np.cos(box.get("heading", 0)), np.sin(box.get("heading", 0))
            rotation_sensor = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
            rotation_world = pose[:3, :3] @ rotation_sensor
            theta = np.arctan2(rotation_world[1, 0], rotation_world[0, 0])
            c, s = np.cos(theta), np.sin(theta)
            center = pose[:3, :3] @ np.array([box["center_x"], box["center_y"], box["center_z"]]) + pose[:3, 3]
            actors.append({"id": obj["id"], "center": center,
                           "rotation": np.array([[c, -s], [s, c]]),
                           "half_extent": np.array([box["length"], box["width"]]) / 2,
                           "height": box["height"]})
        frame = {"raw": raw, "meta": meta, "pose": pose, "camera": camera,
                 "K": intrinsic, "rgb": rgb.astype(np.float32) / 255,
                 "city": city, "confidence": confidence, "body": body,
                 "actors": actors,
                 "target": next((o for o in actors if o["id"] == TARGET), None),
                 "body_rgb_max_difference": body_rgb_max_difference}
        points_sensor = np.fromfile(self.source / "velodyne" / (fid + ".bin"), np.float32).reshape(-1, 6)[:, :3].astype(np.float64)
        world = points_sensor @ pose[:3, :3].T + pose[:3, 3]
        uv, z = project(world, frame)
        # 不将后方或超大投影转成 int，避免无关回波触发溢出 warning。
        inside = (z > 0.1) & np.isfinite(uv).all(1) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
        pix = np.full((len(uv), 2), -1, np.int32)
        pix[inside] = np.floor(uv[inside]).astype(np.int32)
        ix = np.flatnonzero(inside)
        pp = pix[ix]
        nearest = np.full((H, W), np.inf, np.float32)
        np.minimum.at(nearest, (pp[:, 1], pp[:, 0]), z[ix])
        all_distance, all_index = distance_transform_edt(~np.isfinite(nearest), return_indices=True)
        frame["all_lidar_distance"] = all_distance.astype(np.float32)
        frame["all_lidar_depth"] = nearest[tuple(all_index)]
        front = z[ix] <= nearest[pp[:, 1], pp[:, 0]] + 0.03
        road = (city[pp[:, 1], pp[:, 0]] == 0) & (confidence[pp[:, 1], pp[:, 0]] >= 0.75) & front
        ri = ix[road]
        rp = pix[ri]
        road_depth = np.full((H, W), np.inf, np.float32)
        np.minimum.at(road_depth, (rp[:, 1], rp[:, 0]), z[ri])
        distance, index = distance_transform_edt(~np.isfinite(road_depth), return_indices=True)
        frame["road_lidar_distance"] = distance.astype(np.float32)
        frame["road_lidar_depth"] = road_depth[tuple(index)]
        frame["road_world"] = world[ri]
        frame["road_uv"] = uv[ri]
        frame["dynamic_guard"] = binary_dilation(city >= 11, iterations=3)
        self.cache[raw] = frame
        return frame


def project(world, frame):
    sensor = (world - frame["pose"][:3, 3]) @ frame["pose"][:3, :3]
    camera = sensor @ frame["camera"][:, :3].T + frame["camera"][:, 3]
    z = camera[:, 2]
    uv = camera[:, :2] / np.maximum(z[:, None], 1e-8)
    uv = uv * [frame["K"][0, 0], frame["K"][1, 1]] + [frame["K"][0, 2], frame["K"][1, 2]]
    return uv, z


def visibility(world, frame, baseline=False):
    uv, z = project(world, frame)
    inside = (z > 0.1) & (uv[:, 0] >= 1) & (uv[:, 0] < W - 2) & (uv[:, 1] >= 1) & (uv[:, 1] < H - 2)
    road = (sample(frame["city"], uv, 0) == 0) & (sample(frame["confidence"], uv, 0) >= (0.75 if baseline else 0.55))
    lidar = (sample(frame["road_lidar_distance"], uv, 0) <= METHOD["lidar_pixel_support_radius"])
    lidar &= np.abs(z - sample(frame["road_lidar_depth"], uv, 0)) <= METHOD["lidar_depth_tolerance_m"]
    if baseline:
        actor_clear = actor_distances(world[:, :2], frame) >= METHOD["source_other_actor_clearance_m"]
        actor_clear &= footprint_distance(world[:, :2], frame["target"]) >= METHOD["source_target_footprint_clearance_m"]
        clear_mask = sample(frame["dynamic_guard"], uv, 0) == 0
    else:
        actor_clear = actor_distances(world[:, :2], frame, exclude_target=True) >= METHOD["query_other_actor_clearance_m"]
        clear_mask = sample(frame["body"], uv, 0) == 0
    valid = inside & road & lidar & actor_clear & clear_mask
    return valid, uv, {"inside": inside, "road": road, "lidar": lidar,
                       "actor_clear": actor_clear, "clear_mask": clear_mask}


def fit_road_plane(frame):
    center = frame["target"]["center"].copy()
    points = frame["road_world"]
    keep = np.linalg.norm(points[:, :2] - center[:2], axis=1) <= METHOD["plane_fit_radius_m"]
    keep &= actor_distances(points[:, :2], frame) > 0.3
    points = points[keep]
    if len(points) < METHOD["plane_min_inliers"]:
        return None, {"valid": False, "reason": "fewer_than_100_actual_local_road_returns", "returns": len(points)}
    design = np.column_stack([points[:, :2] - center[:2], np.ones(len(points))])
    values = points[:, 2] - center[2]
    active = np.ones(len(points), bool)
    for _ in range(6):
        coefficients = np.linalg.lstsq(design[active], values[active], rcond=None)[0]
        residual = np.abs(design @ coefficients - values)
        threshold = min(0.25, max(0.08, 3 * np.median(residual[active])))
        updated = residual < threshold
        if updated.sum() < METHOD["plane_min_inliers"]:
            break
        active = updated
    residual = np.abs(design @ coefficients - values)
    active = residual < METHOD["plane_inlier_residual_m"]
    if active.sum() >= METHOD["plane_min_inliers"]:
        coefficients = np.linalg.lstsq(design[active], values[active], rcond=None)[0]
    median = float(np.median(np.abs(design[active] @ coefficients - values[active]))) if active.any() else np.inf
    valid = active.sum() >= METHOD["plane_min_inliers"] and median <= METHOD["plane_max_median_residual_m"] and np.linalg.norm(coefficients[:2]) < 0.2
    plane = {"origin": center, "coefficients": coefficients}
    return plane, {"valid": bool(valid), "actual_road_returns": len(points), "inlier_returns": int(active.sum()),
                   "median_abs_residual_m": median, "origin_world": center.tolist(),
                   "z_minus_origin_z_equals": coefficients.tolist(), "inlier_threshold_m": 0.12}


def plane_height(xy, plane):
    return (xy - plane["origin"][:2]) @ plane["coefficients"][:2] + plane["origin"][2] + plane["coefficients"][2]


def plane_pixels(frame, plane):
    yy, xx = np.mgrid[:H, :W]
    uv = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    ray_camera = np.column_stack([(uv[:, 0] - frame["K"][0, 2]) / frame["K"][0, 0],
                                  (uv[:, 1] - frame["K"][1, 2]) / frame["K"][1, 1], np.ones(len(uv))])
    origin_sensor = -frame["camera"][:, :3].T @ frame["camera"][:, 3]
    origin = origin_sensor @ frame["pose"][:3, :3].T + frame["pose"][:3, 3]
    rays = ray_camera @ frame["camera"][:, :3] @ frame["pose"][:3, :3].T
    slope = plane["coefficients"][:2]
    numerator = plane_height(origin[None, :2], plane)[0] - origin[2]
    denominator = rays[:, 2] - rays[:, :2] @ slope
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = numerator / denominator
    world = origin + scale[:, None] * rays
    valid = np.isfinite(world).all(1) & (scale > 1) & (scale < 100)
    return world, valid


def fit_exposure(query, source, anchors, anchor_rgb):
    valid, uv, _ = visibility(anchors, source, baseline=True)
    source_rgb = sample(source["rgb"], uv)
    valid &= np.isfinite(source_rgb).all(1) & (luminance(source_rgb) > 0.05) & (luminance(anchor_rgb) > 0.05)
    if valid.sum() < METHOD["exposure_min_anchors"]:
        return None, {"valid": False, "anchors": int(valid.sum()), "reason": "insufficient_actual_shared_road_anchors"}
    s, q = source_rgb[valid], anchor_rgb[valid]
    gain = np.median(q / np.maximum(s, 0.03), axis=0)
    error = np.mean(np.abs(s * gain - q), axis=1)
    inlier = error <= max(0.035, min(0.08, 2.5 * np.median(error)))
    if inlier.sum() >= METHOD["exposure_min_anchors"]:
        gain = np.median(q[inlier] / np.maximum(s[inlier], 0.03), axis=0)
        error = np.mean(np.abs(s[inlier] * gain - q[inlier]), axis=1)
    median = float(np.median(error))
    valid_fit = bool(inlier.sum() >= METHOD["exposure_min_anchors"] and median <= METHOD["exposure_max_median_rgb_error"] and np.all((gain > 0.65) & (gain < 1.5)))
    return gain if valid_fit else None, {"valid": valid_fit, "anchors": int(valid.sum()), "inlier_anchors": int(inlier.sum()),
                                         "gain_rgb_source_to_query": gain.tolist(), "median_abs_rgb_error": median,
                                         "p90_abs_rgb_error": float(np.percentile(error, 90)), "additive_offset": 0}


def collect_references(world, source_frames, gains, excluded_raw=None):
    n = len(world)
    values = np.full((len(source_frames), n, 3), np.nan, np.float32)
    source_raw = np.array([f["raw"] for f in source_frames], np.int16)
    for i, frame in enumerate(source_frames):
        if frame["raw"] not in gains or frame["raw"] == excluded_raw:
            continue
        valid, uv, _ = visibility(world, frame, baseline=True)
        rgb = sample(frame["rgb"], uv) * gains[frame["raw"]]
        valid &= np.isfinite(rgb).all(1) & (luminance(rgb) > 0.05) & (luminance(rgb) < 0.95)
        values[i, valid] = rgb[valid]
    observed = np.isfinite(values[..., 0])
    count = observed.sum(0)
    # nanmedian 的缺测预期存在，显式处理而非填入合成颜色。
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        median = np.nanmedian(values, axis=0)
        deviation = np.nanmedian(np.mean(np.abs(values - median[None]), axis=2), axis=0)
    first = np.min(np.where(observed, source_raw[:, None], 32767), axis=0)
    last = np.max(np.where(observed, source_raw[:, None], -1), axis=0)
    enough = (count >= 3) & ((last - first) >= 5)
    stable = np.isfinite(deviation) & (deviation <= METHOD["reference_max_median_rgb_deviation"])
    # 三帧展示只选最接近稳健中位数的实际观测，不把中位数当作真实源图。
    error = np.mean(np.abs(values - median[None]), axis=2)
    error[~observed] = np.inf
    selected = np.argsort(error, axis=0)[:3].T
    selected_raw = source_raw[selected]
    selected_valid = np.take_along_axis(observed.T, selected, axis=1)
    selected_raw[~selected_valid] = -1
    return {"rgb": median, "count": count.astype(np.uint8), "enough": enough,
            "stable": stable, "valid": enough & stable, "deviation": deviation,
            "span_raw": (last - first).astype(np.int16), "source_raw": selected_raw,
            "all_values": values}


def darkening(rgb, reference):
    ref = reference["rgb"]
    ql, bl = luminance(rgb), luminance(ref)
    ratio = ql / np.maximum(bl, 0.02)
    drop = bl - ql
    attenuation_error = np.mean(np.abs(rgb - ratio[:, None] * ref), axis=1)
    supported = reference["valid"] & (ratio <= METHOD["query_max_luminance_ratio"]) & (ratio >= 0.25)
    supported &= (drop >= METHOD["query_min_luminance_drop"]) & (attenuation_error <= METHOD["attenuation_max_rgb_residual"])
    strength = np.clip((1 - ratio - 0.10) / 0.22, 0, 1) * np.clip(drop / 0.07, 0, 1)
    strength *= np.clip(1 - attenuation_error / 0.07, 0, 1)
    return supported, strength, ratio, drop, attenuation_error


def as_map(value, indices, dtype=None, fill=0):
    shape = (H * W,) + value.shape[1:]
    out = np.full(shape, fill, dtype=dtype or value.dtype)
    out[indices] = value
    return out.reshape((H, W) + value.shape[1:])


def outline(mask):
    return mask & ~binary_erosion(mask)


def overlays(output, raw, frame, arrays):
    rgb = np.round(frame["rgb"] * 255).astype(np.uint8)
    shadow = rgb.astype(float).copy()
    shadow[arrays["shadow_candidate"]] = 0.45 * shadow[arrays["shadow_candidate"]] + 0.55 * np.array([205, 60, 245])
    shadow[outline(arrays["candidate_domain"])] = [245, 165, 40]
    shadow[outline(frame["body"])] = [0, 235, 255]
    bg = rgb.astype(float).copy()
    good = arrays["road_counterevidence_valid"]
    bg[good] = 0.5 * bg[good] + 0.5 * np.array([60, 210, 110])
    bg[outline(frame["body"])] = [0, 235, 255]
    panels = Image.new("RGB", (W * 3, H + 42), "white")
    for i, panel in enumerate([rgb, shadow, bg]):
        panels.paste(Image.fromarray(np.clip(panel, 0, 255).astype(np.uint8)), (W * i, 42))
    draw = ImageDraw.Draw(panels)
    labels = ["raw %03d original RGB" % raw, "purple: candidate; cyan: body; orange: domain", "green: weak road counterevidence; certified=0"]
    for i, text in enumerate(labels):
        draw.text((W * i + 7, 8), text, fill="black")
    panels.save(output / ("%03d_overlay.png" % raw))
    Image.fromarray(rgb).save(output / ("%03d_original.png" % raw))


def components(mask, score, world_map):
    regions, n = label(mask)
    rows = []
    kept = np.zeros_like(mask)
    for component in range(1, n + 1):
        yy, xx = np.where(regions == component)
        if len(xx) < METHOD["candidate_min_component_pixels"]:
            continue
        kept[yy, xx] = True
        rows.append({"pixels": len(xx), "bbox_xyxy_inclusive": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())],
                     "mean_score": float(score[yy, xx].mean()),
                     "world_centroid": world_map[yy, xx].mean(0).tolist(),
                     "reasons": ["actual_LiDAR_supported_road_plane", "same_world_surface_brighter_when_target_away",
                                 "exposure_matched_original_RGB", "multiplicative_RGB_attenuation",
                                 "darkening_observed_at_target_carried_surface_in_2_or_more_neighbor_frames"],
                     "human_verdict": None})
    return kept, rows


def source_gallery(output, raw, query, arrays, clip):
    eligible = arrays["shadow_evidence_valid"]
    if not eligible.any():
        return None
    score = np.where(eligible, arrays["shadow_score"], -1)
    y, x = np.unravel_index(np.argmax(score), score.shape)
    world = arrays["world_point"][y, x].astype(float)
    sources = [int(v) for v in arrays["source_raw"][y, x] if v >= 0]
    panels = Image.new("RGB", (200 * (1 + len(sources)), 190), "white")
    for i, frame in enumerate([query] + [clip.get(s) for s in sources]):
        uv, _ = project(world[None], frame)
        u, v = uv[0]
        image = Image.fromarray(np.round(frame["rgb"] * 255).astype(np.uint8))
        left, top = int(round(u)) - 50, int(round(v)) - 40
        crop = image.crop((left, top, left + 100, top + 80)).resize((200, 160))
        draw = ImageDraw.Draw(crop)
        draw.line((94, 80, 106, 80), fill="magenta", width=2)
        draw.line((100, 74, 100, 86), fill="magenta", width=2)
        panels.paste(crop, (i * 200, 30))
        ImageDraw.Draw(panels).text((i * 200 + 5, 5), ("query " if i == 0 else "actual source ") + str(frame["raw"]), fill="black")
    panels.save(output / ("%03d_same_world_point_sources.png" % raw))
    return {"query_pixel_xy": [int(x), int(y)], "query_score": float(arrays["shadow_score"][y, x]),
            "world_point": world.tolist(), "source_raw": sources,
            "selection": "highest available evidence score, may still fail candidate gate"}


def query_evidence(clip, raw, source_frames, output):
    started = time.monotonic()
    query = clip.get(raw)
    if query["target"] is None:
        raise ValueError("query 无目标 annotation")
    plane, plane_report = fit_road_plane(query)
    if plane is None:
        # 保留明确失败文件；不借平面缺失生成背景或阴影。
        arrays = {"body": query["body"], "shadow_candidate": np.zeros((H, W), bool),
                  "shadow_score": np.zeros((H, W), np.float32), "shadow_evidence_valid": np.zeros((H, W), bool),
                  "candidate_domain": np.zeros((H, W), bool), "road_counterevidence": np.zeros((H, W), np.float32),
                  "road_counterevidence_valid": np.zeros((H, W), bool), "road_counterevidence_certified": np.zeros((H, W), bool),
                  "world_point": np.full((H, W, 3), np.nan, np.float64),
                  "source_raw": np.full((H, W, 3), -1, np.int16), "source_count": np.zeros((H, W), np.uint8),
                  "reason_bits": np.full((H, W), REASONS["local_plane_uncertain"], np.uint16)}
        np.savez_compressed(output / ("%03d.npz" % raw), **arrays)
        overlays(output, raw, query, arrays)
        return {"raw": raw, "plane": plane_report, "failure": "no_local_plane", "shadow_candidate_pixels": 0}
    world, ray_valid = plane_pixels(query, plane)
    target_distance = footprint_distance(world[:, :2], query["target"])
    domain = ray_valid & (target_distance <= METHOD["candidate_footprint_distance_m"])
    index = np.flatnonzero(domain)
    points = world[index]
    query_rgb = query["rgb"].reshape(-1, 3)[index]
    query_visible, _, details = visibility(points, query)
    if not plane_report["valid"]:
        query_visible[:] = False
    anchors = query["road_world"]
    auv = query["road_uv"]
    anchor_ok = footprint_distance(anchors[:, :2], query["target"]) > 7.0
    anchor_ok &= actor_distances(anchors[:, :2], query, exclude_target=True) > 3.0
    anchor_ok &= np.abs(anchors[:, 2] - plane_height(anchors[:, :2], plane)) < 0.15
    anchors, auv = anchors[anchor_ok], auv[anchor_ok]
    if len(anchors) > 4000:
        choose = np.linspace(0, len(anchors) - 1, 4000).astype(int)
        anchors, auv = anchors[choose], auv[choose]
    anchor_rgb = sample(query["rgb"], auv)
    exposure = {}
    gains = {raw: np.ones(3)}
    witness_raw = [raw + d for d in METHOD["motion_witness_offsets_raw"] if raw + d in clip.frames]
    all_needed = {f["raw"]: f for f in source_frames}
    all_needed.update({r: clip.get(r) for r in witness_raw})
    for frame in all_needed.values():
        gain, report = fit_exposure(query, frame, anchors, anchor_rgb)
        exposure[str(frame["raw"])] = report
        if gain is not None:
            gains[frame["raw"]] = gain
    reference = collect_references(points, source_frames, gains, excluded_raw=raw)
    dark, photo_score, ratio, drop, attenuation_error = darkening(query_rgb, reference)
    witness_valid = np.zeros((len(points), len(witness_raw)), bool)
    witness_support = witness_valid.copy()
    witness_ratios = np.full(witness_valid.shape, np.nan, np.float32)
    witness_report = []
    local = (points[:, :2] - query["target"]["center"][:2]) @ query["target"]["rotation"]
    # 只给query具有原始暗化证据的点计算随动见证；不使用任何编辑图。
    possible = query_visible & reference["valid"] & dark
    pi = np.flatnonzero(possible)
    for wi, wr in enumerate(witness_raw):
        frame = clip.get(wr)
        row = {"raw": wr, "evidence_points_examined": len(pi)}
        if frame["target"] is None or wr not in gains:
            row["reason"] = "target_or_valid_exposure_missing"
            witness_report.append(row)
            continue
        displacement = float(np.linalg.norm(frame["target"]["center"][:2] - query["target"]["center"][:2]))
        row["target_displacement_m"] = displacement
        if displacement < METHOD["motion_witness_min_target_displacement_m"] or len(pi) == 0:
            row["reason"] = "insufficient_target_translation_or_no_query_photometric_evidence"
            witness_report.append(row)
            continue
        carried_xy = local[pi] @ frame["target"]["rotation"].T + frame["target"]["center"][:2]
        carried = np.column_stack([carried_xy, plane_height(carried_xy, plane)])
        vis, uv, _ = visibility(carried, frame)
        carried_rgb = sample(frame["rgb"], uv) * gains[wr]
        carried_reference = collect_references(carried, source_frames, gains, excluded_raw=wr)
        cdark, _, cratio, _, _ = darkening(carried_rgb, carried_reference)
        valid = vis & carried_reference["valid"]
        support = valid & cdark
        witness_valid[pi, wi] = valid
        witness_support[pi, wi] = support
        witness_ratios[pi, wi] = cratio
        row.update(valid_surface_points=int(valid.sum()), darkening_support_points=int(support.sum()))
        witness_report.append(row)
    support_count = witness_support.sum(1)
    valid_count = witness_valid.sum(1)
    evidence_valid = query_visible & reference["valid"]
    score = np.where(evidence_valid, photo_score * np.minimum(support_count / 2, 1), 0).astype(np.float32)
    candidate = evidence_valid & dark & (support_count >= METHOD["motion_witness_min_support"])
    counter_residual = np.mean(np.abs(query_rgb - reference["rgb"]), axis=1)
    counter_valid = evidence_valid & (counter_residual <= 0.035) & (np.abs(ratio - 1) <= 0.10)
    counter_score = np.where(counter_valid, np.clip(1 - counter_residual / 0.05, 0, 1) * np.clip(reference["count"] / 5, 0, 1), 0).astype(np.float32)
    reasons = np.full(len(points), REASONS["source_shadow_state_unknown"], np.uint16)
    for name, mask in [("body_occludes_surface", ~details["clear_mask"]),
                       ("query_not_confident_road", ~details["road"]),
                       ("query_no_matching_real_lidar", ~details["lidar"]),
                       ("other_actor_near_query_surface", ~details["actor_clear"]),
                       ("insufficient_target_away_source_observations", ~reference["enough"]),
                       ("source_rgb_not_stable", ~reference["stable"]),
                       ("no_query_darkening", (ratio > METHOD["query_max_luminance_ratio"]) | (drop < METHOD["query_min_luminance_drop"])),
                       ("query_rgb_not_multiplicative_attenuation", attenuation_error > METHOD["attenuation_max_rgb_residual"]),
                       ("insufficient_target_motion_witness", support_count < METHOD["motion_witness_min_support"]),
                       ("weak_road_counterevidence_only", counter_valid)]:
        reasons[mask] |= REASONS[name]
    if not plane_report["valid"]:
        reasons |= REASONS["local_plane_uncertain"]
    if not any(v["valid"] for v in exposure.values()):
        reasons |= REASONS["no_exposure_matched_source"]
    arrays = {
        "body": query["body"], "candidate_domain": domain.reshape(H, W),
        "shadow_candidate": as_map(candidate, index), "shadow_score": as_map(score, index),
        "shadow_evidence_valid": as_map(evidence_valid, index),
        "road_counterevidence": as_map(counter_score, index), "road_counterevidence_valid": as_map(counter_valid, index),
        "road_counterevidence_certified": np.zeros((H, W), bool),
        "world_point": as_map(points, index, np.float64, np.nan),
        "source_raw": as_map(reference["source_raw"], index, np.int16, -1),
        "source_count": as_map(reference["count"], index),
        "source_span_raw": as_map(reference["span_raw"], index, np.int16, -1),
        "reference_rgb_exposure_matched": as_map(reference["rgb"], index, np.float32, np.nan),
        "reference_rgb_deviation": as_map(reference["deviation"], index, np.float32, np.nan),
        "luminance_ratio": as_map(ratio, index, np.float32, np.nan),
        "luminance_drop": as_map(drop, index, np.float32, np.nan),
        "attenuation_rgb_residual": as_map(attenuation_error, index, np.float32, np.nan),
        "motion_witness_raw": np.array(witness_raw, np.int16),
        "motion_witness_valid": as_map(witness_valid, index),
        "motion_witness_support": as_map(witness_support, index),
        "motion_witness_luminance_ratio": as_map(witness_ratios, index, np.float32, np.nan),
        "motion_witness_valid_count": as_map(valid_count.astype(np.uint8), index),
        "motion_witness_support_count": as_map(support_count.astype(np.uint8), index),
        "reason_bits": as_map(reasons, index, np.uint16, REASONS["outside_candidate_domain"]),
        "source_shadow_state_unknown": np.array(True),
    }
    arrays["shadow_candidate"], candidate_components = components(arrays["shadow_candidate"], arrays["shadow_score"], arrays["world_point"])
    # 与body保持互斥；实际输出只做删减，没有膨胀/补齐阴影标签。
    assert not (arrays["shadow_candidate"] & query["body"]).any()
    assert not arrays["road_counterevidence_certified"].any()
    np.savez_compressed(output / ("%03d.npz" % raw), **arrays)
    overlays(output, raw, query, arrays)
    gallery = source_gallery(output, raw, query, arrays, clip)
    yy, xx = np.where(arrays["candidate_domain"])
    report = {
        "raw": raw, "plane": plane_report, "body_pixels": int(query["body"].sum()),
        "original_png_vs_existing_dataset_rgb_max_abs_difference": query["body_rgb_max_difference"],
        "candidate_domain_pixels": len(index),
        "candidate_domain_bbox_xyxy_inclusive": [int(xx.min()), int(yy.min()), int(xx.max()), int(yy.max())] if len(xx) else None,
        "query_real_road_surface_valid_pixels": int(query_visible.sum()),
        "same_world_rgb_evidence_valid_pixels": int(evidence_valid.sum()),
        "query_photometric_darkening_pixels": int((evidence_valid & dark).sum()),
        "target_motion_supported_pixels_before_component_filter": int(candidate.sum()),
        "shadow_candidate_pixels": int(arrays["shadow_candidate"].sum()),
        "weak_road_counterevidence_pixels": int(counter_valid.sum()),
        "certified_clean_background_pixels": 0,
        "source_shadow_state_unknown": True, "candidate_components": candidate_components,
        "exposure_anchor_count": len(anchors), "exposure_sources": exposure,
        "motion_witnesses": witness_report,
        "reason_pixel_counts_in_domain": {name: int(((reasons & bit) != 0).sum()) for name, bit in REASONS.items()},
        "source_gallery": gallery, "human_verdict": None,
        "wall_seconds": time.monotonic() - started,
    }
    (output / ("%03d.json" % raw)).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


PROPAGATION = {
    "schema": "raw-observation-shadow-evidence-v3-proposal",
    "strict_input_preserved": True,
    "maximum_seed_geodesic_m": 1.5,
    "maximum_neighbor_world_step_m": 0.8,
    "maximum_neighbor_query_mean_abs_rgb_delta": 0.045,
    "maximum_neighbor_reference_mean_abs_rgb_delta": 0.045,
    "maximum_neighbor_luminance_ratio_delta": 0.12,
    "foreground_actual_lidar_pixel_radius": 2.0,
    "foreground_depth_margin_m": 0.65,
    "propagated_score_cap": 0.49,
    "photometric_thresholds": "unchanged from strict r2; every propagated pixel needs its own original same-world RGB evidence",
    "hard_stops": ["weak_road_counterevidence", "body", "other_actor", "nonroad_or_low_confidence", "actual_foreground_depth", "no_independent_raw_photo_darkening"],
    "zero_seed_frames": "no propagation; retain photo_supported_unknown",
    "state_meaning": {"-1": "unknown/missing evidence, including photo-supported but unconfirmed",
                      "0": "weak same-world RGB counterevidence only; not certified nonshadow",
                      "1": "positive candidate evidence; not human GT"},
}


def raw_photo_gate(arrays):
    reference = (arrays["source_count"] >= 3) & (arrays["source_span_raw"] >= 5)
    reference &= arrays["reference_rgb_deviation"] <= METHOD["reference_max_median_rgb_deviation"]
    photo = (arrays["luminance_ratio"] <= METHOD["query_max_luminance_ratio"])
    photo &= arrays["luminance_ratio"] >= 0.25
    photo &= arrays["luminance_drop"] >= METHOD["query_min_luminance_drop"]
    photo &= arrays["attenuation_rgb_residual"] <= METHOD["attenuation_max_rgb_residual"]
    return reference & photo


def constrained_surface_growth(seed, eligible, world, rgb, reference, ratio):
    """从strong seed沿真实光度支持点传播；不跨缺证、反证或颜色边缘。"""
    distance = np.full((H, W), np.inf, np.float32)
    owner = np.full((H, W), -1, np.int32)
    queue = []
    for y, x in zip(*np.where(seed)):
        distance[y, x] = 0
        owner[y, x] = y * W + x
        heapq.heappush(queue, (0.0, int(y), int(x)))
    while queue:
        dist, y, x = heapq.heappop(queue)
        if dist > distance[y, x] + 1e-6:
            continue
        for dy, dx in [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]:
            yy, xx = y + dy, x + dx
            if yy < 0 or yy >= H or xx < 0 or xx >= W or not eligible[yy, xx]:
                continue
            # 斜向步不得从两个硬阻断角之间挤过。
            if dy and dx and (not eligible[y, xx] or not eligible[yy, x]):
                continue
            step = float(np.linalg.norm(world[yy, xx] - world[y, x]))
            if step > PROPAGATION["maximum_neighbor_world_step_m"] or not np.isfinite(step):
                continue
            if np.mean(np.abs(rgb[yy, xx] - rgb[y, x])) > PROPAGATION["maximum_neighbor_query_mean_abs_rgb_delta"]:
                continue
            if np.mean(np.abs(reference[yy, xx] - reference[y, x])) > PROPAGATION["maximum_neighbor_reference_mean_abs_rgb_delta"]:
                continue
            if abs(float(ratio[yy, xx] - ratio[y, x])) > PROPAGATION["maximum_neighbor_luminance_ratio_delta"]:
                continue
            proposed = dist + step
            if proposed <= PROPAGATION["maximum_seed_geodesic_m"] and proposed < distance[yy, xx]:
                distance[yy, xx] = proposed
                owner[yy, xx] = owner[y, x]
                heapq.heappush(queue, (proposed, yy, xx))
    return np.isfinite(distance), distance, owner


def propagation_overlay(output, raw, query, arrays):
    rgb = np.round(query["rgb"] * 255).astype(np.uint8)
    panels = Image.new("RGB", (W * 4, H + 42), "white")
    images = [rgb.copy().astype(float) for _ in range(4)]
    strict, propagated = arrays["shadow_candidate_strict"], arrays["shadow_propagated"]
    unknown = arrays["photo_supported_unknown"]
    for i in [1, 2]:
        images[i][strict] = images[i][strict] * 0.4 + np.array([205, 60, 245]) * 0.6
    images[2][propagated] = images[2][propagated] * 0.45 + np.array([225, 120, 245]) * 0.55
    images[3][unknown] = images[3][unknown] * 0.45 + np.array([245, 170, 55]) * 0.55
    for i in [1, 2, 3]:
        images[i][outline(query["body"])] = [0, 235, 255]
    texts = ["raw %03d original RGB" % raw, "strict candidate (purple)",
             "proposal: strict + propagated (light purple)", "amber: photo-supported unknown; not nonshadow"]
    for i, image in enumerate(images):
        panels.paste(Image.fromarray(np.clip(image, 0, 255).astype(np.uint8)), (i * W, 42))
        ImageDraw.Draw(panels).text((i * W + 5, 8), texts[i], fill="black")
    panels.save(output / ("%03d_propagation_overlay.png" % raw))


def zero_frame_analysis(strict_input):
    summary = json.loads((strict_input / "summary.json").read_text())
    rows = []
    for report in summary["queries"]:
        if report["shadow_candidate_pixels"]:
            continue
        raw = report["raw"]
        with np.load(strict_input / ("%03d.npz" % raw)) as saved:
            arrays = {k: saved[k] for k in ["shadow_evidence_valid", "source_count", "source_span_raw", "reference_rgb_deviation",
                                           "luminance_ratio", "luminance_drop", "attenuation_rgb_residual", "motion_witness_valid_count", "motion_witness_support_count"]}
        photo = raw_photo_gate(arrays) & arrays["shadow_evidence_valid"]
        missing = photo & (arrays["motion_witness_valid_count"] < 2)
        not_repeated = photo & (arrays["motion_witness_valid_count"] >= 2) & (arrays["motion_witness_support_count"] < 2)
        small = photo & (arrays["motion_witness_support_count"] >= 2)
        counts = {"motion_witness_missing": int(missing.sum()), "motion_darkening_not_repeated": int(not_repeated.sum()),
                  "component_smaller_than_6_pixels": int(small.sum())}
        rows.append({"raw": raw, "strict_candidate_pixels": 0, "photo_supported_unknown_pixels": int(photo.sum()),
                     "unknown_reason_counts": counts, "dominant_reason": max(counts, key=counts.get),
                     "verdict": "abstain/unknown; zero strict candidates is not no-shadow",
                     "human_GT": None})
    return {"schema": "zero-strict-shadow-frame-abstention-audit-v1", "zero_strict_frames": len(rows),
            "photo_supported_unknown_pixels": sum(r["photo_supported_unknown_pixels"] for r in rows),
            "reason_totals": {name: sum(r["unknown_reason_counts"][name] for r in rows) for name in
                              ["motion_witness_missing", "motion_darkening_not_repeated", "component_smaller_than_6_pixels"]},
            "frames": rows, "meaning": "missing motion evidence is unknown, never a nonshadow label"}


def propagate_query(clip, raw, strict_input, output):
    started = time.monotonic()
    query = clip.get(raw)
    strict_report = json.loads((strict_input / ("%03d.json" % raw)).read_text())
    with np.load(strict_input / ("%03d.npz" % raw)) as saved:
        arrays = {k: saved[k].copy() for k in saved.files}
    strict = arrays["shadow_candidate"].copy()
    strict_score = arrays["shadow_score"].copy()
    strict_valid = arrays["shadow_evidence_valid"].copy()
    world = arrays["world_point"]
    finite = np.isfinite(world).all(2)
    indices = np.flatnonzero(finite.ravel())
    point = world.reshape(-1, 3)[indices]
    _, uv, detail = visibility(point, query)
    _, expected_depth = project(point, query)
    observed_depth = sample(query["all_lidar_depth"], uv, 0)
    observed_distance = sample(query["all_lidar_distance"], uv, 0)
    foreground = (observed_distance <= PROPAGATION["foreground_actual_lidar_pixel_radius"]) & (observed_depth < expected_depth - PROPAGATION["foreground_depth_margin_m"])
    foreground_map = as_map(foreground, indices)
    road_visible_assumption = detail["inside"] & detail["road"] & detail["actor_clear"] & detail["clear_mask"] & ~foreground
    road_visible_assumption = as_map(road_visible_assumption, indices)
    photo_eligible = raw_photo_gate(arrays) & finite & arrays["candidate_domain"] & road_visible_assumption
    photo_eligible &= ~arrays["road_counterevidence_valid"] & ~arrays["body"]
    if not strict_report["plane"]["valid"]:
        photo_eligible[:] = False
    active_seed = strict & photo_eligible
    reached, geodesic, owner = constrained_surface_growth(active_seed, photo_eligible, world, query["rgb"], arrays["reference_rgb_exposure_matched"], arrays["luminance_ratio"])
    propagated = reached & ~strict
    proposal = strict | propagated
    source_conf = np.clip(arrays["source_count"] / 5, 0, 1)
    photo_strength = np.clip((1 - arrays["luminance_ratio"] - 0.10) / 0.22, 0, 1)
    photo_strength *= np.clip(arrays["luminance_drop"] / 0.07, 0, 1)
    propagated_score = np.zeros((H, W), np.float32)
    yy, xx = np.where(propagated)
    if len(xx):
        seed_scores = strict_score.ravel()[owner[yy, xx]]
        propagated_score[yy, xx] = np.minimum(PROPAGATION["propagated_score_cap"], seed_scores * 0.65 * photo_strength[yy, xx] * source_conf[yy, xx] * np.exp(-geodesic[yy, xx] / 2))
    proposal_score = np.where(strict, strict_score, propagated_score).astype(np.float32)
    photo_unknown = photo_eligible & ~strict
    motion_missing = photo_unknown & (arrays["motion_witness_valid_count"] < 2)
    motion_not_repeated = photo_unknown & (arrays["motion_witness_valid_count"] >= 2) & (arrays["motion_witness_support_count"] < 2)
    small_components = photo_unknown & (arrays["motion_witness_support_count"] >= 2)
    lidar_supported = as_map(detail["lidar"], indices)
    strict_state = np.full((H, W), -1, np.int8)
    strict_state[arrays["road_counterevidence_valid"]] = 0
    strict_state[strict] = 1
    proposal_state = strict_state.copy()
    proposal_state[propagated] = 1
    arrays.update({
        "shadow_candidate_strict": strict, "shadow_score_strict": strict_score,
        "shadow_evidence_valid_strict": strict_valid,
        "shadow_candidate_proposal": proposal, "shadow_score_proposal": proposal_score,
        "shadow_candidate": proposal, "shadow_score": proposal_score,
        "shadow_evidence_valid": strict_valid | propagated,
        "shadow_propagated": propagated, "shadow_propagated_score": propagated_score,
        "propagation_eligible": photo_eligible, "propagation_seed_geodesic_m": geodesic,
        "propagation_seed_pixel_index": owner,
        "photo_supported_unknown": photo_unknown,
        "photo_supported_unknown_motion_missing": motion_missing,
        "photo_supported_unknown_motion_not_repeated": motion_not_repeated,
        "photo_supported_unknown_small_component": small_components,
        "photo_supported_unknown_unattached": photo_unknown & ~propagated,
        "query_surface_lidar_supported": lidar_supported,
        "query_actual_foreground_depth_block": foreground_map,
        "strict_shadow_evidence_state": strict_state,
        "proposal_shadow_evidence_state": proposal_state,
    })
    assert not (propagated & arrays["road_counterevidence_valid"]).any()
    assert not (propagated & arrays["body"]).any()
    assert not (propagated & foreground_map).any()
    np.savez_compressed(output / ("%03d.npz" % raw), **arrays)
    propagation_overlay(output, raw, query, arrays)
    report = dict(strict_report)
    report.update({"schema": PROPAGATION["schema"], "strict_input": str(strict_input / ("%03d.npz" % raw)),
                   "strict_candidate_pixels": int(strict.sum()), "shadow_candidate_pixels": int(proposal.sum()),
                   "strict_seeds_blocked_by_extra_propagation_safeguards": int((strict & ~photo_eligible).sum()),
                   "propagated_added_pixels": int(propagated.sum()),
                   "propagated_without_query_lidar_support_pixels": int((propagated & ~lidar_supported).sum()),
                   "propagated_without_two_motion_support_pixels": int((propagated & (arrays["motion_witness_support_count"] < 2)).sum()),
                   "propagated_overlap_weak_bg_counterevidence_pixels": int((propagated & arrays["road_counterevidence_valid"]).sum()),
                   "propagated_overlap_body_pixels": int((propagated & arrays["body"]).sum()),
                   "propagated_overlap_actual_foreground_depth_pixels": int((propagated & foreground_map).sum()),
                   "photo_supported_unknown_pixels": int(photo_unknown.sum()),
                   "photo_unknown_reason_counts": {"motion_witness_missing_or_not_computed": int(motion_missing.sum()),
                                                   "motion_darkening_not_repeated": int(motion_not_repeated.sum()),
                                                   "small_strict_component": int(small_components.sum())},
                   "photo_supported_unknown_unattached_pixels": int((photo_unknown & ~propagated).sum()),
                   "propagation_wall_seconds": time.monotonic() - started,
                   "source_shadow_state_unknown": True, "certified_clean_background_pixels": 0, "human_GT": None})
    (output / ("%03d.json" % raw)).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--raw", nargs="+", default=["98", "113", "114"])
    parser.add_argument("--source-stride", type=int, default=4)
    parser.add_argument("--propagate-from", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("保留旧现场，请使用新的输出目录: " + str(args.output))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    clip = RawClip(args.root)
    raws = [int(f["raw"]) for f in clip.queries] if args.raw == ["all"] else [int(r) for r in args.raw]
    if args.propagate_from is not None:
        strict_contract = json.loads((args.propagate_from / "contract.json").read_text())
        contract = dict(strict_contract)
        contract.update({"schema": PROPAGATION["schema"], "propagation": PROPAGATION,
                         "strict_input": str(args.propagate_from), "query_raws": raws})
        (args.output / "contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
        zero = zero_frame_analysis(args.propagate_from)
        (args.output / "zero_strict_frames_analysis.json").write_text(json.dumps(zero, indent=2), encoding="utf-8")
        reports = []
        for raw in raws:
            report = propagate_query(clip, raw, args.propagate_from, args.output)
            reports.append(report)
            print("PROPAGATION_QUERY", json.dumps({k: report[k] for k in ["raw", "strict_candidate_pixels", "propagated_added_pixels", "shadow_candidate_pixels", "photo_supported_unknown_pixels", "photo_supported_unknown_unattached_pixels"]}), flush=True)
        summary = {"schema": PROPAGATION["schema"], "context": CTX, "queries": reports,
                   "source_raws": strict_contract["source_raws"], "strict_input": str(args.propagate_from),
                   "wall_seconds": time.monotonic() - started, "training_updates": 0, "GPU_calls": 0,
                   "source_shadow_state_unknown": True, "certified_clean_background_pixels": 0, "human_GT": None}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return
    source_raws = sorted(clip.frames)[::args.source_stride]
    method = dict(METHOD)
    method["source_stride"] = args.source_stride
    method["source_raws"] = source_raws
    method["query_raws"] = raws
    method["root"] = str(args.root)
    method["reason_bits"] = REASONS
    (args.output / "contract.json").write_text(json.dumps(method, indent=2), encoding="utf-8")
    sources = []
    for raw in source_raws:
        sources.append(clip.get(raw))
        print("RAW_SOURCE_LOADED", raw, "road_returns", len(sources[-1]["road_world"]), flush=True)
    reports = []
    for raw in raws:
        report = query_evidence(clip, raw, sources, args.output)
        reports.append(report)
        summary = {k: report.get(k) for k in ["raw", "body_pixels", "candidate_domain_pixels", "same_world_rgb_evidence_valid_pixels", "query_photometric_darkening_pixels", "shadow_candidate_pixels", "weak_road_counterevidence_pixels", "wall_seconds"]}
        print("SHADOW_QUERY", json.dumps(summary), flush=True)
    summary = {"schema": METHOD["schema"], "context": CTX, "source_raws": source_raws,
               "queries": reports, "wall_seconds": time.monotonic() - started,
               "training_updates": 0, "GPU_calls": 0,
               "source_shadow_state_unknown": True, "certified_clean_background_pixels": 0,
               "human_GT": None, "failure_ledger_delta": "none"}
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("SHADOW_DONE", json.dumps({k: v for k, v in summary.items() if k not in ["queries", "source_raws"]}), flush=True)


if __name__ == "__main__":
    main()
