"""Read existing frozen complete Gaussian exports; never load/edit a model.

The output is a geometric point/ellipsoid diagnostic, NOT f_geom, a rendered
coverage measure, physical ownership, or a formal V43 eligibility decision.
Both static and dynamic source branches are included before any target crop.
"""
from pathlib import Path
from hashlib import sha256
import argparse
import base64
import json
import math
import zlib
import numpy as np
from scipy.spatial import cKDTree
from PIL import Image, ImageDraw, ImageFont

BASE = Path(__file__).resolve().parents[3]
GPU = BASE / "outputs/V43_R4_GPU_20261031"
SHARED = Path("C:/Users/dengyunlong/Documents/Codex/2026-10-22/ni-x-a/outputs/GS_Scene_Diagnostics_20261027")
CONTEXTS = ["14705303724557273004_3105_000_3125_000", "4295449061847708198_3769_000_3789_000", "473735159277431842_630_095_650_095"]

def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))

def assignment(path):
    return json.JSONDecoder().raw_decode(Path(path).read_text(encoding="utf-8-sig").split("=", 1)[1])[0]

def decode(path):
    data = assignment(path)
    blob = zlib.decompress(base64.b64decode(data["data"]))
    arrays = {}
    for key, meta in data["arrays"].items():
        arrays[key] = np.frombuffer(blob, dtype=np.dtype(meta["dtype"]), count=int(np.prod(meta["shape"])), offset=meta["offset"]).reshape(meta["shape"]).copy()
    return arrays

def load_scene(context, cache):
    folder = SHARED / context
    manifest = assignment(folder / "cloud_full_manifest.js")
    receipt = {"context_id": context, "files": [], "branches": {}, "geometry": str(folder / "edit_geometry.json"), "geometry_sha256": sha256((folder / "edit_geometry.json").read_bytes()).hexdigest()}
    scene = read(folder / "scene.json")
    geometry = read(folder / "edit_geometry.json")
    out = {}
    for branch in ("static", "dynamic"):
        parts = []
        branch_meta = manifest[branch]
        for part in branch_meta["chunks"]:
            path = folder / part["path"]
            digest = sha256(path.read_bytes()).hexdigest()
            assert digest == part["sha256"], str(path)
            arrays = decode(path)
            assert arrays["xyz"].shape[0] == part["rows"]
            assert np.isfinite(arrays["xyz"]).all()
            parts.append(arrays)
            receipt["files"].append({"path": str(path), "sha256": digest, "rows": part["rows"], "branch": branch})
        keys = parts[0].keys()
        arrays = {key: np.concatenate([part[key] for part in parts], axis=0) for key in keys}
        assert len(arrays["xyz"]) == branch_meta["total"]
        assert len(np.unique(arrays["ids"])) == len(arrays["ids"])
        arrays["cycle"] = np.array(branch_meta["cycle"], dtype=np.float64)
        out[branch] = arrays
        receipt["branches"][branch] = {"total_original_rows": len(arrays["xyz"]), "cycle_model_units": branch_meta["cycle"], "all_source_rows_loaded": True}
        cache.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache / (context + "_" + branch + ".npz"), **arrays)
    with (cache / (context + "_geometry.json")).open("w", encoding="utf-8") as stream:
        json.dump({"scene": scene, "geometry": geometry, "receipt": receipt}, stream, ensure_ascii=False)
    return out, scene, geometry, receipt

def at_time(data, model_time):
    # Same exported SHM/time opacity formula as the established native geometry
    # viewer. This is a center diagnostic; it does not execute splat compositing.
    dt = model_time - data["t"].reshape(-1)
    sigma = data["sigma_t"].reshape(-1)
    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
        marginal = np.exp(-.5 * dt * dt / (sigma * sigma))
    marginal = np.where(np.isfinite(marginal), marginal, 0.)
    opacity = data["opacity"].reshape(-1) * marginal
    omega = 2 * math.pi / float(data["cycle"])
    xyz = data["xyz"] + data["velocity"] * (np.sin(dt * omega) / omega)[:, None]
    return xyz, opacity

def plot_comparison(path, gs, lidar, dimensions, title):
    image = Image.new("RGB", (1200, 700), "#0b1422")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",18)
    small = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc",16)
    # Fixed metric axes derived from the source OBB, identical for G and LiDAR.
    for column, (a, b, label) in enumerate([(0, 1, "俯视 x/y"), (0, 2, "侧视 x/z"), (1, 2, "正视 y/z")]):
        extent = max(float(dimensions[a]), float(dimensions[b])) + 1.6
        scale = 340 / extent
        for row, (points, color, name) in enumerate([(lidar, "#6fe2d0", "框内实测点"), (gs, "#ffad66", "高斯中心（静态＋动态）")]):
            x0 = column * 400 + 200
            y0 = row * 300 + 160
            half_a, half_b = dimensions[a] * scale / 2, dimensions[b] * scale / 2
            draw.rectangle([x0-half_a,y0-half_b,x0+half_a,y0+half_b], outline="#46566b", width=1)
            for point in points:
                x, y = x0 + float(point[a]) * scale, y0 - float(point[b]) * scale
                if column * 400 + 8 < x < (column + 1) * 400 - 8 and row * 300 + 35 < y < (row + 1) * 300 - 4:
                    draw.ellipse((x-1, y-1, x+1, y+1), fill=color)
            draw.text((column*400+12, row*300+10), label+"："+name, fill="white",font=font)
    draw.text((12, 603), title, fill="white",font=font)
    draw.text((12, 632), "高斯诊断域：opacity≥0.01，框外扩0.5米；显示上限：高斯3,000 / LiDAR 5,000点。", fill="#b9cce0",font=small)
    draw.text((12, 658), "单位：米。框内测点可能含遮挡物或地面；高斯中心不是渲染覆盖或完整表面。", fill="#b9cce0",font=small)
    image.save(path)

def make_comparison(context, media_root, cache, out):
    source = read(cache / (context + "_geometry.json"))
    scene, geometry = source["scene"], source["geometry"]
    media = read(media_root / "scenes" / context / "media.json")
    assert media["context_id"] == context and media["frame_count"] == media["full_raw_frame_denominator"]
    queue = [row for row in read(GPU / "actor_preparation_queue.json")["rows"] if row["context_id"] == context]
    assert len(queue) == 3
    clouds = {branch: dict(np.load(cache / (context+"_"+branch+".npz"))) for branch in ("static", "dynamic")}
    local_from_world = np.array(geometry["local_from_world"], dtype=np.float64)
    assert np.allclose(local_from_world[:3,:3].T @ local_from_world[:3,:3], np.eye(3), atol=1e-9)
    results, evidence = [], {}
    for row in queue:
        track = row["track_id"]
        actor = media["actors"][track]
        poses = {point["raw"]: point for point in actor["trajectory"]}
        raw_points = np.load(media_root / actor["lidar"]["npz"])
        first, last = row["recommended_window_first"], row["recommended_window_last"]
        anchors = [first, (first+last)//2, last]
        track_frames = []
        for raw in anchors:
            if raw not in poses:
                results.append({"context_id": context, "track_id": track, "raw": raw, "valid": False, "reason": "missing raw annotation pose"})
                continue
            pose = poses[raw]
            rotation_world = np.array(pose["orientation_world"]).reshape(3,3)
            center_world = np.array(pose["center_world"])
            rotation_local = local_from_world[:3,:3] @ rotation_world
            center_local = local_from_world[:3,:3] @ center_world + local_from_world[:3,3]
            dimensions = np.array(pose["dimensions"])
            model_time = scene["frames"][str(raw)]["model_time"]
            pose_parity = None
            if track == scene["target_track"]:
                native_frame = scene["frames"][str(raw)]
                center_error = float(np.max(np.abs(center_local-np.array(native_frame["target_center"]))))
                rotation_error = float(np.max(np.abs(rotation_local-np.array(native_frame["target_rotation"]))))
                assert center_error < 2e-5 and rotation_error < 2e-5, (context,raw,center_error,rotation_error)
                pose_parity = {"source": "existing original native geometry export target pose", "max_center_error_m": center_error,
                    "max_rotation_element_error": rotation_error, "passed": True}
            points, ids, branches, opacities = [], [], [], []
            branch_counts = {}
            for branch in ("static", "dynamic"):
                xyz, opacity = at_time(clouds[branch], model_time)
                distance = np.max(np.abs(xyz-center_local), axis=1)
                rough = (distance < np.linalg.norm(dimensions)/2+1.5) & (opacity >= .01)
                indices = np.flatnonzero(rough)
                actor_local = (xyz[indices]-center_local) @ rotation_local
                keep = np.all(np.abs(actor_local) <= dimensions/2+.5, axis=1)
                indices = indices[keep]
                actor_local = actor_local[keep]
                branch_counts[branch] = len(indices)
                points.append(actor_local)
                ids.append(clouds[branch]["ids"][indices])
                branches.extend([branch]*len(indices))
                opacities.append(opacity[indices])
            gs = np.concatenate(points)
            original_ids = np.concatenate(ids)
            alpha = np.concatenate(opacities)
            lidar = raw_points["actor_local_xyz"][raw_points["raw_frame"] == raw]
            assert np.isfinite(gs).all() and np.isfinite(lidar).all()
            distances = cKDTree(gs).query(lidar, workers=1)[0] if len(gs) and len(lidar) else np.array([])
            # All opacity-filtered centers in this diagnostic OBB crop enter NN;
            # only the plotted/exported center count has an extra display cap.
            keep = np.linspace(0, len(gs)-1, min(3000, len(gs)), dtype=int) if len(gs) else np.array([], dtype=int)
            lidar_keep = np.linspace(0,len(lidar)-1,min(5000,len(lidar)),dtype=int) if len(lidar) else np.array([],dtype=int)
            stem = media["scene_key"]+"_"+actor["alias"]+"_raw"+str(raw)
            path = out / (stem+".png")
            plot_comparison(path, gs[keep], lidar[lidar_keep], dimensions, media["display_name"]+" / "+actor["alias"]+" / 原始帧 "+str(raw))
            record = {"context_id": context, "track_id": track, "actor_alias": actor["alias"], "raw": raw, "valid": True,
                "model_time": model_time, "measured_lidar_points": len(lidar), "original_G_centers_in_expanded_OBB": len(gs), "branch_counts": branch_counts,
                "nearest_center_distance_median_m": float(np.median(distances)) if len(distances) else None,
                "nearest_center_distance_p90_m": float(np.quantile(distances,.9)) if len(distances) else None,
                "numeric_denominator": len(distances), "formal_f_geom": None, "formal_gate": None,
                "raw_to_native_target_pose_parity": pose_parity,
                "geometry_is_surface_completeness": False, "used_edit_result": False,
                "image": "geometry/"+path.name,
                "limitations": "Measured source LiDAR inside annotation OBB can include occluders/ground. G contains both branches and nearby non-target rows. NN is distance to centers, not footprint/depth/alpha*T coverage or ownership. Opacity .01 and OBB margin .5m define this diagnostic numeric crop as well as its display, never a formal eligibility exclusion."}
            results.append(record)
            track_frames.append({**record, "xyz": np.round(gs[keep],4).tolist(), "source_branch": [branches[i] for i in keep],
                "original_row_id": original_ids[keep].astype(int).tolist(), "effective_opacity": np.round(alpha[keep],5).tolist(),
                "lidar_xyz": np.round(lidar[lidar_keep],4).tolist(), "dimensions": dimensions.tolist(),
                "coordinates": "actor_local_x_forward_y_left_z_up_box_center", "display_points": len(keep)})
        evidence[track] = {"schema": "original_G_actor_comparison_v1", "source_context": context, "source_branch_scope": ["static", "dynamic"],
            "native_window": [first,last], "frames": track_frames, "formal_f_geom": None}
    return evidence, results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=BASE / "work/actor_diagnostics_20261101/root/gs_cache")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--media-root", type=Path)
    parser.add_argument("--out", type=Path, default=BASE / "outputs/Actor_Diagnostics_20261101/geometry")
    args = parser.parse_args()
    receipts = []
    for context in CONTEXTS:
        existing = args.cache / (context + "_geometry.json")
        if args.prepare_only or not existing.exists():
            _, _, _, receipt = load_scene(context, args.cache)
            receipts.append(receipt)
            print(json.dumps({"prepared": context, "branches": receipt["branches"]}), flush=True)
    if args.prepare_only:
        (args.cache / "source_receipt.json").write_text(json.dumps(receipts, ensure_ascii=False, indent=2), encoding="utf-8")
        return
    if not args.media_root:
        raise ValueError("Exact raw-media actor poses are required; no yaw-only surrogate.")
    args.out.mkdir(parents=True, exist_ok=True)
    evidence, results = {}, []
    for context in CONTEXTS:
        if not (args.media_root / "scenes" / context / "media.json").exists():
            print(json.dumps({"waiting_for_media": context}), flush=True)
            continue
        evidence[context], records = make_comparison(context,args.media_root,args.cache,args.out)
        results.extend(records)
    payload = {"schema": "original_complete_G_source_LiDAR_comparison_v1", "rows": results, "sources": str(args.cache / "source_receipt.json"),
        "source_role": "original_none_geometry_only_no_edited_reference_or_student_output", "formal_eligibility": None,
        "used_current_dynamic_branch_only": False, "source_G_rows_loaded": sum(sum(b["total_original_rows"] for b in r["branches"].values()) for r in read(args.cache/"source_receipt.json")),
        "notes": "Neither point-to-center distances nor OBB crop occupancy prove visible surface render coverage or complete geometry."}
    (args.out/"comparison.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")
    (args.out/"gs_evidence.js").write_text("window.ACTOR_GS_EVIDENCE="+json.dumps(evidence,ensure_ascii=False,separators=(",",":"),allow_nan=False)+";\n",encoding="utf-8")
    print(json.dumps({"comparisons":len(results),"actors":sum(len(a) for a in evidence.values())}),flush=True)

if __name__ == "__main__":
    main()
