"""One-frame joint-T conditional geometry of the raw105 dark carrier.

24-feature accumulation only; RGB/alpha must match the unmodified raster.
Road tangent-plane height is geometric evidence, not actor identity or history.
"""
import os
for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree

CAM = Path("/root/autodl-tmp/drivingeditor_repro/campaign")
GROUPS = ("asset", "a37", "road", "legacy_static")


def stats(a):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    if not len(a): return dict(count=0)
    q = np.quantile(a, [0, .5, .9, .99, 1])
    return dict(count=len(a), min=q[0], median=q[1], p90=q[2], p99=q[3], max=q[4], mean=float(a.mean()))


def main():
    p = argparse.ArgumentParser(); p.add_argument("--output", required=True); args = p.parse_args()
    sys.path.insert(0, str(CAM/"scene008_a6_asset_controls_20261105/source"))
    import torch
    from scene_runtime import Runtime, raster, select_rows
    rt = Runtime(); d = rt.raw(105); c = d["cam"]; start = time.time()
    folder = Path(args.output); folder.mkdir(parents=True, exist_ok=False)
    ref = rt.exp.ref; inverse = np.linalg.inv(np.asarray(rt.exp.meta["world_to_model"])@np.asarray(rt.exp.meta["coordinate_anchor"]))
    road_normals = ref.get("normal_local")
    if road_normals is None:
        road_normals = ref["normal_world"]@np.asarray(rt.exp.meta["coordinate_anchor"])[:3, :3]
    road_normals = road_normals/np.maximum(np.linalg.norm(road_normals, axis=1, keepdims=True), 1e-12)
    tree = cKDTree(ref["local_xyz"][:, :2]); states = {}
    with torch.no_grad():
        for delete in (True, False):
            scene = rt.scene(105, delete=delete, baseline=True)
            if delete:
                scene = dict(scene, rows=select_rows(scene["rows"], scene["rows"]["features"][:, 3] < .5))
            rows = scene["rows"]; base = raster(c, rt.pipe, rt.n.static.active_sh_degree, rows, scene["bg"])
            feature = rows["features"].new_zeros((len(rows["features"]), 24)); feature[:, :6] = rows["features"][:, :6]
            for g in range(4):
                own = rows["features"][:, 2+g]
                feature[:, 6+g] = rows["features"][:, 0]*own
                feature[:, 12+g] = rows["features"][:, 0].square()*own
            static = rows["features"][:, 5] > .5
            xyz = rows["means3D"][static].cpu().numpy().astype(float)
            local = xyz@inverse[:3, :3].T+inverse[:3, 3]
            distance, index = tree.query(local[:, :2], k=1)
            height = ((local-ref["local_xyz"][index])*road_normals[index]).sum(1)
            valid = (distance <= .35) & np.isfinite(height)
            h = torch.as_tensor(np.where(valid, height, 0), device="cuda", dtype=feature.dtype)
            ok = torch.as_tensor(valid, device="cuda", dtype=feature.dtype)
            feature[static, 10] = h
            feature[static, 11] = ok
            feature[static, 16] = h.square()
            marked = raster(c, rt.pipe, rt.n.static.active_sh_degree, dict(rows, features=feature), scene["bg"])
            rgb_error = float((marked["rgb"]-base["rgb"]).abs().max()); alpha_error = float((marked["alpha"]-base["alpha"]).abs().max())
            assert rgb_error == 0 and alpha_error == 0
            assert torch.equal(marked["depth_sum"], base["depth_sum"])
            ex = marked["extra"].cpu().numpy(); rgb = (base["rgb"]+(1-base["alpha"])*d["env"]).cpu().numpy()
            states["delete" if delete else "keep"] = dict(groups=ex[:4], depth_sum=ex[4:8]/rt.scale,
                depth2_sum=ex[10:14]/rt.scale**2, height_sum=ex[8], height_valid_mass=ex[9],
                height2_sum=ex[14], rgb=rgb, RGB_feature_instrumentation_maxabs=rgb_error,
                alpha_feature_instrumentation_maxabs=alpha_error,
                active_a37=int(((rows["features"][:, 3] > .5)&rows["mask"]).sum()))
        box = [239, 256, 294, 291]; x0, y0, x1, y1 = box
        roi = np.zeros((320, 480), bool); roi[y0:y1, x0:x1] = True
        lum = np.einsum("c,chw->hw", [.2126, .7152, .0722], states["delete"]["rgb"])
        masks = dict(fixed_ROI=roi, same_dark_core=roi&(lum < .25)); result = {}
        for name, mask in masks.items():
            r = dict(pixels=int(mask.sum()), states={})
            for state, data in states.items():
                groups = {}
                for g, group in enumerate(GROUPS):
                    mass = data["groups"][g]; valid = mask&(mass > 1e-4)
                    z = data["depth_sum"][g]/np.maximum(mass, 1e-12)
                    variance = np.maximum(data["depth2_sum"][g]/np.maximum(mass, 1e-12)-z*z, 0)
                    groups[group] = dict(joint_alphaT_mass=float(mass[mask].sum()),
                        pooled_conditional_camera_z_m=float(data["depth_sum"][g][mask].sum()/mass[mask].sum()) if mass[mask].sum()>1e-8 else None,
                        per_pixel_conditional_z_m=stats(z[valid]), per_pixel_conditional_std_z_m=stats(np.sqrt(variance[valid])))
                hm = data["height_valid_mass"]; good = mask&(hm > 1e-4)
                r["states"][state] = dict(groups=groups,
                    static_road_plane_valid_mass=float(hm[mask].sum()),
                    static_height_above_local_road_plane_m=stats((data["height_sum"]/np.maximum(hm, 1e-12))[good]),
                    pooled_static_height_above_road_m=float(data["height_sum"][mask].sum()/hm[mask].sum()) if hm[mask].sum()>1e-8 else None)
            original = d["obs"]; measured = mask & original["depth_valid"][1].astype(bool)
            r["same_frame_a37_associated_LiDAR_camera_z_m"] = stats(original["depth_m"][1][measured])
            result[name] = r
        assert states["delete"]["active_a37"] == 0 and rt.modelunchanged()
        report = dict(status="PASS_SAME_SORT_CONDITIONAL_GEOMETRY", raw=105, ROI_xyxy=box, fixed_domains=result,
            feature_count=24, RGB_and_alpha_bitwise_unchanged_by_feature_instrumentation=True,
            depth_sum_bitwise_unchanged_by_feature_instrumentation=True,
            delete_a37_active_rows=0, parent_parameters_unchanged=True, model_updates=0, seconds=time.time()-start,
            height_reference="Nearest existing measured local road tangent plane within .35m XY; signed centre height along road normal",
            limitation="Joint-T centre-depth/height evidence does not establish actor identity, physical shadow semantics, or historical Gaussian birth")
        (folder/"conditional_geometry.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")
        np.savez_compressed(folder/"accumulated_features.npz", **{state+"_"+key:value for state,data in states.items() for key,value in data.items() if isinstance(value,np.ndarray)}, ROI=roi, dark_core=masks["same_dark_core"])
        print(json.dumps(report), flush=True)


if __name__ == "__main__": main()
