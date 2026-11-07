"""CPU full-image ray/ground-pool support audit, independent of the old grid.

The fixed ROI is read only after the full-frame geometry decision, for reporting.
No RGB is opened and no unknown geometry is labelled never-visible.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[key] = "1"
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import cKDTree


def count(x): return int(np.count_nonzero(x))


class Surface:
    def __init__(self, points, plane):
        self.points = points.astype(float)
        self.tree = cKDTree(points[:, :2])
        self.plane = -plane[[0, 1, 3]] / plane[2]

    def height(self, xy):
        d, ids = self.tree.query(xy, k=8, distance_upper_bound=.8)
        finite = np.isfinite(d)
        points = self.points[np.minimum(ids, len(self.points)-1)]
        residual = points[:, :, 2] - points[:, :, :2] @ self.plane[:2] - self.plane[2]
        weights = np.where(finite, 1 / np.maximum(d, .05)**2, 0)
        height = xy @ self.plane[:2] + self.plane[2] + (weights*residual).sum(1) / np.maximum(weights.sum(1), 1e-12)
        return height, finite.sum(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=Path("/root/autodl-tmp/drivingeditor_repro/campaign"))
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--raw", type=int, default=137)
    args = p.parse_args()
    started = time.time()
    base, run, raw = args.base, args.run, args.raw
    out = run / "observations"
    output = out / f"{raw:03d}_ray_geometry_frontfill_v2.json"
    if output.exists(): raise FileExistsError("Preserve earlier geometry audit")
    out.mkdir(exist_ok=True, parents=True)
    bundle = base / "r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json"
    meta = json.loads(bundle.read_text())
    plane = np.asarray(json.loads(Path(meta["original_readiness"]).read_text())["SGCM_geometry"]["ground_fit"]["plane_anchor"])
    with np.load(run / "surface/measured_ground_support_pool.npz", allow_pickle=False) as a:
        surface = Surface(a["local_xyz"], plane)
    with np.load(base / f"r15_gn11_fullclip_cpu_20261026/frames/{raw:03d}.npz", allow_pickle=False) as a:
        K, w2l = a["K"], a["w2l"]
    inverse = np.linalg.inv(w2l)
    yy, xx = np.mgrid[:320, :480]
    rays = np.stack([xx+.5, yy+.5, np.ones_like(xx)], -1).reshape(-1,3) @ np.linalg.inv(K).T @ inverse[:3,:3].T
    origin = inverse[:3,3]
    denominator = rays[:,2] - rays[:,:2] @ surface.plane[:2]
    numerator = origin[:2] @ surface.plane[:2] + surface.plane[2] - origin[2]
    depth = np.divide(numerator, denominator, out=np.full(len(rays), 1e5), where=np.abs(denominator)>1e-7)
    first = origin + rays * depth[:,None]
    potential = ((depth>.1)&(depth<120)&(first[:,0]>=.5)&(first[:,0]<=90)&(np.abs(first[:,1])<=8))
    selected = np.flatnonzero(potential)
    # Fixed iteration count and gates for the entire frame; no ROI adjustment.
    for _ in range(8):
        point = origin + rays[selected] * depth[selected,None]
        height, support = surface.height(point[:,:2])
        delta = np.divide(height-point[:,2], denominator[selected], out=np.zeros(len(selected)),
                          where=(np.abs(denominator[selected])>1e-7)&(support>=3))
        depth[selected] += np.clip(delta,-10,10)
    local = origin+rays*depth[:,None]
    height, support = surface.height(local[selected,:2])
    valid = np.zeros(len(rays),bool)
    valid[selected] = ((support>=3)&(np.abs(height-local[selected,2])<.03)&(depth[selected]>.1)&(depth[selected]<120))
    support_full = np.zeros(len(rays),np.uint8)
    support_full[selected] = support
    with np.load(run/"surface/frontfill_v2/road_reference_199.npz", allow_pickle=False) as a:
        cell, lo, shape, grid = float(a["cell_m"]), a["grid_keys_min"], a["grid_shape_xy"], a["grid_index"]
        keys = np.floor(local[:,:2]/cell).astype(np.int64)-lo
        inside = (keys>=0).all(1)&(keys<shape).all(1)
        safe = np.clip(keys,[0,0],shape-1)
        candidate_id = grid[safe[:,0],safe[:,1]]
        candidate_id[~inside] = -1
        candidate = valid&(candidate_id>=0)
        active = candidate & a["active"][np.maximum(candidate_id,0)]
        candidate_geometric_samples = a["source_valid_count"][np.maximum(candidate_id,0)]
        candidate_clean_samples = a["agree_count"][np.maximum(candidate_id,0)]
    # The old ROI is a diagnostic denominator only, after the fixed full-frame rule.
    old = base / "scene008_road_prior_assessment_20261105/coverage"
    with np.load(old/f"{raw:03d}_world_maps.npz", allow_pickle=False) as a:
        old_valid = a["geometry_valid"].ravel().astype(bool)
    with np.load(old/f"{raw:03d}_masks.npz", allow_pickle=False) as a:
        masks = {k:a[k].ravel().astype(bool) for k in ("fixed_roi","under_tail_band")}
    masks["full_image"] = np.ones(len(rays),bool)
    groups = {}
    for name, mask in masks.items():
        unknown = mask&~old_valid
        groups[name] = dict(pixels=count(mask), old_geometry_valid=count(mask&old_valid),
            old_geometry_unknown=count(unknown), new_pooled_geometry_valid=count(mask&valid),
            old_unknown_now_pool_supported=count(unknown&valid), old_unknown_still_geometry_unknown=count(unknown&~valid),
            new_geometry_in_expanded_candidate=count(mask&candidate), new_geometry_in_active_candidate=count(mask&active),
            old_unknown_in_expanded_candidate=count(unknown&candidate), old_unknown_in_active_candidate=count(unknown&active),
            old_unknown_with_any_cell_rgb_witness=count(unknown&candidate&(candidate_geometric_samples>0)),
            old_unknown_with_ge3_cell_clean_witness=count(unknown&candidate&(candidate_clean_samples>=3)),
            local_bounds_supported=[local[mask&valid].min(0).tolist(),local[mask&valid].max(0).tolist()] if (mask&valid).any() else None,
            old_unknown_supported_bounds=[local[unknown&valid].min(0).tolist(),local[unknown&valid].max(0).tolist()] if (unknown&valid).any() else None)
    shape2=(320,480)
    np.savez_compressed(out/f"{raw:03d}_ray_geometry_frontfill_v2.npz",local_xyz=local.reshape(320,480,3).astype(np.float32),
        depth_m=depth.reshape(shape2).astype(np.float32),geometry_valid=valid.reshape(shape2),
        support_count=support_full.reshape(shape2),old_geometry_valid=old_valid.reshape(shape2),
        candidate_id=candidate_id.reshape(shape2),candidate= candidate.reshape(shape2),active=active.reshape(shape2))
    report=dict(schema="R0_FULLFRAME_GROUND_POOL_RAY_SUPPORT_V1",raw=raw,CPU_only=True,RGB_images_opened=0,
        rule="Entire480x320: anchor ground-plane start,8 measured-pool IDW updates; >=3of8 actual deduplicated returns within0.8m, final height residual<0.03m; t(0.1,120), initial x[0.5,90],y+-8. No old-grid containment required.",
        groups=groups,geometry_origin="Measured-support interpolated ray/surface intersection, not dense LiDAR.",
        limits=["Old unknown becoming pool-supported is geometry evidence only; cell RGB witness does not prove exact ray-point RGB.",
                "Sparse failures remain geometry unknown; no never-visible conclusion.",
                "Only the fixed full-image geometry rule determined support; ROI is reporting only."],
        seconds=time.time()-started)
    output.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(report),flush=True)


if __name__=="__main__": main()
