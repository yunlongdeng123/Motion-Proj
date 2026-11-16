"""根据正确三角投影重建覆盖请求，只激活已有合法实测颜色的道路节点。"""
from pathlib import Path
import argparse
import json
import shutil
import numpy as np
from scipy.ndimage import distance_transform_edt


def load(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ["asset","projection","output"]:
        parser.add_argument("--"+key,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    road=load(args.asset/"road.npz");p=load(args.asset/"provenance.npz");atlas=load(args.asset/"atlas.npz")
    n=len(road["active"]);request=p["requested"].copy();new_request=request.copy();rows=[]
    for raw in [86,110,134,179,198]:
        view=load(args.projection/f"{raw:03d}.npz")
        mask=view["frozen_evaluation_domain"]&view["surface_intersection_valid"]&~view["K"]
        triangles=view["triangle_world_rows"][mask].reshape(-1)
        corners=view["source_corner_world_rows"][mask].reshape(-1)
        ids=np.unique(np.r_[triangles,corners]);ids=ids[ids>=0]
        new_request[ids]=True
        rows.append(dict(raw=raw,corrected_ray_pixels=int(mask.sum()),referenced_world_rows=len(ids),
            outside_old_requested_rows=int((~request[ids]).sum())))
    guard=p.get("guarded_world_rows",np.zeros(n,bool))
    added=new_request&~road["active"]&p["known"]&p["surface_supported"]&~guard&~p["protected_geometry"]
    # 物理表面边界决定新高斯足迹；生成W不决定已观测道路是否存在。
    grid=atlas["grid_world_row"]
    support=p["surface_supported"][grid]
    edge=distance_transform_edt(np.pad(support,1),sampling=.02)[1:-1,1:-1]
    distance=np.zeros(n);distance[grid.ravel()]=edge.ravel()
    sigma=np.minimum(.013,.30*distance)
    added&=sigma>.001
    road["active"][added]=True
    road["gaussian_sigma_m"][added,:2]=sigma[added,None]
    road["reference_srgb"][added]=p["observed_reference_srgb"][added]
    if not np.array_equal(road["reference_srgb"][added],p["row_rgb"][added]):
        raise RuntimeError("新增实测道路颜色没有来自同一固定F")
    p["old_requested"]=request
    p["coverage_requested"]=new_request
    p["observed_surface_activated"]=added
    p["coverage_trusted"]=p["trusted_geometry"]&~added
    p["coverage_added"]|=added
    p["material_updated"][added]=True
    p["preserved_legacy_rgb"][added]=False
    p["effective_row_rgb"]=road["reference_srgb"].copy()
    p["effective_active"]=road["active"].copy()
    p["effective_sigma_m"]=road["gaussian_sigma_m"].copy()
    p["coverage_boundary_distance_m"]=distance.astype(np.float32)
    np.savez_compressed(args.output/"road.npz",**road)
    np.savez_compressed(args.output/"provenance.npz",**p)
    np.savez_compressed(args.output/"coverage_request_diff.npz",old_requested=request,
        coverage_requested=new_request,new_observed_active=added,source_type=p["source"],
        physical_surface_boundary_distance_m=distance.astype(np.float32))
    for source in args.asset.iterdir():
        if source.is_file() and source.name not in ["road.npz","provenance.npz","receipt.json"]:
            shutil.copy2(source,args.output/source.name)
    receipt=json.loads((args.asset/"receipt.json").read_text())
    receipt.update(output_road=str(args.output/"road.npz"),active_nodes=int(road["active"].sum()),
        coverage_request_repair=dict(source_projection=str(args.projection),
            old_requested_rows=int(request.sum()),new_coverage_requested_rows=int(new_request.sum()),
            outside_old_request_rows=int((new_request&~request).sum()),
            newly_activated_observed_surface_rows=int(added.sum()),original_generated_W_changed=False,
            existing_active_geometry_or_RGB_updates=0,new_surface_colors="exact fixed observed F at each world point",
            geometry_scope="correct nearest positive triangle ray hits in the same frozen five-frame evaluation domain",
            footprint="sigma<=.013m and 3sigma<=physical supported-road boundary; not eroded by image generation W",
            rows=rows),runtime_material_write_domain="old requested; extra new known active consumes immutable observed F only; generated W unchanged")
    (args.output/"receipt.json").write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(receipt["coverage_request_repair"],ensure_ascii=False),flush=True)


if __name__=="__main__":main()
