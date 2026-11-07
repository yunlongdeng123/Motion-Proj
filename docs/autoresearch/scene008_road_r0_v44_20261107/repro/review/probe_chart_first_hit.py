"""35-ray read-only first-hit probe on existing canonical active-UV triangles.

Not a closed vehicle mesh, native renderer, hidden GT, or complete MOG method.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial import Delaunay


def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def stat(v):
    return dict(count=len(v),min=float(np.min(v)) if len(v) else None,median=float(np.median(v)) if len(v) else None,max=float(np.max(v)) if len(v) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);args=p.parse_args();run=args.run;start=time.time()
    out=run/"review/asset_ground_conflict";target=out/"canonical_chart_35ray_first_hit_probe.json"
    if target.exists():raise FileExistsError("Preserve the bounded first-hit probe")
    geometry=json.loads((out/"geometry_depth_evidence_v2.json").read_text())
    selected=[r for r in geometry["pixels"] if r["raw"]==130 and r["strong_baseline_gt09"]]
    assert len(selected)==35
    path=run.parent/"scene008_a6_insert_artifacts_20261105/surface_fit/combined_v1/observed_fit/formal/complete_a6.npz"
    a=load(path);chart=a["representation_kind"]==1;uv=a["resample_reference_pixel_xy"][chart];vertices=a["xyz"][chart].astype(float)
    alpha=np.minimum(a["opacity"][chart]/np.float32(.99),1)
    triangles=Delaunay(uv).simplices
    edge=np.stack([np.linalg.norm(uv[triangles[:,i]]-uv[triangles[:,j]],axis=1) for i,j in ((0,1),(1,2),(2,0))],1)
    triangles=triangles[edge.max(1)<=1.5]
    v0,v1,v2=(vertices[triangles[:,i]] for i in range(3));e1=v1-v0;e2=v2-v0
    obs=load(run.parent/"scene008_instances_20261105_r1/observations/frames/130.npz")
    scale=.1;camera_to_actor=np.linalg.inv(obs["model_from_local"][0])@np.linalg.inv(obs["model_to_camera"]);origin=camera_to_actor[:3,3]
    records=[]
    for row in selected:
        pixel=np.array(row["pixel_xy"]);q=np.r_[pixel+.5,1.]@np.linalg.inv(obs["K"]).T
        direction=q@camera_to_actor[:3,:3].T*scale
        pvec=np.cross(direction[None],e2);det=np.einsum("ij,ij->i",e1,pvec);safe=np.where(np.abs(det)>1e-12,det,1)
        tvec=origin-v0;u=np.einsum("ij,ij->i",tvec,pvec)/safe;qvec=np.cross(tvec,e1);v=qvec@direction/safe
        distance=np.einsum("ij,ij->i",e2,qvec)/safe
        base=(np.abs(det)>1e-12)&(u>=-1e-8)&(v>=-1e-8)&(u+v<=1+1e-8)&(distance>.1)
        hits={}
        for matte in (.01,.5):
            valid=base&(alpha[triangles].min(1)>=matte)
            if valid.any():
                tid=int(np.flatnonzero(valid)[np.argmin(distance[valid])]);d=float(distance[tid]);ground=row["road_ray_camera_z_m"]
                diff=ground-d;winner="chart_first" if diff>.05 else "ground_first" if diff<-.05 else "near_contact_005m"
                hits[str(matte)]=dict(hit=True,triangle_id=tid,camera_z_m=d,ground_minus_chart_depth_m=diff,ordering=winner,
                    triangle_min_matte_alpha=float(alpha[triangles[tid]].min()),reference_uv_vertices=uv[triangles[tid]].tolist(),
                    difference_from_local_continuous_S_root_m=d-row["continuous_C2_chart_intersection"]["camera_z_m"])
            else:hits[str(matte)]=dict(hit=False,ordering="no_active_chart_triangle_hit")
        records.append(dict(pixel_xy=row["pixel_xy"],baseline_assetW=row["baseline_assetW"],assetW_loss=row["assetW_loss"],
            original_state="a37 observed; a6 hidden/unknown",ground_camera_z_m=row["road_ray_camera_z_m"],matte_thresholds=hits))
    summaries=[]
    for matte in (.01,.5):
        for severe in (False,True):
            r=[x for x in records if not severe or x["assetW_loss"]>.5];h=[x["matte_thresholds"][str(matte)] for x in r]
            summaries.append(dict(matte_threshold=matte,cohort="loss>.5" if severe else "loss>.01",pixels=len(r),
                chart_first=sum(x["ordering"]=="chart_first" for x in h),ground_first=sum(x["ordering"]=="ground_first" for x in h),
                near_contact=sum(x["ordering"]=="near_contact_005m" for x in h),no_chart_hit=sum(not x["hit"] for x in h),
                triangle_vs_continuous_S_depth_difference_m=stat([x["difference_from_local_continuous_S_root_m"] for x in h if x["hit"]])))
    report=dict(schema="BOUNDED_35_RAY_EXISTING_C2_CHART_FIRST_HIT_PROBE_V1",raw=130,rays=35,canonical_chart_vertices=len(vertices),
        triangles_after_UV_edge_gate=len(triangles),UV_edge_max_pixels=1.5,two_sided=True,matte_thresholds=[.01,.5],summary=summaries,pixels=records,
        geometry="Existing immutable C2 chart positions on reference UV1px grid; Delaunay then reject UV edges>1.5px to avoid bridging holes. Compare nearest triangle hit to independently measured-pool ground ray.",
        limitations=["Open chart sheet and matte-boundary approximation, not a closed car surface; retained non-chart Gaussians are excluded from triangle mesh.",
            "Hard matte thresholds are fixed boundary sensitivity checks, not fitted to results; this is an opaque reference rather than translucent native Gaussian rendering.",
            "No original hidden a6 first-hit truth exists at these a37-visible pixels. Internal chart-vs-ground ordering cannot validate hidden physical geometry.",
            "This bounded probe is not a complete MOG renderer, training method, scene-adoption result or replacement for the native paired evaluation."],
        GPU_used=False,model_updates=0,seconds=time.time()-start)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n");print(json.dumps(dict(summary=summaries,seconds=report["seconds"])),flush=True)


if __name__=="__main__":main()
