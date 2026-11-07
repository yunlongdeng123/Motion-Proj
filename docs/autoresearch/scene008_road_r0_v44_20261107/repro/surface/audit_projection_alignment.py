"""CPU compare source pinhole, observation and actual saved raster projections.

Uses the identical forward active surface points under all three mappings.
No current runtime, renderer, model/checkpoint or torch is imported.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates


def project(points,K,w):
    q=points@w[:3,:3].T+w[:3,3]
    h=q@K.T
    return h[:,:2]/q[:,2:]- .5,q[:,2]


def raster_project(points,full,width,height,dtype=np.float64):
    hom=np.c_[points,np.ones(len(points))].astype(dtype)@full.astype(dtype)
    ndc=hom[:,:2]/hom[:,3:4]
    uv=(ndc+1)*np.array([width,height],dtype=dtype)/2-.5
    return uv,hom[:,3]


def sample(rgb,uv):
    return np.stack([map_coordinates(rgb[:,:,c].astype(float),[uv[:,1],uv[:,0]],order=1,mode="constant",cval=np.nan) for c in range(3)],1)


def pair_stats(a,b,mask):
    delta=a[mask]-b[mask]
    if not len(delta):return dict(points=0)
    length=np.linalg.norm(delta,axis=1)
    return dict(points=len(delta),mean_vector=delta.mean(0).tolist(),max_abs_component=float(np.abs(delta).max()),
                median_euclidean=float(np.median(length)),max_euclidean=float(length.max()),p95_euclidean=float(np.quantile(length,.95)))


def colour_stats(a,b):
    finite=np.isfinite(a).all(1)&np.isfinite(b).all(1)
    d=a[finite]-b[finite]
    return dict(points=len(d),mean_difference_rgb=d.mean(0).tolist() if len(d) else None,
                MAE_srgb=float(np.abs(d).mean()) if len(d) else None,
                max_abs_channel=float(np.abs(d).max()) if len(d) else None,
                p95_abs_channel=float(np.quantile(np.abs(d),.95)) if len(d) else None)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True)
    p.add_argument("--base",type=Path,default=Path("/root/autodl-tmp/drivingeditor_repro/campaign"))
    args=p.parse_args();base,run=args.base,args.run
    out=run/"surface/frontfill_v2/projection_alignment.json"
    if out.exists():raise FileExistsError("Preserve prior projection audit")
    started=time.time();rows=[];arrays={}
    with np.load(run/"surface/frontfill_v2/road_reference_199.npz",allow_pickle=False) as a:
        ids=np.flatnonzero(a["active"]&(a["local_xyz"][:,0]>19.75))
        local=a["local_xyz"][ids].astype(float);model=a["model_xyz"][ids]
        source_valid=a["source_valid"][:,ids];reference=a["reference_srgb"][ids]
        accepted_source_rgb=a["source_rgb"][:,ids]
    for raw in (105,113,130,137):
        with np.load(base/f"r15_gn11_fullclip_cpu_20261026/frames/{raw:03d}.npz",allow_pickle=False) as f:
            KA,w=f["K"],f["w2l"]
        with np.load(base/f"scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz",allow_pickle=False) as f:
            KB,model_to_camera=f["K"],f["model_to_camera"]
            original=f["rgb"].astype(float)
        if original.max()>1:original/=255
        metrics=base/f"scene008_v44_rgb_asset_20261107_r1/results/full49_r1/v44/{raw:03d}/metrics.json"
        camera=json.loads(metrics.read_text())["actual_camera_pose"]
        full=np.asarray(camera["full_projection"],float);WV=np.asarray(camera["world_view"],float)
        width,height=int(camera["width"]),int(camera["height"])
        source_image=np.asarray(Image.open(base/f"r15_gn11_fullclip_cpu_20261026/rgb/{raw:03d}.png"),float)/255
        if source_image.shape!=original.shape or original.shape[:2]!=(height,width):
            raise ValueError(("Resolution mismatch",raw,source_image.shape,original.shape,(height,width)))
        A,zA=project(local,KA,w);B,zB=project(model,KB,model_to_camera)
        C,zC=raster_project(model,full,width,height)
        C32,_=raster_project(model,full,width,height,np.float32)
        WVuv,_=project(model,KB,WV.T)
        front=(zA>.1)&(zB>.01)&(zC>.01)
        inframe=front&(A[:,0]>=0)&(A[:,0]<width-1)&(A[:,1]>=0)&(A[:,1]<height-1)&(B[:,0]>=0)&(B[:,0]<width-1)&(B[:,1]>=0)&(B[:,1]<height-1)&(C[:,0]>=0)&(C[:,0]<width-1)&(C[:,1]>=0)&(C[:,1]<height-1)
        qualified=source_valid[raw]&inframe
        sA=sample(source_image,A[qualified]);oA=sample(original,A[qualified])
        oB=sample(original,B[qualified]);oC=sample(original,C[qualified])
        record=dict(raw=raw,active_front_points=len(ids),common_front=count(front),common_inframe=count(inframe),
            same_frame_qualified_points=count(qualified),source_resolution_wh=[source_image.shape[1],source_image.shape[0]],
            observation_resolution_wh=[original.shape[1],original.shape[0]],raster_resolution_wh=[width,height],
            source_K=KA.tolist(),observation_K=KB.tolist(),K_maxabs=float(np.abs(KA-KB).max()),
            observation_model_to_camera_vs_saved_worldview_maxabs=float(np.abs(model_to_camera-WV.T).max()),
            A_source_vs_B_observation_common_inframe_px=pair_stats(A,B,inframe),
            A_source_vs_C_native_raster_common_inframe_px=pair_stats(A,C,inframe),
            B_observation_vs_C_native_raster_common_inframe_px=pair_stats(B,C,inframe),
            A_source_vs_B_observation_qualified_px=pair_stats(A,B,qualified),
            A_source_vs_C_native_raster_qualified_px=pair_stats(A,C,qualified),
            B_observation_vs_saved_worldview_qualified_px=pair_stats(B,WVuv,qualified),
            saved_raster_float64_vs_float32_qualified_px=pair_stats(C,C32,qualified),
            same_frame_srgb=dict(
                cached_source_vs_sourcePNG_at_A=colour_stats(accepted_source_rgb[raw,qualified],sA),
                sourcePNG_at_A_vs_observation_at_A=colour_stats(sA,oA),
                sourcePNG_at_A_vs_observation_at_B=colour_stats(sA,oB),
                sourcePNG_at_A_vs_observation_at_C=colour_stats(sA,oC),
                reference_vs_sourcePNG_at_A=colour_stats(reference[qualified],sA),
                reference_vs_observation_at_B=colour_stats(reference[qualified],oB),
                reference_vs_observation_at_C=colour_stats(reference[qualified],oC)),
            whole_image_same_pixel_sourcePNG_observation_srgb_MAE=float(np.abs(source_image-original).mean()))
        rows.append(record)
        for name,val in (("A",A),("B",B),("C",C),("qualified",qualified),("inframe",inframe)):
            arrays[f"raw{raw}_{name}"]=val
    np.savez_compressed(out.with_suffix(".npz"),asset_ids=ids,local_xyz=local,model_xyz=model,**arrays)
    report=dict(schema="R0_FRONTFILL_THREE_PROJECTION_CPU_AUDIT_V1",CPU_only=True,
        paths=dict(A="full199 frames/{raw}.npz w2l,K on anchor-local metres",B="instances observations model_to_camera,K on model_xyz",
                   C="V44 full49 actual_camera_pose.full_projection; homogeneous model_xyz row-vector multiply and native ndc2Pix"),
        pixel_convention="Pinhole uv=K*[x,y,z]/z -0.5; native raster uv=((NDC+1)*[W,H]-1)/2.",
        rows=rows,GPU_used=False,torch_imported=False,checkpoint_loads=0,optimizer_updates=0,seconds=time.time()-started)
    out.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps([dict(raw=r["raw"],qualified=r["same_frame_qualified_points"],
        A_B_px=r["A_source_vs_B_observation_qualified_px"],A_C_px=r["A_source_vs_C_native_raster_qualified_px"],
        source_vs_obs_C=r["same_frame_srgb"]["sourcePNG_at_A_vs_observation_at_C"],
        ref_vs_obs_C=r["same_frame_srgb"]["reference_vs_observation_at_C"]) for r in rows]),flush=True)


def count(x):return int(np.count_nonzero(x))


if __name__=="__main__":main()
