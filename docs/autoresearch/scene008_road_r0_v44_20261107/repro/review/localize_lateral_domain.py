"""Associate exact keyframe regression with a fixed measured-ground ray map.

This is a CPU domain diagnosis, not a rerendered strip counterfactual or road
semantic GT. Actual saved extension alpha*T identifies contribution separately.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS"):os.environ[k]="1"
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
from scipy.spatial import cKDTree


def load(p):
    with np.load(p,allow_pickle=False) as f:return {k:f[k] for k in f.files}
def count(x):return int(np.count_nonzero(x))


class Surface:
    def __init__(self,p,plane):self.p=p.astype(float);self.tree=cKDTree(p[:,:2]);self.plane=-plane[[0,1,3]]/plane[2]
    def height(self,xy):
        d,ids=self.tree.query(xy,k=8,distance_upper_bound=.8);valid=np.isfinite(d);p=self.p[np.minimum(ids,len(self.p)-1)]
        r=p[:,:,2]-p[:,:,:2]@self.plane[:2]-self.plane[2];w=np.where(valid,1/np.maximum(d,.05)**2,0)
        return xy@self.plane[:2]+self.plane[2]+(w*r).sum(1)/np.maximum(w.sum(1),1e-12),valid.sum(1)


def rays(surface,K,w):
    c=np.linalg.inv(w);yy,xx=np.mgrid[:320,:480]
    ray=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1).reshape(-1,3)@np.linalg.inv(K).T@c[:3,:3].T;o=c[:3,3]
    den=ray[:,2]-ray[:,:2]@surface.plane[:2];num=o[:2]@surface.plane[:2]+surface.plane[2]-o[2]
    depth=np.divide(num,den,out=np.full(len(ray),1e5),where=np.abs(den)>1e-7)
    first=o+ray*depth[:,None];potential=(depth>.1)&(depth<120)&(first[:,0]>=.5)&(first[:,0]<=90)&(np.abs(first[:,1])<=8);ids=np.flatnonzero(potential)
    for _ in range(8):
        p=o+ray[ids]*depth[ids,None];h,support=surface.height(p[:,:2])
        delta=np.divide(h-p[:,2],den[ids],out=np.zeros(len(ids)),where=(np.abs(den[ids])>1e-7)&(support>=3))
        depth[ids]+=np.clip(delta,-10,10)
    p=o+ray*depth[:,None];h,support=surface.height(p[ids,:2]);valid=np.zeros(len(ray),bool)
    valid[ids]=(support>=3)&(np.abs(h-p[ids,2])<.03)&(depth[ids]>.1)&(depth[ids]<120)
    return p.reshape(320,480,3),valid.reshape(320,480)


def main():
    p=argparse.ArgumentParser();p.add_argument("--run",type=Path,required=True);args=p.parse_args();run=args.run;start=time.time()
    target=run/"review/full49_worldfield_lateral_localization.json"
    if target.exists():raise FileExistsError("Preserve prior lateral-domain audit")
    with np.load(run/"surface/frontfill_v2/road_reference_199.npz",allow_pickle=False) as f:
        selected=f["active"]&(f["local_xyz"][:,0]>19.75);xyz=f["local_xyz"]
        strip=selected&(xyz[:,1]>=-3.35)&(xyz[:,1]<=2.71)
        candidate_stats=dict(selected_cells=count(selected),inside_original_lateral_cells=count(strip),outside_original_lateral_cells=count(selected&~strip),
            original_y_m=[-3.35,2.71],selected_bounds=[xyz[selected].min(0).tolist(),xyz[selected].max(0).tolist()])
    print(json.dumps(candidate_stats),flush=True)
    bundle=json.loads((run.parent/"r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json").read_text())
    plane=np.asarray(json.loads(Path(bundle["original_readiness"]).read_text())["SGCM_geometry"]["ground_fit"]["plane_anchor"])
    pool=load(run/"surface/measured_ground_support_pool.npz")["local_xyz"];surface=Surface(pool,plane)
    accum={};records=[]
    for raw in (105,113,130,137):
        f=load(run.parent/f"r15_gn11_fullclip_cpu_20261026/frames/{raw:03d}.npz");local,valid=rays(surface,f["K"],f["w2l"])
        obs=load(run.parent/f"scene008_instances_20261105_r1/observations/frames/{raw:03d}.npz")
        road=obs["road"].astype(bool);protected=obs["protect"].astype(bool)&~obs["visible"].any(0)
        gt=obs["rgb"].astype(np.float32).transpose(2,0,1)
        if gt.max()>1:gt/=255
        lateral_inside=(local[:,:,1]>=-3.35)&(local[:,:,1]<=2.71)
        front=(local[:,:,0]>19.75)&(local[:,:,0]<=31.84)
        classes=dict(front_inside_original_y=valid&front&lateral_inside,
            front_outside_original_y=valid&front&~lateral_inside,
            front_y_below_original=valid&front&(local[:,:,1]<-3.35),front_y_above_original=valid&front&(local[:,:,1]>2.71),
            old_or_far_x=valid&~front,geometry_unknown=~valid)
        masks=dict(observed_road=road,protected=protected,protected_nonroad=protected&~road)
        for state in ("keep","delete"):
            folder=run/f"results/full49_worldfield/{raw:03d}"
            base=load(folder/f"baseline_{state}_float.npz");ca=load(folder/f"calibrated_{state}_float.npz")
            errors=(np.abs(ca["rgb"]-gt).mean(0)-np.abs(base["rgb"]-gt).mean(0)).astype(float)
            positive=np.maximum(errors,0);W=ca["extension_alpha_T"]
            for domain,mask in masks.items():
                total=float(positive[mask].sum());totalnet=float(errors[mask].sum())
                row=dict(raw=raw,state=state,domain=domain,pixels=count(mask),positive_excess_sum=total,net_excess_sum=totalnet,classes={})
                for label,klass in classes.items():
                    m=mask&klass
                    rec=dict(pixels=count(m),positive_excess_sum=float(positive[m].sum()),net_excess_sum=float(errors[m].sum()),
                        actual_extension_W_gt001=count(m&(W>.01)),positive_excess_with_W_gt001=float(positive[m&(W>.01)].sum()))
                    row["classes"][label]=rec
                    key=(state,domain,label)
                    if key not in accum:accum[key]=dict(pixels=0,positive_excess_sum=0.,net_excess_sum=0.,actual_extension_W_gt001=0,positive_excess_with_W_gt001=0.)
                    for k in rec:accum[key][k]+=rec[k]
                key=(state,domain,"TOTAL")
                if key not in accum:accum[key]=dict(pixels=0,positive_excess_sum=0.,net_excess_sum=0.)
                for k,v in (("pixels",count(mask)),("positive_excess_sum",total),("net_excess_sum",totalnet)):accum[key][k]+=v
                records.append(row)
        print(json.dumps(dict(raw=raw,seconds=time.time()-start)),flush=True)
    rows=[]
    for (state,domain,label),value in accum.items():
        if label=="TOTAL":continue
        total=accum[(state,domain,"TOTAL")]
        rows.append(dict(state=state,domain=domain,world_class=label,**value,
            fraction_of_domain_pixels=value["pixels"]/total["pixels"],
            fraction_of_positive_excess=value["positive_excess_sum"]/total["positive_excess_sum"] if total["positive_excess_sum"] else None,
            fraction_of_net_excess=value["net_excess_sum"]/total["net_excess_sum"] if total["net_excess_sum"] else None))
    report=dict(schema="R0_FIXED_LATERAL_DOMAIN_KEY4_EXACT_FLOAT_AUDIT_V1",candidate_stats=candidate_stats,
        source="Original obs fixed masks; saved full49_worldfield exact float baseline/calibrated, raws105/113/130/137.",
        world_mapping="Independent full-image measured-ground-pool ray intersection, fixed8 IDW iterations and >=3of8 actual ground returns within0.8m; no predicted depth used to assign XY.",
        rows=rows,per_frame=records,
        interpretation="Associates appearance regression with ground-projected world XY; does not rerender a strip counterfactual or certify ground as road semantics. Unknown ray geometry remains separate.",
        proposed_strip="Fixed selected active x>19.75 with original-road lateral envelope y[-3.35,2.71], identical across camera/keep/delete. Candidate comparison only; requires real shared-sort rerender before adoption.",
        GPU_used=False,optimizer_updates=0,seconds=time.time()-start)
    target.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps(rows),flush=True)


if __name__=="__main__":main()
