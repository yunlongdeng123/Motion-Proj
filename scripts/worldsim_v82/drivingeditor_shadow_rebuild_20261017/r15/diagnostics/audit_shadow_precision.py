"""阴影分块敏感性、太阳批处理和 FP64 密集参考；不修改生产源码。"""
import argparse
import json
import math
from pathlib import Path
import sys
import time
import torch

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
BASE=ROOT/'backend_restore_r15'
sys.path.insert(0,str(ROOT/'implementation_r15_train_v3/shadow_rebuild_v3'))
from shadow_rebuild.config import Config,LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state
from shadow_rebuild.math3d import disk_directions,covariance
import shadow_rebuild.tracing as tracing


def timed(f):
    torch.cuda.synchronize();start=time.perf_counter();v=f();torch.cuda.synchronize()
    return v,time.perf_counter()-start


def stable_integral(o,d,m,p,amplitude,lower=0.,upper=None):
    delta=o[:,None]-m[None]
    a=torch.einsum('ri,gij,rj->rg',d,p,d).clamp_min(1e-12)
    b=torch.einsum('ri,gij,rgj->rg',d,p,delta)
    residual=delta-(b/a)[...,None]*d[:,None]
    transverse=torch.einsum('rgi,gij,rgj->rg',residual,p,residual).clamp_min(0)
    lo=torch.sqrt(a/2)*(lower+b/a)
    assert upper is None
    return amplitude[None]*torch.exp(-.5*transverse)*torch.sqrt(2*math.pi/a)*(.5*torch.erfc(lo)).clamp_min(0)


@torch.no_grad()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',required=True);args=ap.parse_args()
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    saved=torch.load(BASE/'paused_checkpoint_004000_20261017/checkpoint.pt',map_location='cpu',weights_only=False)
    raw=saved['config'].copy();raw['weights']=LossWeights(**raw['weights']);cfg=Config(**raw).validate()
    data=Dataset(BASE/'data','cuda',True)
    model=SceneModel(seed_from_state(saved['state']),cfg).cuda();model.load_state_dict(saved['state']);model.set_stage('geometry');model.requires_grad_(False)
    frame=data.frame(18);g,p=model.buffers(frame,len(data.actor_ids),'cuda',isolated=False)
    h,w=p['alpha'].shape;s=cfg.shadow_train_stride
    ys=torch.linspace(0,h-1,math.ceil(h/s)+1,device='cuda').round().long()
    xs=torch.linspace(0,w-1,math.ceil(w/s)+1,device='cuda').round().long()
    yy,xx=torch.meshgrid(ys,xs,indexing='ij');ids=(yy*w+xx).flatten()
    active=p['road_alpha'].flatten()[ids]>1e-5
    position=p['road_position'].reshape(-1,3)[ids][active];normal=p['road_normal'].reshape(-1,3)[ids][active]
    result=dict(checkpoint_step=4000,frame=frame.frame_id,active_receiver_rays=len(position),sun_samples=cfg.sun_samples,production_SGD_updates=0,production_source_modified=False)
    v512,t512=timed(lambda:model.light.visibility(position,normal,g))
    cfg.gaussian_chunk=2048
    v2048,t2048=timed(lambda:model.light.visibility(position,normal,g))
    cfg.gaussian_chunk=512
    radius=torch.deg2rad(5*model.light.radius_logit.sigmoid())
    directions=disk_directions(model.light.direction(),radius,cfg.sun_samples)
    origins=position+normal*cfg.ray_bias_m
    def batched():
        nr=len(origins);ns=len(directions)
        flat=tracing.geometry_transmittance(origins.repeat(ns,1),directions[:,None,:].expand(ns,nr,3).reshape(-1,3),g,cfg.ray_chunk,cfg.gaussian_chunk)
        return flat.reshape(ns,nr).mean(0)
    vbatch,tbatch=timed(batched)
    result['variants']=dict(chunk512_seconds=t512,chunk2048_seconds=t2048,batched_same_5_sun_seconds=tbatch,
        chunk_max_abs_visibility_delta=float((v512-v2048).abs().max()),
        chunk_mean_abs_visibility_delta=float((v512-v2048).abs().mean()),
        batched_max_abs_visibility_delta=float((v512-vbatch).abs().max()))
    selected=(v512-v2048).abs().topk(min(5,len(position))).indices
    o=origins[selected].double();nr=len(o);ns=len(directions)
    flat_o=o.repeat(ns,1);flat_d=directions.double()[:,None,:].expand(ns,nr,3).reshape(-1,3)
    keep=g['owner']>=0;rows=g['rows'][keep];means=g['xyz'][keep].double()
    q=model.field.quaternion[rows].double();scales=model.field.log_scales[rows].double().exp().clamp(cfg.scale_min_m,cfg.scale_max_m)
    rotations=frame.poses[g['owner'][keep],:3,:3].double()
    cov64=rotations@covariance(q,scales)@rotations.transpose(-1,-2)
    precision64=torch.linalg.inv(cov64)
    cov32=g['cov'][keep]
    precision32=torch.linalg.inv(cov32)
    opacity=g['opacity'][keep,0].double().clamp(1e-6,1-1e-6)
    amp=-torch.log1p(-opacity)/(math.sqrt(2*math.pi)*scales.amin(-1))
    def dense_reference(precision,stable=True):
        tau=torch.zeros(len(flat_o),device='cuda',dtype=torch.float64)
        for g0 in range(0,len(means),8192):
            fn=stable_integral if stable else tracing._precision_integral
            tau+=fn(flat_o,flat_d,means[g0:g0+8192],precision[g0:g0+8192].double(),amp[g0:g0+8192]).sum(-1)
        return torch.exp(-tau.clamp_max(80)).reshape(ns,nr).mean(0)
    reference64,tref=timed(lambda:dense_reference(precision64))
    ref32matrix,tref32=timed(lambda:dense_reference(precision32))
    result['selected_ray_references']=dict(indices=selected.cpu().tolist(),chunk512=v512[selected].cpu().tolist(),
        chunk2048=v2048[selected].cpu().tolist(),dense_FP64_physical_covariance=reference64.cpu().tolist(),
        dense_FP64_using_original_FP32_inverse=ref32matrix.cpu().tolist(),
        reference_seconds=tref,FP32_inverse_reference_seconds=tref32,
        scope='Five highest chunk-difference road-grid rays, five sun samples, all valid Actor Gaussians; no AABB culling. Physical reference reconstructs covariance in FP64 from saved quaternion/scales and fixed pose.')
    condition=(scales.amax(-1)/scales.amin(-1)).square()
    covariance_eigen=torch.linalg.eigvalsh(cov32.double())
    precision_eigen=torch.linalg.eigvalsh((precision32.double()+precision32.double().transpose(-1,-2))*.5)
    result['conditioning']=dict(valid_actor_gaussians=len(means),
        physical_condition_gt_1e6=int((condition>1e6).sum()),physical_condition_gt_1e7=int((condition>1e7).sum()),
        physical_condition_p99=float(torch.quantile(condition,.99)),physical_condition_max=float(condition.max()),
        rounded_FP32_covariance_not_positive_definite=int((covariance_eigen[:,0]<=0).sum()),
        FP32_inverse_symmetric_part_not_positive_definite=int((precision_eigen[:,0]<=0).sum()),
        note='Eigenvalue checks run in FP64 on the actual rounded FP32 covariance/inverse; physical condition comes from saved principal scales.')
    del precision64,cov64,precision32,covariance_eigen,precision_eigen
    torch.cuda.empty_cache()
    original=tracing._precision_integral
    tracing._precision_integral=stable_integral
    try:
        vs512,ts512=timed(lambda:model.light.visibility(position,normal,g))
        cfg.gaussian_chunk=2048
        vs2048,ts2048=timed(lambda:model.light.visibility(position,normal,g))
        result['stable_residual_FP32']=dict(chunk512_seconds=ts512,chunk2048_seconds=ts2048,
            finite=bool(torch.isfinite(vs512).all()&torch.isfinite(vs2048).all()),
            nonfinite_receiver_count_512=int((~torch.isfinite(vs512)).sum()),
            nonfinite_receiver_count_2048=int((~torch.isfinite(vs2048)).sum()),
            max_chunk_delta=float((vs512-vs2048).abs().max()),
            selected_chunk512=vs512[selected].cpu().tolist(),selected_chunk2048=vs2048[selected].cpu().tolist())
    finally:
        tracing._precision_integral=original;cfg.gaussian_chunk=512
    result['status']='COMPLETE_TRAINING_REMAINS_PAUSED'
    def finite_json(x):
        if isinstance(x,float) and not math.isfinite(x):return None
        if isinstance(x,list):return [finite_json(v) for v in x]
        if isinstance(x,dict):return {k:finite_json(v) for k,v in x.items()}
        return x
    result=finite_json(result)
    (out/'shadow_precision.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result,allow_nan=False),flush=True)


if __name__=='__main__':main()
