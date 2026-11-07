"""Freeze candidate-independent same-scene world masks, reusing old five files.

CPU only. Reads original road/guard/raw LiDAR ground; never reads a new target,
candidate render, fitted residual, model alpha, or road_edit_rgb/depth.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import argparse,hashlib,json,shutil,time
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

KEY5=(105,113,128,130,137)
EDGES=(('near_x',0,19.8,1),('far_x',0,31.8,1),('lateral_y',1,2.71,0))
PATCH=(19.,32.6,.5,3.5)


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):
    with np.load(p,allow_pickle=False) as a:return {k:a[k] for k in a.files}
def write(p,a):p.write_text(json.dumps(a,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def pair_summary(m):
    out={};xyz=m['local_xyz']
    for name,axis,value,ta in EDGES:
        minus=m[name+'_minus'];plus=m[name+'_plus'];selected=minus|plus
        bins=np.zeros(minus.shape,np.int32);bins[selected]=np.floor(xyz[...,ta][selected]/.2).astype(np.int32)
        pairs=[]
        for b in np.unique(bins[selected]):
            mm=minus&(bins==b);mp=plus&(bins==b)
            if mm.sum()>=3 and mp.sum()>=3:pairs.append(dict(tangent_bin=int(b),minus_count=int(mm.sum()),plus_count=int(mp.sum())))
        out[name]=dict(band_minus_count=int(minus.sum()),band_plus_count=int(plus.sum()),paired_bins=len(pairs),
            matched_minus_count=sum(r['minus_count'] for r in pairs),matched_plus_count=sum(r['plus_count'] for r in pairs),bins=pairs)
    return out


def build(obs_path,frame_path):
    obs=load(obs_path);f=load(frame_path);ground=f['ground_local'].astype(float)
    h,w=obs['road'].shape;eligible=obs['road'].astype(bool)&~f['guard'].astype(bool)
    y,x=np.nonzero(eligible)
    xyz_full=np.full((h,w,3),np.nan,np.float32);clean=np.zeros((h,w),bool);qualified=clean.copy()
    masks={'world_patch_clean_road':clean,'world_patch_direct_depth_qualified_road':qualified}
    for name,axis,value,ta in EDGES:masks[name+'_minus']=clean.copy();masks[name+'_plus']=clean.copy()
    pg=ground[(ground[:,0]>=PATCH[0]-1)&(ground[:,0]<=PATCH[1]+1)&(ground[:,1]>=PATCH[2]-1)&(ground[:,1]<=PATCH[3]+1)]
    if len(pg)<3 or not len(x):
        return dict(**masks,local_xyz=xyz_full),dict(status='NO_RAW_GROUND_OR_ORIGINAL_ELIGIBLE_PIXELS',original_eligible_pixels=len(x),raw_patch_ground_points=len(pg))
    # Vector operations are independent per ray. Selecting only original
    # road&~guard saves CPU without changing any eligible ray's mapping rule.
    tree=cKDTree(ground[:,:2]);ray=np.c_[x+.5,y+.5,np.ones(len(x))]@np.linalg.inv(obs['K']).T
    c2l=np.linalg.inv(f['w2l']);ray=ray@c2l[:3,:3].T;origin=c2l[:3,3]
    initial=float(np.median(pg[:,2]));validray=abs(ray[:,2])>1e-8
    t=(initial-origin[2])/np.where(validray,ray[:,2],1)
    t=np.where(np.isfinite(t)&(t>0)&(t<100),t,1.)
    for _ in range(12):
        xyz=origin+t[:,None]*ray;dist,ids=tree.query(xyz[:,:2],k=8);z=ground[ids,2]
        med=np.median(z,axis=-1);good=(dist<=.8)&(abs(z-med[:,None])<=.15)
        weight=np.where(good,1/np.maximum(dist,.04)**2,0);den=weight.sum(-1)
        target_z=(weight*z).sum(-1)/np.maximum(den,1e-20)
        next_t=(target_z-origin[2])/np.where(validray,ray[:,2],1)
        converged=abs(next_t-t)*np.linalg.norm(ray,axis=-1)<.02;t=next_t
        support=good.sum(-1);spread=np.sqrt((weight*(z-target_z[:,None])**2).sum(-1)/np.maximum(den,1e-20))
    xyz=origin+t[:,None]*ray
    mapped=validray&(t>0)&(t<100)&(support>=3)&(spread<=.08)&converged
    patch=(xyz[:,0]>=PATCH[0])&(xyz[:,0]<=PATCH[1])&(xyz[:,1]>=PATCH[2])&(xyz[:,1]<=PATCH[3])
    ok=mapped&patch;clean[y,x]=ok;xyz_full[y,x]=xyz.astype(np.float32)
    measured=obs['scene_depth_m'][y,x]>0
    qualified[y,x]=ok&measured&(abs(t-obs['scene_depth_m'][y,x])<=.3)
    for name,axis,value,ta in EDGES:
        signed=xyz[:,axis]-value
        masks[name+'_minus'][y,x]=ok&(signed>=-.4)&(signed<=-.08)
        masks[name+'_plus'][y,x]=ok&(signed>=.08)&(signed<=.4)
    return dict(**masks,local_xyz=xyz_full),dict(status='ORIGINAL_ROAD_GUARD_GROUND_ONLY',original_eligible_pixels=len(x),raw_patch_ground_points=len(pg),initial_height_m=initial)


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--previous-run',type=Path,required=True);p.add_argument('--output',type=Path)
    a=p.parse_args();run=a.run;old=a.previous_run;out=a.output or run/'red_review/world_masks49_v1'
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True);started=time.time();campaign=run.parent
    originals=campaign/'scene008_instances_20261105_r1/observations/frames';cache=campaign/'r15_gn11_fullclip_cpu_20261026/frames'
    old_masks=old/'blue/fivekey_world_patch_masks_v1';old_registry=json.loads((old_masks/'registry.json').read_text())
    rows=[]
    for raw in range(89,138):
        obs_path=(old/f'diagnostic/layers_v1/{raw:03d}/observations.npz') if raw in KEY5 else originals/f'{raw:03d}.npz'
        frame_path=cache/f'{raw:03d}.npz';target=out/f'{raw:03d}.npz'
        if raw in KEY5:
            source=old_masks/f'{raw:03d}.npz';shutil.copy2(source,target);m=load(target)
            assert sha(source)==sha(target)
            meta=dict(status='EXACT_OLD_KEY5_FILE_REUSE',source=str(source),source_sha256=sha(source))
        else:
            m,meta=build(obs_path,frame_path);np.savez_compressed(target,**m)
        row=dict(raw=raw,mask_file=target.name,mask_sha256=sha(target),metadata=meta,
            counts={k:int(v.sum()) for k,v in m.items() if v.dtype==bool},edges=pair_summary(m),
            obs_path=str(obs_path),obs_sha256=sha(obs_path),raw_ground_frame_path=str(frame_path),raw_ground_frame_sha256=sha(frame_path))
        rows.append(row);print(json.dumps(dict(raw=raw,clean=row['counts']['world_patch_clean_road'],elapsed_seconds=time.time()-started)),flush=True)
    primary={}
    for name in ('fixed_masks_raw128.npz','roi_registration.json'):
        src=old/'blue'/name;dst=out/'primary_raw128'/name;dst.parent.mkdir(exist_ok=True);shutil.copy2(src,dst)
        assert sha(src)==sha(dst);primary[name]=dict(path=str(dst),sha256=sha(dst),source=str(src))
    registry=dict(schema='FROZEN_49_WORLD_PATCH_ORIGINAL_INPUT_MASKS_V1',status='FROZEN_BEFORE_OPENING_NEW_TARGET_OR_CANDIDATE_ARRAYS',
        raws=list(range(89,138)),states=['keep','delete'],world_patch_anchor_local_m=dict(x=[19.,32.6],y=[.5,3.5]),
        edges={n:dict(axis=ax,anchor_local_m=v,tangent_axis=ta) for n,ax,v,ta in EDGES},
        support_rule=dict(neighbours=8,radius_m=.8,z_median_gate_m=.15,weighted_z_spread_max_m=.08,iterations=12,convergence_position_m=.02,
            direct_scene_depth_agreement_m=.3,tangent_bin_m=.2,min_pixels_per_side=3,band_abs_signed_m=[.08,.4]),
        inherited_key5_registry=str(old_masks/'registry.json'),inherited_key5_registry_sha256=sha(old_masks/'registry.json'),
        key5_file_bytes_unchanged=True,primary_raw128_separate=primary,frames=rows,seconds=time.time()-started,
        evidence_scope='Same-scene development extension to remaining44 views; shares BUILD measurements with the road model. Not independent test, hidden GT, or all-camera evidence.',
        exclusions=['No new target/render/residual/alpha reads','No candidate-based denominator changes','No unsupported edge scored as zero'],GPU_used=False)
    write(out/'registry.json',registry)
    print(json.dumps(dict(registry=str(out/'registry.json'),sha256=sha(out/'registry.json'),seconds=registry['seconds'],key5_exact=True)),flush=True)


if __name__=='__main__':main()
