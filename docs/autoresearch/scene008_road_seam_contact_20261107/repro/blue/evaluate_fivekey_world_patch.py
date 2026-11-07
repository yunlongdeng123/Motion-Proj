"""CPU secondary view audit on the already frozen blue world patch and edges.

All 5 original-input masks are saved before any method array is opened.
The user blue bbox remains primary only in raw128; this is supporting world-view
evidence, not a new circle selected per image or a substitute heldout benchmark.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
from scipy.spatial import cKDTree

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_road_seam_contact_20261107_r1')
BASE=RUN/'blue'
CACHE=RUN.parent/'r15_gn11_fullclip_cpu_20261026/frames'
RAWS=(105,113,128,130,137)
EDGES=(('near_x',0,19.8,1),('far_x',0,31.8,1),('lateral_y',1,2.71,0))
PATCH=(19.,32.6,.5,3.5)


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for c in iter(lambda:f.read(2**20),b''):h.update(c)
    return h.hexdigest()


def write(p,v):
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')


def build_mask(raw,out):
    op=RUN/f'diagnostic/layers_v1/{raw:03d}/observations.npz';fp=CACHE/f'{raw:03d}.npz'
    obs=np.load(op);frame=np.load(fp);ground=frame['ground_local'].astype(float);tree=cKDTree(ground[:,:2])
    h,w=obs['road'].shape;yy,xx=np.mgrid[:h,:w]
    rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],-1)@np.linalg.inv(obs['K']).T
    c2l=np.linalg.inv(frame['w2l']);ray=rays@c2l[:3,:3].T;origin=c2l[:3,3]
    pg=ground[(ground[:,0]>=PATCH[0]-1)&(ground[:,0]<=PATCH[1]+1)&(ground[:,1]>=PATCH[2]-1)&(ground[:,1]<=PATCH[3]+1)]
    if len(pg)<3:raise RuntimeError('No measured raw ground support for registered patch')
    initial=float(np.median(pg[:,2]));validray=abs(ray[...,2])>1e-8
    t=(initial-origin[2])/np.where(validray,ray[...,2],1)
    t=np.where(np.isfinite(t)&(t>0)&(t<100),t,1.)
    for _ in range(12):
        xyz=origin+t[...,None]*ray;dist,ids=tree.query(xyz[...,:2],k=8);z=ground[ids,2]
        med=np.median(z,axis=-1);good=(dist<=.8)&(abs(z-med[...,None])<=.15)
        weight=np.where(good,1/np.maximum(dist,.04)**2,0);den=weight.sum(-1)
        target_z=(weight*z).sum(-1)/np.maximum(den,1e-20)
        next_t=(target_z-origin[2])/np.where(validray,ray[...,2],1)
        converged=abs(next_t-t)*np.linalg.norm(ray,axis=-1)<.02;t=next_t
        support=good.sum(-1);spread=np.sqrt((weight*(z-target_z[...,None])**2).sum(-1)/np.maximum(den,1e-20))
    xyz=origin+t[...,None]*ray;mapped=validray&(t>0)&(t<100)&(support>=3)&(spread<=.08)&converged
    patch=(xyz[...,0]>=PATCH[0])&(xyz[...,0]<=PATCH[1])&(xyz[...,1]>=PATCH[2])&(xyz[...,1]<=PATCH[3])
    clean=obs['road'].astype(bool)&~frame['guard'].astype(bool)&mapped&patch
    measured=obs['scene_depth_m']>0;qualified=clean&measured&(abs(t-obs['scene_depth_m'])<=.3)
    masks={'world_patch_clean_road':clean,'world_patch_direct_depth_qualified_road':qualified}
    for edge,axis,value,tangent in EDGES:
        signed=xyz[...,axis]-value
        masks[f'{edge}_minus']=clean&(signed>=-.4)&(signed<=-.08)
        masks[f'{edge}_plus']=clean&(signed>=.08)&(signed<=.4)
    file=out/f'{raw:03d}.npz'
    np.savez_compressed(file,**masks,local_xyz=xyz.astype(np.float32))
    return dict(raw=raw,counts={k:int(v.sum()) for k,v in masks.items()},
                masks_sha256=sha(file),original_input_sha256={str(op):sha(op),str(fp):sha(fp)})


def metric(d,target,m):
    if not m.any():return dict(count_pixels=0,abs_RGB_error_sum=0.,abs_RGB_error_mean=None)
    e=abs(d['rgb'][:,m]-target[:,m]).mean(0)
    w=d['group_alpha_T'];total=float(m.sum())
    return dict(count_pixels=int(m.sum()),abs_RGB_error_sum=float(e.sum()),abs_RGB_error_mean=float(e.mean()),
                newroad_alphaT_mean=float(d['newroad_alpha_T'][m].mean()),oldroad_alphaT_mean=float(d['oldroad_alpha_T'][m].mean()),
                static_alphaT_mean=float(w[3][m].mean()))


def main():
    start=time.time();out=BASE/'fivekey_world_patch_masks_v1';out.mkdir(exist_ok=False)
    reg=json.loads((BASE/'roi_registration.json').read_text())
    frozen=[build_mask(raw,out) for raw in RAWS]
    registry=dict(status='ALL_ORIGINAL_INPUT_MASKS_SAVED_BEFORE_METHOD_ARRAY_READS',
        original_raw128_registry_sha256=sha(BASE/'roi_registration.json'),
        inherited_world_patch_anchor_local_m=reg['world_patch_anchor_local_m'],inherited_edges=reg['edge_centres'],
        mapping=reg['world_mapping'],counts=frozen,
        interpretation='Supporting5FRONT views of same previously registered world patch. Raw128 approximate user bbox remains primary; no per-view circle is invented. These supporting views were prepared during same-scene development after pilot outcomes were open, but selection uses original geometry/road/guard only.',
        tangent_pair_rule='.2m tangent bins;>=3 original-eligible pixels on each side; equal-bin mean absolute RGB residual jump after subtracting original observed colour.',
        dense_hidden_GT=False)
    write(out/'registry.json',registry)
    methods={'R0':RUN/'diagnostic/layers_v1','n2':RUN/'results/n2_full49',
             'AA':RUN/'results/n2_mip_full49','n2_material':RUN/'results/n2_material_full49',
             'AA_frozen_material':RUN/'results/n2_mip_material_pilot'}
    results=[];sources={}
    for raw in RAWS:
        mask=np.load(out/f'{raw:03d}.npz');obs=np.load(RUN/f'diagnostic/layers_v1/{raw:03d}/observations.npz')
        target=obs['rgb'].transpose(2,0,1);xyz=mask['local_xyz']
        for state in ('keep','delete'):
            for method,root in methods.items():
                path=root/f'{raw:03d}'/(f'r0_{state}.npz' if method=='R0' else f'candidate_{state}.npz')
                d=np.load(path);sources[str(path)]=sha(path)
                row=dict(raw=raw,state=state,method=method,regions={k:metric(d,target,mask[k]) for k in mask.files if mask[k].dtype==bool},edges={})
                resid=d['rgb']-target
                for edge,axis,value,ta in EDGES:
                    mm=mask[f'{edge}_minus'];mp=mask[f'{edge}_plus'];bins=np.floor(xyz[...,ta]/.2).astype(np.int32)
                    pairs=[]
                    for b in np.unique(bins[mm|mp]):
                        mn=mm&(bins==b);pl=mp&(bins==b)
                        if mn.sum()>=3 and pl.sum()>=3:
                            jump=resid[:,pl].mean(1)-resid[:,mn].mean(1)
                            pairs.append(dict(tangent_bin=int(b),minus_count=int(mn.sum()),plus_count=int(pl.sum()),signed_residual_jump_rgb=jump.astype(float).tolist(),abs_channel_jump=float(abs(jump).mean())))
                    row['edges'][edge]=dict(paired_bins=len(pairs),equal_bin_abs_residual_jump_mean=float(np.mean([p['abs_channel_jump'] for p in pairs])) if pairs else None,bins=pairs)
                results.append(row)
    payload=dict(schema='BLUE_SECONDARY_FIVEKEY_FIXED_WORLD_PATCH_V1',registry_sha256=sha(out/'registry.json'),
                 primary_raw128_fixed_bbox_is_separate=True,methods=list(methods),raws=list(RAWS),states=['keep','delete'],
                 per_view=results,sources_sha256=sources,seconds=time.time()-start,
                 evidence_limit='All same-scene FRONT evidence; do not pool unsupported edges as zeros or relabel this as heldout/full-camera validation.')
    write(BASE/'fivekey_fixed_world_patch_comparison.json',payload)
    print(json.dumps(dict(output=str(BASE/'fivekey_fixed_world_patch_comparison.json'),seconds=time.time()-start,counts=frozen)),flush=True)


if __name__=='__main__':main()
