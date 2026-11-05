"""One fixed three-view LOD test. Frozen S/F, no training or parameter search."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import copy,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
import torch
from PIL import Image,ImageDraw,ImageFont
from scale_probe import Runtime,SHAsset,sha,write,row_hash,cpu,camera_hr,environment,board,crop,down_result,decorate
from scene_runtime import select_rows,tagged,merge_rows,raster
from relative_area_asset import RelativeAreaOpacityAsset
from surface_lod_asset import SurfaceLODAsset
import diff_gaussian_rasterization
from diff_gaussian_rasterization import _C

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ART=CAM/'scene008_a6_insert_artifacts_20261105'
FORMAL=CAM/'scene008_a6_appearance_formal_20261105'
ASSET=ART/'surface_fit/combined_v1/observed_fit/formal/complete_a6.npz'
POST=ART/'surface_fit/masked_depth_v1/posterior.npz'
OLD=FORMAL/'output_adapter/projective_surface/formal_observed/formal/complete_a6.npz'
DEST=ART/'scale/surface_lod_v1'
RAWS=(106,113,130)
CALLS=0

def scene(rt,raw,cam,asset,delete):
    """Same frozen Runtime row assembly with explicit viewport camera."""
    d=rt.raw(raw);rt.assert_unchanged()
    static=rt._static(cam);dynamic=rt.n.actor_kwargs(cam,delete_a37=False)
    channels=static['shs'].shape[1] if static['shs'] is not None else 1
    precomputed=static['colors_precomp'] is not None
    a37=select_rows(dynamic,~rt.n.keep_a6)
    if delete:a37=select_rows(a37,torch.zeros_like(a37['mask']))
    A=asset.rows(d['pose'],cam,rt.pipe,channels,precomputed=precomputed)
    assert len(A['means3D'])==len(asset.xyz)
    rows=merge_rows(merge_rows(static,tagged(A,'asset')),tagged(a37,'a37'))
    rows=merge_rows(rows,rt._road(cam,channels,precomputed))
    assert bool((rows['features'][:,6:]==0).all())
    amask=rows['features'][:,2]>.5
    rows=decorate(rows,asset,amask)
    return rows,select_rows(rows,amask),dict(native_a6_rows_input=0,native_a37_packed_rows=len(a37['mask']),asset_rows=len(A['means3D']),delete=delete)

def render(rt,cam,rows):
    global CALLS
    o=raster(cam,rt.pipe,rt.n.static.active_sh_degree,rows,rt.n.bg.new_zeros(3));CALLS+=1
    torch.cuda.synchronize()
    err=float((o['extra'][:4].sum(0)-o['alpha'][0]).abs().max());assert err<2e-5
    result={k:cpu(v) for k,v in dict(rgb=o['rgb'],alpha=o['alpha'],asset_rgb=o['extra'][4:7],asset_alpha=o['extra'][:1],depth_sum=o['depth_sum'],group_alpha=o['extra'][:4]).items()}
    assert all(np.isfinite(x).all() for x in result.values())
    return result

def metrics(o,ref,mask,interior,rt):
    norm=lambda a:a['depth_sum']/np.maximum(a['alpha'],1e-8)/rt.scale
    x={}
    for k in ('rgb','alpha','asset_rgb','asset_alpha'):
        e=np.abs(o[k]-ref[k]).mean(0)
        x[k+'_MAE_vs_D4HR']=float(e[mask].mean())
    x['expected_depth_MAE_vs_D4HR_m']=float(abs(norm(o)-norm(ref))[0,interior].mean())
    x['interior_alpha_mean']=float(o['asset_alpha'][0,interior].mean())
    x['interior_alpha_below09']=int((o['asset_alpha'][0,interior]<.9).sum())
    x['interior_pixels']=int(interior.sum());x['object_pixels']=int(mask.sum())
    return x

def actual_observation(o,d,rt):
    m=cpu(d['a6_visible']).astype(bool)
    if m.ndim==3:m=m[0]
    gt=cpu(d['rgb']);a=o['alpha'][0]
    rgb=np.abs(o['rgb']-gt).mean(0)
    valid=cpu(d['a6_depth_valid']).astype(bool)
    if valid.ndim==3:valid=valid[0]
    depth=cpu(d['depth']);
    if depth.ndim==3:depth=depth[0]
    z=o['depth_sum'][0]/np.maximum(a,1e-8)/rt.scale
    return dict(fixed_visible_pixels=int(m.sum()),RGB_MAE=float(rgb[m].mean()),alpha_mean=float(a[m].mean()),
        alpha_below09=int((a[m]<.9).sum()),fixed_depth_pixels=int(valid.sum()),
        measured_camera_z_MAE_m=float(abs(z-depth)[valid].mean()) if valid.any() else None)

def save_outputs(outdir,outputs,E,box):
    outdir.mkdir()
    for state,o in outputs.items():
        np.savez_compressed(outdir/(state+'_crop_float.npz'),**{k:crop(v,box) for k,v in o.items()},crop_xyxy=np.array(box))
    board([crop(outputs['solo']['rgb'],box),crop(outputs['keep']['rgb']+(1-outputs['keep']['alpha'])*E,box),crop(outputs['delete']['rgb']+(1-outputs['delete']['alpha'])*E,box)],
        ['same resolution solo','same resolution keep a37','same resolution delete a37'],outdir/'fair.png')

@torch.no_grad()
def main():
    DEST.mkdir(parents=True,exist_ok=False);start=time.time()
    assert sha(ASSET)=='0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d'
    assert sha(_C.__file__)=='0eaff4d7a9b8ef6e711627f4c19824bf427c4387f7c6453fcb769a2c919c920e'
    source=[ASSET,POST,OLD,Path(__file__),Path(__file__).with_name('surface_lod_asset.py'),Path(__file__).with_name('relative_area_asset.py'),Path(_C.__file__)]
    before={str(p):sha(p) for p in source}
    rt=Runtime();base=SHAsset.from_geometry(ASSET).cuda();old=SHAsset.from_geometry(OLD).cuda()
    area=RelativeAreaOpacityAsset(base);lod=SurfaceLODAsset(base,POST)
    version=(base.coefficients._version,base.opacity._version)
    torch.cuda.reset_peak_memory_stats()
    records=[]
    for raw in RAWS:
        d=rt.raw(raw);lr=d['cam'];hr=camera_hr(lr);per=DEST/f'raw{raw}';per.mkdir()
        eh=environment(rt,hr,d);el=environment(rt,lr,d)
        record=dict(raw=raw,variants={},dataset_role='internal development diagnostic',source_SF_hash=lod.persistent_SF_hash)
        refs=None;lr_outputs={}
        for name,cam,asset in [('HR_default',hr,base)]+([('HR_LOD_reference_replay',hr,lod)] if raw==106 else [])+[(n,lr,a) for n,a in [('default',base),('area',area),('LOD',lod)]]:
            J,A,meta=scene(rt,raw,cam,asset,False)
            ah=row_hash(A);nA=len(A['means3D'])
            outputs=dict(solo=render(rt,cam,A),keep=render(rt,cam,J))
            lodrec=copy.deepcopy(lod.last_lod_record) if asset is lod else None
            del J,A
            J,Ad,md=scene(rt,raw,cam,asset,True)
            assert row_hash(Ad)==ah,'Keep/delete asset rows differ'
            assert md['native_a37_packed_rows']==0 and meta['native_a6_rows_input']==0
            outputs['delete']=render(rt,cam,J);del J,Ad
            vr=dict(asset_rows=nA,keep_delete_A_rows_bitwise_exact=True,meta_keep=meta,meta_delete=md)
            if lodrec:vr['LOD']=lodrec
            if name=='HR_default':
                refs={s:down_result(o) for s,o in outputs.items()}
                yy,xx=np.where(outputs['solo']['alpha'][0]>.01)
                boxh=(max(0,(int(xx.min())-48)//4*4),max(0,(int(yy.min())-48)//4*4),min(1920,(int(xx.max())+52)//4*4),min(1280,(int(yy.max())+52)//4*4))
                boxl=tuple(x//4 for x in boxh)
                mask=refs['solo']['alpha'][0]>.01
                interior=binary_erosion(refs['solo']['alpha'][0]>.99,iterations=2)
                save_outputs(per/name,outputs,eh,boxh)
                if raw==106:hr_identity=outputs
                else:del outputs
            elif name=='HR_LOD_reference_replay':
                vr['canonical_grid_replay_maxabs']={s:{k:float(abs(outputs[s][k]-hr_identity[s][k]).max()) for k in ('rgb','alpha','asset_rgb','asset_alpha','depth_sum')} for s in outputs}
                assert max(v for q in vr['canonical_grid_replay_maxabs'].values() for v in q.values())==0
                save_outputs(per/name,outputs,eh,boxh);del outputs,hr_identity
            else:
                vr['vs_D4HR']={s:metrics(o,refs[s],mask,interior,rt) for s,o in outputs.items()}
                vr['real_observation']={s:actual_observation(o,d,rt) for s,o in outputs.items()}
                save_outputs(per/name,outputs,el,boxl)
                lr_outputs[name]=outputs
            record['variants'][name]=vr
        box=boxl
        board([crop(lr_outputs[n]['solo']['rgb'],box) for n in ('default','area','LOD')]+[crop(refs['solo']['rgb'],box)],
              ['final native480','final area480 control','fixed S/F LOD480','D4 actual canonical HR'],per/'solo_sampling_comparison.png')
        board([crop(lr_outputs[n]['delete']['rgb']+(1-lr_outputs[n]['delete']['alpha'])*el,box) for n in ('default','area','LOD')],
              ['final native480 street','final area480 street','fixed S/F LOD480 street'],per/'street_sampling_comparison.png')
        if raw==113:
            Jo,Ao,mo=scene(rt,raw,lr,old,True);oo=render(rt,lr,Jo);del Jo,Ao
            np.savez_compressed(per/'old_delete_crop_float.npz',**{k:crop(v,box) for k,v in oo.items()},crop_xyxy=np.array(box))
            board([crop(oo['rgb']+(1-oo['alpha'])*el,box),crop(lr_outputs['default']['delete']['rgb']+(1-lr_outputs['default']['delete']['alpha'])*el,box),crop(lr_outputs['LOD']['delete']['rgb']+(1-lr_outputs['LOD']['delete']['alpha'])*el,box)],
                  ['old formal native480','C2 + matte native480','C2 + matte LOD480 (control)'],DEST/'before_after_LOD_control.png')
            board([crop(oo['rgb']+(1-oo['alpha'])*el,box),crop(lr_outputs['default']['delete']['rgb']+(1-lr_outputs['default']['delete']['alpha'])*el,box)],
                  ['old formal native480','C2 + matte native480'],DEST/'before_after_without_LOD.png')
        record['crop_HR_xyxy']=boxh;record['crop_LR_xyxy']=boxl
        assert lod.assert_persistent_unchanged() and rt.modelunchanged()
        assert version==(base.coefficients._version,base.opacity._version)
        write(per/'receipt.json',record);records.append(record)
        print(json.dumps(dict(stage='LOD_view_complete',raw=raw,du=record['variants']['LOD']['LOD']['du'],chart_rows=record['variants']['LOD']['LOD']['chart_rows'],
            solo={n:record['variants'][n]['vs_D4HR']['solo'] for n in ('default','area','LOD')},calls=CALLS,seconds=time.time()-start)),flush=True)
        del d,lr_outputs,refs,eh,el;rt._data.clear();torch.cuda.empty_cache()
    assert before=={str(p):sha(p) for p in source}
    result=dict(status='PASS_FIXED_THREE_VIEW_LOD_TEST_NOT_AUTOMATIC_ADOPTION',raws=list(RAWS),sources=before,views=records,
        source_SF_hash=lod.persistent_SF_hash,render_calls=CALLS,seconds=time.time()-start,
        peak_cuda_allocated=torch.cuda.max_memory_allocated(),peak_cuda_reserved=torch.cuda.max_memory_reserved(),
        kernel_and_global_neg_fov_unchanged=True,training_steps=0,PVD_weight_updates=0,geometry_fit_steps=0,colour_fit_steps=0,
        source_asset_opacity_updates=0,source_retained_rows_unchanged=True,keep_delete_sampling_independent=True,
        method='Only camera-frequency sampling and fixed premultRGBA mip filtering of persistent S/F; .99 peak opacity, no area attenuation',
        self_consistency_reference='D4 actual canonical HR premultRGB/alpha, not hidden appearance/geometry truth',
        dataset_role='internal development diagnostics, not independent final confirmation',automatic_adoption=False)
    write(DEST/'receipt.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','views')},indent=2),flush=True)

if __name__=='__main__':main()
