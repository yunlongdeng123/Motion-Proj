"""Fixed-geometry candidate-matte safety control on the old formal material.

This is a diagnosis, not the final merge: all40BUILD observed rows and real106
rows are protected; the new continuous-surface branch restarts from prototype.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import hashlib,json,time
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
import torch

RUN=Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_a6_appearance_formal_20261105')
ROOT=RUN.parent/'scene008_a6_insert_artifacts_20261105/vehicle/matte_contract'
BASE=RUN/'output_adapter/projective_surface/formal_observed/formal/complete_a6.npz'
PROTO=RUN/'output_adapter/projective_surface/projective_surface_asset.npz'
sys.path.insert(0,str(RUN/'source'))
from highres_fit import camera_hr
from scene_runtime import Runtime,raster
from sh_asset import SHAsset,hash_geometry,C0
from full_pvd import GEOMETRY_FIELDS


def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def cpu(t):return t.detach().cpu().numpy()
def rgb8(a):return np.uint8(np.clip(a,0,1)*255+.5)
def stats(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return dict(count=0)
    q=np.quantile(a,[0,.5,.9,.99,1]);return dict(count=len(a),min=q[0],median=q[1],p90=q[2],p99=q[3],max=q[4],mean=float(a.mean()))
def tile(images,names,box,path):
    x0,y0,x1,y1=box;w=x1-x0;h=y1-y0
    out=Image.new('RGB',(w*len(images),h+30),(237,243,247));draw=ImageDraw.Draw(out)
    for i,(image,title) in enumerate(zip(images,names)):
        if image.ndim==2:image=np.repeat(image[...,None],3,2)
        out.paste(Image.fromarray(rgb8(image[y0:y1,x0:x1])),(i*w,30));draw.text((i*w+5,8),title,fill=(12,25,38))
    out.save(path)
def streetpanel(images,names,path):
    panels=[]
    for im,title in zip(images,names):
        p=Image.fromarray(rgb8(im));p=p.resize((720,480),Image.Resampling.LANCZOS)
        q=Image.new('RGB',(720,510),(237,243,247));q.paste(p,(0,30));ImageDraw.Draw(q).text((8,8),title,fill=(12,25,38));panels.append(q)
    out=Image.new('RGB',(1440,1020))
    for i,p in enumerate(panels):out.paste(p,((i%2)*720,(i//2)*510))
    out.save(path)


@torch.no_grad()
def main():
    start=time.time();base=bank(BASE);proto=bank(PROTO);texture=bank(ROOT/'matte_texture_samples.npz')
    support=bank(RUN/'output_adapter/projective_surface/formal_observed/inputs.npz')
    N=len(base['xyz']);rows=texture['original_asset_row'];assert N==50322 and len(rows)==47544
    assert np.array_equal(rows,np.arange(2778,N))
    candidate_alpha=np.ones(N,np.float32);candidate_alpha[rows]=texture['sampled_alpha']
    real=base['resample_rgb_real'].astype(bool);retained=base['representation_kind']==0
    observed=~support['unknown'].astype(bool)
    protected=observed|real|retained
    free=~protected
    gate=np.ones(N,np.float32);gate[free]=candidate_alpha[free]
    assert not np.any(free&retained) and not np.any(free&real)
    out={k:v.copy() for k,v in base.items()}
    # SHAsset requires positive stored opacity;1e-8 is far below the native
    # 1/255 alpha cutoff and creates no rendered point for matte-zero canvas.
    opacity_floor=np.float32(1e-8)
    out['opacity'][free]=np.maximum(base['opacity'][free]*gate[free],opacity_floor)
    fallback=base['resample_rgb_prior_fallback'].astype(bool)&~retained
    update=free&(candidate_alpha>1e-6)&((candidate_alpha<1-1e-7)|fallback)
    corrected=np.zeros((N,3),np.float32);corrected[rows]=texture['sampled_unpremultiplied_RGB']
    out['canonical_sh'][update,0]=(corrected[update]-.5)/C0
    out['candidate_matte_alpha']=candidate_alpha
    out['candidate_matte_gate_applied']=gate
    out['candidate_matte_protected_observed']=observed
    out['candidate_matte_protected_real106']=real
    out['candidate_matte_sampling_updated']=update
    out['geometry_hash']=np.asarray(hash_geometry(out));out.pop('provenance_hash',None)
    meta=json.loads(str(base['metadata_json']));meta['matte_only_diagnostic']=dict(
        source_asset=str(BASE),source_asset_sha256=sha(BASE),source_geometry_hash=str(base['geometry_hash']),
        candidate_RGB_sha256=sha(RUN/'inputs/raw106/strong_candidate_v2_512.png'),
        matte_sha256=sha(ROOT/'car_matte_512.png'),texture_samples_sha256=sha(ROOT/'matte_texture_samples.npz'),
        point_positions_depth_sigma_normals_and_count_unchanged=True,
        opacity_changes_only_on_free_unobserved_chart=True,protected40BUILD_and_real106_and_prior=True,
        stored_zero_opacity_floor=float(opacity_floor),native_alpha_cutoff=1/255,
        premult_RGB_alpha_filter_then_unpremultiply=True,PVD_training_updates=0,colour_fit_steps=0,
        final_merge_instruction='New continuous surface must use prototype initial colours and then new40BUILD colour calibration; do not permanently reuse this old-formal safety material')
    asset=SHAsset(out,meta).cuda();baseline=SHAsset.from_geometry(BASE).cuda()
    result=asset.export(ROOT/'matte_only_asset.npz');reloaded=bank(ROOT/'matte_only_asset.npz')
    for k in GEOMETRY_FIELDS:
        if k!='opacity':assert np.array_equal(base[k],reloaded[k]),k
    assert np.array_equal(base['canonical_sh'][protected],reloaded['canonical_sh'][protected])
    assert np.array_equal(base['opacity'][protected],reloaded['opacity'][protected])
    for k in ('resample_source_row','resample_source_parent','resample_reference_pixel_xy'):
        assert np.array_equal(base[k],reloaded[k])
    dark=(proto['rgb'].max(1)<16/255)&~retained
    outside=dark&(candidate_alpha<=1e-6)&~real
    inside=dark&(candidate_alpha>.99)
    partial=dark&(candidate_alpha>1e-6)&(candidate_alpha<=.99)
    markers=np.column_stack((outside,inside,partial,(candidate_alpha<=1e-6)&free))
    group=np.zeros(N,np.int64);group[real]=1;group[base['resample_rgb_generated']]=2
    group[(base['resample_rgb_prior_fallback'])&~retained]=3
    tg=torch.as_tensor(group,device='cuda');tm=torch.as_tensor(markers,device='cuda')
    def native(rt,d,a,c):
        rr=a.rows(d['pose'],c,rt.pipe,1,precomputed=True);features=rr['features'].new_zeros((N,24));features[:,:2]=rr['features']
        features[torch.arange(N,device='cuda'),2+tg]=1
        for g in range(4):features[tg==g,6+g*3:9+g*3]=rr['colors_precomp'][tg==g]
        features[:,18:22]=tm.to(features.dtype)
        return raster(c,rt.pipe,0,dict(rr,features=features),rt.n.bg.new_zeros(3))
    rt=Runtime();dest=ROOT/'evaluation';dest.mkdir(exist_ok=False);records=[]
    for raw in (106,113,130):
        per=dest/f'raw{raw}';per.mkdir();d=rt.raw(raw);c=camera_hr(d['cam']);packages={}
        for name,a in (('before',baseline),('matte_only',asset)):
            r=native(rt,d,a,c);rgb=cpu(r['rgb']).transpose(1,2,0);alpha=cpu(r['alpha'])[0];extra=cpu(r['extra'])
            assert np.abs(extra[:4].sum(0)-alpha).max()<2e-5
            packages[name]=dict(rgb=rgb,alpha=alpha,groups=extra[:4],markers=extra[16:20])
            Image.fromarray(rgb8(rgb)).save(per/f'{name}_black_hr.png')
            Image.fromarray(np.uint16(np.clip(alpha,0,1)*65535+.5)).save(per/f'{name}_alpha16_hr.png')
        b=packages['before'];m=packages['matte_only'];union=(b['alpha']>.1)|(m['alpha']>.1)
        y,x=np.where(union);side=max(380,int(max(x.max()-x.min(),y.max()-y.min())+30));cx=(x.min()+x.max()+1)/2;cy=(y.min()+y.max()+1)/2
        box=(int(cx-side/2),int(cy-side/2),int(cx-side/2)+side,int(cy-side/2)+side)
        tile([b['rgb'],m['rgb'],b['alpha'],m['alpha']],['old formal','matte only /same xyz','old alpha','matte alpha'],box,per/'black_compare.png')
        tile([b['markers'][0],m['markers'][0],b['markers'][1],m['markers'][1]],
            ['canvas-dark alphaT before','canvas-dark alphaT after','dark FG alphaT before','dark FG alphaT after'],box,per/'dark_provenance.png')
        known=np.repeat(np.repeat(d['a6_visible'].cpu().numpy(),4,0),4,1)
        record=dict(raw=raw,known_HR_RGB_delta=stats(np.abs(b['rgb']-m['rgb']).mean(2)[known]),
            known_alpha_delta_maxabs=float(np.abs(b['alpha'][known]-m['alpha'][known]).max()),
            before_opaque_dark=int(((b['alpha']>.98)&(b['rgb'].max(2)<16/255)).sum()),
            matte_opaque_dark=int(((m['alpha']>.98)&(m['rgb'].max(2)<16/255)).sum()),
            source_canvas_dark_contribution_before=float(b['markers'][0].sum()),
            source_canvas_dark_contribution_after=float(m['markers'][0].sum()),
            source_dark_foreground_contribution_before=float(b['markers'][1].sum()),
            source_dark_foreground_contribution_after=float(m['markers'][1].sum()),
            retained_max_alphaT_before=float(b['groups'][0].max()),retained_max_alphaT_after=float(m['groups'][0].max()))
        # Actual native joint scene rendering, no2D image paste.
        yy,xx=torch.meshgrid(torch.arange(1280,device='cuda',dtype=torch.float32),torch.arange(1920,device='cuda',dtype=torch.float32),indexing='ij')
        c.grid=torch.stack((xx,yy),-1);environment=rt.n.env(c.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
        scenes=[];scenechecks=[]
        for delete in (False,True):
            for name,a in (('before',baseline),('matte_only',asset)):
                sc=rt.scene(raw,a,delete=delete);rr=raster(c,rt.pipe,rt.n.static.active_sh_degree,sc['rows'],sc['bg'])
                rgb=rr['rgb']+(1-rr['alpha'])*environment
                conserve=float((rr['extra'][:4].sum(0)-rr['alpha'][0]).abs().max());assert conserve<2e-5
                assert sc['meta']['native_a6_rows_input']==0
                if delete:assert sc['meta']['native_a37_packed_rows']==0
                im=cpu(rgb).transpose(1,2,0);state='delete' if delete else 'keep'
                Image.fromarray(rgb8(im)).save(per/f'{name}_{state}_hr.png');scenes.append(im)
                scenechecks.append(dict(asset=name,state=state,group_conservation_maxabs=conserve,**sc['meta']))
                del sc,rr
        streetpanel(scenes,['old /keep','matte /keep','old /delete','matte /delete'],per/'street_compare.png')
        record['native_scenes']=scenechecks;write(per/'metrics.json',record);records.append(record)
        print(json.dumps(dict(stage='matte_render_complete',**record)),flush=True)
        del packages,scenes,environment;rt._data.clear();torch.cuda.empty_cache()
    assert rt.modelunchanged() and asset.assert_geometry_unchanged() and baseline.assert_geometry_unchanged()
    assert sha(BASE)==meta['matte_only_diagnostic']['source_asset_sha256']
    np.savez_compressed(ROOT/'matte_application.npz',candidate_alpha=candidate_alpha,applied_gate=gate,
        free_chart=free,protected_observed40=observed,protected_real106=real,protected_retained=retained,
        corrected_colour_updated=update,source_canvas_dark=outside,source_dark_foreground=inside,
        original_opacity=base['opacity'],new_opacity=reloaded['opacity'])
    receipt=dict(status='PASS_MATTE_ONLY_SAFETY_CONTROL',asset=result,source_asset=str(BASE),
        point_geometry_and_count_unchanged=True,only_GEOMETRY_FIELD_changed='opacity',
        protected_point_coefficients_and_opacity_bitwise_exact=True,protected_observed40_rows=int(observed.sum()),
        protected_real106_rows=int(real.sum()),free_chart_rows=int(free.sum()),
        opacity_changed_rows=int((base['opacity']!=reloaded['opacity']).sum()),
        free_chart_alpha_zero=int((free&(candidate_alpha<=1e-6)).sum()),
        free_chart_alpha_partial=int((free&(candidate_alpha>0)&(candidate_alpha<1)).sum()),
        source_dark_canvas_rows_suppressed=int((free&outside).sum()),
        dark_foreground_rows_alpha_gt099_preserved=int(inside.sum()),
        colour_edge_or_fallback_rows_corrected=int(update.sum()),
        no_brightness_deletion_rule=True,premult_alpha_unpremultiply=True,
        input_teacher_sha256=sha(RUN/'inputs/raw106/strong_candidate_v2_512.png'),
        matte_sha256=sha(ROOT/'car_matte_512.png'),frames=records,
        PVD_weight_updates=0,colour_optimizer_updates=0,old_assets_modified=False,
        final_merge='Use prototype initial colours + new continuous depth surface, then re-fit40BUILD; this diagnostic preserves old formal only for safety attribution',
        seconds=time.time()-start,peak_vram_bytes=int(torch.cuda.max_memory_allocated()))
    write(ROOT/'application_receipt.json',receipt)
    print(json.dumps(dict(stage='matte_control_complete',**receipt)),flush=True)


if __name__=='__main__':main()
