"""Read-only M2 end-to-end appearance provenance and null-space diagnostic.

The diagnostic recolouring is explicitly not a recovered material proposal.
No native/PVD asset, scene checkpoint, geometry or fit objective is changed.
"""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import json,sys,hashlib,time
from pathlib import Path
import numpy as np
import torch
from scipy.sparse import load_npz
from PIL import Image,ImageDraw

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
OLD=CAM/'scene008_a6_asset_controls_20261105'
RUN=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(OLD/'source'))
from sh_asset import SHAsset,C0
from scene_runtime import Runtime,raster

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def arr(t):return t.detach().cpu().numpy() if torch.is_tensor(t) else np.asarray(t)
def stats(a):
    a=np.asarray(a)
    return dict(n=int(a.size),min=float(a.min()),mean=float(a.mean()),max=float(a.max())) if a.size else dict(n=0)
def tile(values,names,crop,path,height=280):
    tiles=[]
    for a,name in zip(values,names):
        a=arr(a)
        if a.shape[0]==3:a=a.transpose(1,2,0)
        im=Image.fromarray(np.uint8(np.clip(a,0,1)*255+.5)).crop(crop)
        im=im.resize((round(im.width*height/im.height),height),Image.Resampling.NEAREST)
        out=Image.new('RGB',(im.width,height+34),(240,245,248));out.paste(im,(0,34))
        ImageDraw.Draw(out).text((6,9),name,fill=(18,35,47));tiles.append(out)
    out=Image.new('RGB',(sum(v.width for v in tiles),height+34));x=0
    for v in tiles:out.paste(v,(x,0));x+=v.width
    out.save(path)

@torch.no_grad()
def main():
    start=time.time();dest=RUN/'audit';dest.mkdir(exist_ok=True);(dest/'frames').mkdir(exist_ok=True)
    path=OLD/'m2/complete_a6.npz';before=sha(path)
    A=load_npz(OLD/'m2/design.npz');inputs=dict(np.load(OLD/'m2/inputs.npz'))
    geometry=dict(np.load(OLD/'geometry_dense.npz'));rgb=dict(np.load(path))['rgb']
    zero=A.getnnz(axis=0)==0;weak=(~zero)&inputs['inferred'];strong=(~zero)&~inputs['inferred']
    assert not np.any(zero & ~inputs['inferred'])
    prior=inputs['prior'];modal,count=np.unique(prior,axis=0,return_counts=True);median=modal[count.argmax()]
    is_default=np.max(abs(prior-median),axis=1)<1e-7
    counter=rgb.copy();counter[zero]=[1,0,1]
    lhs=A@rgb;rhs=A@counter
    assert np.array_equal(lhs,rhs),'Zero columns must not alter even one BUILD fit prediction'
    geo_path=CAM/'scene008_a37_C2_20261103_r1/inputs/a6_geometry_minimal.npz'
    obs=dict(np.load(geo_path));minz=float(obs['canonical_xyz'][:,2].min())
    lower=geometry['xyz'][:,2]<minz
    summary=dict(status='PASS_DIAGNOSTIC',asset_sha256=before,geometry_hash=str(inputs['geometry_hash']),
       design_sha256=sha(OLD/'m2/design.npz'),fit_pixels=A.shape[0],points=A.shape[1],
       zero_column_points=int(zero.sum()),weak_points=int(weak.sum()),multi_frame_mass_supported_points=int(strong.sum()),
       default_fallback_rgb=median.tolist(),default_prior_points=int(is_default.sum()),
       zero_column_final_equals_prior_points=int((np.max(abs(rgb-prior),axis=1)[zero]<2e-7).sum()),
       zero_column_prior_is_global_default_points=int((zero&is_default).sum()),
       original_real199_lowest_point_z_m=minz,proposal_below_real199_lowest_z=int(lower.sum()),
       proposal_below_real199_lowest_z_zero_columns=int((lower&zero).sum()),
       lower_final_colour_range=[rgb[lower].min(0).tolist(),rgb[lower].max(0).tolist()],
       exact_zero_column_recolour_BUILD_prediction_unchanged=True,
       interpretation='Changing all zero-column colours leaves EVERY BUILD fit prediction exact. Such material is unidentifiable under this observation contract; magenta is only diagnostic.',
       source_real199_geometry_sha256=sha(geo_path),geometry_changed=False,training_steps=0,assets_adopted=0)
    bins=np.linspace(float(geometry['xyz'][:,2].min()),float(geometry['xyz'][:,2].max()),11)
    summary['vertical_bins']=[dict(z_range_m=[float(a),float(b)],points=int(((geometry['xyz'][:,2]>=a)&(geometry['xyz'][:,2]<(b if i<9 else b+1))).sum()),
       zero_points=int((((geometry['xyz'][:,2]>=a)&(geometry['xyz'][:,2]<(b if i<9 else b+1)))&zero).sum())) for i,(a,b) in enumerate(zip(bins[:-1],bins[1:]))]
    np.savez_compressed(dest/'appearance_provenance.npz',xyz=geometry['xyz'],rgb=rgb,prior=prior,
       no_RGB_fit_support=zero,weak_RGB_fit_support=weak,multiframe_RGB_fit_support=strong,
       global_fallback_prior=is_default,below_all199_observed_geometry=lower,
       support_mass=inputs['mass'],support_views=inputs['views'],design_nnz=A.getnnz(axis=0),
       category=np.where(zero,0,np.where(weak,1,2)).astype(np.uint8))
    rt=Runtime();asset=SHAsset.from_geometry(path).cuda();records=[];maxcon=0.
    for raw in range(89,138):
        d=rt.raw(raw);rows=asset.rows(d['pose'],d['cam'],rt.pipe,16,precomputed=False)
        f=rows['features'].new_zeros((len(asset.xyz),24));f[:,:2]=rows['features'][:,:2]
        for j,mask in enumerate((zero,weak,strong,lower)):f[:,2+j]=f.new_tensor(mask.astype(np.float32))
        out=raster(d['cam'],rt.pipe,3,dict(rows,features=f),rt.n.bg.new_zeros(3))
        al=arr(out['alpha'][0]);group=arr(out['extra'][:4]);colour=arr(out['rgb']);
        maxcon=max(maxcon,float(abs(group[:3].sum(0)-al).max()))
        support=al>.5;visible=arr(d['a6_visible']).astype(bool);reviewed=arr(d['obs']['visible_reviewed'][0]).astype(bool)
        # Exact alpha*T from same sort, source categories partition the whole object.
        rec=dict(raw=raw)
        for key,mask in (('all_asset_pixels',support),('confirmed_visible',visible),('outside_reviewed_visible',support&~reviewed)):
            den=al[mask].sum()
            rec[key]=dict(pixels=int(mask.sum()),zero_support_contribution_fraction=float(group[0,mask].sum()/den) if den else None,
                weak_support_contribution_fraction=float(group[1,mask].sum()/den) if den else None,
                below_real199_geometry_fraction=float(group[3,mask].sum()/den) if den else None)
        records.append(rec)
        if raw in (105,113,130):
            diagcol=asset.rgb.clone();diagcol[torch.as_tensor(zero,device='cuda')]=diagcol.new_tensor([1.,0.,1.])
            cr=dict(rows,shs=None,colors_precomp=diagcol,features=f)
            diag=raster(d['cam'],rt.pipe,0,cr,rt.n.bg.new_zeros(3))
            assert torch.equal(diag['alpha'],out['alpha']) and torch.equal(diag['depth_sum'],out['depth_sum'])
            palette=np.array([[1,0,1],[1,.72,0],[.1,.95,.25]],np.float32)
            supportcol=np.einsum('ghw,gc->chw',group[:3],palette)
            lowercol=np.repeat(group[3:4],3,0)
            yy,xx=np.where((al>.01)|visible);crop=(max(0,int(xx.min())-8),max(0,int(yy.min())-8),min(480,int(xx.max())+9),min(320,int(yy.max())+9))
            tile([d['rgb']*d['a6_visible'][None],colour,diag['rgb'],supportcol,lowercol],
                ['Real observed RGB','M2 unchanged','NO-RGB points -> magenta','RGB support provenance','Below lowest real199 point'],crop,dest/'frames'/f'{raw:03d}_provenance.png')
            np.savez_compressed(dest/'frames'/f'{raw:03d}_replay.npz',rgb=colour,alpha=al,category_alphaT=group,visible=visible)
        rt._data.clear();torch.cuda.empty_cache()
    assert rt.modelunchanged() and asset.assert_geometry_unchanged() and sha(path)==before and maxcon<2e-5
    summary.update(frames=records,group_conservation_maxabs=maxcon,all49_complete=True,seconds=time.time()-start,
        frozen_native_road_environment_unchanged=True,original_M2_bytes_unchanged=True)
    write(dest/'current_audit.json',summary);print(json.dumps({k:v for k,v in summary.items() if k not in ('frames','vertical_bins')}),flush=True)

if __name__=='__main__':main()
