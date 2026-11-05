"""Short native-GPU artifact isolation, independent diagnostic run only.

No fit, no PVD updates, no old asset edits. A flat-depth chart is an ORACLE
to isolate geometric depth/sort effects, never a proposed physical solution.
"""
import os
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[name]='1'
import hashlib,json,time
from pathlib import Path
import sys
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import map_coordinates,distance_transform_edt,label
import torch

CAM=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=CAM/'scene008_a6_appearance_formal_20261105'
DEST=CAM/'scene008_a6_insert_artifacts_20261105/vehicle'
SURFACE=RUN/'output_adapter/projective_surface'
PROTO=SURFACE/'projective_surface_asset.npz'
FORMAL=SURFACE/'formal_observed/formal/complete_a6.npz'
STRONG=RUN/'inputs/raw106/strong_candidate_v2_512.png'
SD=RUN/'candidates_raw106/prompt2_seed3/composed.png'
sys.path.insert(0,str(RUN/'source'))
from highres_fit import camera_hr
from scene_runtime import Runtime,raster,select_rows
from sh_asset import SHAsset,hash_geometry


def bank(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rgb8(a):return np.uint8(np.clip(a,0,1)*255+.5)
def stats(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return dict(count=0)
    q=np.quantile(a,[0,.5,.9,.99,1]);return dict(count=len(a),min=q[0],median=q[1],p90=q[2],p99=q[3],max=q[4],mean=float(a.mean()))
def project(model,P):
    hp=np.c_[model,np.ones(len(model))]@P
    return ((hp[:,:2]/hp[:,3,None]+1)*[1920,1280]-1)/2
def tile(images,names,box,path):
    x0,y0,x1,y1=box;w=x1-x0;h=y1-y0
    out=Image.new('RGB',(w*len(images),h+30),(237,243,247));dr=ImageDraw.Draw(out)
    for i,(v,name) in enumerate(zip(images,names)):
        if v.ndim==2:v=np.repeat(v[...,None],3,2)
        out.paste(Image.fromarray(rgb8(v[y0:y1,x0:x1])),(i*w,30));dr.text((i*w+5,8),name,fill=(12,25,38))
    out.save(path)


def native(rt,d,asset,cam,mask=None,custom=None):
    rows=asset.rows(d['pose'],cam,rt.pipe,1,precomputed=True)
    if custom is not None:rows=dict(rows,features=custom(rows))
    else:
        f=rows['features'];pad=f.new_zeros((len(f),24));pad[:,:2]=f;rows=dict(rows,features=pad)
    if mask is not None:rows=select_rows(rows,torch.as_tensor(mask,device='cuda'))
    out=raster(cam,rt.pipe,0,rows,rt.n.bg.new_zeros(3))
    return out


@torch.no_grad()
def main():
    DEST.mkdir(parents=True,exist_ok=False);start=time.time()
    chart=bank(SURFACE/'reference_chart.npz');p=bank(PROTO);f=bank(FORMAL)
    source_hashes={str(q):sha(q) for q in (PROTO,FORMAL,STRONG,SD)}
    N=len(f['xyz']);cut=int((f['representation_kind']==0).sum());M=N-cut
    assert (N,cut,M)==(50322,2778,47544)
    proto=SHAsset.from_geometry(PROTO).cuda();formal=SHAsset.from_geometry(FORMAL).cuda()
    assert proto.geometry_hash==formal.geometry_hash
    retained=np.arange(N)<cut;chartmask=~retained
    real=f['resample_rgb_real'].astype(bool)&chartmask
    gen=f['resample_rgb_generated'].astype(bool)&chartmask
    fallback=f['resample_rgb_prior_fallback'].astype(bool)&chartmask
    groups=np.zeros(N,np.int64);groups[real]=1;groups[gen]=2;groups[fallback]=3
    assert np.array_equal(retained|real|gen|fallback,np.ones(N,bool))
    strong=np.asarray(Image.open(STRONG).convert('RGB')).astype(np.float32)/255
    sd=np.asarray(Image.open(SD).convert('RGB')).astype(np.float32)/255
    uv=chart['generated_xy'];near=np.clip(np.floor(uv+.5).astype(int),[0,0],[511,511])
    fg=np.asarray(Image.open(RUN/'inputs/raw106/object_mask_512.png'))>0
    dark2d=strong.max(2)<16/255;labs,_=label(dark2d)
    outside_ids=np.unique(np.r_[labs[0],labs[-1],labs[:,0],labs[:,-1]])
    connected_black=np.isin(labs,outside_ids)&dark2d
    fgdist=distance_transform_edt(fg)
    dist=map_coordinates(fgdist,[uv[:,1],uv[:,0]],order=1,mode='nearest',prefilter=False)
    source_dark=np.zeros(N,bool);source_dark[cut:]=chart['color_generated']&(chart['color_rgb'].max(1)<16/255)
    external_black=np.zeros(N,bool);external_black[cut:]=chart['color_generated']&connected_black[near[:,1],near[:,0]]
    boundary_dark=np.zeros(N,bool);boundary_dark[cut:]=source_dark[cut:]&(dist<=3)
    newly_dark_real=real&(f['rgb'].max(1)<16/255)&(p['rgb'].max(1)>=16/255)
    # A constant-depth ORACLE preserves the actual ref106 centre projection
    # and each row's angular sigma. Only local depth/sort geometry changes.
    pose=chart['reference_pose'];view=chart['reference_world_view'];P=chart['reference_full_projection']
    center=chart['reference_camera_center'];xy=chart['pixel_xy'];z0=float(np.median(chart['expected_depth_model']))
    ndc=np.c_[(2*xy[:,0]+1)/1920-1,(2*xy[:,1]+1)/1280-1,np.full(M,.5),np.ones(M)]
    hp=ndc@np.linalg.inv(P);ray=hp[:,:3]/hp[:,3,None]-center
    world=center+ray*(z0/(ray@view[:3,2]))[:,None]
    local=((world-pose[:3,3])@np.linalg.inv(pose[:3,:3]).T).astype(np.float32)
    scale=float(np.linalg.norm(pose[:3,:3],axis=0).mean())
    fx=float((np.linalg.inv(view)@P)[0,0]*1920/2)
    flat_sigma=np.float32(.6*z0/fx/scale)
    oracle_assets={}
    for name,base in (('flat_proto',p),('flat_formal',f)):
        a={k:v.copy() for k,v in base.items()};a['xyz'][cut:]=local
        a['sigma'][cut:]=flat_sigma;a['thickness'][cut:]=flat_sigma
        a['geometry_hash']=np.asarray(hash_geometry(a))
        meta=json.loads(str(a['metadata_json']));meta['diagnostic_oracle']=dict(physical_solution=False,flat_ref106_camera_z_model=z0,angular_sigma_ref_HR_pixels=.6)
        oracle_assets[name]=SHAsset(a,meta).cuda()
    refworld=local.astype(float)@pose[:3,:3].T+pose[:3,3]
    xy_error=np.linalg.norm(project(refworld,P)-xy,axis=1)
    orig_sigma_px=p['sigma'][cut:]*scale*fx/chart['expected_depth_model']
    flat_z=refworld@view[:3,2]+view[3,2]
    flat_sigma_px=flat_sigma*scale*fx/flat_z
    assert xy_error.max()<.01 and np.abs(flat_sigma_px-orig_sigma_px).max()<1e-5
    write(DEST/'oracle_contract.json',dict(oracle_only=True,not_a_physical_repair=True,
        reference_xy_error_HR_px_max=float(xy_error.max()),angular_sigma_change_HR_px_max=float(np.abs(flat_sigma_px-orig_sigma_px).max()),
        flat_depth_m=z0/.1,source_depth_m=stats(chart['expected_depth_m']),
        no_RGB_change=True,no_opacity_change=True,all_variants_same_chart_point_count=M))
    tint={k:v.copy() for k,v in f.items()};tint_coeff=tint['canonical_sh'].copy()
    tint_coeff[source_dark]=0;tint_coeff[source_dark,0]=(np.array([1.,0.,1.])-.5)/.28209479177387814
    tint['canonical_sh']=tint_coeff
    tint_asset=SHAsset(tint,json.loads(str(tint['metadata_json']))).cuda()
    group_tensor=torch.as_tensor(groups,device='cuda')
    marker_tensor=torch.as_tensor(np.column_stack((source_dark,external_black,boundary_dark,newly_dark_real)),device='cuda')
    def custom(rows):
        features=rows['features'].new_zeros((N,24));features[:,:2]=rows['features']
        idx=torch.arange(N,device='cuda');features[idx,2+group_tensor]=1
        for g in range(4):features[group_tensor==g,6+3*g:9+3*g]=rows['colors_precomp'][group_tensor==g]
        features[:,18:22]=marker_tensor.to(features.dtype)
        return features
    rt=Runtime();records=[]
    variants=[('formal_full',formal,None),('proto_full',proto,None),('formal_chart',formal,chartmask),
        ('formal_retained',formal,retained),('formal_real',formal,real),('formal_gen',formal,gen),
        ('formal_fallback',formal,fallback),('proto_chart',proto,chartmask),
        ('flat_proto_chart',oracle_assets['flat_proto'],chartmask),('flat_formal_chart',oracle_assets['flat_formal'],chartmask),
        ('mark_source_dark',tint_asset,None)]
    for raw in (106,113,130):
        per=DEST/f'raw{raw}';per.mkdir();d=rt.raw(raw);c=camera_hr(d['cam']);images={};alphas={};fullout=None
        for name,asset,mask in variants:
            out=native(rt,d,asset,c,mask,custom if name=='formal_full' else None)
            im=out['rgb'].permute(1,2,0).cpu().numpy();alpha=out['alpha'][0].cpu().numpy()
            images[name]=im;alphas[name]=alpha
            if name=='formal_full':fullout={key:val.detach().cpu().numpy() for key,val in out.items() if key in ('rgb','alpha','extra','depth_sum')}
            del out
        union=(alphas['formal_full']>.1)|(alphas['flat_formal_chart']>.1)
        yy,xx=np.where(union);cx=(xx.min()+xx.max()+1)/2;cy=(yy.min()+yy.max()+1)/2
        side=max(380,int(max(xx.max()-xx.min(),yy.max()-yy.min())+30))
        box=(int(cx-side/2),int(cy-side/2),int(cx-side/2)+side,int(cy-side/2)+side)
        group_alpha=fullout['extra'][:4];group_rgb=fullout['extra'][4:16].reshape(4,3,1280,1920).transpose(0,2,3,1)
        markers=fullout['extra'][16:20]
        assert np.max(np.abs(group_alpha.sum(0)-alphas['formal_full']))<2e-5
        assert np.max(np.abs(group_rgb.sum(0)-images['formal_full']))<2e-5
        darkout=(images['formal_full'].max(2)<16/255)&(alphas['formal_full']>.98)
        core=(alphas['formal_full']>.98)
        metrics=dict(raw=raw,full_rgb_group_conservation_maxabs=float(np.abs(group_rgb.sum(0)-images['formal_full']).max()),
            group_alpha_conservation_maxabs=float(np.abs(group_alpha.sum(0)-alphas['formal_full']).max()),
            opaque_dark_output_pixels=int(darkout.sum()),
            source_dark_marker_mass_on_opaque_dark=stats(markers[0][darkout]),
            external_connected_black_mass_on_opaque_dark=stats(markers[1][darkout]),
            source_dark_opaque_dark_pixels_mass_gt09=int((darkout&(markers[0]>.9)).sum()),
            boundary_source_dark_contribution_pixels_gt01=int((markers[2]>.1).sum()),
            retained_contribution_max=float(group_alpha[0].max()),retained_contribution_mass=float(group_alpha[0].sum()),
            retained_visible_pixels_gt01=int((group_alpha[0]>.1).sum()),
            chart_only_vs_full_RGB_L1=stats(np.abs(images['formal_chart']-images['formal_full']).mean(2)[union]),
            raw_vs_flat_proto_RGB_L1=stats(np.abs(images['proto_chart']-images['flat_proto_chart']).mean(2)[union]),
            raw_vs_flat_formal_RGB_L1=stats(np.abs(images['formal_chart']-images['flat_formal_chart']).mean(2)[union]),
            formal_vs_proto_RGB_L1=stats(np.abs(images['formal_full']-images['proto_full']).mean(2)[union]),
            caveat='Opaque dark-marker contribution includes valid tires/seams; marker panels diagnose provenance, not a deletion rule')
        write(per/'metrics.json',metrics);records.append(metrics)
        tile([images[n] for n in ('proto_full','formal_full','formal_chart','formal_retained','mark_source_dark')],
             ['proto full','formal full','formal chart only','retained2778 only','source-dark to magenta'],box,per/'decomposition.png')
        tile([images[n] for n in ('proto_chart','flat_proto_chart','formal_chart','flat_formal_chart')],
             ['prior depth /proto','FLAT oracle /proto','prior depth /formal','FLAT oracle /formal'],box,per/'depth_oracle.png')
        tile([group_rgb[g] for g in range(4)],['actual retained RGB','actual real RGB','actual generated RGB','actual fallback RGB'],box,per/'actual_group_RGB.png')
        tile([group_alpha[g] for g in range(4)]+[markers[0],markers[2]],
             ['retained alphaT','real alphaT','gen alphaT','fallback alphaT','source-dark alphaT','boundary-dark alphaT'],box,per/'actual_group_alpha.png')
        straight=images['formal_full']/np.maximum(alphas['formal_full'][...,None],1e-6)
        tile([images['formal_full'],np.clip(straight,0,1),alphas['formal_full']],['premult RGB on black','RGB /alpha display','alpha'],box,per/'normalization.png')
        x0,y0,x1,y1=box
        np.savez_compressed(per/'actual_contribution_crop.npz',crop_xyxy=np.array(box),
            group_alpha_T=group_alpha[:,y0:y1,x0:x1],group_RGB=group_rgb[:,y0:y1,x0:x1],
            markers_alpha_T=markers[:,y0:y1,x0:x1],full_alpha=alphas['formal_full'][y0:y1,x0:x1],
            full_rgb=images['formal_full'][y0:y1,x0:x1])
        if raw==106:
            # Same1920 pixel centres and old512affine, not an enlarged board.
            gy,gx=np.mgrid[:1280,:1920];alluv=(np.stack((gx,gy,np.ones_like(gx)),-1)@chart['highres_to_generated_pixel_center'].T)[...,:2]
            teachers=[]
            for im in (strong,sd):teachers.append(np.stack([map_coordinates(im[...,j],[alluv[...,1],alluv[...,0]],order=1,mode='constant',cval=0,prefilter=False) for j in range(3)],-1))
            seed=np.zeros_like(images['formal_full']);seed.reshape(-1,3)[chart['pixels']]=chart['color_rgb']
            tile(teachers+[seed,images['proto_full'],images['formal_full']],
                ['ACTUAL hosted strongv2','SD selected p2seed3','mixed real/gen point RGB','actual106 proto native','actual106 formal native'],box,DEST/'matched_teachers_106.png')
        print(json.dumps(dict(stage='native_decomposition_complete',**metrics)),flush=True)
        del fullout,images,alphas;rt._data.clear();torch.cuda.empty_cache()
    assert rt.modelunchanged()
    for asset in (proto,formal,*oracle_assets.values(),tint_asset):assert asset.assert_geometry_unchanged()
    assert all(sha(q)==value for q,value in source_hashes.items())
    write(DEST/'receipt.json',dict(status='PASS_NATIVE_GPU_DIAGNOSIS',frames=records,source_hashes=source_hashes,
        PVD_weight_updates=0,colour_optimizer_updates=0,old_assets_modified=False,
        oracle_not_a_physical_solution=True,source_dark_not_used_as_a_tire_deletion_rule=True,
        seconds=time.time()-start,peak_vram_bytes=int(torch.cuda.max_memory_allocated())))
    files=[q for q in DEST.rglob('*') if q.is_file() and q.name!='file_manifest.json']
    write(DEST/'file_manifest.json',dict(files=[dict(path=q.relative_to(DEST).as_posix(),bytes=q.stat().st_size,sha256=sha(q)) for q in sorted(files)]))
    print(json.dumps(dict(stage='complete',status='PASS',seconds=time.time()-start)),flush=True)


if __name__=='__main__':main()
