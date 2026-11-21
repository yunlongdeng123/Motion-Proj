"""从已认证真实观测中检索完整道路高斯块；CPU筛选，不重算测距、不调用扩散。"""
from __future__ import annotations
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for _k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[_k]='2'
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy.ndimage import binary_dilation,maximum_filter,uniform_filter,label,gaussian_filter
from scipy.spatial import cKDTree
from PIL import Image,ImageDraw

C=Path('/root/autodl-tmp/drivingeditor_repro/campaign')
RUN=C/'road_gaussian_transfer_20261122_r1';OUT=RUN/'source_selection'
S=C/'road_multisource_20261122_r1/sources'
ASSET=C/'road_surface_residual_20261121_r2/appearance_fit/asset'
RENDERS=ASSET.parent/'rendered'
BANK=Path('/root/autodl-tmp/codex_tmp/waymo_val065_allframes_20261109_r1/inputs')
CERT=C/'road_surface_adaptive_20261121_r1/proposal/surface_certificate.json'
CELL=.04
SIZES=((.6,.6),(.8,.8),(1.,1.),(1.2,.8),(1.2,1.2),(1.6,1.))

def read(p):return json.loads(Path(p).read_text())
def load(p):
    with np.load(p,allow_pickle=False) as z:return {k:z[k] for k in z.files}
def write(p,d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def save(p,**d):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**d)
    p.with_suffix('.tmp').replace(p)
def srgb(c):return np.where(c<=.0031308,c*12.92,1.055*np.maximum(c,0)**(1/2.4)-.055)
def as_hwc(a):return a.transpose(1,2,0) if a.ndim==3 and a.shape[0]==3 else a
def img(a):return Image.fromarray(np.rint(np.clip(a,0,1)*255).astype(np.uint8))
def project(X,K,T):
    q=X@T[:3,:3].T+T[:3,3];p=q@K.T
    return p[:,:2]/np.maximum(p[:,2:],1e-8)-.5,q[:,2]
def bilinear(a,uv):
    from scipy.ndimage import map_coordinates
    return np.stack([map_coordinates(a[:,:,ch],[uv[:,1],uv[:,0]],order=1,mode='nearest') for ch in range(3)],1)
def stats(v):
    v=np.asarray(v)
    return dict(count=int(v.size),min=float(v.min()),median=float(np.median(v)),p95=float(np.quantile(v,.95)),max=float(v.max())) if v.size else dict(count=0)

def tangent_frame(X,n):
    n=np.asarray(n,float);n/=np.linalg.norm(n);O=np.median(X,axis=0)
    _,_,V=np.linalg.svd(X-O,full_matrices=False);u=V[0]-n*(V[0]@n);u/=np.linalg.norm(u)
    if u[np.argmax(abs(u))]<0:u=-u
    v=np.cross(n,u);R=np.stack((u,v,n),1)
    T=np.eye(4);T[:3,:3]=R;T[:3,3]=O
    return O,R,T

def make_grid(uv,good,risk):
    keys=np.floor(uv[:,:2]/CELL).astype(np.int64);low=keys[good|risk].min(0);high=keys[good|risk].max(0)
    shape=high-low+1;inside=((keys>=low)&(keys<=high)).all(1);safe=np.clip(keys-low,0,shape-1)
    occupied=np.zeros(shape,bool);paint=np.zeros(shape,bool)
    p=safe[good];occupied[p[:,0],p[:,1]]=True;p=safe[risk];paint[p[:,0],p[:,1]]=True
    paint=binary_dilation(paint,iterations=int(np.ceil(.08/CELL)))
    return keys,low,shape,safe,inside,occupied,paint

def target_interface(g,frame):
    manifest=read(C/'road_multisource_20261122_r1/contract.json');raws=manifest['train_frames']
    X=[];N=[];R=[];P=[];role=[];counts=[]
    for raw in raws:
        a=load(S/f'targets/{raw:03d}.npz');cs=a['requested_C']|a['requested_S']
        mask=cs&a['geometry_valid']&~a['same_mask']&~a['K'];y,x=np.nonzero(mask)
        X.append(a['world_xyz'][mask]);N.append(a['normal_world'][mask]);R.append(np.full(len(y),raw,np.int16));P.append(np.c_[x,y].astype(np.int32))
        role.append(np.where(a['requested_C'][mask],1,2).astype(np.uint8))
        counts.append(dict(raw=raw,CS=int(cs.sum()),geometry_valid_CS=int((cs&a['geometry_valid']).sum()),
                           same_position_preserved=int((cs&a['same_mask']).sum()),target_query_count=len(y),
                           unsupported_CS=int((cs&~a['geometry_valid']).sum())))
    X=np.concatenate(X);uv=(X-frame[:3,3])@frame[:3,:3]
    save(OUT/'target_interface.npz',world_xyz=X,normal_world=np.concatenate(N),raw=np.concatenate(R),pixel_xy=np.concatenate(P),role=np.concatenate(role),
         tangent_uv=uv[:,:2],source_tangent_frame_world=frame,role_names=np.array(['C_hole','S_shadow']))
    write(OUT/'target_interface.json',dict(complete=True,inputs='Existing fixed C/S masks intersect same frozen geometry; no manual target world ROI.',rows=counts,
          same_position_priority_preserved=True,unknown_geometry_policy='No Gaussian transfer query invented outside certified surface.'))
    return X,uv[:,:2],counts

def current_native_views(g,f,clean):
    views={};N=len(g['world_xyz'])
    for raw in np.unique(g['target_raw']):
        raw=int(raw);cache=RENDERS/f'{raw:03d}.npz'
        if not cache.exists():continue
        a=load(cache);h=load(S/f'maps/{raw:03d}.npz');ob=load(S/f'qualified_observations/{raw:03d}.npz')
        q=np.flatnonzero(g['target_raw']==raw);qual=np.zeros(N,bool);qual[ob['query_id']]=True
        px=g['target_flat_pixel'][q];obs=read(BANK/f'observations/{raw:03d}.json')
        photo=np.asarray(Image.open(BANK/obs['image_path']).resize((1152,768),Image.Resampling.LANCZOS),np.float32)/255
        rgb=as_hwc(a['road_rgb']);alpha=a['road_alpha'];eff=np.divide(rgb,alpha[:,:,None],out=np.zeros_like(rgb),where=alpha[:,:,None]>1e-8)
        joint=as_hwc(a['joint_rgb']);p=photo.reshape(-1,3)[px];r=eff.reshape(-1,3)[px];j=joint.reshape(-1,3)[px]
        exact=clean[q]&qual[q]&(alpha.ravel()[px]>=.95)&~a['K'].ravel()[px]
        hf_photo=photo-gaussian_filter(photo,(1.2,1.2,0));hf_native=eff-gaussian_filter(eff,(1.2,1.2,0))
        views[raw]=dict(query_id=q,valid=exact,photo=p,native=r,joint=j,alpha=alpha.ravel()[px],
           rgb_error=abs(r-p).mean(1),joint_error=abs(j-p).mean(1),channel_error=abs(r-p).max(1),
           hf_photo=hf_photo.reshape(-1,3)[px],hf_native=hf_native.reshape(-1,3)[px],
           K=h['camera_K'],T=h['world_to_camera'],photo_image=photo,native_image=eff,joint_image=joint)
    return views

def photo_metrics(ids,views):
    m=np.zeros(max(int(v['query_id'].max()) for v in views.values())+1,bool);m[ids]=True;rows=[]
    for raw,v in views.items():
        use=v['valid']&m[v['query_id']]
        if use.sum()<32:continue
        a=v['hf_photo'][use].ravel();b=v['hf_native'][use].ravel();corr=float(np.corrcoef(a,b)[0,1]) if a.std()>1e-6 and b.std()>1e-6 else 0.
        row=dict(raw=raw,pixels=int(use.sum()),RGB_MAE=float(v['rgb_error'][use].mean()),
                 RGB_maxchannel_p90=float(np.quantile(v['channel_error'][use],.9)),
                 joint_RGB_MAE=float(v['joint_error'][use].mean()),native_alpha_p05=float(np.quantile(v['alpha'][use],.05)),
                 photo_HF_std=float(a.std()),native_HF_std=float(b.std()),HF_correlation=corr,
                 HF_MAE=float(abs(a-b).mean()))
        row['passed']=row['RGB_MAE']<=.04 and row['RGB_maxchannel_p90']<=.1 and row['joint_RGB_MAE']<=.05
        rows.append(row)
    return sorted(rows,key=lambda r:(not r['passed'],r['RGB_MAE'],-r['pixels']))

def rectangles(occupied,paint,low):
    proposals=[];seen=set()
    for size in SIZES:
        cells=np.maximum(1,np.rint(np.array(size)/CELL).astype(int));density=uniform_filter(occupied.astype(float),size=tuple(cells),mode='constant')
        bad=maximum_filter(paint.astype(np.uint8),size=tuple(cells+2*int(np.ceil(.18/CELL))),mode='constant')>0
        maxima=density>=maximum_filter(density,size=7,mode='constant')-1e-10
        ix=np.argwhere(maxima&(density>=.70)&~bad)
        order=np.argsort(-density[tuple(ix.T)],kind='stable')
        accepted=[]
        for p in ix[order]:
            center=(low+p+.5)*CELL
            if any(np.linalg.norm(center-q)<.18 for q in accepted):continue
            accepted.append(center);key=tuple(np.rint(np.r_[center,size]/CELL).astype(int))
            if key not in seen:
                seen.add(key);proposals.append(dict(center=center,size=np.array(size),density=float(density[tuple(p)])))
            if len(accepted)>=30:break
    return proposals

def iou(a,b):
    alo=a['center']-a['size']/2;ahi=a['center']+a['size']/2;blo=b['center']-b['size']/2;bhi=b['center']+b['size']/2
    area=np.prod(np.maximum(0,np.minimum(ahi,bhi)-np.maximum(alo,blo)))
    return float(area/(np.prod(a['size'])+np.prod(b['size'])-area))

def preview(cid,center,size,O,R,plane_w,metrics,views):
    corners=np.array([[-1,-1],[1,-1],[1,1],[-1,1]])*size/2+center
    world=O+np.c_[corners,np.full(4,plane_w)]@R.T;records=[]
    for m in metrics[:2]:
        raw=m['raw'];v=views[raw];uv,z=project(world,v['K'],v['T'])
        low=np.maximum(0,np.floor(uv.min(0)-24).astype(int));high=np.minimum([1152,768],np.ceil(uv.max(0)+25).astype(int))
        if (high<=low).any() or (z<=0).any():continue
        parts=[]
        for key,title in [('photo_image','Original photograph'),('native_image','R2 native road / alpha'),('joint_image','R2 native joint')]:
            im=img(v[key]);d=ImageDraw.Draw(im);d.line([tuple(p) for p in np.r_[uv,uv[:1]]],fill=(255,210,50),width=2)
            crop=im.crop((int(low[0]),int(low[1]),int(high[0]),int(high[1])));parts.append((crop,title))
        w=max(p.width for p,_ in parts);h=max(p.height for p,_ in parts);canvas=Image.new('RGB',(3*w,h+28),(18,22,30));draw=ImageDraw.Draw(canvas)
        for j,(p,title) in enumerate(parts):canvas.paste(p,(j*w,28));draw.text((j*w+5,7),title,fill=(230,235,245))
        path=OUT/f'previews/{cid}_raw{raw:03d}.png';path.parent.mkdir(parents=True,exist_ok=True);canvas.save(path)
        records.append(dict(raw=raw,model_crop_xyxy=[*low.tolist(),*high.tolist()],source_roi_uv=uv.tolist(),comparison=str(path)))
    return world,records

def validate_selection():
    manifest=read(OUT/'manifest.json');road=load(ASSET/'road.npz');p=load(ASSET/'provenance.npz')
    g=load(S/'geometry.npz');f=load(S/'fusion.npz');rows=[]
    for entry in manifest['candidates']:
        z=load(entry['packet']);ids=z['world_row_ids'];core=z['core_mask'];q=z['fresh_query_ids']
        assert len(ids)==len(np.unique(ids)) and np.all(np.diff(ids)>0)
        assert road['active'][ids].all() and (road['minimum_model_depth'][ids]==0).all()
        for k in ['normal_world','gaussian_sigma_m','road_opacity','reference_srgb','minimum_model_depth']:
            assert np.array_equal(z[k],road[k][ids]),(entry['candidate_id'],k)
        assert np.array_equal(z['world_xyz'],p['world_xyz'][ids])
        T=z['source_tangent_frame_world'];reconstructed=z['local_tangent_xyz']@T[:3,:3].T+z['source_center_world']
        error=float(abs(reconstructed-z['world_xyz']).max());assert error<1e-9
        assert f['real_same_position'][q].all() and (f['source_count'][q]>=5).all()
        assert np.array_equal(g['world_xyz'][q],z['fresh_world_xyz'])
        # A nearest fresh point is a geometric support descriptor, never a new material-source truth flag.
        tree=cKDTree(z['fresh_world_xyz']);distance,near=tree.query(z['world_xyz'],workers=2)
        sidecar=Path(entry['packet']).with_name(entry['candidate_id']+'_support_audit.npz')
        fields={key:p[key][ids] for key in ['source','known','effective_material_source','effective_material_available'] if key in p}
        save(sidecar,world_row_ids=ids,nearest_fresh_query_id=q[near],nearest_fresh_query_distance_m=distance,
             source_prior_fields_are_observed_truth=np.array(False),**{'legacy_'+k:v for k,v in fields.items()})
        row=dict(candidate_id=entry['candidate_id'],passed=True,gaussian_parameter_copy_exact=True,
                 local_to_world_reconstruction_max_m=error,core_gaussians=int(core.sum()),support_gaussians=len(ids),
                 fresh_support_distance_core_m=stats(distance[core]),fresh_support_distance_halo_m=stats(distance[~core]),
                 legacy_known_core_counts={str(k):int(v) for k,v in zip(*np.unique(p['known'][ids[core]],return_counts=True))},
                 source_prior_ledger_not_promoted=True,native_isolated_block_verified=False,support_audit=str(sidecar))
        rows.append(row)
    write(OUT/'validation.json',dict(complete=True,passed=True,rows=rows,scope='CPU input integrity and fresh-reference traceability; native isolated block fidelity remains a separate root gate.',GPU_calls=0,new_diffusion_calls=0,training_updates=0))
    return rows

def write_native_requests():
    manifest=read(OUT/'manifest.json');bank_ids={int(r['raw']):r['id'] for r in read(BANK/'receipt.json')['frames']};requests=[]
    for item in manifest['candidates']:
        cached=[r['raw'] for r in item['cached_native_photo_views'] if r['passed']]
        frames=cached[:2]
        for witness in item['original_RGB_witnesses']:
            if len(frames)>=2:break
            if witness['raw'] not in frames:frames.append(witness['raw'])
        for raw in frames:
            d=read(BANK/f'observations/{raw:03d}.json')
            cal={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in (BANK/'calib'/(bank_ids[raw]+'.txt')).read_text().splitlines()}
            K=cal['P0'].reshape(3,4)[:,:3].copy();K[:2]*=1152/d['camera_size'][0]
            T=np.eye(4);T[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);T=T@np.linalg.inv(np.array(d['ego_world']).reshape(4,4))
            requests.append(dict(candidate_id=item['candidate_id'],packet=item['packet'],raw=raw,camera_K=K.tolist(),world_to_camera=T.tolist(),
                 render_wh=[1152,768],original_RGB=str(BANK/d['image_path']),cached_full_parent_render=str(RENDERS/f'{raw:03d}.npz') if (RENDERS/f'{raw:03d}.npz').exists() else None,
                 roi_world_corners=item['world_roi_corners'],core_gaussian_count=item['core_gaussians'],support_gaussian_count=item['support_gaussians']))
    write(OUT/'native_verification_requests.json',dict(complete=True,source_asset=str(ASSET),requests=requests,
         renderer_input='Exact R2 parent packet subset indexed by world_row_ids; preserve order, RGB, XYZ, normal, sigma and opacity. Verify isolated core against full parent before accepting transfer.',
         photograph_check='Read original PNG at the provided camera; compare certified witness UV samples and the visible projected core/halo. Audit the halo for markings, shadows or foreign objects.',
         source_qualification_recomputed=False,new_diffusion_calls=0,GPU_calls_by_selector=0,training_updates=0))
    return requests

def main():
    start=time.monotonic();OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'manifest.json').exists():raise FileExistsError('Preserve existing selection; use a new explicitly versioned output for changed policy.')
    g=load(S/'geometry.npz');f=load(S/'fusion.npz');p=load(ASSET/'provenance.npz');road=load(ASSET/'road.npz')
    X=g['world_xyz'];color=srgb(f['reference_common_linear']);L=f['reference_common_linear']@np.array([.2126,.7152,.0722])
    asphalt=f['real_same_position']&(color.max(1)<.65)&(np.ptp(color,axis=1)<.18)
    base_luma=float(np.quantile(L[asphalt],.75));clean=asphalt&(L<1.25*base_luma)&(L>.035)&(f['source_count']>=5)&(f['confidence']>=.6)
    risk=f['real_same_position']&((color.max(1)>=.65)|(np.ptp(color,axis=1)>=.18)|(L>1.35*base_luma))
    O,R,T=tangent_frame(X[clean],read(CERT)['plane_normal_world']);uv=(X-O)@R
    keys,low,shape,safe,inside,occupied,paint=make_grid(uv,clean,risk)
    _,components=label(occupied);target_X,target_uv,target_counts=target_interface(g,T);target_tree=cKDTree(target_uv)
    ids=np.flatnonzero(road['active']);wx=p['world_xyz'][ids];guv=(wx-O)@R;tree=cKDTree(guv[:,:2])
    support=p.get('adaptive_effective_surface_supported',p['surface_supported'])[ids]
    sig=road['gaussian_sigma_m'][ids];nc=abs(road['normal_world'][ids]@R[:,2]);op=road['road_opacity'][ids]
    good_geometry=support&(nc>=.9)&(sig[:,:2].min(1)>=.002-1e-8)&(sig[:,:2].max(1)<=.035)&(sig[:,2]<=.006)&(op>=.95)&(road['minimum_model_depth'][ids]==0)
    radii=4.5*sig.max(1)+.04;maxradius=float(radii.max())
    views=current_native_views(g,f,clean);proposals=rectangles(occupied,paint,low)
    audit=[];eligible=[]
    for number,prop in enumerate(proposals):
        center,size=prop['center'],prop['size'];q=np.flatnonzero(clean&(abs(uv[:,:2]-center)<=size/2).all(1))
        rows=np.sort(np.asarray(tree.query_ball_point(center,float(np.linalg.norm(size/2+maxradius))),np.int64))
        core=(abs(guv[rows,:2]-center)<=size/2).all(1)
        include=(abs(guv[rows,:2]-center)<=size/2+radii[rows,None]).all(1);rows=rows[include];core=core[include]
        if not len(rows) or not core.any():continue
        counts=photo_metrics(q,views);goodviews=[m for m in counts if m['passed']]
        height=guv[rows,2];hm=float(np.median(height[core]));height_p95=float(np.quantile(abs(height[core]-hm),.95))
        distance=target_tree.query(guv[rows[core],:2],workers=2)[0]
        core_quality=float(good_geometry[rows[core]].mean());halo_quality=float(good_geometry[rows].mean())
        reasons=[]
        if core_quality<.98:reasons.append('core_gaussian_geometry_scale_opacity_quality')
        if halo_quality<.95:reasons.append('halo_gaussian_geometry_scale_opacity_quality')
        if np.any(road['minimum_model_depth'][ids[rows]]!=0):reasons.append('camera_depth_cropped_rows_present')
        if height_p95>.04:reasons.append('nonplanar_source_core')
        if distance.min()<.08:reasons.append('source_overlaps_target_world_domain')
        if len(goodviews)<1:reasons.append('no_cached_native_photo_fidelity_pass')
        if len(q)<100:reasons.append('insufficient_fresh_dense_queries')
        source_count=float(np.median(f['source_count'][q])) if len(q) else 0.
        fidelity=min((r['RGB_MAE'] for r in goodviews),default=1.)
        texture=max((r['photo_HF_std']*max(r['HF_correlation'],0) for r in goodviews),default=0.)
        score=float(np.prod(size)*prop['density']*(1-fidelity/.08)*min(source_count/8,1)*(1+10*texture))
        row=dict(proposal_id=number,center=center.tolist(),size=size.tolist(),occupied_fraction=prop['density'],fresh_query_count=len(q),
                 core_gaussians=int(core.sum()),support_gaussians=len(rows),core_geometry_good_fraction=core_quality,
                 support_geometry_good_fraction=halo_quality,core_height_abs_deviation_p95_m=height_p95,
                 minimum_core_to_target_distance_m=float(distance.min()),source_witness_count_median=source_count,
                 cached_native_photo_views=counts,eligible_CPU=not reasons,rejection_reasons=reasons,score=score)
        audit.append(row)
        if not reasons:eligible.append(dict(row=row,center=center,size=size,query=q,active_local_ids=rows,core=core,plane_w=hm,metrics=counts,score=score))
    eligible.sort(key=lambda x:-x['score']);chosen=[]
    for a in eligible:
        if any(iou(a,b)>.25 for b in chosen):continue
        chosen.append(a)
        if len(chosen)>=6:break
    # Preserve high-ranked overlapping alternatives for native verification if the physical source is small.
    alternatives=[a for a in eligible if not any(a is b for b in chosen)][:6]
    full_evidence=[load(S/f'evidence/{int(raw):03d}.npz') for raw in np.unique(g['target_raw'])]
    e_q=np.concatenate([a['query_id'] for a in full_evidence]);e_raw=np.concatenate([a['source_raw'] for a in full_evidence]);e_uv=np.concatenate([a['source_uv_native'] for a in full_evidence])
    outputs=[]
    for rank,a in enumerate(chosen):
        cid=f'patch_{rank:02d}';r=a['active_local_ids'];world_ids=ids[r];q=a['query'];qm=np.zeros(len(X),bool);qm[q]=True;take=qm[e_q]
        frames,counts=np.unique(e_raw[take],return_counts=True);witness=sorted([dict(raw=int(x),query_records=int(y)) for x,y in zip(frames,counts)],key=lambda r:-r['query_records'])
        world_corners,previews=preview(cid,a['center'],a['size'],O,R,a['plane_w'],a['metrics'],views)
        source_center=O+R@np.r_[a['center'],a['plane_w']]
        save(OUT/f'candidates/{cid}.npz',world_row_ids=world_ids,core_mask=a['core'],world_xyz=wx[r],normal_world=road['normal_world'][world_ids],
             gaussian_sigma_m=sig[r],road_opacity=op[r],reference_srgb=road['reference_srgb'][world_ids],minimum_model_depth=road['minimum_model_depth'][world_ids],
             source_asset=np.array(str(ASSET)),source_tangent_frame_world=T,source_center_world=source_center,
             core_center_tangent=a['center'],core_size_m=a['size'],source_world_roi_corners=world_corners,local_tangent_xyz=guv[r]-np.r_[a['center'],a['plane_w']],
             fresh_query_ids=q,fresh_world_xyz=X[q],fresh_source_count=f['source_count'][q],fresh_confidence=f['confidence'][q],
             witness_query_id=e_q[take],witness_source_raw=e_raw[take],witness_source_uv_native=e_uv[take],model_from_world=road['model_from_world'],
             source_quality_all_rows=good_geometry[r],source_world_margin_radius_m=radii[r])
        record=dict(a['row'],candidate_id=cid,rank=rank,packet=str(OUT/f'candidates/{cid}.npz'),
                    world_center=source_center.tolist(),world_roi_corners=world_corners.tolist(),photograph_comparisons=previews,
                    original_RGB_witnesses=witness[:12],native_isolated_block_verified=False,
                    verification='Render this exact Gaussian subset using listed source cameras, compare core photo pixels and full-parent cache; require isolation/core agreement before transfer.')
        write(OUT/f'candidates/{cid}.json',record);outputs.append(record)
    save(OUT/'source_distribution.npz',clean_query_ids=np.flatnonzero(clean),world_xyz=X[clean],tangent_xyz=uv[clean],
         source_count=f['source_count'][clean],confidence=f['confidence'][clean],occupied_grid=occupied,marking_guard_grid=paint,grid_low_key=low,cell_m=np.array(CELL),tangent_frame_world=T)
    write(OUT/'proposal_audit.json',dict(complete=True,rows=audit,overlapping_alternatives=[a['row'] for a in alternatives]))
    manifest=dict(schema='road_gaussian_transfer_source_selection.v1',complete=True,CPU_candidates_ready=bool(outputs),
         source_asset=str(ASSET),source_asset_role='R2 ordinary-Gaussian parent, no new diffusion-distilled asset',
         source_bank=str(S),new_diffusion_calls=0,GPU_calls=0,training_updates=0,measured_source_requalification_calls=0,
         source_dense_query_total=int(f['real_same_position'].sum()),clean_asphalt_queries=int(clean.sum()),
         occupied_world_grid_cells=int(occupied.sum()),occupied_area_grid_proxy_m2=float(occupied.sum()*CELL**2),connected_components=int(components),
         world_bounds=[X[clean].min(0).tolist(),X[clean].max(0).tolist()],tangent_frame_world=T.tolist(),
         selection_policy=dict(cell_m=CELL,source_count_min=5,source_confidence_min=.6,maximum_asphalt_sRGB=.65,
             maximum_asphalt_RGB_range=.18,minimum_occupied_fraction=.70,marking_guard_m=.08,
             Gaussian_support='All active R2 rows whose4.5sigma+4cm world footprint intersects the core rectangle; halo rows are retained, not silently filtered.',
             minimum_core_geometry_fraction=.98,minimum_support_geometry_fraction=.95,minimum_normal_cosine=.9,
             tangent_sigma_m=[.002,.035],normal_sigma_max_m=.006,opacity_min=.95,minimum_model_depth_required=0,
             cached_native_RGB_MAE_max=.04,cached_native_maxchannel_p90_max=.1,cached_joint_RGB_MAE_max=.05,
             source_target_world_separation_m=.08),
         candidate_count=len(outputs),eligible_CPU_proposals=len(eligible),proposals=len(audit),candidates=outputs,target_domain_rows=target_counts,
         interpretation='Similar-position Gaussian blocks are inferred donors. Dense source RGB qualifies donor location; native source fidelity and isolated block verification are separate gates. Not hidden-road ground truth.',
         pending_gate='Exact isolated source-block native rendering and original-photo inspection by root before accepting transfer.',
         shadow_evidence='Existing near-object AND local deficit shadow proxy plus upper-radiance multiview qualification. No independent shadow ground-truth labels.',
         seconds=time.monotonic()-start)
    write(OUT/'manifest.json',manifest)
    # World-domain map, using standard scientific plotting.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(11,5),facecolor='#111827');ax.set_facecolor('#111827')
    ax.scatter(uv[clean,0][::8],uv[clean,1][::8],s=2,c='#40c99b',alpha=.6,label='Fresh real asphalt queries')
    ax.scatter(target_uv[::30,0],target_uv[::30,1],s=1,c='#b780f2',alpha=.15,label='Target C/S world rays')
    for r in outputs:
        u,v=r['center'];w,h=r['size'];ax.add_patch(plt.Rectangle((u-w/2,v-h/2),w,h,fill=False,color='#ffca68',linewidth=2));ax.text(u,v,r['candidate_id'],color='white',fontsize=8)
    ax.set_aspect('equal');ax.set_xlabel('Road tangent U (m)',color='white');ax.set_ylabel('Road tangent V (m)',color='white');ax.tick_params(colors='white');ax.legend(facecolor='#1f2937',labelcolor='white');fig.tight_layout();fig.savefig(OUT/'source_distribution.png',dpi=150);plt.close(fig)
    validate_selection()
    write_native_requests()
    print(json.dumps(dict(complete=True,candidates=len(outputs),eligible=len(eligible),proposals=len(audit),seconds=manifest['seconds'])),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--validate',action='store_true');args=parser.parse_args()
    if args.validate:validate_selection()
    else:main()
