"""新B10k的单a37道路证据、各向异性范围及周围RGB自适应校准。"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import json,time,warnings
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,binary_closing,binary_dilation,distance_transform_edt,gaussian_filter,label
from scipy.spatial import cKDTree
RUN=Path(__file__).resolve().parent.parent
OLD=RUN.parent/'shadow_a5_a37_199_20261103_r4/inputs'
BASE=RUN.parent/'scene008_a37_only_B10k_20261103_r1/evaluation'
FULL=RUN.parent/'r15_gn11_fullclip_cpu_20261026'
META=RUN.parent/'r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json'
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):return np.where(x<=.0031308,x*12.92,1.055*np.maximum(x,1e-10)**(1/2.4)-.055)
def transform(x,T):return x@T[:3,:3].T+T[:3,3]
def write(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')

def main():
    start=time.time();dest=RUN/'inputs';dest.mkdir(exist_ok=False);(dest/'views').mkdir()
    settings=json.loads((RUN/'settings.json').read_text());r=dict(np.load(OLD/'road_reference.npz'));meta=json.loads(META.read_text())
    N=len(r['keys']);ij=r['keys']-r['keys'].min(0);shape=tuple(ij.max(0)+1)
    def chart(x):
        z=np.zeros(shape,dtype=x.dtype);z[ij[:,0],ij[:,1]]=x;return z
    def gather(x):return x[ij[:,0],ij[:,1]]
    bad=np.zeros(N);witness=np.zeros(N);yellow_votes=np.zeros(N);vis=np.zeros(N);body_witness=np.zeros(N);cached={}
    for raw in range(89,138):
        d=dict(np.load(OLD/'views'/f'{raw:03d}.npz'));uid=d['unit_id'];safe=np.maximum(uid,0)
        b=np.asarray(Image.open(BASE/f'{raw:03d}/delete_a37.png'),np.float32)/255.;bl=linear(b)
        other=d['other_vehicle_guard']|binary_dilation((d['actor_labels']>=1)&(d['actor_labels']<=4),iterations=1)
        body=(uid>=0)&r['domain_a37'][safe]&(d['actor_labels']==5)&~other
        body_witness+=np.bincount(uid[body],minlength=N)
        eligible=(uid>=0)&r['domain_a37'][safe]&~other&r['active'][safe]
        target=r['reference_linear'][safe];lum=bl@np.array([.2126,.7152,.0722]);refLum=target@np.array([.2126,.7152,.0722])
        deficit=eligible&(refLum-lum>.025)&(lum<.80*np.maximum(refLum,1e-6))
        witness+=np.bincount(uid[eligible&d['shadow_proxy']],minlength=N)
        bad+=np.bincount(uid[deficit],minlength=N);vis+=np.bincount(uid[eligible],minlength=N)
        observed=np.asarray(Image.open(BASE/f'{raw:03d}/gt.png'),np.float32)/255.
        yellow=(b[:,:,0]>1.22*b[:,:,2])&(b[:,:,1]>1.08*b[:,:,2])&(b[:,:,0]>.32)
        white=(b.min(2)>.55)&(b.max(2)-b.min(2)<.18)&(observed.min(2)>.50)
        yy=(yellow|white)&(np.abs(b-observed).mean(2)<.08)&(d['actor_labels']==0)&eligible
        cc,num=label(yy);sizes=np.bincount(cc.ravel());sizes[0]=0;yy&=sizes[cc]>=15
        yellow_votes+=np.bincount(uid[yy],minlength=N)
        d['other_vehicle_guard']=other;d['marking_mask']=binary_dilation(yy,iterations=1);d['new_shadow_proxy']=deficit
        cached[raw]=(d,b,bl)
    real_support=r['active']&r['domain_a37']&(r['source_valid'].sum(0)>=3)
    real_ids=np.flatnonzero(real_support);distance_to_real,nearest_real=cKDTree(r['local_xyz'][real_ids,:2]).query(r['local_xyz'][:,:2])
    inferred_parent=r['domain_a37']&~real_support&(body_witness>0)&(distance_to_real<=settings['inference']['max_neighbour_distance_m'])
    support=real_support|inferred_parent
    mandatory=support&((witness>0)|(bad>=2)|(inferred_parent&(body_witness>0)));seed=binary_closing(chart(mandatory),iterations=2)|chart(mandatory)
    cov=settings['coverage'];distance=distance_transform_edt(~seed,sampling=(.08/cov['longitudinal_outer_m'],.08/cov['lateral_outer_m']))
    weight=gather(np.clip((1-distance)/(1-cov['core_fraction']),0,1)).astype(np.float32)*support
    marked=(yellow_votes>=3)&(yellow_votes/np.maximum(vis,1)>.12)
    marked=gather(binary_dilation(chart(marked),iterations=1))
    weight[marked]=0
    parents=np.flatnonzero(weight>.005);sub=int(round(.08/settings['detail_cell_m']));cell=.08/sub
    ox,oy=np.meshgrid(np.arange(sub),np.arange(sub),indexing='ij');offset=np.c_[ox.ravel(),oy.ravel()]
    parent=np.repeat(parents,sub*sub);keys=(r['keys'][parents,None]*sub+offset[None]).reshape(-1,2)
    xy=(keys+.5)*cell;geometry_parent=parent.copy();geometry_parent[inferred_parent[parent]]=real_ids[nearest_real[parent[inferred_parent[parent]]]]
    delta=xy-r['local_xyz'][geometry_parent,:2];normal=r['normal_local'][geometry_parent]
    xyz=np.c_[xy,r['local_xyz'][geometry_parent,2]-(normal[:,:2]*delta).sum(1)/np.maximum(normal[:,2],.2)]
    M=len(parent);rgb=np.empty((199,M,3),np.float32);valid=np.zeros((199,M),bool);scores=np.zeros((199,M),np.float32)
    for raw in range(199):
        f=np.load(FULL/'frames'/f'{raw:03d}.npz');p=transform(xyz,f['w2l']);uvh=p@f['K'].T;uv=uvh[:,:2]/np.maximum(p[:,2:],1e-8)-.5
        ok=r['source_valid'][raw,parent]&(p[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
        ok&=map_coordinates(f['guard'].astype(float),[uv[:,1],uv[:,0]],order=0,mode='constant',cval=1)<.5
        image=np.asarray(Image.open(FULL/'rgb'/f'{raw:03d}.png'),np.float32)/255.
        sampled=np.stack([map_coordinates(image[:,:,ch],[uv[:,1],uv[:,0]],order=1,mode='nearest') for ch in range(3)],1)
        rgb[raw]=linear(sampled);valid[raw]=ok
        ray=np.linalg.inv(f['w2l'])[:3,3]-xyz;length=np.linalg.norm(ray,axis=1)
        cosine=np.abs((ray/np.maximum(length[:,None],1e-8)*normal).sum(1));scores[raw]=np.where(ok,cosine/np.maximum(p[:,2],.1)**2,0)
        if raw%40==0:print(json.dumps(dict(stage='real199',raw=raw,cells=M,seconds=time.time()-start)),flush=True)
    luma=rgb@np.array([.2126,.7152,.0722],np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');q75=np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)
    clean=valid&(luma>=q75[None]-.025);chosen=np.argmax(np.where(clean,scores,-1),axis=0)
    reference=rgb[chosen,np.arange(M)].copy();agree=clean&(np.abs(rgb-reference[None]).max(2)<.08)
    real_active=agree.sum(0)>=3
    colour_srgb=srgb(reference);asphalt=real_active&(colour_srgb.max(1)-colour_srgb.min(1)<.18)&(colour_srgb.max(1)<.65)
    real_fine=np.flatnonzero(asphalt);dist,near=cKDTree(xyz[real_fine,:2]).query(xyz[:,:2],k=8)
    inferred=(~real_active)&(dist[:,0]<=settings['inference']['max_neighbour_distance_m'])
    neighbours=real_fine[near];weights=1/np.maximum(dist,.02)**2
    reference[inferred]=(reference[neighbours[inferred]]*weights[inferred,:,None]).sum(1)/weights[inferred].sum(1)[:,None]
    active=real_active|inferred;chosen[inferred]=-1
    del rgb,luma,scores
    lo=keys.min(0);fine_shape=tuple(keys.max(0)-lo+1);lookup=np.full(fine_shape,-1,np.int32);fineij=keys-lo;lookup[fineij[:,0],fineij[:,1]]=np.arange(M)
    def view_index(raw,d):
        f=np.load(FULL/'frames'/f'{raw:03d}.npz');yy,xx=np.mgrid[:320,:480]
        rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],2)@np.linalg.inv(f['K']).T;inv=np.linalg.inv(f['w2l'])
        point=rays@inv[:3,:3].T*d['depth_m'][:,:,None]+inv[:3,3]
        k=np.floor(point[:,:,:2]/cell).astype(int)-lo;inside=(k>=0).all(2)&(k<np.array(fine_shape)).all(2)
        kk=np.clip(k,[0,0],np.array(fine_shape)-1);index=lookup[kk[:,:,0],kk[:,:,1]];inside&=index>=0
        return np.maximum(index,0),inside
    count=np.zeros(N);residual=np.zeros((N,3));pixel_map={}
    for raw,(d,b,bl) in cached.items():
        index,inside=view_index(raw,d);pixel_map[raw]=(index,inside)
        good=inside&real_active[index]&(d['actor_labels']==0)&~d['other_vehicle_guard']&~d['marking_mask']
        good&=(np.abs(bl-reference[index]).max(2)<.05)&(d['unit_id']>=0)&~d['new_shadow_proxy']&~d['shadow_proxy']
        ids=d['unit_id'][good];count+=np.bincount(ids,minlength=N)
        for ch in range(3):residual[:,ch]+=np.bincount(ids,weights=(bl-reference[index])[:,:,ch][good],minlength=N)
    cfg=settings['adaptive_rgb'];sigma=(cfg['sigma_longitudinal_m']/.08,cfg['sigma_lateral_m']/.08)
    den=gaussian_filter(chart(count),sigma);delta=np.stack([gather(gaussian_filter(chart(residual[:,ch]),sigma)/np.maximum(den,1e-8)) for ch in range(3)],1)
    correction=np.clip(delta,-cfg['max_linear_delta'],cfg['max_linear_delta'])*float(cfg['enabled'])*cfg['strength']
    uncalibrated=reference.copy();reference=np.clip(reference+correction[parent],0,1)
    world=transform(xyz,np.asarray(meta['coordinate_anchor']));model=transform(world,np.asarray(meta['world_to_model']))
    opacity=np.where(weight[parent]>.999,.99005,1-(1-weight[parent])**.4).astype(np.float32)
    np.savez_compressed(dest/'road_reference.npz',active=active,keys=keys,local_xyz=xyz,model_xyz=model,
        normal_world=r['normal_world'][geometry_parent],gaussian_sigma_m=np.tile([cell*.65,cell*.65,settings['normal_sigma_m']],(M,1)),
        reference_linear=reference,reference_srgb=srgb(reference),uncalibrated_linear=uncalibrated,native_correction_linear=correction[parent],
        road_opacity=opacity,parent_coarse_id=parent,edit_weight=weight[parent],source_valid_count=valid.sum(0),agree_count=agree.sum(0),inferred=inferred,real_active=real_active,
        best_source_raw=chosen.astype(np.int16),source_raw=np.arange(199),source_valid=valid,source_agree=agree,cell_m=cell)
    np.savez_compressed(dest/'coarse_envelope.npz',keys=r['keys'],weight=weight,mandatory=mandatory,marking=marked,clean_count=count,correction=correction)
    for raw,(d,b,bl) in cached.items():
        index,inside=pixel_map[raw];view_scope=d['domain_a37']|(d['actor_labels']==5)|d['new_shadow_proxy']
        repair=inside&active[index]&~d['other_vehicle_guard']&view_scope
        w=weight[parent[index]]*repair*np.where(inferred[index],settings['inference']['loss_confidence'],1);target=d['reference_linear'].copy();target[inside]=reference[index[inside]]
        keep=np.maximum(1-weight[parent[index]]*repair,d['other_vehicle_guard']).astype(np.float32)
        keep[(d['actor_labels']==5)&~d['other_vehicle_guard']]=0
        d.update(repair_mask=repair,fit_weight=w,recovery_reference_linear=target,native_keep_weight=keep,
            protect_mask=~repair,world_edit_weight=w,inferred_fine=inferred[index]&repair)
        # Old strong reference remains an unchanged cross-run proxy; new fit domain is separately reported.
        np.savez_compressed(dest/'views'/f'{raw:03d}.npz',**d)
    write(dest/'receipt.json',dict(target=37,parent='fresh single-a37 B10k',source_frames=199,source_role='BUILD, not independent clean-road GT',
        coarse_domain_cells=int((r['domain_a37']&r['active']).sum()),coarse_envelope_cells=len(parents),mandatory_cells=int(mandatory.sum()),
        fine_cells=M,active_real_reference_cells=int(real_active.sum()),inferred_cells=int(inferred.sum()),unknown_fine_cells=int((~active).sum()),
        no_new_colour_interpolation=False,inferred_policy='up to0.6m, eight measured-asphalt neighbours IDW; separate labels/confidence; excluded from real-reference metrics',geometry='2cm tangent-plane subcells; unknown parents use nearest measured plane within0.6m; not new LiDAR measurements',
        calibration='world-stable anisotropic clean-neighbour native-minus-real RGB offset; recomputed on B10k; capped per channel',
        settings=settings,seconds=time.time()-start,updates=0))
    print((dest/'receipt.json').read_text(),flush=True)
if __name__=='__main__':main()
