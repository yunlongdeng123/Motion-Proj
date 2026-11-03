"""Real-image supersampling of the measured road, preserving paint detail.

Subcells share the measured parent patch geometry, with explicit tangent-plane
interpolation. They are not new LiDAR witnesses. Every colour is sampled from
a geometrically eligible real source and rechecked against its image guard.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,shutil,warnings
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import map_coordinates,gaussian_filter

RUN=Path(__file__).resolve().parent.parent
COARSE=RUN.with_name('shadow_quality_20261103_r7')
COARSE_INPUT=RUN/'coarse_inputs' if (RUN/'coarse_inputs').exists() else COARSE/'inputs'
BASELINE=RUN.with_name('shadow_a5_a37_199_20261103_r4')/'evaluation/all49'
FULL=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_fullclip_cpu_20261026')
META=Path('/root/autodl-tmp/drivingeditor_repro/campaign/r15_gn11_stage1_v3_20261020/road_multiview_002/bundle.json')
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,1e-10)**(1/2.4)-.055)
def transform(x,T):return x@T[:3,:3].T+T[:3,3]
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False))
def main():
    start=time.time();out=RUN/'inputs';out.mkdir(exist_ok=False);(out/'views').mkdir()
    settings=json.loads((RUN/'quality_settings.json').read_text());sub=int(round(.08/settings['detail']['cell_m']));assert sub>=1 and abs(sub*settings['detail']['cell_m']-.08)<1e-8
    r=dict(np.load(COARSE_INPUT/'road_reference.npz'));m=json.loads(META.read_text());ids=np.flatnonzero(r['active']);cell=.08/sub
    ox,oy=np.meshgrid(np.arange(sub),np.arange(sub),indexing='ij');offset=np.c_[ox.ravel(),oy.ravel()];parent=np.repeat(ids,sub*sub)
    keys=(r['keys'][ids,None,:]*sub+offset[None,:,:]).reshape(-1,2);xy=(keys+.5)*cell
    delta=xy-r['local_xyz'][parent,:2];normal=r['normal_local'][parent]
    z=r['local_xyz'][parent,2]-(normal[:,:2]*delta).sum(1)/np.maximum(normal[:,2],.2)
    xyz=np.c_[xy,z];N=len(xyz);allrgb=np.empty((199,N,3),np.float32);valid=np.zeros((199,N),bool);score=np.zeros((199,N),np.float32)
    for raw in range(199):
        f=np.load(FULL/'frames'/('%03d.npz'%raw));c=transform(xyz,f['w2l']);uvh=c@f['K'].T;uv=uvh[:,:2]/np.maximum(c[:,2:],1e-8)-.5
        v=r['source_valid'][raw,parent]&(c[:,2]>.1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)
        v&=map_coordinates(f['guard'].astype(float),[uv[:,1],uv[:,0]],order=0,mode='constant',cval=1)<.5
        image=np.asarray(Image.open(FULL/'rgb'/('%03d.png'%raw)),np.float32)/255.
        sampled=np.stack([map_coordinates(image[:,:,ch],[uv[:,1],uv[:,0]],order=1,mode='nearest') for ch in range(3)],1)
        allrgb[raw]=linear(sampled);valid[raw]=v
        origin=np.linalg.inv(f['w2l'])[:3,3];ray=origin-xyz;length=np.linalg.norm(ray,axis=1)
        cosine=np.abs((ray/np.maximum(length[:,None],1e-8)*normal).sum(1));score[raw]=np.where(v,cosine/np.maximum(c[:,2],.1)**2,0)
        if raw%30==0:print(json.dumps(dict(raw=raw,subcells=N,seconds=time.time()-start)),flush=True)
    luma=allrgb@np.array([.2126,.7152,.0722],np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');q75=np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)
    clean=valid&(luma>=q75[None]-.025)
    chosen=np.argmax(np.where(clean,score,-1),axis=0);has=clean.any(0)
    ref=allrgb[chosen,np.arange(N)].copy();ref[~has]=r['reference_linear'][parent[~has]]-r['native_correction_linear'][parent[~has]]
    # Keep single real high-resolution source samples rather than averaging
    # narrow painted stripes with grey asphalt from many registered views.
    del allrgb,luma,score
    lo=keys.min(0);shape=tuple(keys.max(0)-lo+1);chart=np.full(shape,-1,np.int32);ij=keys-lo;chart[ij[:,0],ij[:,1]]=np.arange(N)
    ckeys=r['keys'];clo=ckeys.min(0);cshape=tuple(ckeys.max(0)-clo+1);cij=ckeys-clo
    counts=np.zeros(len(ckeys));residual=np.zeros((len(ckeys),3));cached={}
    def lookup(raw,d):
        f=np.load(FULL/'frames'/('%03d.npz'%raw));yy,xx=np.mgrid[:320,:480];rays=np.stack([xx+.5,yy+.5,np.ones_like(xx)],2)@np.linalg.inv(f['K']).T
        inv=np.linalg.inv(f['w2l']);p=rays@inv[:3,:3].T*d['depth_m'][:,:,None]+inv[:3,3]
        k=np.floor(p[:,:,:2]/cell).astype(int)-lo;inside=(k>=0).all(2)&(k<np.array(shape)).all(2)
        kk=np.clip(k,[0,0],np.array(shape)-1);index=chart[kk[:,:,0],kk[:,:,1]];inside&=index>=0
        return np.maximum(index,0),inside
    for raw in range(89,138):
        d=dict(np.load(COARSE_INPUT/'views'/('%03d.npz'%raw)));index,inside=lookup(raw,d);cached[raw]=(d,index,inside)
        base=np.asarray(Image.open(BASELINE/('%03d_baseline_delete5.png'%raw)),np.float32)/255.;b=linear(base);target=ref[index]
        cleanmask=inside&d['repair_mask']&(d['actor_labels']==0)&~d['other_vehicle_guard']&(np.abs(b-target).max(2)<.035)
        yellow=(base[:,:,0]>1.22*base[:,:,2])&(base[:,:,1]>1.08*base[:,:,2])&(base[:,:,0]>.32);cleanmask&=~yellow
        uid=d['unit_id'];cleanmask&=uid>=0
        counts+=np.bincount(uid[cleanmask],minlength=len(ckeys))
        for ch in range(3):residual[:,ch]+=np.bincount(uid[cleanmask],weights=(b-target)[:,:,ch][cleanmask],minlength=len(ckeys))
    def cchart(x):
        a=np.zeros(cshape);a[cij[:,0],cij[:,1]]=x;return a
    rgb=settings['adaptive_rgb'];sigma=(rgb['sigma_longitudinal_m']/.08,rgb['sigma_lateral_m']/.08);den=gaussian_filter(cchart(counts),sigma)
    correction=np.stack([(gaussian_filter(cchart(residual[:,ch]),sigma)/np.maximum(den,1e-8))[cij[:,0],cij[:,1]] for ch in range(3)],1)
    correction=np.clip(correction,-rgb['max_linear_delta'],rgb['max_linear_delta'])*rgb['strength']*float(rgb['enabled']);ref=np.clip(ref+correction[parent],0,1)
    world=transform(xyz,np.asarray(m['coordinate_anchor']));model=transform(world,np.asarray(m['world_to_model']))
    np.savez_compressed(out/'road_reference.npz',active=np.ones(N,bool),model_xyz=model,local_xyz=xyz,keys=keys,
        normal_world=r['normal_world'][parent],gaussian_sigma_m=np.tile([cell*.65,cell*.65,settings['detail']['normal_sigma_m']],(N,1)),
        reference_linear=ref.astype(np.float32),reference_srgb=srgb(ref).astype(np.float32),road_opacity=r['road_opacity'][parent],
        source_raw=np.arange(199),source_valid=valid,source_clean=clean,best_source_raw=chosen.astype(np.int16),
        has_real_source=has,parent_coarse_id=parent,cell_m=cell,native_correction_linear=correction[parent])
    for raw,(d,index,inside) in cached.items():
        target=d['recovery_reference_linear'].copy();target[inside]=ref[index[inside]];d['recovery_reference_linear']=target
        np.savez_compressed(out/'views'/('%03d.npz'%raw),**d)
    shutil.copy2(COARSE_INPUT/'contributors.npz',out/'contributors.npz');shutil.copy2(COARSE_INPUT/'road_reference.npz',out/'coarse_reference.npz')
    rec=json.loads((COARSE_INPUT/'receipt.json').read_text());rec.update(settings=settings,coarse_active_cells=rec['active_cells'],refined_cells=N,
        detail_cell_m=cell,normal_sigma_m=settings['detail']['normal_sigma_m'],real_colour_subcells=int(has.sum()),
        inherited_fallback_cells=int((~has).sum()),detail_source_policy='highest projected resolution real source within clean upper radiance cluster; source image guard rechecked',
        geometry_policy='tangent-plane interpolation inside measured 8cm parent patch, parent visibility eligibility inherited; not new measured LiDAR points',seconds=time.time()-start)
    write(out/'receipt.json',rec);print(json.dumps(rec),flush=True)

if __name__=='__main__':main()
