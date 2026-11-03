"""Build real199 receiver evidence behind the joint a37/a6 image footprints.

R4 and original sensor files are read-only. Added 8cm cells use measured ground
patch qualification and box visibility; unobserved colours stay unavailable.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,warnings
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation,binary_closing,binary_fill_holes
sys.path.insert(0,'/root/autodl-tmp')
from fullclip_road_cpu_20261026 import loadmeta,Surface,transform,project,sample,measured_patch_evidence,ray_box_blocked
from repair_surface import FixedSurface

RUN=Path(__file__).resolve().parent.parent;ROOT=RUN.parent
OLD=ROOT/'shadow_a5_a37_199_20261103_r4';FULL=ROOT/'r15_gn11_fullclip_cpu_20261026';DENSE=ROOT/'r15_gn11_stage1_v41_20261023_dense_r7/frames';CELL=.08
def linear(x):return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def srgb(x):return np.where(x<=.0031308,12.92*x,1.055*np.maximum(x,1e-10)**(1/2.4)-.055)
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
    start=time.time();out=RUN/'inputs';out.mkdir(exist_ok=False);(out/'views').mkdir()
    m,anchor,inv,plane=loadmeta();initial=dict(np.load(FULL/'full199_initialization.npz'));old=dict(np.load(OLD/'inputs/road_reference.npz'))
    surface=FixedSurface(m,dict(keys=initial['keys'],world=initial['world_xyz'],surface_supported=initial['surface_supported'],
        provenance=initial['eligible'].astype(np.uint8),base_weight=initial['source_valid'],normal_world=initial['normal_world'],cell_m=.2))
    cache={};seed=[]
    for raw in range(89,138):
        f=np.load(FULL/'frames'/('%03d.npz'%raw));dense=np.load(DENSE/('0008%03d.npz'%raw));labels=dense['actor_labels']
        masks={a:binary_dilation(binary_closing(labels==m['actors'].index(a)+1,iterations=2),iterations=5) for a in [6,37]}
        p=surface.camera(f['K'],f['w2l']@inv);supported=p['supported'].reshape(320,480);local=transform(p['world'],inv).reshape(320,480,3)
        domain=masks[6]|masks[37];seed.extend(np.floor(local[domain&supported,:2]/CELL).astype(np.int32))
        cache[raw]=(masks,supported,local,labels)
    keys=np.unique(np.asarray(seed),axis=0);lo=keys.min(0)-6;hi=keys.max(0)+6;chart=np.zeros(tuple(hi-lo+1),bool);ij=keys-lo;chart[ij[:,0],ij[:,1]]=True
    chart=binary_fill_holes(binary_closing(chart,iterations=3));chart=binary_dilation(chart,iterations=4);keys=np.argwhere(chart)+lo
    xy=(keys+.5)*CELL;ok=(xy[:,0]>=.5)&(xy[:,0]<=90)&(np.abs(xy[:,1])<=8);keys=keys[ok];xy=xy[ok]
    oldset={tuple(x) for x in old['keys'][old['active']]};keep=np.array([tuple(x) not in oldset for x in keys]);keys=keys[keep];xy=xy[keep]
    pool=np.load(FULL/'full199_ground_pool.npz')['local_xyz'];s=Surface(pool,plane);height,supported=s.height(xy);normal=s.normal(xy);xyz=np.c_[xy,height]
    keys=keys[supported];xyz=xyz[supported];normal=normal[supported];N=len(keys)
    print(json.dumps(dict(stage='receiver_domain',cells=N,bounds=[xyz.min(0).tolist(),xyz.max(0).tolist()],seconds=time.time()-start)),flush=True)
    quality=np.array([x['sharpness'] for x in json.loads((FULL/'extraction.json').read_text())['frames']]);quality=np.minimum(quality/max(np.median(quality[89:138]),1e-9),1.)
    valid=np.zeros((199,N),bool);colours=np.zeros((199,N,3),np.float32)
    for raw in range(199):
        f=np.load(FULL/'frames'/('%03d.npz'%raw));uv,z=project(xyz,f['K'],f['w2l']);origin=np.linalg.inv(f['w2l'])[:3,3]
        direction=origin-xyz;length=np.linalg.norm(direction,axis=1);cosine=np.abs((direction/np.maximum(length[:,None],1e-8)*normal).sum(1))
        area=f['K'][0,0]*f['K'][1,1]*.2**2*cosine/np.maximum(z*z,1e-8)
        candidate=(z>.1)&(uv[:,0]>=0)&(uv[:,0]<479)&(uv[:,1]>=0)&(uv[:,1]<319)&(sample(f['guard'],uv)==0)&(cosine>=.12)&(area>=.25)&(quality[raw]>=.2)
        ids=np.flatnonzero(candidate)
        if len(ids):
            ok,dep,tri,conflict=measured_patch_evidence(f['cloud'],f['ground'],f['K'],f['w2l'],xyz[ids],uv[ids],normal[ids])
            boxes=[dict(center=c,half=h,axes=a) for c,h,a in zip(f['box_center'],f['box_half'],f['box_axes'])]
            valid[raw,ids]=ok&~ray_box_blocked(xyz[ids],origin,boxes)
        rgb=np.asarray(Image.open(FULL/'rgb'/('%03d.png'%raw)),np.float32)/255.;colours[raw]=linear(sample(rgb,uv)).astype(np.float32)
        if raw%20==0 or raw==198:print(json.dumps(dict(raw=raw,qualified=int(valid[raw].sum()),seconds=time.time()-start)),flush=True)
    luma=colours@np.array([.2126,.7152,.0722],np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');q75=np.nanquantile(np.where(valid,luma,np.nan),.75,axis=0)
        upper=valid&(luma>=q75[None]-.025)&(luma<=q75[None]+.06);ref=np.nanmedian(np.where(upper[:,:,None],colours,np.nan),axis=0)
    ref=np.nan_to_num(ref,nan=0).astype(np.float32);agree=valid&(np.abs(colours-ref[None]).max(2)<.035);count=agree.sum(0)
    active=(count>=3)&(ref.min(1)>.012)&(ref.max(1)<.88)
    world=transform(xyz,anchor);model=transform(world,np.asarray(m['world_to_model']));normalworld=normal@anchor[:3,:3].T
    np.savez_compressed(out/'extra_road_reference.npz',keys=keys,local_xyz=xyz,world_xyz=world,model_xyz=model,normal_world=normalworld,
        gaussian_sigma_m=np.tile([CELL*.65,CELL*.65,.015],(N,1)),reference_linear=ref,reference_srgb=srgb(ref),active=active,
        source_raw=np.arange(199),source_valid=valid,source_agree=agree,agree_count=count,source_rgb_linear=colours,cell_m=CELL)
    lo=keys.min(0);shape=tuple(keys.max(0)-lo+1);lookup=np.full(shape,-1,np.int32);ij=keys-lo;lookup[ij[:,0],ij[:,1]]=np.arange(N)
    rows=[]
    for raw,(masks,supported,local,labels) in cache.items():
        d=dict(np.load(OLD/'inputs/views'/('%03d.npz'%raw)));k=np.floor(local[:,:,:2]/CELL).astype(int)-lo
        inside=(k>=0).all(2)&(k<np.array(shape)).all(2);k=np.clip(k,[0,0],np.array(shape)-1);uid=lookup[k[:,:,0],k[:,:,1]];safe=np.maximum(uid,0)
        extra_valid=inside&supported&(uid>=0)&active[safe];old_uid=d['unit_id'];old_safe=np.maximum(old_uid,0);old_valid=(old_uid>=0)&old['active'][old_safe]
        known=extra_valid|old_valid;target=d['recovery_reference_linear'].copy();target[extra_valid]=ref[safe[extra_valid]]
        coherent=binary_closing(masks[6]|masks[37],iterations=3);guard=d['other_vehicle_guard'];fit=coherent&known&~guard
        view=dict(a6_mask=masks[6],a37_mask=masks[37],coherent_domain=coherent,joint_fit_mask=fit,
            extra_reference_valid=extra_valid,reference_linear=target,protect_mask=~coherent|guard,other_vehicle_guard=guard,
            fit_a6=fit&masks[6],fit_a37=fit&masks[37],actor_labels=labels)
        np.savez_compressed(out/'views'/('%03d.npz'%raw),**view)
        rows.append(dict(raw=raw,actors={str(a):dict(domain=int(masks[a].sum()),fit=int((fit&masks[a]).sum()),new_measured_reference=int((extra_valid&masks[a]).sum())) for a in [6,37]}))
    receipt=dict(source_frames=199,new_receiver_cells=N,new_measured_active_cells=int(active.sum()),unknown_cells=int((~active).sum()),
        baseline='R4 immutable, existing a5/a37 repaired road and trained gate retained',joint_actors=[37,6],deletion_set=[5,26,18,6,37],
        new_colour_inference=False,source_policy='real measured triangle + box visibility; >=3 colour-agreeing views; R4 upper-radiance median',
        cell_m=CELL,seconds=time.time()-start,rows=rows,updates=0)
    write(out/'receipt.json',receipt);print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
