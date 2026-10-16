"""CPU cross-time road evidence; no invented clean RGB or dense semantic GT."""
import os,sys,json,time
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import binary_erosion,binary_dilation
from PIL import Image,ImageDraw
from observation_guards import annotation_boxes,project_guard
ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
OLD=ROOT/'shadow_BC_20261016_v1';NEW=ROOT/'shadow_sequence_20261016_v2'
sys.path.insert(0,str(OLD/'implementation_v1'))
from prepare_e0 import project,rays_to_plane,sample

def lin(x):
    x=x.clip(0,1)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def main():
    start=time.time();out=NEW/'surface_guarded_data';out.mkdir(exist_ok=False)
    spec=json.loads((ROOT/'rigid_specs/scene0008_089_137.json').read_text());src=Path(spec['source_scene'])
    ev=ROOT/'rigid_post30k/scene0008_089_137/evaluation'
    frames=json.loads((ev/'scene_evaluation_manifest.json').read_text())['streams']['observed']['frames']
    with np.load(OLD/'training_data_v2/training_data.npz') as f:
        plane=f['plane_model'].copy();scale=float(f['world_scale']);poses=f['poses'];sizes=f['sizes_model'];valid=f['pose_valid'];gains=f['gains']
    with np.load(Path(spec['prepared'])/'instances.npz') as f:norm=f['normalization'];selected=set(f['instance_ids'].tolist())
    n=plane[:3];seed=np.eye(3)[np.argmin(abs(n))];u=np.cross(n,seed);u/=np.linalg.norm(u);v=np.cross(n,u);basis=np.stack([u,v],1)
    origin=-plane[3]*n
    lidar=[];lidar_receipt=[];images=[];bodies=[];depth=[];guards=[];all_boxes=[]
    # All 49 real camera observations and LiDAR sweeps. Motion is used only for
    # body exclusion and shadow attribution, not to fabricate road observations.
    for i,row in enumerate(frames):
        fid=row['frame_id'];meta=json.loads((src/'raw_front_metadata'/f'{fid}.json').read_text())
        raw=np.fromfile(src/'velodyne'/f'{fid}.bin',np.float32).reshape(-1,6)[:,:3]
        pose=np.array(meta['frame_pose_world_from_vehicle']).reshape(4,4)
        world=raw@pose[:3,:3].T+pose[:3,3];q=world@norm[:3,:3].T+norm[:3,3]
        uv,front=project(q,row['camera']);height=(q@n+plane[3])/scale
        road=front&(uv[:,0]>12)&(uv[:,0]<468)&(uv[:,1]>185)&(uv[:,1]<310)&(abs(height)<.14)
        lidar.append(q[road]);lidar_receipt.append(int(road.sum()))
        with np.load(ev/row['npz']) as f:
            boxes=annotation_boxes(json.loads((src/'annotations'/f'{fid}.json').read_text()),norm,selected)
            guard=binary_dilation(project_guard(boxes,row['camera']),iterations=2)
            guards.append(guard);all_boxes.append([b.tolist() for b in boxes])
            images.append(np.moveaxis(f['gt'],0,-1));bodies.append(binary_dilation(f['observed_masks'].any(0),iterations=3)|guard)
        print(json.dumps({'load_frame':i+1,'road_lidar':int(road.sum())}),flush=True)
    cloud=np.concatenate(lidar);xy=(cloud-origin)@basis/scale
    # A fixed Euclidean chart in metres. Grid points without actual multi-view
    # photometric and LiDAR support remain unavailable for reconstruction.
    lo=np.floor(np.quantile(xy,.005,axis=0)/.08)*.08;hi=np.ceil(np.quantile(xy,.995,axis=0)/.08)*.08
    step=.08;shape=np.ceil((hi-lo)/step).astype(int)+1
    if np.prod(shape)>150000:
        step=math.ceil((np.sqrt(np.prod(shape)/150000)*step)*100)/100;shape=np.ceil((hi-lo)/step).astype(int)+1
    xx,yy=np.meshgrid(np.arange(shape[0]),np.arange(shape[1]),indexing='xy');chart=np.stack([xx.ravel(),yy.ravel()],1)*step+lo
    xyz=origin+chart@basis.T*scale
    dist,_=cKDTree(cloud/scale).query(xyz/scale,k=1,workers=2);lidar_support=dist<.23
    obs=[];free=[];uvs=[];nearactor=[]
    for i,row in enumerate(frames):
        uv,front=project(xyz,row['camera']);values,inview=sample(images[i],uv)
        x=np.rint(uv[:,0]).astype(int).clip(0,479);y=np.rint(uv[:,1]).astype(int).clip(0,319)
        good=inview&front&~bodies[i][y,x]&(uv[:,1]>185)&(uv[:,1]<312)&lidar_support
        obs.append(lin(values)/gains[i]);free.append(good);uvs.append(uv)
        d=xyz[None]-poses[i,:,:3,3][:,None];local=np.einsum('kni,kij->knj',d,poses[i,:,:3,:3])/scale
        ds=np.linalg.norm(np.maximum(abs(local[:,:,:2])-sizes[i,:,None,:2]/scale/2,0),axis=-1);ds[~valid[i]]=np.inf
        nearactor.append(ds.min(0))
    obs=np.asarray(obs,dtype=np.float32);free=np.asarray(free);uvs=np.asarray(uvs,dtype=np.float32);nearactor=np.asarray(nearactor)
    luma=obs@np.array([.2126,.7152,.0722],np.float32)
    masked=np.where(free,luma,np.nan)
    # Upper stable observations suppress transient dark cast shadows. Require
    # >=3 agreeing observations and temporal separation; brightness alone is
    # insufficient. Static dark markings and stable environmental shadows stay.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');q75=np.nanquantile(masked,.75,axis=0)
        upper=free&(luma>=q75[None]-.025)&(luma<=q75[None]+.06)
        ref=np.nanmedian(np.where(upper[:,:,None],obs,np.nan),axis=0)
    ref=np.nan_to_num(ref,nan=0.)
    agree=free&(abs(obs-ref[None]).max(2)<.035)
    count=agree.sum(0);first=np.argmax(agree,axis=0);last=48-np.argmax(agree[::-1],axis=0)
    supported=lidar_support&(count>=3)&(last-first>=6)&(ref.min(1)>.012)&(ref.max(1)<.88)
    # Do not grow unsupported holes by image interpolation. Isolated texels and
    # outer boundary get no authority to replace the parent scene.
    mask=supported.reshape(shape[1],shape[0]);interior=binary_erosion(mask,iterations=1)
    confidence=interior.ravel().astype(np.float32)
    # Linear RGB reference variation within agreeing observations is measured.
    clean=free&supported[None]&(abs(obs-ref[None]).max(2)<.045)
    ratio=(obs@np.array([.2126,.7152,.0722]))/np.maximum(ref@np.array([.2126,.7152,.0722]),.01)[None]
    chroma=np.std(np.log(np.maximum(obs/np.maximum(ref[None],.015),.02)),axis=2)
    # Road marking registration ambiguity is excluded from shadow fitting.
    lum=(ref@np.array([.2126,.7152,.0722])).reshape(shape[1],shape[0]);edge=np.zeros_like(mask)
    edge[:,1:]|=abs(np.diff(lum,axis=1))>.07;edge[1:]|=abs(np.diff(lum,axis=0))>.07
    edge=binary_dilation(edge,iterations=1).ravel()
    shadow=free&supported[None]&(ratio<.70)&(chroma<.55)&(nearactor<4)&~edge[None]&~clean
    np.savez_compressed(out/'surface.npz',xyz=xyz.astype('float32'),reference=ref.astype('float32'),confidence=confidence,
        support=supported,clean=clean,shadow=shadow,free=free,observed=obs,uv=uvs,lo=lo.astype('float32'),step=step,shape=shape,
        plane=plane.astype('float32'),basis=basis.astype('float32'),origin=origin.astype('float32'),world_scale=scale,gains=gains,
        frames=np.array([r['frame_id'] for r in frames]),lidar=cloud.astype('float32'),protection=np.array(guards))
    (out/'protection_boxes.json').write_text(json.dumps(all_boxes))
    im=np.where(ref<=.0031308,ref*12.92,1.055*np.maximum(ref,1e-10)**(1/2.4)-.055)
    picture=np.rint(im.clip(0,1).reshape(shape[1],shape[0],3)*255).astype('uint8')
    Image.fromarray(picture).save(out/'atlas_rgb.png');Image.fromarray((mask*255).astype('uint8')).save(out/'atlas_support.png')
    summary={'frames':49,'grid_shape':shape.tolist(),'grid_spacing_m':step,'grid_texels':len(xyz),'supported_texels':int(supported.sum()),
        'interior_texels':int(confidence.sum()),'supported_surface_area_m2':float(supported.sum()*step**2),
        'clean_samples':int(clean.sum()),'shadow_pseudo_samples':int(shadow.sum()),'lidar_counts':lidar_receipt,
        'source_is_real_observation':True,'clean_reference_is_confidence_filtered_not_ground_truth':True,
        'no_unseen_frame_claim':True,'elapsed_seconds':time.time()-start,'coordinate_basis':basis.tolist(),
        'min_chart_m':lo.tolist(),'max_chart_m':hi.tolist(),'unselected_actor_protection':True,'protection_pixels_per_frame':[int(g.sum()) for g in guards]}
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':
    import math
    main()
