"""Prepare real, local road evidence for an explicitly scoped reconstruction pilot."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
import argparse,datetime,json,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import cKDTree
from prepare_e0 import project,rays_to_plane,sample

ROOT=Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
def read(p):return json.loads(Path(p).read_text())
def lin(x):
    x=np.clip(x,0,1)
    return np.where(x<=.04045,x/12.92,((x+.055)/1.055)**2.4)
def dilate(a,r=2):
    h,w=a.shape;out=np.zeros_like(a)
    for dy in range(-r,r+1):
        for dx in range(-r,r+1):
            out[max(dy,0):min(h,h+dy),max(dx,0):min(w,w+dx)]|=a[max(-dy,0):min(h,h-dy),max(-dx,0):min(w,w-dx)]
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    sid='scene0008_089_137';spec=read(ROOT/'rigid_specs'/f'{sid}.json');src=Path(spec['source_scene'])
    ev=ROOT/'rigid_post30k'/sid/'evaluation';mf=read(ev/'scene_evaluation_manifest.json');frames=mf['streams']['observed']['frames'];target=frames[12]
    with np.load(Path(spec['prepared'])/'instances.npz') as f:
        scale=float(f['world_scale']);norm=f['normalization'];poses=f['poses'];sizes=f['box_size'];posevalid=f['valid'];tracks=f['instance_ids'];times=f['timestamps']
    plane=np.array(read(ROOT/'metadata'/sid/'CPU_readiness.json')['SGCM_geometry']['plane_model'])
    plane/=np.linalg.norm(plane[:3])
    if np.mean(poses[:,:,:3,2]@plane[:3])<0:plane=-plane
    yy,xx=np.mgrid[225:285,40:180];pixels=np.c_[xx.ravel(),yy.ravel()];xyz=rays_to_plane(pixels,target['camera'],plane)
    source_indices=[0,6];check_indices=[24]
    lidar=[];lidar_rows=[]
    for i in source_indices+check_indices:
        row=frames[i];fid=row['frame_id'];meta=read(src/'raw_front_metadata'/f'{fid}.json')
        raw=np.fromfile(src/'velodyne'/f'{fid}.bin',np.float32).reshape(-1,6)[:,:3]
        pose=np.array(meta['frame_pose_world_from_vehicle']).reshape(4,4)
        world=raw@pose[:3,:3].T+pose[:3,3]
        model=world@norm[:3,:3].T+norm[:3,3]
        uv,fr=project(model,target['camera']);height=(model@plane[:3]+plane[3])/scale
        region=fr&(uv[:,0]>25)&(uv[:,0]<195)&(uv[:,1]>215)&(uv[:,1]<300)
        candidate=region&(abs(height)<.25)
        lidar.append(model[candidate]);lidar_rows.append({'frame_id':fid,'local_road_candidates':int(candidate.sum()),
            'plane_abs_residual_median_m':float(np.median(abs(height[candidate])))})
    cloud=np.concatenate(lidar);tree=cKDTree(cloud/scale)
    dist,near=tree.query(xyz/scale,k=1,workers=2)
    support=dist<.35
    # Current data provide independent local LiDAR support, not per-pixel dense depth GT.
    observed=[];bodyfree=[];grids=[];frame_metadata=[]
    sheet=Image.new('RGB',(560*2,272*4),'white')
    selected=[0,6,10,12,18,24,36,48]
    for i,row in enumerate(frames):
        uv,fr=project(xyz,row['camera']);grids.append(uv)
        with np.load(ev/row['npz']) as f:
            rgb=np.moveaxis(f['gt'],0,-1);body=dilate(f['observed_masks'].any(0),2)
        values,inview=sample(rgb,uv);x=np.rint(uv[:,0]).astype(int).clip(0,479);y=np.rint(uv[:,1]).astype(int).clip(0,319)
        free=inview&fr&~body[y,x]&support
        observed.append(values);bodyfree.append(free)
        frame_metadata.append({'frame_id':row['frame_id'],'camera':row['camera'],'timestamp':row['model_time']})
    observed=np.asarray(observed);bodyfree=np.asarray(bodyfree);grids=np.asarray(grids)
    accepted=bodyfree[source_indices].all(0)&support
    atlas=np.median(observed[source_indices],axis=0)
    # Exposure is a bounded nuisance correction estimated only from temporally
    # agreeing road pixels outside selected body boundaries; no shadow-fit gain.
    base=lin(atlas);gt=lin(observed);gains=[];gain_count=[]
    for i in range(49):
        ratio=gt[i]/np.maximum(base,.01)
        stable=bodyfree[i]&accepted&(np.abs(ratio-1).max(1)<.18)&(base.min(1)>.03)
        gain=np.median(ratio[stable],axis=0) if stable.sum()>=150 else np.ones(3)
        gains.append(gain.clip(.92,1.08));gain_count.append(int(stable.sum()))
    gains=np.array(gains);gains[0]=1
    corrected=gt/gains[:,None,:]
    ratio=corrected/np.maximum(base,.01)
    luminance=lambda x:x@np.array([.2126,.7152,.0722])
    lr=luminance(corrected)/np.maximum(luminance(base)[None],.01)
    chroma=np.std(np.log(np.maximum(ratio,.02)),axis=-1)
    distances=[]
    for i in range(49):
        delta=xyz[None,:,:]-poses[i,:,:3,3][:,None,:]
        local=np.einsum('kni,kij->knj',delta,poses[i,:,:3,:3])/scale
        extent=sizes[i,:,:2]/scale/2
        d=np.linalg.norm(np.maximum(abs(local[:,:,:2])-extent[:,None,:],0),axis=-1)
        d[~posevalid[i]]=np.inf;distances.append(d.min(0))
    distances=np.array(distances)
    valid=bodyfree&accepted[None]
    clean=valid&(abs(lr-1)<.18)&(np.abs(corrected-base[None]).max(-1)<.08)
    base_luma=luminance(base).reshape(60,140)
    edge=np.zeros_like(base_luma,dtype=bool)
    edge[1:]|=abs(np.diff(base_luma,axis=0))>.015
    edge[:,1:]|=abs(np.diff(base_luma,axis=1))>.015
    # Registration errors at road markings must not become shadow supervision.
    edge=dilate(edge,2).ravel()
    shadow=valid&(lr<.70)&(chroma<.60)&(distances<4.0)&~edge[None]
    shadow&=~clean
    unknown=valid&~(clean|shadow)
    check=accepted&bodyfree[24]
    checks={'coordinate_roundtrip_max_px':float(np.max(abs(project(xyz,target['camera'])[0]-pixels))),
       'local_LiDAR_support_fraction':float(support.mean()),'accepted_reference_pixels':int(accepted.sum()),
       'atlas_check_sRGB_MAE':float(abs(atlas[check]-observed[24,check]).mean()),
       'atlas_check_linear_MAE_after_gain':float(abs(base[check]-corrected[24,check]).mean()),
       'clean_pixels_all49':int(clean.sum()),'shadow_candidate_pixels_all49':int(shadow.sum()),
       'target_clean_pixels':int(clean[12].sum()),'target_shadow_candidate_pixels':int(shadow[12].sum()),
       'lidar_frames':lidar_rows,'gains':gains.tolist(),'gain_control_counts':gain_count,
       'clean_and_shadow_labels_are_confidence_filtered_pseudo_labels':True,
       'local_LiDAR_not_dense_visibility_certificate':True,'lane_texture_edge_shadow_exclusion_pixels':int(edge.sum()),
       'unselected_static_occluders_checked_by_RGB_agreement_in_reference':True}
    assert checks['coordinate_roundtrip_max_px']<1e-3
    assert accepted.sum()>3000 and shadow.sum()>1000 and checks['atlas_check_sRGB_MAE']<.05,checks
    for j,i in enumerate(selected):
        im=(observed[i].reshape(60,140,3).clip(0,1)*255).astype('uint8')
        label=im.copy().reshape(-1,3).astype(float)
        label[clean[i]]=label[clean[i]]*.5+[0,100,0]
        label[shadow[i]]=label[shadow[i]]*.4+[120,0,0]
        label[unknown[i]]=label[unknown[i]]*.4+[90,60,0]
        left=Image.fromarray(im).resize((280,240));right=Image.fromarray(label.clip(0,255).astype('uint8').reshape(60,140,3)).resize((280,240))
        x=(j%2)*560;y=(j//2)*272;sheet.paste(left,(x,y+32));sheet.paste(right,(x+280,y+32))
        ImageDraw.Draw(sheet).text((x+5,y+8),frames[i]['frame_id']+' RGB | green clean / red shadow / amber unknown',fill='black')
    sheet.save(out/'training_label_review.png')
    # Sensitivity to the fitted road surface height; reference sampling only.
    perturb=[]
    for shift_m in [-.05,.05]:
        shifted=plane.copy();shifted[3]+=shift_m*scale
        q=rays_to_plane(pixels,target['camera'],shifted)
        vals=[]
        for i in source_indices:
            with np.load(ev/frames[i]['npz']) as f:rgb=np.moveaxis(f['gt'],0,-1)
            uv,_=project(q,frames[i]['camera']);vals.append(sample(rgb,uv)[0])
        perturb.append({'plane_offset_m':shift_m,'atlas_sRGB_MAE_change':float(abs(np.median(vals,axis=0)[accepted]-atlas[accepted]).mean())})
    checks['plane_perturbation']=perturb
    np.savez_compressed(out/'training_data.npz',xyz_model=xyz.astype('float32'),pixels=pixels.astype('float32'),
        uv=grids.astype('float32'),reference_srgb=atlas.astype('float32'),reference_linear=base.astype('float32'),
        observed_linear=corrected.astype('float32'),observed_srgb=observed.astype('float32'),
        clean=clean,shadow=shadow,unknown=unknown,body_free=bodyfree,accepted_reference=accepted,
        poses=poses.astype('float32'),sizes_model=sizes.astype('float32'),pose_valid=posevalid,
        frame_ids=np.array([r['frame_id'] for r in frames]),times=times,plane_model=plane,world_scale=np.array(scale),
        gains=gains.astype('float32'),local_lidar=cloud.astype('float32'),plane_nearest_m=dist.astype('float32'))
    (out/'camera_metadata.json').write_text(json.dumps(frame_metadata))
    (out/'preparation_receipt.json').write_text(json.dumps({'schema':'shadow_local_training_data_v1',
        'scope':'local road window at primary raw101; not whole driving scene or fully labelled shadow benchmark',
        'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'GPU_started':False,
        'manual_dense_review_pending':True,'checks':checks},indent=2))
    print(json.dumps({k:v for k,v in checks.items() if k not in ['gains','gain_control_counts']}))
if __name__=='__main__':main()
