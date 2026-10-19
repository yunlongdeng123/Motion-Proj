"""Reconstruct exact donor UV table against the frozen training targets, CPU-only.

Writes separate audit evidence. Does not edit input_bundle_002 or any model.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from PIL import Image,ImageDraw
from prepare_real_inputs import read,box,sparse_visibility_maps,RAW
from world_road import RoadSurface,transform,query_from_pixels,select_asphalt_donors,fill_from_donor


def audit(C,E):
    bundle=C/'input_bundle_002';out=C/'donor_provenance_audit'
    if out.exists():raise FileExistsError('preserve existing donor audit')
    meta=read(bundle/'bundle.json');config=meta['config']['road'];source=E.parent/'inputs/scene0008_089_137'
    A=E/'full5_pipeline/inputs/scene0008_089_137';B=E/'all_vehicle_pipeline/inputs/scene0008_089_137'
    ready=read(E/'full5_pipeline/metadata/scene0008_089_137/CPU_readiness.json');geo=ready['SGCM_geometry']
    anchor=np.loadtxt(geo['anchor_pose']);inv=np.linalg.inv(anchor);plane=np.asarray(geo['ground_fit']['plane_anchor'])
    measured=[];first=None
    for raw in RAW:
        fid=f'0008{raw:03d}';pose_path=source/'pose'/f'{fid}.txt';ann_path=source/'annotations'/f'{fid}.json';lidar_path=source/'velodyne'/f'{fid}.bin'
        for p in [pose_path,ann_path,lidar_path]:
            if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['source_hashes'][fid][str(p)]:raise RuntimeError('frozen geometry input changed')
        pose=np.loadtxt(pose_path);ann=read(ann_path);lidar=np.fromfile(lidar_path,np.float32).reshape(-1,6)[:,:3].astype(float)
        world=transform(lidar,pose);local=transform(world,inv);in_object=np.zeros(len(world),bool)
        for label in ann['laser_labels']:
            if min(label['box'][k] for k in ['length','width','height'])<=0:continue
            obb=box(label,pose);in_object|=(np.abs((world-obb['center'])@obb['axes'])<=obb['half']).all(1)
        ground=(np.abs(local@plane[:3]+plane[3])<=.15)&(~in_object)
        ground&=(local[:,0]>=3)&(local[:,0]<=90)&(local[:,1]>=-8)&(local[:,1]<=8)
        measured.append(local[ground])
        if raw==89:first=(pose,world,ground)
    measured=np.concatenate(measured);_,indices=np.unique(np.floor(measured[:,:2]/.1).astype(int),axis=0,return_index=True)
    surface=RoadSurface(anchor,-plane[[0,1,3]]/plane[2],measured[indices],config['surface_support_radius_m'])
    fid='0008089';pose,world,ground=first
    cal_path=source/'calib'/f'{fid}.txt';rgb_path=source/'image/image_0'/f'{fid}.png';label_path=A/'matched_training_labels'/f'{fid}.png'
    for p in [cal_path,rgb_path,label_path]:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['source_hashes'][fid][str(p)]:raise RuntimeError('frozen donor source changed')
    cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ') for s in cal_path.read_text().splitlines() if ':' in s}
    tr=np.eye(4);tr[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);w2c=tr@np.linalg.inv(pose)
    K=cal['P0'].reshape(3,4)[:,:3].copy();K[:2]*=.25
    rgb=np.asarray(Image.open(rgb_path).convert('RGB').resize((480,320),Image.Resampling.LANCZOS))
    labels=np.asarray(Image.open(label_path))
    mask_b_path=B/'dynamic_masks/dynamic_mask_0'/f'{fid}.png';proposal_path=E/'all_vehicle_native49/native_predictions'/f'{fid}.npz'
    for p in [mask_b_path,proposal_path]:
        if hashlib.sha256(p.read_bytes()).hexdigest()!=meta['source_hashes'][fid][str(p)]:raise RuntimeError('frozen donor occlusion changed')
    mask_b=np.asarray(Image.open(mask_b_path).resize((480,320),Image.Resampling.NEAREST))>0
    with np.load(proposal_path,allow_pickle=False) as z:native=z['masks'].any(0)
    native=np.asarray(Image.fromarray(native).resize((480,320),Image.Resampling.NEAREST))>0
    occluder=(labels>0)|mask_b|native;_,_,road=sparse_visibility_maps(world,ground,K,w2c)
    candidates,regions,marking=select_asphalt_donors(rgb,config['donor_polygons_raw89'],occluder,road)
    y,x=np.nonzero(candidates);pixels=np.c_[x,y];dw,valid,supported=query_from_pixels(pixels,K,w2c,surface);ok=valid&supported
    donor_pixels=pixels[ok];donor_world=dw[ok];donor_xy=transform(donor_world,inv)[:,:2]
    donor_rgb=rgb[y[ok],x[ok]]/255.;donor_regions=regions[y[ok],x[ok]]
    with np.load(bundle/'query_evidence.npz',allow_pickle=False) as z:
        qworld=z['world'];expected_rgb=z['rgb'];expected_pro=z['provenance'];expected_index=z['donor_index']
        candidate_count=z['candidate_count'];source_count=z['source_count']
    pro=expected_pro.copy();pro[pro==2]=0
    fused=dict(rgb=expected_rgb.copy(),provenance=pro,confidence=np.zeros(len(pro)),candidate_count=candidate_count,source_count=source_count)
    reconstructed=fill_from_donor(transform(qworld,inv)[:,:2],fused,donor_xy,donor_rgb,True,donor_regions=donor_regions)
    delta=float(np.max(np.abs(reconstructed['rgb']-expected_rgb)))
    if not np.array_equal(reconstructed['donor_index'],expected_index) or not np.array_equal(reconstructed['provenance'],expected_pro) or delta>1e-12:
        raise RuntimeError('donor reconstruction differs from frozen training inputs')
    out.mkdir();np.savez_compressed(out/'donor_table.npz',raw89_pixels=donor_pixels,world=donor_world,road_xy=donor_xy,RGB=donor_rgb,region=donor_regions)
    prior=np.flatnonzero(expected_pro==2);np.savez_compressed(out/'prior_links.npz',query_index=prior,donor_index=expected_index[prior])
    retained=np.zeros((320,480),bool);retained[donor_pixels[:,1],donor_pixels[:,0]]=True
    preview=rgb.copy();preview[retained]=(preview[retained]*.5+np.array([25,200,255])*.5).astype(np.uint8)
    im=Image.fromarray(preview);d=ImageDraw.Draw(im)
    for polygon in config['donor_polygons_raw89']:d.line([tuple(p) for p in polygon]+[tuple(polygon[0])],fill='#ffbf47',width=2)
    im.save(out/'actual_donor_support.png')
    result=dict(status='PASSED_EXACT_FROZEN_DONOR_SOURCE_RECONSTRUCTION',training_targets_changed=False,
        bundle_sha256=hashlib.sha256((bundle/'bundle.json').read_bytes()).hexdigest(),
        road_anchor=meta['coordinate_anchor_frame'],model_anchor=meta['model_anchor_frame'],
        depth_and_color_candidates=[int(((regions==i)&candidates).sum()) for i in range(2)],
        retained_donor_samples=[int((donor_regions==i).sum()) for i in range(2)],
        unique_used_donor_samples=int(len(np.unique(expected_index[prior]))),
        prior_query_samples=len(prior),reconstructed_RGB_max_abs=delta,GPU_used=False,optimizer_updates=0,
        note='Explicit different-world-position texture prior; exact source UV and RGB retained separately, not query-location ground truth.')
    (out/'receipt.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--engineering',type=Path,required=True)
    a=p.parse_args();audit(a.campaign,a.engineering)
