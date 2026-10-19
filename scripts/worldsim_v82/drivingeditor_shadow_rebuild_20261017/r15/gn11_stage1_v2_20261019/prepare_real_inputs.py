"""CPU-only, read-only input adapter for the SAME prepared A five-ID49 window.

Run on the powered-on server before any GPU gate. Only --output receives writes.
Sparse measured depth and a measured height field are used; missing evidence is
never replaced with the faulty old static render. No checkpoint is read here.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np
from PIL import Image, ImageDraw
from scipy.ndimage import distance_transform_edt, laplace
from world_road import (RoadSurface, transform, project, query_from_pixels,
                        observations, fuse, fill_from_donor, select_asphalt_donors, ray_box_blocked)

IDS=[5,26,18,6,37]
RAW=list(range(89,138))


def read(path):return json.loads(path.read_text(encoding='utf-8'))
def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def box(label,pose):
    b=label['box'];angle=b['heading'];c,s=np.cos(angle),np.sin(angle)
    rot=np.array([[c,-s,0],[s,c,0],[0,0,1.]])
    return dict(center=transform(np.array([[b['center_x'],b['center_y'],b['center_z']]]),pose)[0],
                half=np.array([b['length'],b['width'],b['height']])/2,axes=pose[:3,:3]@rot)


def sparse_visibility_maps(xyz,ground,K,w2c):
    """Nearest measured return, then <=1.5px nearest-neighbor extension.

    Interpolation is never done across a selected point with a different depth.
    The tolerance is exposed in evidence; this is conservative sparse support,
    not dense ground-truth semantic segmentation.
    """
    uv,z,inside=project(xyz,K,w2c,(480,320))
    ix=np.flatnonzero(inside);pixel=np.floor(uv[ix]+0.5).astype(int)
    depth=np.full((320,480),np.inf);road=np.zeros((320,480),bool)
    # Far-to-near sorted assignment keeps nearest return for each target pixel.
    order=ix[np.argsort(z[ix])[::-1]]
    pix=np.floor(uv[order]+0.5).astype(int)
    for i,(x,y) in zip(order,pix):depth[y,x]=z[i];road[y,x]=ground[i]
    valid=np.isfinite(depth)
    if not valid.any():return np.zeros_like(depth),valid,road
    distance,nearest=distance_transform_edt(~valid,return_indices=True)
    valid=distance<=1.5
    return np.where(valid,depth[tuple(nearest)],0),valid,road[tuple(nearest)]&valid


def prepare(engineering,output,config):
    if output.exists():raise FileExistsError('preserve existing CPU bundle; inspect before replacement')
    root=engineering.parent;source=root/'inputs/scene0008_089_137'
    a=engineering/'full5_pipeline/inputs/scene0008_089_137'
    b=engineering/'all_vehicle_pipeline/inputs/scene0008_089_137'
    metadata=engineering/'full5_pipeline/metadata/scene0008_089_137'
    readiness=read(metadata/'CPU_readiness.json')
    spec=read(metadata/'scene_spec.json')
    input_receipt=read(engineering/'matched5_input_receipt.json')
    if input_receipt['frames']!=49 or len(input_receipt['track_ids'])!=5:
        raise ValueError('original five-ID input binding missing')
    tracks=dict(zip(IDS,input_receipt['track_ids']))
    hashes={r['frame_id']:r['label_sha256'] for r in input_receipt['records']}
    model=np.asarray(readiness['transform_matrix'],float)
    scale=float(readiness['actual_meter_to_model_scale'])
    if not np.allclose(np.linalg.svd(model[:3,:3])[1],scale):raise ValueError('nonuniform model scale')
    # Camera-model orientation and SGCM ground-fit coordinates use different anchors.
    model_anchor_frame=spec['fixed_pose_anchor']
    if model_anchor_frame!=readiness['fixed_pose_anchor']:raise ValueError('scene/model anchor mismatch')
    geometry=readiness['SGCM_geometry']
    anchor_path=Path(geometry['anchor_pose']);anchor_frame=anchor_path.stem
    anchor=np.loadtxt(anchor_path);anchor_inv=np.linalg.inv(anchor)
    if not np.allclose(anchor,np.loadtxt(source/'pose'/f'{anchor_frame}.txt'),atol=1e-9):
        raise ValueError('SGCM road anchor pose differs from actual source')
    plane=np.asarray(geometry['ground_fit']['plane_anchor'])
    plane_world=anchor_inv.T@plane;plane_model=np.linalg.inv(model).T@plane_world
    plane_model/=np.linalg.norm(plane_model[:3])
    expected_plane=np.asarray(geometry['plane_model']);expected_plane/=np.linalg.norm(expected_plane[:3])
    if not np.allclose(plane_model,expected_plane,atol=1e-6,rtol=1e-6):
        raise ValueError('road anchor/model plane conversion disagrees with actual SGCM plane')
    frames=[];ground_local=[];source_hashes={}
    for raw in RAW:
        fid=f'0008{raw:03d}'
        label_path=a/'matched_training_labels'/f'{fid}.png'
        if sha(label_path)!=hashes[fid]:raise ValueError('A mask changed: '+fid)
        labels=np.asarray(Image.open(label_path))
        if labels.shape!=(320,480) or not set(np.unique(labels)).issubset(set(range(6))):
            raise ValueError('label dimensions or classes changed')
        pose_path=source/'pose'/f'{fid}.txt';cal_path=source/'calib'/f'{fid}.txt'
        ann_path=source/'annotations'/f'{fid}.json';lidar_path=source/'velodyne'/f'{fid}.bin'
        rgb_path=source/'image/image_0'/f'{fid}.png'
        pose=np.loadtxt(pose_path);ann=read(ann_path)
        cal={s.split(':',1)[0]:np.fromstring(s.split(':',1)[1],sep=' ')
             for s in cal_path.read_text().splitlines() if ':' in s}
        tr=np.eye(4);tr[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4)
        w2c=tr@np.linalg.inv(pose)
        K=cal['P0'].reshape(3,4)[:,:3].copy();K[:2]*=.25
        rgb=np.asarray(Image.open(rgb_path).convert('RGB').resize((480,320),Image.Resampling.LANCZOS))
        lidar=np.fromfile(lidar_path,np.float32).reshape(-1,6)[:,:3].astype(float)
        world=transform(lidar,pose);local=transform(world,anchor_inv)
        boxes=[];selected=[];other_boxes=[];in_object=np.zeros(len(world),bool)
        for label in ann['laser_labels']:
            if min(label['box'][k] for k in ['length','width','height'])<=0:continue
            obb=box(label,pose);boxes.append(obb)
            in_object|=(np.abs((world-obb['center'])@obb['axes'])<=obb['half']).all(1)
            if label['id'] in tracks.values():selected.append(obb)
            else:other_boxes.append(obb)
        ground=(np.abs(local@plane[:3]+plane[3])<=.15)&(~in_object)
        ground&=(local[:,0]>=3)&(local[:,0]<=90)&(local[:,1]>=-8)&(local[:,1]<=8)
        ground_local.append(local[ground])
        depth,valid,road=sparse_visibility_maps(world,ground,K,w2c)
        mask_b=np.asarray(Image.open(b/'dynamic_masks/dynamic_mask_0'/f'{fid}.png').resize((480,320),Image.Resampling.NEAREST))>0
        proposal_path=engineering/'all_vehicle_native49/native_predictions'/f'{fid}.npz'
        with np.load(proposal_path,allow_pickle=False) as z:mask_native=z['masks'].any(0)
        mask_native=np.asarray(Image.fromarray(mask_native).resize((480,320),Image.Resampling.NEAREST))>0
        occluder=(labels>0)|mask_b|mask_native
        sharpness=float(np.var(laplace(rgb.astype(float).mean(2)/255)))
        frames.append(dict(fid=fid,raw=raw,K=K,world_to_camera=w2c,rgb=rgb,labels=labels,
                           occluder_mask=occluder,depth_m=depth,depth_valid=valid,road_mask=road,
                           boxes=boxes,selected=selected,other_boxes=other_boxes,sharpness=sharpness))
        mask_b_path=b/'dynamic_masks/dynamic_mask_0'/f'{fid}.png'
        source_hashes[fid]={str(p):sha(p) for p in [pose_path,cal_path,ann_path,lidar_path,rgb_path,label_path,proposal_path,mask_b_path]}
    measured=np.concatenate(ground_local)
    # Deterministic 0.1m XY grid reduction limits density bias in repeated stationary frames.
    _,idx=np.unique(np.floor(measured[:,:2]/.1).astype(int),axis=0,return_index=True)
    measured=measured[idx]
    surface=RoadSurface(anchor,-plane[[0,1,3]]/plane[2],measured,
                        config['road']['surface_support_radius_m'])
    median=max(np.median([f['sharpness'] for f in frames]),1e-9)
    for f in frames:f['quality']=np.full((320,480),min(f['sharpness']/median,1.))
    query_mask=np.isin(frames[0]['labels'],[IDS.index(i)+1 for i in config['road']['query_actors']])
    yy,xx=np.nonzero(query_mask);pixels=np.c_[xx,yy]
    world,valid,supported=query_from_pixels(pixels,frames[0]['K'],frames[0]['world_to_camera'],surface)
    query_body_pixels=len(pixels);rejected_pixels=pixels[~valid].copy()
    pixels,world,supported=pixels[valid],world[valid],supported[valid]
    normals=surface.normals(world)
    colors=[];weights=[];uvs=[];reasons=[]
    for f in frames:
        c,w,uv,why=observations(world,normals,supported,f,config['road'])
        colors.append(c);weights.append(w);uvs.append(uv);reasons.append(why)
    colors,weights=np.asarray(colors),np.asarray(weights)
    kwargs=dict(min_views=config['road']['min_views'],color_threshold=config['road']['color_threshold_linear_rgb'])
    late=np.array([r in config['road']['source_raw'] for r in RAW])
    primary=fuse(colors[late],weights[late],np.array(RAW)[late].tolist(),**kwargs)
    all_views=fuse(colors,weights,RAW,**kwargs)
    # Full49 fusion is the fallback before any neighboring-asphalt prior.
    final={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in all_views.items()}
    preferred=primary['provenance']==1
    for key in ['rgb','confidence','source_count','provenance']:final[key][preferred]=primary[key][preferred]
    final['contributors'][:,preferred]=False
    final['contributors'][np.ix_(late,preferred)]=primary['contributors'][:,preferred]
    donor_xy=np.empty((0,2));donor_rgb=np.empty((0,3))
    donor_pixels=[];donor_regions=np.empty(0,np.int16)
    donor_cfg=config['road']
    if donor_cfg['donor_approved']:
        donor_mask,region_map,marking_mask=select_asphalt_donors(frames[0]['rgb'],
            donor_cfg['donor_polygons_raw89'],frames[0]['occluder_mask'],frames[0]['road_mask'])
        dy,dx=np.nonzero(donor_mask);donor_pixels=np.c_[dx,dy]
        dw,dvalid,dsupport=query_from_pixels(donor_pixels,frames[0]['K'],frames[0]['world_to_camera'],surface)
        ok=dvalid&dsupport
        donor_xy=transform(dw[ok],anchor_inv)[:,:2]
        donor_rgb=frames[0]['rgb'][dy[ok],dx[ok]]/255.
        donor_regions=region_map[dy[ok],dx[ok]]
        if len(donor_xy)<20:raise RuntimeError('approved donor lacks measured clean-road support')
    final=fill_from_donor(transform(world,anchor_inv)[:,:2],final,donor_xy,donor_rgb,
                          donor_cfg['donor_approved'],donor_regions=donor_regions)
    # Do not declare joint-module training ready when the road branch has no target.
    if not np.any(final['provenance']):raise RuntimeError('no verified same-world or approved donor target; preserve evidence and review input support')
    output.mkdir(parents=True);(output/'targets').mkdir();(output/'fullframes').mkdir()
    frame_meta={}
    evidence=dict(world=world,pixels_raw89=pixels,invalid_geometry_pixels_raw89=rejected_pixels,surface_supported=supported,
                  source_uv=np.asarray(uvs),candidate_weights=weights,
                  rgb=final['rgb'],provenance=final['provenance'],source_count=final['source_count'],
                  contributors=final['contributors'],candidate_count=final['candidate_count'],
                  primary_observed=preferred,donor_index=final['donor_index'])
    for key in reasons[0]:evidence['gate_'+key]=np.asarray([r[key] for r in reasons])
    np.savez_compressed(output/'query_evidence.npz',**evidence)
    for f in frames:
        uv,depth,inside=project(world,f['K'],f['world_to_camera'],(480,320))
        unrelated_blocked=ray_box_blocked(world,np.linalg.inv(f['world_to_camera'])[:3,3],f['other_boxes'])
        p=np.floor(uv+0.5).astype(int)
        target_rgb=np.zeros((3,320,480),np.float32);target_depth=np.ones((1,320,480),np.float32)
        provenance=np.zeros((1,320,480),np.uint8);confidence=np.zeros((1,320,480),np.float32)
        for i in np.argsort(depth)[::-1]:
            if not inside[i] or unrelated_blocked[i] or final['provenance'][i]==0:continue
            x,y=p[i]
            # Never paint through an unrelated object in another frame.
            if f['occluder_mask'][y,x] and f['labels'][y,x]==0:continue
            target_rgb[:,y,x]=final['rgb'][i];target_depth[:,y,x]=depth[i]*scale
            provenance[:,y,x]=final['provenance'][i];confidence[:,y,x]=final['confidence'][i]
        path=output/'targets'/f'{f["fid"]}.npz'
        np.savez_compressed(path,rgb=target_rgb,depth=target_depth,provenance=provenance,confidence=confidence)
        selected=[]
        for obb in f['selected']:
            selected.append(dict(center=transform(obb['center'][None],model)[0].tolist(),
                half=(obb['half']*scale).tolist(),axes=(model[:3,:3]@obb['axes']/scale).tolist()))
        frame_meta[f['fid']]=dict(world_to_camera=f['world_to_camera'].tolist(),selected_boxes_model=selected,
                                road_target=str(path.relative_to(output)),road_target_sha256=sha(path))
        if f['raw']==89 or f['raw']>=131:
            im=Image.fromarray(f['rgb']);draw=ImageDraw.Draw(im)
            for i in np.flatnonzero(inside):
                x,y=p[i];color={0:'#ff5261',1:'#32df88',2:'#d786ff'}[int(final['provenance'][i])]
                draw.point((int(x),int(y)),fill=color)
            if f['raw']==89:
                for x,y in rejected_pixels:draw.point((int(x),int(y)),fill='#ff5261')
                for poly in donor_cfg['donor_polygons_raw89']:draw.line([tuple(t) for t in poly]+[tuple(poly[0])],fill='#ffbf47',width=2)
            im.save(output/'fullframes'/f'{f["fid"]}.png')
    receipt=dict(status='CPU_INPUT_BUNDLE_VERIFIED',actors=IDS,config=config,
        frames=frame_meta,world_to_model=model.tolist(),meter_to_model=scale,
        ground_plane_model=plane_model.tolist(),source_hashes=source_hashes,
        query_raw=89,coordinate_anchor_frame=anchor_frame,model_anchor_frame=model_anchor_frame,
        SGCM_plane_coordinate_roundtrip_verified=True,
        query_body_pixels_total=query_body_pixels,query_pixels=len(pixels),
        geometry_invalid_unknown_pixels=len(rejected_pixels),geometry_unsupported=int((~supported).sum()),
        primary_raw131_137_observed=int(preferred.sum()),full49_observed=int((final['provenance']==1).sum()),
        prior=int((final['provenance']==2).sum()),unknown_valid_geometry=int((final['provenance']==0).sum()),
        unknown=int((final['provenance']==0).sum())+len(rejected_pixels),
        visibility_evidence='real nearest LiDAR depth <=1.5px support + measured ground + all vehicle masks + all labeled OBB segment tests',
        depth_is_dense_ground_truth=False,RGB_fusion_is_hidden_ground_truth=False,
        GPU_used=False,training_updates=0,quality_accepted=False,
        limitations=['Sparse measured depth leaves many unknowns.','Road plane/height uncertainty remains.',
                     'Blur screening is relative whole-frame Laplacian energy, not optical calibration.',
                     'Shadows and illumination may differ across source frames.','Prior is separate provenance, not real query-location evidence.'])
    write(output/'bundle.json',receipt)
    return receipt


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--engineering',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--config',type=Path,default=Path(__file__).with_name('config.json'))
    a=p.parse_args();r=prepare(a.engineering,a.output,read(a.config))
    print(json.dumps({k:r[k] for k in ['status','query_pixels','primary_raw131_137_observed','full49_observed','prior','unknown']}))
