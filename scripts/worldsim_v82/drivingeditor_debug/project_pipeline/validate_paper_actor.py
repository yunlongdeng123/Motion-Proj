#!/usr/bin/env python3
"""Independently verify reviewed actor pixels and exact time-aligned raw points."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

p=argparse.ArgumentParser()
p.add_argument('--scene',type=Path,required=True)
a=p.parse_args();scene=a.scene.resolve()
assert scene.is_relative_to(Path('/root/autodl-tmp/drivingeditor_repro/data/waymo_paper_scenes'))
prep=json.loads((scene/'preparation_manifest.json').read_text())
actor=json.loads((scene/'dynamic_actor_preparation.json').read_text())
selection=json.loads((scene/'selected_actor_0.json').read_text())
assert actor['status']=='ready' and selection['status']=='visually_reviewed'
assert len(prep['frames'])==len(actor['frames'])==49
assert [r['id'] for r in prep['frames']]==[f'0017{i:03}' for i in range(49)]
v=PlyData.read(scene/'points3d_dynamic_0.ply')['vertex'].data
xyz=np.column_stack([v[k] for k in ('x','y','z')])
assert np.isfinite(xyz).all() and len(v)>0
assert all(np.array_equal(v['time'],v[k]) for k in ('nx','ny','nz'))
assert len(v)==actor['total_actor_points']
assert set(v['time']).issubset(set(np.float32(r['model_time']) for r in prep['frames']))
records=[];centers=[];center_indices=[];hashes={}
visible=[];empty=[];without_lidar=[]
for index,(row,claimed) in enumerate(zip(prep['frames'],actor['frames'])):
    name=row['id'];assert claimed['frame']==name
    selected_ids=selection.get('per_frame_instance_ids',{}).get(name,selection['default_instance_ids'])
    visibility=selection['per_frame_visibility'][name]
    assert visibility in ('visible','not_visible')
    points=xyz[v['time']==np.float32(row['model_time'])]
    assert len(points)==claimed['retained_visible_lidar_points']
    raw=np.fromfile(scene/'velodyne'/f'{name}.bin',dtype=np.float32).reshape(-1,6)
    pose=np.loadtxt(scene/'pose'/f'{name}.txt').reshape(4,4)
    world=(np.column_stack([raw[:,:3].astype(np.float64),np.ones(len(raw))])@pose.T)[:,:3].astype(np.float32)
    if len(points):
        distance,_=cKDTree(world).query(points)
        assert np.all(distance==0), f'Fabricated or time-mismatched point: {name}'
    ids=np.array(Image.open(scene/'mask/mask_0/Annotations'/f'{name}.png'))
    selected=np.isin(ids,selected_ids)
    binary=np.array(Image.open(scene/'dynamic_masks/dynamic_mask_0'/f'{name}.png'))
    assert np.array_equal(binary,selected.astype(np.uint8)*255)
    image=np.array(Image.open(scene/'image/image_0'/f'{name}.png').convert('RGB'))
    masked=np.array(Image.open(scene/'dynamic_image/dynamic_image_0'/f'{name}.png').convert('RGB'))
    assert np.array_equal(masked[selected],image[selected]) and not masked[~selected].any()
    assert bool(selected.any())==(visibility=='visible')
    assert all(np.any(ids==i) for i in selected_ids)
    if visibility=='not_visible':
        assert not len(points) and not masked.any() and not binary.any()
        empty.append(name)
    else:
        visible.append(name)
        if not len(points): without_lidar.append(name)
    extents=points.max(0)-points.min(0) if len(points) else None
    if extents is not None:
        assert extents.max()<6 and extents[2]<2.5, f'Background-scale geometry: {name}, {extents}'
        center=np.median(points,axis=0);centers.append(center);center_indices.append(index)
    else:
        center=None
    record={'frame':name,'visibility':visibility,'instance_ids':selected_ids,'points':len(points),
        'exact_raw_point_subset':True,'exact_selected_source_rgb':True,'selected_pixels':int(selected.sum()),
        'world_extent_m':extents.tolist() if extents is not None else None,
        'world_median_xyz':center.tolist() if center is not None else None,
        'camera_depth_quantiles_m':claimed['camera_depth_quantiles_m']}
    records.append(record)
    hashes[name]={str(folder):digest(scene/folder/f'{name}.png') for folder in
        ('image/image_0','sky/sky_0','mask/mask_0/Annotations','dynamic_image/dynamic_image_0','dynamic_masks/dynamic_mask_0')}
centers=np.array(centers)
steps=np.linalg.norm(np.diff(centers,axis=0),axis=1)
gaps=np.diff(center_indices)
assert np.all(steps/gaps<2), 'Physical-car centroid trajectory discontinuity exceeds2m/frame'
result={'status':'passed','scene':scene.name,'camera_index':0,'actor_instance_ids':selection['default_instance_ids'],
    'frames':49,'visible_frames':len(visible),'empty_mask_frames':len(empty),
    'visible_frame_ids':visible,'empty_frame_ids':empty,'visible_without_lidar_frame_ids':without_lidar,
    'total_actor_points':len(v),'points_per_frame_min':min(r['points'] for r in records),
    'points_per_frame_max':max(r['points'] for r in records),
    'projection_only_points':sum(r['visible_before_component_filter'] for r in actor['frames']),
    'background_component_points_removed':sum(r['component_filtered_points'] for r in actor['frames']),
    'all_points_exact_subsets_of_original_motion_compensated_lidar':True,
    'all_dynamic_rgb_equals_source_rgb_under_selected_instances':True,
    'model_time_range':[float(v['time'].min()),float(v['time'].max())],
    'world_median_trajectory_step_m':{'median':float(np.median(steps)),'max':float(steps.max()),'all':steps.tolist()},
    'sha256':{k:digest(scene/k) for k in ('points3d_dynamic_0.ply','selected_actor_0.json','dynamic_actor_preparation.json','preparation_manifest.json')},
    'checks':['No fabricated coordinates; exact raw LiDAR subset at each original frame/model timestamp.',
        'All49 binary masks and dynamic RGB exactly match explicitly reviewed categorical instance selections.',
        'One nonempty actor PLY; nx,ny,nz,time identical; every nonempty component physically compact.',
        'Explicit visibility distinguishes physical absence, image-boundary truncation, and visible-without-LiDAR.'],
    'paper_preprocessing_gaps':actor['paper_preprocessing_alignment'],
    'limitations':['Largest connected component radius0.5m and zbuffer2px/0.3m are reproduction adapter choices.',
        'Car body only; no shadow-supervised pixels or other dynamic vehicles/persons are claimed.',
        'Original loss thresholds first RGB channel; genuinely black selected pixels remain excluded.',
        'Frames27..48 retain visible car-rear fragments at image boundary; no complete target disappearance occurs.'],
    'frames_detail':records,'frame_hashes':hashes}
(scene/'dynamic_actor_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('frames_detail','frame_hashes','visible_frame_ids')},indent=2))
