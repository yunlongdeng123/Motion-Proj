#!/usr/bin/env python3
"""Independent exact source-subset, RGB-mask and temporal consistency checks."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from plyfile import PlyData
from scipy.spatial import cKDTree

scene=Path('/root/autodl-tmp/drivingeditor_repro/data/waymo_scenes/0017085_demo_000_020')
prep=json.loads((scene/'preparation_manifest.json').read_text())
actor=json.loads((scene/'dynamic_actor_preparation.json').read_text())
selection=json.loads((scene/'selected_actor_0.json').read_text())
assert actor['status']=='ready' and selection['default_instance_ids']==[154]
v=PlyData.read(scene/'points3d_dynamic_0.ply')['vertex'].data
xyz=np.column_stack([v[k] for k in ('x','y','z')])
assert np.isfinite(xyz).all()
assert all(np.array_equal(v['time'],v[k]) for k in ('nx','ny','nz'))
assert len(v)==actor['total_actor_points'] and len(np.unique(v['time']))==21
records=[];centers=[]
for row,claimed in zip(prep['frames'],actor['frames']):
    name=row['id'];assert claimed['frame']==name
    points=xyz[v['time']==np.float32(row['model_time'])]
    assert len(points)==claimed['retained_visible_lidar_points'] and len(points)>0
    raw=np.fromfile(scene/'velodyne'/f'{name}.bin',dtype=np.float32).reshape(-1,6)
    pose=np.loadtxt(scene/'pose'/f'{name}.txt').reshape(4,4)
    world=(np.column_stack([raw[:,:3].astype(np.float64),np.ones(len(raw))])@pose.T)[:,:3].astype(np.float32)
    distance,_=cKDTree(world).query(points)
    assert np.all(distance==0), f'Actor contains a point absent from its raw LiDAR source: {name}'
    ids=np.array(Image.open(scene/'mask/mask_0/Annotations'/f'{name}.png'))
    binary=np.array(Image.open(scene/'dynamic_masks/dynamic_mask_0'/f'{name}.png'))
    assert np.array_equal(binary,(ids==154).astype(np.uint8)*255)
    image=np.array(Image.open(scene/'image/image_0'/f'{name}.png').convert('RGB'))
    masked=np.array(Image.open(scene/'dynamic_image/dynamic_image_0'/f'{name}.png').convert('RGB'))
    assert np.array_equal(masked[ids==154],image[ids==154]) and not masked[ids!=154].any()
    extents=points.max(0)-points.min(0)
    assert extents.max()<6 and extents[2]<2.5, f'Background-scale geometry remains: {name}'
    center=np.median(points,axis=0);centers.append(center)
    records.append({'frame':name,'points':len(points),'exact_raw_point_subset':True,
        'exact_selected_source_rgb':True,'selected_pixels':int((ids==154).sum()),
        'world_extent_m':extents.tolist(),'world_median_xyz':center.tolist(),
        'camera_depth_quantiles_m':claimed['camera_depth_quantiles_m']})
centers=np.array(centers);steps=np.linalg.norm(np.diff(centers,axis=0),axis=1)
assert steps.max()<2, 'Unexpected discontinuity in the physical-car point trajectory'
result={'status':'passed','scene':scene.name,'instance_id':154,'frames':21,
    'total_actor_points':len(v),'points_per_frame_min':min(r['points'] for r in records),
    'points_per_frame_max':max(r['points'] for r in records),
    'projection_only_points':sum(r['visible_before_component_filter'] for r in actor['frames']),
    'background_component_points_removed':sum(r['component_filtered_points'] for r in actor['frames']),
    'all_points_exact_subsets_of_original_motion_compensated_lidar':True,
    'all_dynamic_rgb_equals_source_rgb_under_selected_instance':True,
    'model_time_range':[float(v['time'].min()),float(v['time'].max())],
    'world_median_trajectory_step_m':{'median':float(np.median(steps)),'max':float(steps.max()),'all':steps.tolist()},
    'checks':['No empty actor frame; no fabricated coordinates; exact raw LiDAR subset per original timestamp.',
              'All21 binary masks exactly equal selected DEVA ID154; all RGBs exactly equal source under that mask.',
              'One nonempty actor PLY; nx,ny,nz,time identical; dominant component physically compact.',
              'World median trajectory has no step above2m between adjacent original0.1s frames.'],
    'limitations':['Largest component radius0.5m is a documented reproduction adapter choice, not an author-supplied preprocessing setting.',
                   'Original training loss thresholds the first RGB channel; genuinely black pixels are excluded by that original implementation.'],
    'frames_detail':records}
(scene/'dynamic_actor_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='frames_detail'},indent=2))
