"""保留E数据，新增连续世界查询面投影保护的source-only revision。"""
from __future__ import annotations
import os
if __name__ == '__main__':
    os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
from pathlib import Path
import argparse,json,copy,time
import numpy as np
from PIL import Image,ImageDraw
from scipy.spatial import ConvexHull
from scipy.ndimage import binary_dilation
if __package__:
    from .world_feature_field import transform_xyz,project_world_to_camera,camera_from_observation
else:
    from world_feature_field import transform_xyz,project_world_to_camera,camera_from_observation


def build(root):
    start=time.monotonic();root=Path(root);m=json.loads((root/'dataset.json').read_text());old_sources=json.loads((root/'sources.json').read_text())
    for name in ['sources_fineguard','observations_fineguard','all_frame_continuous_query_masks']:(root/name).mkdir(exist_ok=False)
    boxes=[];centers=[];corners=[];random_points=[];rng=np.random.default_rng(m['seed'])
    for sample in m['samples']:
        hole=np.asarray(Image.open(root/sample['hole_mask']))>0;y,x=np.nonzero(hole);matrix=np.asarray(sample['world_from_uv'])
        low=np.array([x.min(),y.min()]);high=np.array([x.max()+1,y.max()+1]);a=np.linspace(0,1,65)
        edge=np.concatenate([np.c_[low[0]+a*(high[0]-low[0]),np.full(len(a),low[1])],
            np.c_[np.full(len(a),high[0]),low[1]+a*(high[1]-low[1])],
            np.c_[high[0]-a*(high[0]-low[0]),np.full(len(a),high[1])],
            np.c_[np.full(len(a),low[0]),high[1]-a*(high[1]-low[1])]])
        bbox_world=transform_xyz(np.c_[edge,np.zeros(len(edge))],matrix)
        boxes.append(dict(id=sample['id'],edge_world=bbox_world,hole_texels=len(x),expanded_bbox_texels=int(np.prod(high-low))))
        centers.append(transform_xyz(np.c_[x+.5,y+.5,np.zeros(len(x))],matrix))
        corners.append(np.concatenate([transform_xyz(np.c_[x+dx,y+dy,np.zeros(len(x))],matrix) for dx,dy in [(0,0),(1,0),(1,1),(0,1)]]))
        jitter=rng.random((len(x),2));random_points.append(transform_xyz(np.c_[x+jitter[:,0],y+jitter[:,1],np.zeros(len(x))],matrix))
    audit_points=dict(fine_centers=np.concatenate(centers),fine_cell_corners=np.concatenate(corners),fine_cell_random=np.concatenate(random_points))
    source_by_raw={s['raw']:s for s in old_sources['sources']};new_sources=[];frame_rows=[];source_rows=[]
    bank=Path(m['source_config']['bank'])
    for raw in m['all_input_frames']:
        obs=json.loads((bank/'observations'/f'{raw:03d}.json').read_text());camera=camera_from_observation(obs);h,w=camera['image_hw']
        mask=Image.new('L',(w,h),0);draw=ImageDraw.Draw(mask);straddled=0
        camargs={k:camera[k] for k in ['camera_K_native','world_to_camera','image_hw','distortion_Brown']}
        for box in boxes:
            uv,depth,_=project_world_to_camera(box['edge_world'],**camargs)
            if depth.max()<=.1:continue
            if depth.min()<=.1:
                # 面穿过相机平面时保守遮住整帧，避免近裁剪/畸变造成未覆盖的连续面。
                draw.rectangle([0,0,w-1,h-1],fill=255);straddled+=1;continue
            uv=np.clip(np.nan_to_num(uv,nan=0,posinf=1e6,neginf=-1e6),-1e6,1e6)
            if len(uv)>=3:
                hull=uv[ConvexHull(uv).vertices]
                draw.polygon([tuple(p) for p in hull],fill=255)
        continuous=binary_dilation(np.asarray(mask)>0,iterations=4)
        Image.fromarray(continuous.astype(np.uint8)*255).save(root/'all_frame_continuous_query_masks'/f'{raw:03d}.png')
        frame_rows.append(dict(raw=raw,continuous_mask_pixels=int(continuous.sum()),camera_plane_straddling_boxes=straddled))
        if raw not in source_by_raw:continue
        old=source_by_raw[raw];allowed=np.asarray(Image.open(root/old['feature_allowed_mask']))>0
        allowed &= ~continuous
        image=np.asarray(Image.open(root/old['feature_input'])).copy();image[~allowed]=128
        new=copy.deepcopy(old);new['feature_input']=f'sources_fineguard/{raw:03d}_feature_input.png'
        new['feature_allowed_mask']=f'sources_fineguard/{raw:03d}_feature_allowed.png'
        new['observations']=f'observations_fineguard/{raw:03d}.npz'
        Image.fromarray(image).save(root/new['feature_input']);Image.fromarray(allowed.astype(np.uint8)*255).save(root/new['feature_allowed_mask'])
        with np.load(root/old['observations']) as d:values={k:d[k].copy() for k in d.files}
        p=np.clip(np.floor(values['uv_native']).astype(int),[0,0],[w-1,h-1]);keep=allowed[p[:,1],p[:,0]]
        np.savez_compressed(root/new['observations'],**{k:v[keep] for k,v in values.items()})
        new['observations_count']=int(keep.sum());new['mask_policy']='continuous world hole bounding rectangles with 65 edge samples/side; projected convex hull; 4 native pixels dilation; old masks additionally retained'
        row=dict(raw=raw,observations_removed=int((~keep).sum()),legal_observations=int(keep.sum()))
        for label,points in audit_points.items():
            uv,_,inside=project_world_to_camera(points,**camargs);p=np.clip(np.floor(uv[inside]).astype(int),[0,0],[w-1,h-1])
            leak=int(allowed[p[:,1],p[:,0]].sum());row[label+'_projected']=int(inside.sum());row[label+'_leak']=leak
            if leak:raise AssertionError((raw,label,leak))
        source_rows.append(row);new_sources.append(new)
    revision=copy.deepcopy(m);revision['sources']='sources_fineguard.json';revision['source_guard_revision']='continuous_world_bbox_native_r1'
    revision['data_revision']='r3_clear64_continuous_sourceguard_r1'
    revision['evidence_limits'].append('Source-only engineering repair: earlier coarse-center mask audit missed fine texels; continuous world bbox projection used before encoding for all methods in the C/D axis. E pixels/geometry/split are unchanged.')
    (root/'sources_fineguard.json').write_text(json.dumps(dict(schema='road_material_sources.v1',sources=new_sources),ensure_ascii=False,indent=2)+'\n')
    (root/'dataset_fineguard.json').write_text(json.dumps(revision,ensure_ascii=False,indent=2)+'\n')
    receipt=dict(status='PASS_CONTINUOUS_WORLD_HOLE_BEFORE_ENCODER_GUARD',seconds=time.monotonic()-start,
        manifest='dataset_fineguard.json',sources='sources_fineguard.json',old_sources_preserved=True,E_images_geometry_valid_split_unchanged=True,
        surface_policy='whole hole world bounding rectangle (ellipse conservatively expanded), 65 edge samples/side, Brown projection convex hull, 4 native pixels dilation',
        all_frame_masks=len(frame_rows),source_frames=len(source_rows),audit_point_counts={k:len(v) for k,v in audit_points.items()},
        fine_centers_leak=sum(r['fine_centers_leak'] for r in source_rows),fine_cell_corners_leak=sum(r['fine_cell_corners_leak'] for r in source_rows),
        fine_cell_random_leak=sum(r['fine_cell_random_leak'] for r in source_rows),sample_bounds=[{k:v for k,v in b.items() if k!='edge_world'} for b in boxes],
        all_frame_rows=frame_rows,source_rows=source_rows)
    (root/'continuous_source_guard_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['all_frame_rows','source_rows','sample_bounds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True,type=Path);a=p.parse_args();build(a.dataset)
