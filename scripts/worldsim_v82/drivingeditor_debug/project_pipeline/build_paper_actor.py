#!/usr/bin/env python3
"""Build one real actor cloud and masked RGB from explicitly reviewed DEVA IDs.

No boxes or semantic ground truth are used. Points come only from the existing
motion-compensated LiDAR arrays, projected into selected FRONT instance masks.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def save_ply(path,xyz,times):
    dtype=np.dtype([(x,'<f4') for x in ('x','y','z','nx','ny','nz')]+
                   [(x,'u1') for x in ('red','green','blue')]+[('time','<f4')])
    array=np.zeros(len(xyz),dtype=dtype)
    for i,key in enumerate(('x','y','z')): array[key]=xyz[:,i]
    for key in ('nx','ny','nz','time'): array[key]=times
    header=f'ply\nformat binary_little_endian 1.0\nelement vertex {len(xyz)}\n'
    header+=''.join('property '+('uchar' if name in ('red','green','blue') else 'float')+' '+name+'\n' for name in dtype.names)
    header+='end_header\n'
    with path.open('xb') as stream:
        stream.write(header.encode('ascii'));array.tofile(stream)

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--scene',type=Path,required=True)
parser.add_argument('--selection',type=Path,required=True)
parser.add_argument('--zbuffer-pixel-size',type=int,default=2)
parser.add_argument('--depth-tolerance',type=float,default=0.30)
parser.add_argument('--component-radius',type=float,default=0.50)
args=parser.parse_args()
assert args.zbuffer_pixel_size>=1 and args.depth_tolerance>=0 and args.component_radius>0
scene=args.scene.resolve()
assert scene.is_relative_to(Path('/root/autodl-tmp/drivingeditor_repro/data/waymo_paper_scenes'))
selection=json.loads(args.selection.read_text())
assert selection['status']=='visually_reviewed', 'Explicit visual actor selection is required'
assert selection['scene']==scene.name and selection['camera_index']==0
assert selection['actor_index']==0, 'This launcher intentionally prepares one reviewed actor'
assert selection.get('evidence') and selection.get('target_description')
default_ids=selection.get('default_instance_ids',[])
overrides=selection.get('per_frame_instance_ids',{})
visibility=selection['per_frame_visibility']
preparation=json.loads((scene/'preparation_manifest.json').read_text())
frames=preparation['frames']
mask_root=scene/'mask/mask_0'
assert json.loads((mask_root/'mask_validation.json').read_text())['status']=='passed'
assert json.loads((mask_root/'inference_manifest.json').read_text())['status']=='validated'
rgb_root=scene/'dynamic_image/dynamic_image_0'
binary_root=scene/'dynamic_masks/dynamic_mask_0'
ply_path=scene/'points3d_dynamic_0.ply'
report_path=scene/'dynamic_actor_preparation.json'
for output in (rgb_root,binary_root,ply_path,report_path):
    assert not output.exists(), f'Refuse to overwrite existing actor artifact: {output}'
rgb_root.mkdir(parents=True);binary_root.mkdir(parents=True)
report={'status':'processing','actor_index':0,'scene':scene.name,'target_description':selection['target_description'],
    'selection_path':str(args.selection),'selection_sha256':digest(args.selection),
    'adapter_sha256':digest(Path(__file__)),
    'input_preparation_manifest_sha256':digest(scene/'preparation_manifest.json'),
    'deva_manifest_sha256':digest(mask_root/'inference_manifest.json'),
    'geometry_source':'Existing official-converter Nx6 LiDAR; vehicle XYZ transformed with each frame ego pose',
    'projection':'Official pinhole P0 and Tr_velo_to_cam_0; positive camera depth and nearest image pixel',
    'occlusion_filter':{'pixel_bin_size':args.zbuffer_pixel_size,'depth_tolerance_m':args.depth_tolerance,
        'rule':'Retain mask-selected points no farther than nearest in-frustum LiDAR depth in the same pixel bin plus tolerance'},
    'component_filter':{'radius_m':args.component_radius,
        'rule':'Retain the largest connected component of real mask-projected LiDAR points using Euclidean-radius adjacency, independently per frame.',
        'reason':'Projection-only audit found background returns up to70m through/near the car silhouette; the actual car is approximately15-20m away. No 3D ground-truth boxes are used.'},
    'model_time':'Exact preparation_manifest frame model_time; written identically to nx,ny,nz,time',
    'uses_3d_ground_truth_boxes':False,
    'paper_preprocessing_alignment':{'instance_method':'DEVA/SAM with visually reviewed silver-sedan ID',
        'shadow_supervision':'not prepared; no synthetic shadows',
        'lidar_motion_method':'mask projection plus0.5m dominant component; M-detector not used',
        'scope':'Single published silver-car actor body; other instances not claimed dynamic'},
    'limitations':['This explicit LiDAR-to-instance adapter fills a preprocessing step not supplied by the author.',
        'Segmentation and sparse LiDAR can leave imperfect actor boundaries; inspect overlay and depth statistics.'],
    'frames':[]}
report_path.write_text(json.dumps(report,indent=2)+'\n')
clouds=[];times=[];samples=[]
sample_indices=set(np.linspace(0,len(frames)-1,min(5,len(frames)),dtype=int).tolist())
for local_index,row in enumerate(frames):
    name=row['id'];ids=overrides.get(name,default_ids)
    state=visibility[name]
    assert state in ('visible','not_visible')
    assert all(isinstance(i,int) and 1<=i<=255 for i in ids), f'Invalid reviewed IDs for {name}'
    assert bool(ids)==(state=='visible'), f'IDs/visibility disagree for {name}'
    rgb=np.array(Image.open(scene/'image/image_0'/f'{name}.png').convert('RGB'))
    id_mask=np.array(Image.open(mask_root/'Annotations'/f'{name}.png'))
    selected_mask=np.isin(id_mask,ids)
    assert bool(selected_mask.any())==(state=='visible'), f'Reviewed visibility disagrees with mask for {name}'
    assert all(np.any(id_mask==i) for i in ids), f'An explicitly selected ID is absent: {name}'
    calib={line.split(':',1)[0]:np.fromstring(line.split(':',1)[1],sep=' ')
           for line in (scene/'calib'/f'{name}.txt').read_text().splitlines() if ':' in line}
    projection=calib['P0'].reshape(3,4)
    extrinsic=calib['Tr_velo_to_cam_0'].reshape(3,4)
    raw=np.fromfile(scene/'velodyne'/f'{name}.bin',dtype=np.float32).reshape(-1,6)
    xyz=raw[:,:3].astype(np.float64)
    homogeneous=np.column_stack((xyz,np.ones(len(xyz))))
    camera=homogeneous@extrinsic.T
    positive=np.isfinite(camera).all(axis=1)&(camera[:,2]>0.01)
    ids_positive=np.flatnonzero(positive)
    uvw=np.column_stack((camera[positive],np.ones(positive.sum())))@projection.T
    uv=np.floor(uvw[:,:2]/uvw[:,2:3]+0.5).astype(np.int64)
    h,w=selected_mask.shape
    inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    point_indices=ids_positive[inside];uv=uv[inside]
    z=camera[point_indices,2]
    bin_w=(w+args.zbuffer_pixel_size-1)//args.zbuffer_pixel_size
    bin_h=(h+args.zbuffer_pixel_size-1)//args.zbuffer_pixel_size
    bins=(uv[:,1]//args.zbuffer_pixel_size)*bin_w+(uv[:,0]//args.zbuffer_pixel_size)
    nearest=np.full(bin_w*bin_h,np.inf)
    np.minimum.at(nearest,bins,z)
    target=selected_mask[uv[:,1],uv[:,0]]
    visible=z<=nearest[bins]+args.depth_tolerance
    keep=target&visible
    chosen=point_indices[keep]
    visible_count=len(chosen)
    if visible_count:
        edges=cKDTree(xyz[chosen]).query_pairs(args.component_radius,output_type='ndarray')
        graph=coo_matrix((np.ones(len(edges),dtype=np.uint8),(edges[:,0],edges[:,1])),
                         shape=(visible_count,visible_count))
        _,labels=connected_components(graph,directed=False)
        counts=np.bincount(labels)
        dominant=labels==counts.argmax()
        assert dominant.sum()>=visible_count*.5, f'No dominant physical actor component: {name}'
        keep[np.flatnonzero(keep)[~dominant]]=False
    else:
        counts=np.array([],dtype=int)
    chosen=point_indices[keep]
    pose=np.loadtxt(scene/'pose'/f'{name}.txt').reshape(4,4)
    world=(homogeneous[chosen]@pose.T)[:,:3].astype(np.float32)
    assert np.isfinite(world).all()
    clouds.append(world);times.append(np.full(len(world),row['model_time'],dtype=np.float32))
    masked_rgb=np.zeros_like(rgb);masked_rgb[selected_mask]=rgb[selected_mask]
    Image.fromarray(masked_rgb).save(rgb_root/f'{name}.png')
    Image.fromarray(selected_mask.astype(np.uint8)*255).save(binary_root/f'{name}.png')
    record={'frame':name,'model_time':row['model_time'],'instance_ids':ids,
        'visibility':state,'geometry_status':'real_lidar_points' if len(chosen) else ('visible_without_lidar' if state=='visible' else 'target_not_visible'),
        'source_lidar_sha256':digest(scene/'velodyne'/f'{name}.bin'),
        'source_instance_mask_sha256':digest(mask_root/'Annotations'/f'{name}.png'),
        'selected_image_pixels':int(selected_mask.sum()),'mask_projected_lidar_points':int(target.sum()),
        'visible_before_component_filter':visible_count,'retained_visible_lidar_points':len(chosen),
        'component_filtered_points':int(visible_count-len(chosen)),
        'largest_component_sizes':sorted(counts.tolist(),reverse=True)[:5],
        'occlusion_filtered_points':int((target&~visible).sum()),
        'camera_depth_quantiles_m':dict(zip(('min','p05','median','p95','max'),map(float,np.quantile(z[keep],[0,.05,.5,.95,1])))) if len(chosen) else None,
        'world_xyz_min':world.min(axis=0).tolist() if len(chosen) else None,
        'world_xyz_max':world.max(axis=0).tolist() if len(chosen) else None}
    report['frames'].append(record)
    if local_index in sample_indices:
        over=rgb.copy();over[selected_mask]=(rgb[selected_mask]*.55+np.array([255,120,20])*.45).astype(np.uint8)
        im=Image.fromarray(over);draw=ImageDraw.Draw(im)
        for u,v in uv[keep]: draw.ellipse((int(u)-2,int(v)-2,int(u)+2,int(v)+2),fill=(0,255,40))
        samples.append((name,Image.fromarray(masked_rgb),im,len(chosen)))
    print(json.dumps(record),flush=True)
xyz=np.concatenate(clouds);t=np.concatenate(times)
assert len(xyz)>0, 'No genuine physical actor points were found in the sequence'
save_ply(ply_path,xyz,t)
report.update(status='ready',total_actor_points=len(xyz),actor_ply=str(ply_path),
    actor_ply_sha256=digest(ply_path),dynamic_rgb_frames=len(frames),dynamic_binary_frames=len(frames),
    visible_frames=sum(x['visibility']=='visible' for x in report['frames']),
    empty_mask_frames=sum(x['visibility']=='not_visible' for x in report['frames']),
    visible_frames_without_lidar=sum(x['geometry_status']=='visible_without_lidar' for x in report['frames']))
report_path.write_text(json.dumps(report,indent=2)+'\n')
fontfile=Path('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf')
font=ImageFont.truetype(str(fontfile),16) if fontfile.exists() else ImageFont.load_default()
sheet=Image.new('RGB',(972,len(samples)*352),(235,238,242));draw=ImageDraw.Draw(sheet)
for index,(name,masked,overlay,count) in enumerate(samples):
    y=index*352;sheet.paste(masked.resize((480,320),Image.Resampling.LANCZOS),(0,y))
    sheet.paste(overlay.resize((480,320),Image.Resampling.LANCZOS),(492,y))
    draw.text((5,y+325),f'{name}: masked silver car / {count} real projected LiDAR points',font=font,fill=(0,0,0))
sheet.save(scene/'dynamic_actor_contact_sheet.jpg',quality=93)
print(json.dumps({'status':'ready','actor_points':len(xyz),'actor_ply':str(ply_path),'frames':len(frames)}),flush=True)
