"""追踪背景错误首交点所用build支持；区分近传感器候选与掠射，不改评价数据。"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import open3d as o3d
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex,_transform
from motion_proj.worldsim_v73.scene_readout import SurfaceBVH

parser=argparse.ArgumentParser(); parser.add_argument('--scene-data',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True); args=parser.parse_args(); started=time.monotonic()
index=NuScenesCameraIndex(Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval'))
scenes=json.loads((args.scene_data/'index.json').read_text())['scenes']; records=[]
for scene in scenes:
    folder=args.scene_data/scene['scene']; surface=np.load(folder/'background.npz')
    vertices=surface['vertices_world_m']; faces=surface['faces']; centers=vertices.reshape(-1,9,3).mean(1)
    token=next(s['token'] for s in index.scenes if s['name']==scene['scene'])
    samples=sorted([s for s in index.samples if s['scene_token']==token],key=lambda s:s['timestamp'])
    near=np.zeros(len(centers),bool); raw_counts=[]
    for frame in scene['build']:
        sample=samples[frame['sample_index']]
        lidar=index.sample_data[index.data_by_sample_channel[(sample['token'],'LIDAR_TOP')]]
        calibration=index.calibrated[lidar['calibrated_sensor_token']]; ego=index.ego_poses[lidar['ego_pose_token']]
        pose=_transform(ego['translation'],ego['rotation'])@_transform(calibration['translation'],calibration['rotation'])
        local=(centers-pose[:3,3])@pose[:3,:3]
        near|=(np.abs(local[:,0])<1)&(np.abs(local[:,1])<1)
        raw=np.fromfile(index.dataset_root/lidar['filename'],np.float32).reshape(-1,5)
        raw_counts.append({'sample_index':frame['sample_index'],'raw_points':len(raw),
            'sdk_close_xy1m':int(((np.abs(raw[:,0])<1)&(np.abs(raw[:,1])<1)).sum()),
            'range_under2m':int((np.linalg.norm(raw[:,:3],axis=-1)<2).sum())})
    bvh=SurfaceBVH(vertices,faces)
    for frame in scene['frames']:
        data=np.load(folder/frame['file']); directions=data['directions_world']
        rays=np.concatenate([np.broadcast_to(data['origin_world_m'],directions.shape),directions],axis=-1).astype(np.float32)
        result=bvh.scene.cast_rays(o3d.core.Tensor(rays),nthreads=2)
        depth=result['t_hit'].numpy(); ids=result['primitive_ids'].numpy()
        observed=data['observed_first_range_m']; valid=np.isfinite(depth)
        early=valid&(depth<observed-.2); severity=np.where(early,observed-.2-depth,0)
        close=np.zeros(len(depth),bool); grazing=np.zeros(len(depth),bool)
        tri=vertices[faces[ids[valid]]]; normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normal/=np.maximum(np.linalg.norm(normal,axis=-1,keepdims=True),1e-12)
        close[valid]=near[ids[valid]//8]
        grazing[valid]=np.abs((normal*directions[valid]).sum(-1))<.1
        group={}
        for name,mask in [('all',np.ones(len(depth),bool)),('close_to_any_build_sensor_xy1m',close),
                          ('grazing_abs_cos_lt01',grazing),('cohort_return',np.isin(data['observed_owner'],scene['cohort_ids']))]:
            take=early&mask
            group[name]={'early_rays':int(take.sum()),'intrusion_sum_m':float(severity[take].sum()),
                'median_first_depth_m':float(np.median(depth[take])) if take.any() else None,
                'median_observed_depth_m':float(np.median(observed[take])) if take.any() else None}
        records.append({'scene':scene['scene'],'sample_index':frame['sample_index'],'rays':len(depth),
            'groups':group,'background_close_patches':int(near.sum()),'build_raw_proximity':raw_counts})
value={'status':'done','scene_data':str(args.scene_data),'frames':records,'wall_s':time.monotonic()-started,
    'boundary':'overlapping diagnostic groups, not causal proof or ego semantic labels; close xy1m follows devkit geometric convention; no point removal or training change'}
args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(value,indent=2)+'\n')
for name in records[0]['groups']:
    print(json.dumps({'group':name,'early_rays':sum(r['groups'][name]['early_rays'] for r in records),
                      'intrusion_sum_m':sum(r['groups'][name]['intrusion_sum_m'] for r in records)}))
