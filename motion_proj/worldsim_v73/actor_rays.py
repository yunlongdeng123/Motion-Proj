"""从原始首回波构建Actor规范射线；被遮挡区域不被误标为空。"""
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
import torch

from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex, _transform
from .native_data import interpolate_pose


def load_actor_rays(dataset_root, scene_id, owner, build_sample_ids,index=None,tracks=None):
    index=index or NuScenesCameraIndex(Path(dataset_root))
    scene=next(s for s in index.scenes if s['name']==scene_id)
    samples=sorted([s for s in index.samples if s['scene_token']==scene['token']],key=lambda s:s['timestamp'])
    sample_ids={s['token'] for s in samples}
    if tracks is None:
        tracks=defaultdict(list)
        with (index.metadata_root/'sample_annotation.json').open('rb') as handle:
            for row in ijson.items(handle,'item'):
                if row['sample_token'] in sample_ids:
                    stamp=int(index.sample_by_token[row['sample_token']]['timestamp'])
                    tracks[row['instance_token']].append((stamp,_transform(row['translation'],row['rotation']),
                                                           np.asarray(row['size'],float)[[1,0,2]]))
        for rows in tracks.values(): rows.sort(key=lambda r:r[0])
    trajectory=tracks[owner]
    selected=[i for i,s in enumerate(samples) if s['token'] in build_sample_ids]
    if not selected: raise ValueError('没有对应的build采样帧')
    records=[]
    # held-out仅为build时间范围内未输入的原始采样帧，metadata位姿只读。
    for i,sample in enumerate(samples):
        if not min(selected)<=i<=max(selected): continue
        if sample['token'] not in build_sample_ids and i%3!=2: continue
        lidar=index.sample_data.get(index.data_by_sample_channel.get((sample['token'],'LIDAR_TOP'),''))
        if lidar is None or not (index.dataset_root/lidar['filename']).is_file(): continue
        stamp=int(lidar['timestamp'])
        pose=interpolate_pose(trajectory,stamp)
        if pose is None: continue
        world=index.sensor_points_world(sample['token']).astype(float)
        calibrated=index.calibrated[lidar['calibrated_sensor_token']]
        ego=index.ego_poses[lidar['ego_pose_token']]
        sensor_pose=_transform(ego['translation'],ego['rotation'])@_transform(calibrated['translation'],calibrated['rotation'])
        origin=(sensor_pose[:3,3]-pose[:3,3])@pose[:3,:3]
        local=(world-pose[:3,3])@pose[:3,:3]
        vector=local-origin
        ranges=np.linalg.norm(vector,axis=1)
        directions=vector/np.maximum(ranges[:,None],1e-8)
        size=min(trajectory,key=lambda r:abs(r[0]-stamp))[2]
        membership=np.zeros(len(world),dtype=int)
        target_inside=np.zeros(len(world),dtype=bool)
        for track,rows in tracks.items():
            # 单时刻轨迹仅在其已知时间可用；interpolate_pose不会对外部时间外推。
            other_pose=interpolate_pose(rows,stamp)
            if other_pose is None: continue
            other=(world-other_pose[:3,3])@other_pose[:3,:3]
            other_size=min(rows,key=lambda r:abs(r[0]-stamp))[2]
            inside=np.all(np.abs(other)<=other_size/2+.10,axis=1)
            membership+=inside
            if track==owner: target_inside=inside
        # 只根据已知box筛选可能穿过Actor邻域的原始束，不根据返回深度或模型结果筛选。
        extent=size/2+.50
        parallel=np.abs(directions)<1e-10
        inv=1/np.where(parallel,1,directions)
        left=(-extent-origin)*inv
        right=(extent-origin)*inv
        lower=np.where(parallel,-np.inf,np.minimum(left,right)).max(1)
        upper=np.where(parallel,np.inf,np.maximum(left,right)).min(1)
        outside_parallel=np.any(parallel&(np.abs(origin)>extent),axis=1)
        near=(upper>=np.maximum(lower,0))&~outside_parallel&(ranges>0)
        positive=target_inside&(membership==1)
        records.append({'sample_id':sample['token'],'sample_index':i,'timestamp_us':stamp,
            'role':'build' if sample['token'] in build_sample_ids else 'heldout_time',
            'origins_actor_m':torch.tensor(np.broadcast_to(origin,(near.sum(),3)).copy(),dtype=torch.float32),
            'directions_actor':torch.tensor(directions[near],dtype=torch.float32),
            'observed_first_range_m':torch.tensor(ranges[near],dtype=torch.float32),
            'positive_actor':torch.tensor(positive[near]),
            'ambiguous_owner':torch.tensor((membership>1)[near]),
            'points_actor_m':torch.tensor(local[near],dtype=torch.float32),
            'size_lwh_m':torch.tensor(size,dtype=torch.float32),
            'raw_scan_points':len(world),'near_box_rays':int(near.sum()),
            'owned_points':int((positive&near).sum())})
    return records


class ActorRayDataset:
    """共享一次传感器索引和注释读取，避免多Actor预处理重复解析完整元数据。"""
    def __init__(self,dataset_root,scene_ids):
        self.index=NuScenesCameraIndex(Path(dataset_root))
        names={s['token']:s['name'] for s in self.index.scenes if s['name'] in scene_ids}
        self.tracks={name:defaultdict(list) for name in names.values()}
        with (self.index.metadata_root/'sample_annotation.json').open('rb') as handle:
            for row in ijson.items(handle,'item'):
                sample=self.index.sample_by_token[row['sample_token']]
                if sample['scene_token'] not in names: continue
                self.tracks[names[sample['scene_token']]][row['instance_token']].append((int(sample['timestamp']),
                    _transform(row['translation'],row['rotation']),np.asarray(row['size'],float)[[1,0,2]]))
        for tracks in self.tracks.values():
            for rows in tracks.values(): rows.sort(key=lambda r:r[0])

    def actor(self,scene_id,owner,build_sample_ids):
        return load_actor_rays(self.index.dataset_root,scene_id,owner,build_sample_ids,
                               index=self.index,tracks=self.tracks[scene_id])
