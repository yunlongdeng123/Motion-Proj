"""按build/轨迹选每日志一个Actor，共享真实束和相机时刻几何，不读取评价质量。"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from motion_proj.worldsim_v73.actor_rays import ActorRayDataset
from motion_proj.worldsim_v73.native_data import interpolate_pose


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--coverage',type=Path,required=True)
    parser.add_argument('--native-run',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--selection',choices=['log_median','all_window_rigid'],default='log_median')
    args=parser.parse_args()
    torch.set_num_threads(4)
    args.output.mkdir(parents=True,exist_ok=False)
    groups=defaultdict(list)
    coverage=json.loads(args.coverage.read_text())['actors']
    for row in coverage:
        groups[(row['role'],row['log_id'])].append(row)
    selected=[]
    for key,rows in sorted(groups.items()):
        moving=[r for r in rows if (r['translation_speed_mps'] or 0)>2]
        pool=sorted(moving or rows,key=lambda r:(r['unique_projected_build_points'],r['scene'],r['owner']))
        selected.append({**pool[len(pool)//2],'selection':'moving if available, then median build support within log'})
    scenes={s['scene_id']:s for s in torch.load(args.native_run/'build_observations.pt',map_location='cpu',weights_only=False)}
    sensor=ActorRayDataset('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval',
                          set(scenes) if args.selection=='all_window_rigid' else {r['scene'] for r in selected})
    if args.selection=='all_window_rigid':
        metadata=sensor.index.metadata_root
        categories={r['token']:r['name'] for r in json.loads((metadata/'category.json').read_text())}
        instances={r['token']:categories[r['category_token']] for r in json.loads((metadata/'instance.json').read_text())}
        rigid=set(json.loads((args.native_run/'config.json').read_text())['data']['rigid_categories'])
        original={(r['scene'],r['owner']):r for r in coverage}
        selected=[]
        for scene_id,scene in scenes.items():
            stamps=sorted({v['lidar_time_us'] for v in scene['views']})
            for owner,trajectory in sensor.tracks[scene_id].items():
                if instances.get(owner) not in rigid: continue
                poses=[(stamp,interpolate_pose(trajectory,stamp)) for stamp in stamps]
                poses=[(stamp,pose) for stamp,pose in poses if pose is not None]
                if not poses: continue
                dt=(poses[-1][0]-poses[0][0])/1e6
                distance=np.linalg.norm(poses[-1][1][:3,3]-poses[0][1][:3,3])
                row=original.get((scene_id,owner),{'scene':scene_id,'log_id':scene['log_id'],'role':scene['role'],
                    'owner':owner,'category':instances[owner],'unique_projected_build_points':0,
                    'observed_views':0,'camera_channels':[],'nearest_camera_m':None})
                selected.append({**row,'window_duration_s':dt,'translation_speed_mps':float(distance/dt) if dt>0 else None,
                    'source_has_camera_lidar_correspondence':(scene_id,owner) in original,
                    'selection':'all known rigid tracks with pose at build LiDAR time; missing inputs retained'})
        selected.sort(key=lambda r:(r['role'],r['log_id'],r['scene'],r['owner']))
    (args.output/'selection.json').write_text(json.dumps(selected,indent=2)+'\n')
    channel_ids={'CAM_FRONT':0,'CAM_FRONT_RIGHT':1,'CAM_BACK_RIGHT':2,'CAM_BACK':3,'CAM_BACK_LEFT':4,'CAM_FRONT_LEFT':5}
    summary=[]
    for row in selected:
        scene=scenes[row['scene']]; owner=row['owner']
        rays=sensor.actor(row['scene'],owner,{v['sample_id'] for v in scene['views']})
        build=[r for r in rays if r['role']=='build']
        parts=[r['points_actor_m'][r['positive_actor']] for r in build]
        points=torch.unique(torch.cat(parts),dim=0) if parts else torch.empty(0,3)
        views=[]; matrices=[]
        for i,view in enumerate(scene['views']):
            # 位姿可由已知轨迹获得，不依赖该scan是否恰好有Actor返回。
            pose=interpolate_pose(sensor.tracks[row['scene']][owner],view['camera_time_us'])
            if pose is not None:
                views.append(i)
                matrices.append(np.linalg.inv(view['world_from_camera'])@pose)
        entry={**row,'build_points':len(points),'actual_views':len(views),
               'heldout_frames':sum(r['role']=='heldout_time' for r in rays),
               'heldout_actor_returns':sum(r['owned_points'] for r in rays if r['role']=='heldout_time')}
        if not build or not len(points):
            entry['status']='unavailable_input'
            entry['reason']='no build LiDAR surface points; visual-only query initialization not wired; retained in cohort'
        else:
            entry['status']='ready'
        subset=[scene['views'][i] for i in views]
        t0=subset[0]['camera_time_us'] if subset else 0
        filename=row['scene']+'__'+owner+'.pt'
        entry['file']=filename
        size=build[0]['size_lwh_m'] if build else torch.tensor(sensor.tracks[row['scene']][owner][0][2],dtype=torch.float32)
        case={'metadata':entry,'rays':rays,'points_actor_m':points,'size_lwh_m':size,
            'view_indices':views,'camera_from_actor':torch.tensor(np.stack(matrices),dtype=torch.float32) if matrices else torch.empty(0,4,4),
            'intrinsics':torch.tensor(np.stack([v['intrinsics'] for v in subset]),dtype=torch.float32) if subset else torch.empty(0,3,3),
            'camera_ids':torch.tensor([channel_ids[v['camera_id']] for v in subset],dtype=torch.long),
            'time_offsets_s':torch.tensor([(v['camera_time_us']-t0)/1e6 for v in subset],dtype=torch.float32),
            'image_hw':list(scene['views'][0]['image'].shape[-2:])}
        torch.save(case,args.output/filename)
        summary.append(entry)
        print(json.dumps(entry),flush=True)
    result={'status':'done','code_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'native_run':str(args.native_run),'cases':summary,
        'selection':args.selection,
        'selection_boundary':'build metadata only, no score selection; window pose scope is not visual/occlusion ground truth',
        'scenes':[{'scene':s['scene_id'],'role':s['role'],'log_id':s['log_id'],'input_views':len(s['views'])} for s in scenes.values()],
        'point_ownership':'annotation box +0.1m with overlap exclusion, proxy membership',
        'pose_boundary':'known trajectory at each camera exposure and LiDAR scan timestamp; no per-point scan time'}
    (args.output/'index.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'ready':sum(r['status']=='ready' for r in summary),'selected':len(summary)}),flush=True)


if __name__=='__main__': main()
