"""只检查输入轨迹与文件时间，不做AI视频评分。"""
import json
from pathlib import Path
import numpy as np
import av

ROOT = Path('/root/autodl-tmp/data/dynamic_editing_v2/drivestudio_processed_10Hz/trainval')
INPUT = Path('/root/autodl-tmp/data/worldsim_v75_downstream_bench/adapter_inputs/omnidreams')
RUN = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-DOWNSTREAM-FULL-01/20260923-r1/omnidreams-r2')
for scene in ['179', '191', '204']:
    poses = {int(p.stem): np.loadtxt(p) for p in (ROOT/scene/'lidar_pose').glob('*.txt')}
    positions = np.array([poses[i][:3,3] for i in sorted(poses)])
    speed = np.linalg.norm(np.diff(positions,axis=0),axis=1)*10
    print(json.dumps({'scene':scene,'frames':len(poses),'ego_speed_by_10frame_mean_mps':[round(float(np.mean(speed[i:i+10])),3) for i in range(0,len(speed),10)]}))
for row in json.loads((INPUT/'index.json').read_text())['cases'][:4]:
    c=row['case']; root=ROOT/c['dataset']['scene_id']; event=c['anchor']['event_frame']
    if c['target']['role']=='ego':
        poses={int(p.stem):np.loadtxt(p) for p in (root/'lidar_pose').glob('*.txt')}
    else:
        ann=json.loads((root/'instances/instances_info.json').read_text())[c['target']['actor_key']]['frame_annotations']
        poses={int(f):np.array(p) for f,p in zip(ann['frame_idx'],ann['obj_to_world'])}
    pts=np.array([poses[i][:3,3] for i in sorted(poses)])
    counts={}
    for branch in ['original-nuscenes','factual','counterfactual']:
        with av.open(str(RUN/c['case_id']/(branch+'.mp4'))) as video:
            stream=video.streams.video[0]
            times=[float(f.pts*f.time_base) for f in video.decode(video=0)]
            counts[branch]={'fps':str(stream.average_rate),'frames':len(times),'first':times[0],'last':times[-1],'monotonic':bool(np.all(np.diff(times)>0))}
    print(json.dumps({'case_id':c['case_id'],'event':event,'track_range':[min(poses),max(poses)],'track_progress_6s_m':float(np.linalg.norm(poses[min(event+55,max(poses))][:3,3]-poses[event][:3,3])),'files':counts}))
