"""仅CPU：train来源候选、连续RGB/GT几何、donor配对与公共盘提取计划。"""
from __future__ import annotations
import argparse
import bisect
import glob
import itertools
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import cv2
import ijson
import numpy as np
from nuscenes.utils.splits import train
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

cv2.setNumThreads(1)
EXPOSED = {'scene-0230', 'scene-0255', 'scene-0061', 'scene-0436', 'scene-0875', 'scene-0242', 'scene-0535', 'scene-0471', 'scene-0998'}
CAMS = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK']
TASK = 'WS-V77-TARGET-PROTECTED-20260929'

def read(p):
    return json.loads(Path(p).read_text())

def dump(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(p)

def stream(p):
    with Path(p).open('rb') as f:
        yield from ijson.items(f, 'item', use_float=True)

def mat(t, q):
    m = np.eye(4)
    m[:3, :3] = Quaternion(q).rotation_matrix
    m[:3, 3] = t
    return m

def project(obj, view):
    b = Box(obj['translation'], obj['size'], Quaternion(obj['rotation']))
    xyz = view['w2c'][:3, :3] @ b.corners() + view['w2c'][:3, 3:4]
    if xyz[2].min() < .5:
        return None
    xy = view['k'] @ xyz
    xy = xy[:2] / xy[2:]
    bounds = np.r_[xy.min(1), xy.max(1)]
    x0, y0, x1, y1 = bounds
    if x1 <= 0 or y1 <= 0 or x0 >= 1024 or y0 >= 576:
        return None
    center = view['w2c'][:3, :3] @ np.asarray(obj['translation']) + view['w2c'][:3, 3]
    forward = view['w2c'][:3, :3] @ Quaternion(obj['rotation']).rotation_matrix[:, 0]
    relative_yaw = math.atan2(forward[0], forward[2]) - math.atan2(center[0], center[2])
    relative_yaw = (relative_yaw + math.pi) % (2*math.pi) - math.pi
    return dict(box_xyxy=np.round(bounds, 3).tolist(), hull=cv2.convexHull(np.float32(xy.T)).reshape(-1, 2).tolist(),
                depth=float(center[2]), yaw_view_deg=math.degrees(relative_yaw),
                area_frac=float((x1-x0)*(y1-y0)/(1024*576)), width=float(x1-x0), height=float(y1-y0),
                border=float(min(x0, y0, 1024-x1, 576-y1)))

def intersects(a, b):
    return max(0., min(a[2], b[2])-max(a[0], b[0])) * max(0., min(a[3], b[3])-max(a[1], b[1]))

def good(p):
    return p and p['border'] >= 8 and p['width'] >= 72 and p['height'] >= 40 and .006 <= p['area_frac'] <= .18

def yawdist(a, b):
    return abs((a-b+180) % 360-180)

def sharding():
    by_log = defaultdict(set)
    patterns = ['/root/autodl-tmp/data/worldsim_v67/*member_shards*.json',
                '/root/autodl-tmp/data/worldsim_v5/manifests/*member_shards*.json',
                '/root/autodl-tmp/data/dynamic_editing_v2/manifests/*member_shards*.json']
    for pat in patterns:
        for p in glob.glob(pat):
            for n, s in read(p).items():
                value = str(s)
                shard = f'{int(value):02}' if value.isdigit() else value.split('trainval')[1][:2]
                by_log[Path(n).name.split('__')[0]].add(shard)
    return by_log

def make_index(meta, root):
    scenes = [s for s in read(meta/'scene.json') if s['name'] in train and s['name'] not in EXPOSED
              and not any(k in s['description'].lower() for k in ['night', 'rain', 'dark'])]
    sc = {s['token']: s for s in scenes}
    samples = {s['token']: s for s in read(meta/'sample.json') if s['scene_token'] in sc}
    ordered = defaultdict(list)
    for t, s in samples.items():
        ordered[s['scene_token']].append(t)
    for arr in ordered.values():
        arr.sort(key=lambda t: samples[t]['timestamp'])
    cat = {c['token']: c['name'] for c in read(meta/'category.json')}
    inst = {r['token']: cat[r['category_token']] for r in read(meta/'instance.json')}
    ann = defaultdict(list)
    for r in stream(meta/'sample_annotation.json'):
        if r['sample_token'] in samples and inst[r['instance_token']].startswith('vehicle.'):
            ann[r['sample_token']].append({k:r[k] for k in ['instance_token','translation','rotation','size','visibility_token','num_lidar_pts']} | {'category':inst[r['instance_token']]})
    print('ANNOTATIONS_READY', len(scenes), sum(map(len, ann.values())), flush=True)
    channels = {r['token']:r['channel'] for r in read(meta/'sensor.json')}
    cal = {r['token']:r for r in read(meta/'calibrated_sensor.json')}
    sd = defaultdict(dict)
    needed_ego = set()
    for r in stream(meta/'sample_data.json'):
        if r['sample_token'] in samples and r['is_key_frame']:
            channel = channels[cal[r['calibrated_sensor_token']]['sensor_token']]
            if channel in CAMS:
                sd[r['sample_token']][channel] = {k:r[k] for k in ['token','filename','timestamp','ego_pose_token','calibrated_sensor_token']}
                needed_ego.add(r['ego_pose_token'])
    ego = {r['token']:r for r in stream(meta/'ego_pose.json') if r['token'] in needed_ego}
    print('KEYFRAMES_READY',len(sd),len(ego),flush=True)
    shard_logs = sharding()
    logs = {r['token']:r for r in read(meta/'log.json')}
    candidates=[]
    for si, scene in enumerate(scenes):
        rows=ordered[scene['token']]
        by_scene=[]
        for start in [3, 11, 19, 27]:
            keys=rows[start:start+8]
            if len(keys)!=8: continue
            tracks=defaultdict(list)
            for key in keys:
                for a in ann[key]: tracks[a['instance_token']].append(a)
            complete={t:rr for t,rr in tracks.items() if len(rr)==8 and all(int(a['visibility_token'])==4 for a in rr)
                      and np.median([a['num_lidar_pts'] for a in rr])>=8 and rr[0]['category'] in ['vehicle.car','vehicle.truck','vehicle.bus.rigid','vehicle.bus.bendy']}
            if not complete:continue
            for channel in CAMS:
                if any(channel not in sd[k] for k in keys):continue
                views=[]
                for k in keys:
                    datum=sd[k][channel]; c=cal[datum['calibrated_sensor_token']]; e=ego[datum['ego_pose_token']]
                    c2w=mat(e['translation'],e['rotation'])@mat(c['translation'],c['rotation'])
                    intr=np.asarray(c['camera_intrinsic'])*np.array([[.64,.64,.64],[.64,.64,.64],[1,1,1]])
                    views.append({'w2c':np.linalg.inv(c2w),'k':intr})
                actors=[]
                for t,rr in complete.items():
                    pp=[project(a,v) for a,v in zip(rr,views)]
                    if not all(good(p) for p in pp):continue
                    sizes=np.array([[p['width'],p['height']] for p in pp])
                    if np.max(sizes[1:]/sizes[:-1])>1.65 or np.min(sizes[1:]/sizes[:-1])<.6:continue
                    nearby=[]
                    for other in ann[keys[3]]:
                        if other['instance_token']==t:continue
                        op=project(other,views[3])
                        if op:
                            overlap=intersects(pp[3]['box_xyxy'],op['box_xyxy'])/(pp[3]['width']*pp[3]['height'])
                            if overlap>.02: nearby.append({'instance_token':other['instance_token'],'overlap':overlap,'depth':op['depth']})
                    foreground=max([n['overlap'] for n in nearby if n['depth']<pp[3]['depth']] or [0])
                    if foreground>.08:continue
                    actor={'instance_token':t,'category':rr[0]['category'],'anchor_projection':pp[3],
                           'keyframe_projections':pp,'keyframe_annotations':rr,'max_front_box_overlap':foreground,
                           'overlapping_GT_neighbors':nearby}
                    actors.append(actor)
                if not actors:continue
                # 可辨主actor与无明显前景包络重叠仅作素材门槛；真实可见性后续逐例看图。
                actors.sort(key=lambda a:(a['max_front_box_overlap'],abs(a['anchor_projection']['area_frac']-.028),a['instance_token']))
                datum=sd[keys[3]][channel]
                logprefix=Path(datum['filename']).name.split('__')[0]
                shards=sorted(shard_logs.get(logprefix,set()))
                if not shards:continue
                by_scene.append({'scene':scene['name'],'scene_token':scene['token'],'description':scene['description'],
                    'location':logs[scene['log_token']]['location'],'log_token':scene['log_token'],
                    'camera':channel,'start_keyframe':start,'keyframe_tokens':keys,
                    'start_timestamp_us':samples[keys[0]]['timestamp'],'keyframe_timestamps':[samples[k]['timestamp'] for k in keys],
                    'shards':shards,'actors':actors[:4],
                    'anchor_rgb':datum['filename'],'anchor_sample_data_token':datum['token']})
        # 每scene最多3个预案，优先完整可辨和有多个protected候选的相机。
        by_scene.sort(key=lambda c:(-len(c['actors']),c['actors'][0]['max_front_box_overlap'],abs(c['actors'][0]['anchor_projection']['area_frac']-.028),c['start_keyframe'],c['camera']))
        candidates.extend(by_scene[:3])
        if si%60==0:print('SCENES_SCAN',si,'CANDIDATES',len(candidates),flush=True)
    result={'task_id':TASK,'run_id':'r1','scene_count':len(scenes),'candidate_count':len(candidates), 'candidates':candidates}
    dump(root/'candidate_pool.json',result)
    return result

def select(pool, root, limit):
    # 在CPU半核下冻结最多2个已有公共分片作为数据工厂试产，不宣称代表完整train分布。
    # 优先选能覆盖最多合格scene的两分片；质量筛选仍只依赖metadata，不看模型输出。
    scores=[]
    for shards in itertools.combinations([f'{i:02}' for i in range(1,11)],2):
        eligible=[c for c in pool['candidates'] if set(c['shards']).issubset(shards)]
        scores.append((len({c['scene'] for c in eligible}), len({c['scene'] for c in eligible if len(c['actors'])>=2}),shards))
    _,_,shards=max(scores)
    elig=[c for c in pool['candidates'] if set(c['shards']).issubset(shards)]
    by_scene=defaultdict(list)
    for c in elig:by_scene[c['scene']].append(c)
    selected=[]
    for name,cc in sorted(by_scene.items()):
        c=min(cc,key=lambda c:(-len(c['actors']),c['actors'][0]['max_front_box_overlap'],c['start_keyframe'],c['camera']))
        selected.append(c)
    # 各location交错，保留最多64例候选用于剔除；不是用劣质例凑到50。
    buckets=defaultdict(list)
    for c in selected:buckets[c['location']].append(c)
    selected=[]
    while any(buckets.values()) and len(selected)<limit:
        for b in sorted(buckets):
            if buckets[b] and len(selected)<limit:selected.append(buckets[b].pop(0))
    for i,c in enumerate(selected):c['source_id']=f'S{i+1:03}'
    result={'task_id':TASK,'run_id':'r1','sample_role':'train_data_factory_pilot_not_test',
            'archive_scope':list(shards),'archive_scope_reason':'CPU0.5 quota; two-shard pilot selected by count of metadata-eligible scenes; not a representative audit',
            'candidate_count':len(selected),'distinct_scenes':len({c['scene'] for c in selected}),
            'locations':dict(Counter(c['location'] for c in selected)),
            'multi_actor_candidates':sum(len(c['actors'])>=2 for c in selected),
            'clips':selected,'human_verdict':None,'failure_ledger_refs':['V77-F02']}
    assert all(c['scene'] in train and c['scene'] not in EXPOSED for c in selected)
    dump(root/'source_selection.json',result)
    print('SELECTED',len(selected),'SHARDS',shards,'MULTI',result['multi_actor_candidates'],flush=True)
    return result

def resolve(meta, root, selection):
    samples={r['token']:r for r in read(meta/'sample.json')}
    wanted={c['scene_token'] for c in selection['clips']}
    channels={r['token']:r['channel'] for r in read(meta/'sensor.json')}
    cal={r['token']:r for r in read(meta/'calibrated_sensor.json')}
    wanted_cams={(c['scene_token'],c['camera']) for c in selection['clips']}
    camera=defaultdict(list)
    for r in stream(meta/'sample_data.json'):
        scene=samples[r['sample_token']]['scene_token']
        if scene not in wanted:continue
        cam=channels[cal[r['calibrated_sensor_token']]['sensor_token']]
        if (scene,cam) in wanted_cams:
            camera[(scene,cam)].append({k:r[k] for k in ['token','sample_token','timestamp','filename','width','height','ego_pose_token','calibrated_sensor_token']})
    for rr in camera.values():rr.sort(key=lambda r:r['timestamp'])
    all_rows={}
    planned=[]
    rejected=[]
    for c in selection['clips']:
        rows=camera[(c['scene_token'],c['camera'])]; times=[r['timestamp'] for r in rows]
        first_ix=bisect.bisect_left(times,c['start_timestamp_us']+50000)
        # 以真实相机曝光对齐10Hz网格，避免LiDAR相位落在两曝光的中点。
        first_exposure=times[first_ix]
        frames=[]
        for f in range(30):
            t=first_exposure+100000*f
            j=bisect.bisect_left(times,t)
            r=min([rows[k] for k in [j-1,j] if 0<=k<len(rows)],key=lambda r:abs(r['timestamp']-t))
            frames.append(r|{'frame':f,'requested_timestamp_us':t,'delta_ms':abs(r['timestamp']-t)/1000})
        stamps=[r['timestamp'] for r in frames]
        gaps=np.diff(stamps)/1000
        if max(r['delta_ms'] for r in frames)>55 or min(gaps)<=0 or max(gaps)>180:
            rejected.append({'source_id':c['source_id'],'reason':'exposure_sampling_gate','max_delta_ms':max(r['delta_ms'] for r in frames),'min_gap_ms':float(min(gaps)),'max_gap_ms':float(max(gaps))})
            continue
        planned.append(c|{'frames':frames,'max_exposure_gap_ms':float(max(gaps))})
        for r in frames:all_rows[r['token']]=r
    need_ego={r['ego_pose_token'] for r in all_rows.values()}
    ego={r['token']:r for r in stream(meta/'ego_pose.json') if r['token'] in need_ego}
    for c in planned:
        for f in c['frames']:
            ca=cal[f['calibrated_sensor_token']]; e=ego[f['ego_pose_token']]
            c2w=mat(e['translation'],e['rotation'])@mat(ca['translation'],ca['rotation'])
            k=np.asarray(ca['camera_intrinsic'],dtype=float);k[0]*=1024/f['width'];k[1]*=576/f['height']
            f['camera_to_world']=c2w.tolist();f['intrinsics_1024']=k.tolist()
            f['actors']=[]
            for a in c['actors']:
                stamps=c['keyframe_timestamps'];t=f['timestamp'];idx=bisect.bisect_right(stamps,t)
                if not (0<idx<len(stamps)):continue
                if not (0 < stamps[idx]-stamps[idx-1] <= 600000):continue
                low,hi=a['keyframe_annotations'][idx-1:idx+1]
                frac=(t-stamps[idx-1])/(stamps[idx]-stamps[idx-1])
                q=Quaternion.slerp(Quaternion(low['rotation']),Quaternion(hi['rotation']),amount=frac)
                obj={'translation':((1-frac)*np.array(low['translation'])+frac*np.array(hi['translation'])).tolist(),
                     'size':((1-frac)*np.array(low['size'])+frac*np.array(hi['size'])).tolist(),
                     'rotation':q.elements.tolist()}
                p=project(obj,{'w2c':np.linalg.inv(c2w),'k':k})
                f['actors'].append({'instance_token':a['instance_token'],'category':a['category'],**obj,'projection':p})
        # 全30帧都必须有主要目标；真实曝光略早于起始annotation时不外推。
        actor_gate={}
        for a in c['actors']:
            token=a['instance_token']
            bad=[f['frame'] for f in c['frames'] if not any(x['instance_token']==token and good(x['projection']) for x in f['actors'])]
            actor_gate[token]={'pass':not bad,'bad_frames':bad}
        c['actor_geometry_gate']=actor_gate
        c['source_geometry_status']='pending_pixel_review' if actor_gate[c['actors'][0]['instance_token']]['pass'] else 'geometry_reject'
    output={'task_id':TASK,'run_id':'r1','selection':str(root/'source_selection.json'),
            'clips':planned,'sampling_rejected':rejected,'unique_rgb_file_count':len({f['filename'] for c in planned for f in c['frames']})}
    dump(root/'source_manifest.json',output)
    print('RESOLVED',len(planned),'FILES',output['unique_rgb_file_count'],'SAMPLING_REJECTED',len(rejected),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--meta',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--limit',type=int,default=64)
    args=p.parse_args();args.root.mkdir(parents=True,exist_ok=True)
    pool=read(args.root/'candidate_pool.json') if (args.root/'candidate_pool.json').exists() else make_index(args.meta,args.root)
    sel=read(args.root/'source_selection.json') if (args.root/'source_selection.json').exists() else select(pool,args.root,args.limit)
    if not (args.root/'source_manifest.json').exists():resolve(args.meta,args.root,sel)

if __name__=='__main__':main()
