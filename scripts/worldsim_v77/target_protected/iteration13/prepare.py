"""仅CPU：固定旧数据、相机/轨迹和同窗LiDAR清单；不生成新遮挡。"""
from common import *
import bisect
from collections import defaultdict
import numpy as np
from pyquaternion import Quaternion
import ijson
from geometry_factory import transform

TRAIN=['M039','M041','M042','M044','M013','M014','M015','M017','M018','M019','M010','M011']
VAL=['M003','M006','M007','M001']

def rows(path):
    with path.open('rb') as f: yield from ijson.items(f,'item',use_float=True)

def at_time(track,t):
    times=[a['timestamp'] for a in track];k=bisect.bisect_left(times,t)
    if k<len(track) and times[k]==t: return dict(track[k])
    if k==0 or k==len(track): return None
    a,b=track[k-1:k+1]
    if b['timestamp']-a['timestamp']>1100000: return None
    f=(t-a['timestamp'])/(b['timestamp']-a['timestamp'])
    return dict(a, translation=((1-f)*np.array(a['translation'])+f*np.array(b['translation'])).tolist(),
                rotation=Quaternion.slerp(Quaternion(a['rotation']),Quaternion(b['rotation']),f).elements.tolist())

def lidar_files(scans):
    roots=[T/f'{r}/factory/rgb' for r in ['r8','r4','r16','r23','r37','r43']]+[A/'rgb']
    missing=[]
    for d in scans:
        p=next((r/d['filename'] for r in roots if (r/d['filename']).is_file()),None)
        d['path']=str(p) if p else None
        if p is None: missing.append(d['filename'])
    return missing

def main():
    if (O/'manifest.json').exists(): print('ALREADY_PREPARED');return
    O.mkdir(parents=True,exist_ok=True)
    dump(O/'prepare_state.json',{'stage':'metadata','pid':os.getpid()})
    catalog=read(T/'r8/dataset_catalog.json')['cases'];byid={c['case_id']:c for c in catalog}
    sources={c['source_id']:c for c in read(T/'r8/factory/source_manifest.json')['clips']}
    contexts={c['source_id']:c for c in read(T/'r8/factory/source_context.json')['clips']}
    cases=[]
    for cid in TRAIN+VAL:
        c=byid[cid];src=sources[c['source_id']];ctx=contexts[c['source_id']]
        assert c['assistant_score']==2 and c['technical_pass']
        tracks=defaultdict(list)
        for stamp,fr in zip(src['keyframe_timestamps'],ctx['frames']):
            for a in fr['annotations']:tracks[a['instance_token']].append(dict(a,timestamp=stamp))
        frames=[]
        for fr in src['frames']:
            actors=[a for tr in tracks.values() if (a:=at_time(tr,fr['timestamp'])) is not None]
            frames.append({k:fr[k] for k in ['timestamp','camera_to_world','intrinsics_1024']}|{'actors':actors})
        scans=[f['sensors']['LIDAR_TOP'] for f in ctx['frames'] if frames[0]['timestamp']<=f['sensors']['LIDAR_TOP']['timestamp']<=frames[-1]['timestamp']]
        cases.append(dict(case_id=cid,kind='synthetic',split=c['split'],scene=c['receiver_scene'],type=c['type'],
            folder=c['folder'],frame_indices=list(range(10)),frames=frames,scans=scans,target_token=None,
            source_case=c['dataset_id'],GT='Y is loss/evaluation only; never read by condition builder'))
    del sources,contexts,catalog
    real=read(T/'r21/real_input_plan.json')['cases'];expected={c['clip_id']:c for c in read(A/'expected_sources.json')['clips']}
    meta=A/'metadata/v1.0-trainval';sample={r['token']:r for r in read(meta/'sample.json')}
    needed_samples={t for c in real for t in expected[c['clip_id']]['keyframe_sample_tokens']}
    categories={r['token']:r['name'] for r in read(meta/'category.json')}
    inst={r['token']:categories[r['category_token']] for r in read(meta/'instance.json')}
    ann=defaultdict(list)
    for r in rows(meta/'sample_annotation.json'):
        if r['sample_token'] in needed_samples:
            ann[r['sample_token']].append({k:r[k] for k in ['instance_token','translation','size','rotation']}|
                {'category':inst[r['instance_token']],'timestamp':sample[r['sample_token']]['timestamp']})
    sensor={r['token']:r for r in read(meta/'sensor.json')}
    cal={r['token']:r for r in read(meta/'calibrated_sensor.json')}
    camera_tokens={expected[c['clip_id']]['frames'][i]['sample_data_token'] for c in real for i in c['frames']}
    sd={};lidar=[]
    for r in rows(meta/'sample_data.json'):
        if r['token'] in camera_tokens:sd[r['token']]=r
        elif r['sample_token'] in needed_samples and r['is_key_frame'] and sensor[cal[r['calibrated_sensor_token']]['sensor_token']]['channel']=='LIDAR_TOP':lidar.append(r)
    needed_ego={r['ego_pose_token'] for r in list(sd.values())+lidar}
    ego={r['token']:r for r in rows(meta/'ego_pose.json') if r['token'] in needed_ego}
    for c in real:
        exp=expected[c['clip_id']];tracks=defaultdict(list)
        for tok in exp['keyframe_sample_tokens']:
            for a in ann[tok]:tracks[a['instance_token']].append(a)
        frames=[]
        for i in c['frames']:
            d=sd[exp['frames'][i]['sample_data_token']];cs=cal[d['calibrated_sensor_token']];ep=ego[d['ego_pose_token']]
            K=np.array(cs['camera_intrinsic']);K[0]*=1024/d['width'];K[1]*=576/d['height']
            actors=[a for tr in tracks.values() if (a:=at_time(sorted(tr,key=lambda a:a['timestamp']),d['timestamp'])) is not None]
            frames.append({'timestamp':d['timestamp'],'camera_to_world':(transform(ep['translation'],ep['rotation'])@transform(cs['translation'],cs['rotation'])).tolist(),
                'intrinsics_1024':K.tolist(),'actors':actors})
        scans=[d|{'ego_pose':ego[d['ego_pose_token']],'calibrated_sensor':cal[d['calibrated_sensor_token']]} for d in lidar
               if d['sample_token'] in exp['keyframe_sample_tokens'] and frames[0]['timestamp']<=d['timestamp']<=frames[-1]['timestamp']]
        cases.append(dict(case_id=c['eval_id'],kind='real',split='real_DEV',scene=c['scene'],type='real_DELETE',folder=c['folder'],
            frame_indices=c['frames'],frames=frames,scans=scans,target_token=c['instance_token'],GT=None))
    missing=sorted(set(p for c in cases for p in lidar_files(c['scans'])))
    assert not ({c['scene'] for c in cases if c['split']=='train'}&{c['scene'] for c in cases if c['split']=='validation'})
    assert len(cases)==24
    plan={'task_id':'WS-V77-TARGET-PROTECTED-20260929','run_id':'r45','cases':cases,
          'steps':160,'lr':1e-4,'seed':6201,'eval_seed':42,'eval_steps':25,'train_size':[320,576],'eval_size':[576,1024],
          'channels':['O','N','U','Q'],'base':'original DriveEditor, frozen including original attention',
          'data_scope':'12 existing train cases / 4 existing held-scene synthetic DEV / 8 fixed real DEV; two dense train cases share one scene and pose but differ in H',
          'geometry_auxiliary':'GT camera and retained vehicle boxes; coarse geometry POC, not estimated semantic silhouettes',
          'N':'positive observed non-actor LiDAR returns only, never complement of O; outside H and dilated full actor envelopes',
          'arms':['adapter_off','all_unknown','conditioned'],'checkpoint_selection':'fixed final step160, no real-case checkpoint selection',
          'failure_ledger_refs':['V77-F02'],'failure_ledger_delta':'none; experiment preparation','human_verdict':None}
    dump(O/'manifest.json',plan);dump(O/'missing_lidar.json',{'files':missing,'count':len(missing)})
    dump(O/'prepare_state.json',{'stage':'CPU_metadata_ready','cases':len(cases),'missing_lidar':len(missing),
        'scans':sum(len(c['scans']) for c in cases),'pid':os.getpid()})
    print('PREPARED',len(cases),'MISSING_LIDAR',len(missing),flush=True)

if __name__=='__main__':main()
