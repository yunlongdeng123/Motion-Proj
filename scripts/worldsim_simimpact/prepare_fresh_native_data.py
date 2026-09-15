"""只提取冻结新日志的完整传感器文件，并保存轻量姿态/标定索引。"""
import json,subprocess,shutil,time
from pathlib import Path
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-FRESH-NATIVE-01/20260915-r1')
M=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval');N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
if (R/'source_manifest.json').exists():raise RuntimeError('Preserve registered data stage')
shutil.copy2(__file__,R/'data_source_snapshot.py');reg=json.loads((R/'registration.json').read_text());assert reg['selection_completed'] and reg['scenes']
names=reg['scenes'];scenes={x['name']:x for x in json.loads((M/'scene.json').read_text()) if x['name'] in names};st={s['token'] for s in scenes.values()}
samples={s['token']:s for s in json.loads((M/'sample.json').read_text()) if s['scene_token'] in st};sd={d['token']:d for d in json.loads((M/'sample_data.json').read_text()) if d['sample_token'] in samples and ('/CAM_' in d['filename'] or '/LIDAR_TOP/' in d['filename'])}
et={x['ego_pose_token'] for x in sd.values()};ct={x['calibrated_sensor_token'] for x in sd.values()}
ego={e['token']:e for e in json.loads((M/'ego_pose.json').read_text()) if e['token'] in et};cal={c['token']:c for c in json.loads((M/'calibrated_sensor.json').read_text()) if c['token'] in ct}
for s in samples.values():s['data']={}
for d in sd.values():
    if d['is_key_frame']:samples[d['sample_token']]['data'][d['filename'].split('/')[1]]=d['token']
(R/'metadata').mkdir();data=R/'data';data.mkdir();(data/'v1.0-trainval').symlink_to(M,target_is_directory=True);(data/'maps').symlink_to((N/'data/maps').resolve(),target_is_directory=True)
for name,scene in scenes.items():
    ss={k:v for k,v in samples.items() if v['scene_token']==scene['token']};ds={k:v for k,v in sd.items() if v['sample_token'] in ss};ets={d['ego_pose_token'] for d in ds.values()};cts={d['calibrated_sensor_token'] for d in ds.values()}
    (R/'metadata'/f'{name}.json').write_text(json.dumps({'scene':{scene['token']:scene},'sample':ss,'sample_data':ds,'ego_pose':{k:ego[k] for k in ets},'calibrated_sensor':{k:cal[k] for k in cts}}))
members=sorted({d['filename'] for d in sd.values()});(R/'raw_members.txt').write_text('\n'.join(members)+'\n')
manifest={'scenes':names,'members':members,'counts':{n:sum(d['sample_token'] in {k for k,s in samples.items() if s['scene_token']==scene['token']} for d in sd.values()) for n,scene in scenes.items()},'start_unix':time.time()}
(R/'source_manifest.json').write_text(json.dumps(manifest,indent=2));print('EXTRACTING',names,len(members),flush=True)
archive='/root/autodl-pub/nuScenes/Fulldatasetv1.0/Trainval/v1.0-trainval01_blobs.tgz'
proc=subprocess.run(['tar','-xzf',archive,'-C',str(data),'--no-recursion','-T',str(R/'raw_members.txt')])
missing=[x for x in members if not (data/x).is_file()];result={'archive':archive,'returncode':proc.returncode,'requested':len(members),'missing':missing,'end_unix':time.time()}
(R/'extraction_result.json').write_text(json.dumps(result,indent=2));print('EXTRACTED',len(members)-len(missing),'missing',len(missing),flush=True);assert not missing
