"""保留指定日志的轻量标定/姿态索引；不依赖解析器已释放的全库对象。"""
import json
from pathlib import Path
N=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-NATIVE-LIDAR-01/20260915-r1')
M=N/'data/v1.0-trainval'
names=json.loads((N/'registration.json').read_text())['scenes']
scenes={x['name']:x for x in json.loads((M/'scene.json').read_text()) if x['name'] in names}
scene_tokens={x['token'] for x in scenes.values()}
samples={x['token']:x for x in json.loads((M/'sample.json').read_text()) if x['scene_token'] in scene_tokens}
sd={x['token']:x for x in json.loads((M/'sample_data.json').read_text()) if x['sample_token'] in samples and ('/CAM_' in x['filename'] or '/LIDAR_TOP/' in x['filename'])}
ego_tokens={x['ego_pose_token'] for x in sd.values()};cal_tokens={x['calibrated_sensor_token'] for x in sd.values()}
ego={x['token']:x for x in json.loads((M/'ego_pose.json').read_text()) if x['token'] in ego_tokens}
cal={x['token']:x for x in json.loads((M/'calibrated_sensor.json').read_text()) if x['token'] in cal_tokens}
for s in samples.values():s['data']={}
for d in sd.values():
    if d['is_key_frame']:samples[d['sample_token']]['data'][d['filename'].split('/')[1]]=d['token']
(N/'metadata').mkdir(exist_ok=True)
for name,scene in scenes.items():
    selected={k:v for k,v in samples.items() if v['scene_token']==scene['token']}
    selected_sd={k:v for k,v in sd.items() if v['sample_token'] in selected}
    et={s['ego_pose_token'] for s in selected_sd.values()};ct={s['calibrated_sensor_token'] for s in selected_sd.values()}
    payload={'scene':{scene['token']:scene},'sample':selected,'sample_data':selected_sd,
        'ego_pose':{k:ego[k] for k in et},'calibrated_sensor':{k:cal[k] for k in ct}}
    path=N/'metadata'/f'{name}.json'
    if path.exists():raise RuntimeError(f'Preserve existing metadata {path}')
    path.write_text(json.dumps(payload))
    print(name,len(selected),len(selected_sd),'records',flush=True)
