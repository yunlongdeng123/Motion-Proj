"""仅查元数据及文件可用性，区分当前队列外载荷与真正未使用的确认日志。"""
import json,os,time
from collections import defaultdict
from pathlib import Path
import ijson
ROOT=Path(__file__).resolve().parents[1]
dataset=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval')
metadata=dataset/'v1.0-trainval'; start=time.monotonic()
def read(name): return json.loads((metadata/(name+'.json')).read_text())
channels=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK','CAM_BACK_LEFT','CAM_BACK_RIGHT','LIDAR_TOP']
present=set()
for channel in channels:
    folder=dataset/'samples'/channel
    if folder.is_dir():
        present.update('samples/'+channel+'/'+entry.name for entry in os.scandir(folder) if entry.is_file())
scenes=read('scene'); samples=defaultdict(list)
for sample in read('sample'): samples[sample['scene_token']].append(sample)
sensors={row['token']:row['channel'] for row in read('sensor')}
calibrated={row['token']:sensors[row['sensor_token']] for row in read('calibrated_sensor')}
available=defaultdict(set)
with (metadata/'sample_data.json').open('rb') as handle:
    for row in ijson.items(handle,'item'):
        if row['is_key_frame'] and row['filename'] in present and calibrated[row['calibrated_sensor_token']] in channels:
            available[row['sample_token']].add(calibrated[row['calibrated_sensor_token']])
split=json.loads((ROOT/'configs/worldsim_v72/e2_visual_split.json').read_text())
current_names=set(split['fit_scenes']+split['development_scenes'])
current_logs={row['log_token'] for row in scenes if row['name'] in current_names}
result=[]
for scene in scenes:
    rows=sorted(samples[scene['token']],key=lambda r:r['timestamp'])
    complete=[i for i,row in enumerate(rows) if set(channels)<=available[row['token']]]
    build=[i for i in complete if i%3!=2 and i not in [0,len(rows)-1]]
    result.append({'scene':scene['name'],'log_id':scene['log_token'],'current_cohort_scene':scene['name'] in current_names,
        'outside_current_logs':scene['log_token'] not in current_logs,'complete_seven_sensor_keyframes':len(complete),
        'available_build_indices':build,'four_build_times_available':len(build)>=4})
eligible=[row for row in result if row['four_build_times_available']]
outside=[row for row in eligible if row['outside_current_logs']]
historical={
    'scene-0139':['configs/worldsim_v64/p4n_fresh_native_voxel_uq_v2.yaml','configs/worldsim_v64/p2e_fresh_evidence_v1.yaml'],
    'scene-0379':['configs/worldsim_v5/m1_sam_diagnostic_scene0379_v1.yaml','configs/worldsim_v5/m3_development_clip_inventory_v1.yaml']}
for row in outside:
    row['historical_usage_references']=historical.get(row['scene'],[])
    row['fresh_confirmation_candidate']=False if row['scene'] in historical else None
roles=json.loads((ROOT/'configs/worldsim_v72/data_roles.json').read_text())['datasets']['nuscenes']['group_roles']
all_logs={row['log_token'] for row in scenes}
assigned=set().union(*(set(values) for values in roles.values()))
output={'wall_s':time.monotonic()-start,'metadata_only':True,'payload_content_read':False,'annotation_content_read':False,
    'model_or_metric_used_for_selection':False,'current_logs':len(current_logs),'metadata_scenes':len(result),
    'complete_window_scenes':len(eligible),'complete_window_logs':len({r['log_id'] for r in eligible}),
    'outside_current_complete_scenes':len(outside),'outside_current_complete_logs':len({r['log_id'] for r in outside}),
    'metadata_logs':len(all_logs),'v72_role_log_counts':{k:len(v) for k,v in roles.items()},
    'logs_without_v72_role':sorted(all_logs-assigned),
    'boundary':'Outside current 25 logs is NOT proof of historical non-exposure; trace V7/V7.1/V7.2 usage before defining final confirmation. Metadata/file-presence inventory only; no prediction or target read.',
    'outside_current_candidates':outside,'all_scene_payload_availability':result}
path=ROOT/'docs/autoresearch/worldsim_v73/coverage/log_payload_inventory_r1.json'
path.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in output.items() if k not in ['all_scene_payload_availability','outside_current_candidates']}))
