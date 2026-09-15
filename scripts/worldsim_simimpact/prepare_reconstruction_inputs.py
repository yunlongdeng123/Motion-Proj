"""冻结本批首两关键帧RGB作为BUILD，后六帧LiDAR仅供后续评价。"""
import json
from pathlib import Path
import numpy as np
from pyquaternion import Quaternion
from PIL import Image
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
M=Path('/root/autodl-tmp/data/worldsim_v4/drivestudio_raw_trainval/v1.0-trainval')
src=json.loads((R/'raw_source_availability.json').read_text())
sd={x['filename']:x for x in json.loads((M/'sample_data.json').read_text()) if x['is_key_frame']}
ego={x['token']:x for x in json.loads((M/'ego_pose.json').read_text())}
cal={x['token']:x for x in json.loads((M/'calibrated_sensor.json').read_text())}
def matrix(rec):
    t=np.eye(4);t[:3,:3]=Quaternion(rec['rotation']).rotation_matrix;t[:3,3]=rec['translation'];return t
order=['CAM_FRONT','CAM_FRONT_LEFT','CAM_FRONT_RIGHT','CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']
(R/'inputs').mkdir(exist_ok=True)
for name,s in src.items():
    views=[]
    for i,token in enumerate(s['samples'][:2]):
        for cam in order:
            record=next(x for x in s['records'] if x['sample_token']==token and f'/{cam}/' in x['filename'])
            d=sd[record['filename']];c=cal[d['calibrated_sensor_token']];e=ego[d['ego_pose_token']]
            path=R/'raw_initial'/record['filename']
            views.append({'sample_token':token,'sample_index':i,'camera':cam,'image':str(path),'timestamp_us':d['timestamp'],
                'world_from_camera':(matrix(e)@matrix(c)).tolist(),'world_from_ego_camera':matrix(e).tolist(),
                'intrinsics_original':c['camera_intrinsic'],'original_wh':list(Image.open(path).size)})
    dest=R/'inputs'/f'{name}.json'
    if dest.exists():raise RuntimeError(f'Already registered {dest}')
    dest.write_text(json.dumps({'scene':name,'log':s.get('log','see metadata'),'role':'BUILD_RGB_ONLY','views':views,
        'query_sample_indices':list(range(2,8)),'selection':'Initial cohort retained; frames chosen after native rollout, discovery only',
        'downstream':'Native simulator geometry replacement is a diagnostic adapter, not an official feed-forward end-to-end system'},indent=2))
    print(name,len(views),flush=True)
