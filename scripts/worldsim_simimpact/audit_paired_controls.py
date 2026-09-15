import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
rows=[]
for name in ['scene-0013','scene-0038','scene-0041']:
    folder=R/'rollouts'/name;path=folder/'fixediter_stride1_r1/trace.json'
    if not path.exists():continue
    base=json.loads(path.read_text());controls=[]
    for stride in [1,2,4]:
        p=folder/f'fixediter_stride{stride}_r1'
        if not (p/'trace.json').exists():continue
        tr=json.loads((p/'trace.json').read_text());n=min(len(tr),len(base))
        diff=[float(np.linalg.norm(np.array(a['post_info']['ego_pos'])-np.array(b['post_info']['ego_pos']))) for a,b in zip(base[:n],tr[:n])]
        controls.append({'stride':stride,'summary':json.loads((p/'summary.json').read_text()),'shared_prefix_poses':n,'shared_prefix_max_position_difference_m':max(diff)})
    model=json.loads(next((R/'assets/extracted').glob(f'**/{name}/meta_data.json')).read_text());inv=np.array(model['inv_pose'])
    inp=json.loads((R/'inputs'/f'{name}.json').read_text())['views'];alignment=[]
    for v in inp:
        pred=inv@np.array(v['world_from_camera']);candidates=[f for f in model['frames'] if '/'+v['camera']+'/' in f['rgb_path']]
        poses=np.array([f['camtoworld'] for f in candidates]);dist=np.linalg.norm(poses[:,:3,3]-pred[:3,3],axis=1);j=np.argmin(dist)
        angle=Rotation.from_matrix(poses[j,:3,:3].T@pred[:3,:3]).magnitude()*180/np.pi
        alignment.append({'camera':v['camera'],'raw_sample':v['sample_index'],'closest_published_frame':candidates[j]['rgb_path'],'translation_difference_m':float(dist[j]),'rotation_difference_deg':float(angle),'published_time_s':candidates[j]['timestamp']})
    identity=np.load(R/'geometry_swap'/name/'identity.npz');pts=identity['points'];native=json.loads((folder/'native_r1/trace.json').read_text())[-1]['post_info']
    rot=Rotation.from_euler('XYZ',native['ego_rot']).as_matrix();local=(pts-np.array(native['ego_pos']))@rot
    inside=(abs(local[:,0])<.8)&(local[:,1]>0)&(local[:,1]<1.5)&(abs(local[:,2])<1.5)
    rows.append({'scene':name,'fixed_iteration_controls':controls,'raw_vs_published_camera_alignment':alignment,
        'native_terminal_collision_centers':int(inside.sum()),'terminal_centers_within_geometry_swap_support':int((inside&identity['visible']).sum()),
        'geometry_swap_visible_centers':int(identity['visible'].sum()),'geometry_swap_total_centers':len(pts)})
(R/'paired_control_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
