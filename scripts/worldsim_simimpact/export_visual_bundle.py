import json,shutil
from pathlib import Path
import numpy as np
from PIL import Image
R=Path('/root/autodl-tmp/runs/worldsim_simimpact/WS-SIM-IMPACT-01/20260915-r1')
out=R/'visual_bundle';out.mkdir(exist_ok=True)
for name in ['scene-0013','scene-0038','scene-0041']:
    dst=out/name;dst.mkdir(exist_ok=True)
    inp=json.loads((R/'inputs'/f'{name}.json').read_text())['views'][0]
    meta=json.loads(next((R/'assets/extracted').glob(f'**/{name}/meta_data.json')).read_text())
    c2w=np.array(meta['inv_pose'])@np.array(inp['world_from_camera'])
    g=np.load(R/'collision_audit'/name/'geometry.npz');data=dict(g)
    p=(g['collision_points']-c2w[:3,3])@c2w[:3,:3]
    K=np.array(inp['intrinsics_original'])*.5;K[2,2]=1
    uvz=p@K.T;data['initial_rgb_uv']=uvz[:,:2]/uvz[:,2:3];data['initial_rgb_camera_z']=p[:,2]
    np.savez_compressed(dst/'geometry.npz',**data)
    Image.open(inp['image']).convert('RGB').resize((800,450),Image.Resampling.LANCZOS).save(dst/'raw_initial.jpg',quality=95)
    shutil.copy2(R/'rollouts'/name/'native_r1/front_final.jpg',dst/'native_terminal.jpg')
    shutil.copy2(R/'collision_audit'/name/'audit.json',dst/'audit.json')
    alignment=np.array(meta['frames'][0]['camtoworld'])
    (dst/'alignment.json').write_text(json.dumps({'first_camera_translation_difference_m':float(np.linalg.norm(c2w[:3,3]-alignment[:3,3])),
        'first_camera_rotation_matrix_difference':float(np.max(np.abs(c2w[:3,:3]-alignment[:3,:3]))),
        'RGB':'Real first BUILD image, annotations are projected published GS centers counted at the later terminal pose. Not phantom GT.'},indent=2))
for name in ['paired_collision_geometry.json','rollout_summary.json','geometry_swap_protocol.json','native_collision_audit.json','ltf_input_dependency.json']:
    shutil.copy2(R/name,out/name)
