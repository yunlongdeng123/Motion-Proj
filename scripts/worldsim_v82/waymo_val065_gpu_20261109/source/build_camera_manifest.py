"""固定BUILD的归一化；留出相机不重算场景变换。"""
from pathlib import Path
import json
import numpy as np
ROOT=Path(__file__).resolve().parent.parent
PREP=Path('/root/autodl-tmp/codex_tmp/waymo_val065_cpu_20261108_r1')
def main():
    old=json.loads((ROOT/'formal10k/cameras.json').read_text());by={int(c['img_name'][-3:]):c for c in old};T=np.loadtxt(ROOT/'inputs/native/transform_matrix.txt');scale=float(np.linalg.det(T[:3,:3])**(1/3));errors=[];rows=[]
    for raw in range(199):
        held=raw in range(79,119);base=ROOT/'heldout' if held else PREP/'inputs'
        obs=json.loads((base/'observations'/f'{raw:03d}.json').read_text());calib=base/'calib'/(f'{raw:03d}.txt' if held else f'0065{raw:03d}.txt')
        cal={s.split(':')[0]:np.fromstring(s.split(':')[1],sep=' ') for s in calib.read_text().splitlines()}
        K=cal['P0'].reshape(3,4)[:,:3];C=np.eye(4);C[:3]=cal['Tr_velo_to_cam_0'].reshape(3,4);pose=T@np.array(obs['ego_world']).reshape(4,4)@np.linalg.inv(C);pose[:3,:3]/=scale
        row=dict(id=raw,img_name=f'0065{raw:03d}',width=1920,height=1280,fx=float(K[0,0]),fy=float(K[1,1]),cx=float(K[0,2]),cy=float(K[1,2]),position=pose[:3,3].tolist(),rotation=pose[:3,:3].tolist(),model_time=(raw-99)*.02,heldout=held)
        if raw in by:
            ref=by[raw];errors.append(float(max(abs(np.array(ref['position'])-pose[:3,3]).max(),abs(np.array(ref['rotation'])-pose[:3,:3]).max())))
            row.update({k:ref[k] for k in ref})
        rows.append(row)
    assert len(errors)==159 and max(errors)<2e-10
    (ROOT/'all_cameras.json').write_text(json.dumps(dict(schema='waymo.v1_camera_export.v1',cameras=rows,BUILD_pose_maxabs=max(errors),heldout_fitted=False),indent=2))
    print('ALL_CAMERAS',len(rows),max(errors))
if __name__=='__main__':main()
