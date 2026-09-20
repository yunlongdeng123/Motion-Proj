"""导出冻结初始状态到实际条件图的可视化证据，不重新渲染或推理。"""
import json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from prepare_argoverse import CORNERS,project_track
from run_natural_rollouts import OUT as SOURCE
from repeat_natural_rollouts import OUT

def main():
    out=OUT/'trace';assert not out.exists();out.mkdir()
    p=json.loads((SOURCE/'protocol.json').read_text());base=Path(p['base_dir'])
    read=json.loads((Path(p['readout_dir'])/'readout_ray_control_result.json').read_text())
    trajectory=np.load(base/'trajectory.npz');ego=trajectory['ego_world'][0]
    camera=trajectory['camera_world'];K=trajectory['K'];rows=[]
    ref=json.loads((SOURCE/'evaluator_reference.json').read_text())
    bounds=ref['frames'][2]['match']['box'];center=(np.array(bounds[:2])+bounds[2:])/2
    crop=[round(center[0]-180),round(center[1]-108),round(center[0]+180),round(center[1]+108)]
    for name in ['gt_clean','dvgt_metric','ordinary_bbox','reference_lidar']:
        scene=json.loads((SOURCE/name/'scene.json').read_text());target=next(t for t in scene['tracks'] if t['id']==p['target'])
        rotation=Rotation.from_quat(target['quaternions'][0]).as_matrix();xyz=np.array(target['centers'][0])
        local=(xyz-ego[:3,3])@ego[:3,:3];world_corners=(CORNERS*np.array(target['dimensions']))@rotation.T+xyz
        ec=(world_corners-ego[:3,3])@ego[:3,:3];xy=np.c_[-ec[:,1],ec[:,0]]
        hull=xy[ConvexHull(xy).vertices]
        projections=[project_track(target,f,camera,K) for f in [0,15,30,45,60]]
        condition=np.load(SOURCE/name/'conditions.npy',mmap_mode='r')
        Image.fromarray(condition[30]).crop(crop).save(out/f'condition-{name}.png')
        rows.append({'variant':name,'center_ego_flu':local.tolist(),'bev_right_forward':[-float(local[1]),float(local[0])],
                     'bev_hull':hull.tolist(),'dimensions':target['dimensions'],'projections':projections,
                     'error_m':0 if name=='gt_clean' else read['readouts'][name]['center_error_m']})
    Image.open(base/'initial_rgb.png').save(out/'initial_rgb.png')
    result={'status':'complete','source_run':str(SOURCE),'created_utc':datetime.now(timezone.utc).isoformat(),
            'condition_frame':30,'figure_time_selection':'fixed 1s before reading seed43 output; quantitative window remains all five scheduled times',
            'crop_xyxy':crop,'initial_reference_box':ref['frames'][0]['reference_bounds'],'states':rows,
            'coordinate_axes':'BEV horizontal=ego right=-left, vertical=ego forward; meters',
            'human_verdict':None,'reconstruction_calls':0,'render_calls':0,'generation_calls':0}
    (out/'trace_data.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'complete','states':len(rows),'path':str(out)}))

if __name__=='__main__':main()
