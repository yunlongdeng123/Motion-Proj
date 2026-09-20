"""复核单段已完成生成的全帧轨迹；GT门控先于配对误差分支。"""
import argparse,json,os
from pathlib import Path
import numpy as np
import torch
from following_geometry import scene
from raster_ground import RasterGround
from rgb_idm_policy import RGBIDMPolicy
from run_approach_state_control import rollout


def assess(folder,arm):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
    p=json.loads((folder/'frozen_state_protocol.json').read_text());base=Path(p['base']);dest=folder/arm
    result=json.loads((dest/'result.json').read_text());assert result['status']=='complete'
    rows=json.loads((dest/'decisions.json').read_text());assert len(rows)==15
    reference=scene(base);tr=np.load(base/'trajectory.npz')
    target=next(t for t in reference['tracks'] if t['id']==p['target'] and 0 in t['frames'])
    extra=np.linalg.inv(tr['ego_world'][0])@tr['camera_world'][0]
    policy=RGBIDMPolicy(tr['ego_world'][:,:3,3],tr['K'],extra,[0,0,0],detector=False)
    terrain=RasterGround(base).mesh_for_route(tr['ego_world'][:,:2,3])
    condition=json.loads((dest/'condition_scene.json').read_text())
    expected=json.loads(Path(p['conditions'][arm]).read_text());assert condition==expected
    trajectory,cameras=rollout(base,tr,policy,terrain,condition,reference,target,rows,True)
    error=float(np.max(abs(cameras-np.load(dest/'camera_trajectory.npz')['camera_world'])))
    assert error<=1e-7
    d=np.array([r['acceleration_difference_mps2'] for r in rows])
    lateral=max(abs(r['route_progress_lateral'][1]) for r in trajectory['frames'])
    check={'status':'passed' if not trajectory['overlap_frames'] and lateral<=1 else 'failed',
           'frames':117,'replay_max_abs':error,'overlap_frames':trajectory['overlap_frames'],
           'max_route_lateral_m':lateral,'human_verdict':None}
    summary={'arm':arm,'seed':p['seed'],'source':str(dest),'progress_m':trajectory['progress_m'],
             'mean_abs_action_error_at_own_ego_mps2':float(np.mean(abs(d))),
             'median_abs_action_error_at_own_ego_mps2':float(np.median(abs(d))),
             'max_underbraking_mps2':float(np.maximum(d,0).max()),'max_overbraking_mps2':float(np.maximum(-d,0).max()),
             'final_target_reference_clearance_m':trajectory['final_target_reference_clearance_m'],
             'final_speed_mps':trajectory['final_speed_mps'],'overlap_frames':trajectory['overlap_frames'],
             'camera_replay_max_abs':error,'camera_replay_frames':117,'max_route_lateral_m':lateral,
             'baseline_admitted':result['baseline_admitted'],'human_verdict':None}
    for name,value in [('dense_replay.json',trajectory),('dense_reference_result.json',check),('assessment.json',summary)]:
        (dest/name).write_text(json.dumps(value,indent=2)+'\n')
    assert not torch.cuda.is_initialized()
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);p.add_argument('--arm',required=True)
    a=p.parse_args();print(json.dumps(assess(a.folder,a.arm)),flush=True)
