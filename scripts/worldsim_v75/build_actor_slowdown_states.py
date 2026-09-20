"""由唯一 DVGT 读出构造三种状态及相同未来减速编辑。"""
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from readout_natural import bbox, read_center


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
SOURCE = ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-QUALIFY-01/20260921-r1'
OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01/20260921-r2'
DIMS = np.array([3.9, 1.6, 1.56])
EVENT = 5


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def fit_class(observed, K, rotation):
    center = (observed[:2]+observed[2:])/2
    ray = np.linalg.inv(K)@np.r_[center, 1.]
    distance = K[1, 1]*DIMS[2]/(observed[3]-observed[1])
    fit = least_squares(lambda c: bbox(c, rotation, DIMS, K)-observed, ray*distance,
                        bounds=([-200, -200, 2], [200, 200, 200]), loss='soft_l1', f_scale=1)
    assert fit.success and np.isfinite(fit.x).all()
    return fit.x, fit.fun


def change_state(scene, actor_id, center_world, dimensions=None, rotation_world=None):
    output = copy.deepcopy(scene)
    target = next(t for t in output['tracks'] if t['id'] == actor_id and 0 in t['frames'])
    original_center = np.asarray(target['centers'][target['frames'].index(0)])
    target['centers'] = (np.asarray(target['centers'])-original_center+center_world).tolist()
    if dimensions is not None:
        target['dimensions'] = np.asarray(dimensions).tolist()
    if rotation_world is not None:
        original = Rotation.from_quat(target['quaternions']).as_matrix()
        initial = original[target['frames'].index(0)]
        target['quaternions'] = Rotation.from_matrix(original@initial.T@rotation_world).as_quat().tolist()
    return output


def slow(scene, actor_id):
    output = copy.deepcopy(scene)
    target = next(t for t in output['tracks'] if t['id'] == actor_id and 0 in t['frames'])
    frames = np.asarray(target['frames'], float)
    tau = np.where(frames >= EVENT, EVENT-1+0.5*(frames-(EVENT-1)), frames)
    centers = np.asarray(target['centers'])
    target['centers'] = np.stack([np.interp(tau, frames, centers[:, i]) for i in range(3)], 1).tolist()
    target['quaternions'] = Slerp(frames, Rotation.from_quat(target['quaternions']))(tau).as_quat().tolist()
    return output


def main():
    prep = json.loads((OUT/'protocol.json').read_text())
    recon = OUT/'reconstruction'
    inference = json.loads((recon/'inference_result.json').read_text())
    readout = json.loads((recon/'readout_ray_control_result.json').read_text())
    assert inference['status'] == 'complete' and readout['status'] == 'complete'
    assert readout['generation_admitted'], readout['stop_reasons']
    p = json.loads((recon/'protocol.json').read_text())
    q = json.loads((SOURCE/'result.json').read_text())
    scene = json.loads(Path(q['selected']['state_inputs']['reference']['source']).read_text())
    actor_id = prep['actor']
    target = next(t for t in scene['tracks'] if t['id'] == actor_id and 0 in t['frames'])
    truth_center = np.asarray(target['centers'][target['frames'].index(0)])
    truth_dims = np.asarray(target['dimensions'])
    truth_rotation = Rotation.from_quat(target['quaternions'][target['frames'].index(0)]).as_matrix()
    camera = np.asarray(p['views'][0]['camera_world']); ego = np.asarray(p['views'][0]['ego_world'])
    K = np.asarray(p['views'][0]['K_network']); observed = np.asarray(p['bbox_network'])
    dvgt_center = np.asarray(readout['readouts']['dvgt_metric']['center_world'])
    yaw = Rotation.from_matrix(ego[:3, :3]).as_euler('xyz')[2]
    class_rotation_world = Rotation.from_euler('z', yaw).as_matrix()
    class_rotation_camera = camera[:3, :3].T@class_rotation_world
    ordinary, residual = fit_class(observed, K, class_rotation_camera)
    arrays = np.load(recon/'readout_ray_control_arrays.npz')
    mask = arrays['central_mask']; depth = arrays['depths'][0][mask]
    yy, xx = np.mgrid[:512, :512]
    rays = np.stack([xx, yy, np.ones_like(xx)], -1)[mask]@np.linalg.inv(K).T
    class_read = read_center(rays, depth, ordinary, class_rotation_camera, DIMS)
    assert class_read is not None and class_read['n'] >= 20 and class_read['face_retention'] >= .8
    class_center = np.asarray(class_read['center_camera'])@camera[:3, :3].T+camera[:3, 3]
    arms = {
        'reference': copy.deepcopy(scene),
        'dvgt_metric': change_state(scene, actor_id, dvgt_center),
        'class_prior': change_state(scene, actor_id, class_center, DIMS, class_rotation_world),
    }
    states = {}
    for name, state in arms.items():
        edited = slow(state, actor_id)
        source_path = OUT/f'condition-{name}.json'
        edited_path = OUT/f'condition-{name}-slowed-after-{EVENT:03d}.json'
        save(source_path, state); save(edited_path, edited)
        t = next(t for t in state['tracks'] if t['id'] == actor_id and 0 in t['frames'])
        states[name] = {
            'source': str(source_path), 'edited': str(edited_path),
            'initial_center_world_m': t['centers'][t['frames'].index(0)],
            'target_dimensions_m': t['dimensions'],
            'target_rotation_world': Rotation.from_quat(t['quaternions'][t['frames'].index(0)]).as_matrix().tolist(),
            'initial_projected_bounds': bbox((np.asarray(t['centers'][t['frames'].index(0)])-camera[:3, 3])@camera[:3, :3],
                                             camera[:3, :3].T@Rotation.from_quat(t['quaternions'][t['frames'].index(0)]).as_matrix(),
                                             np.asarray(t['dimensions']), K).tolist(),
            'exact_through_frame': EVENT-1,
            'other_tracks_exact_reference': True, 'map_exact_reference': True,
        }
    geometry = {
        'truth_center_world_m': truth_center.tolist(), 'truth_dimensions_m': truth_dims.tolist(),
        'dvgt_center_error_m': float(np.linalg.norm(dvgt_center-truth_center)),
        'class_center_error_m': float(np.linalg.norm(class_center-truth_center)),
        'class_dimension_error_m': (DIMS-truth_dims).tolist(),
        'class_yaw_error_deg': float(np.rad2deg(Rotation.from_matrix(class_rotation_world.T@truth_rotation).magnitude())),
        'ordinary_class_bbox_residual_px': residual.tolist(), 'class_readout': class_read,
    }
    result = {
        'status': 'passed_pending_raster',
        'selected': {'log': prep['log'], 'actor': actor_id, 'behind': prep['behind'], 'event_frame': EVENT,
                     'state_inputs': states, 'conditioning': q['selected']['conditioning'], 'base': prep['base']},
        'geometry': geometry, 'reference_g1_passed': True, 'generation_admitted': False,
        'raster_probe_required': True, 'new_reconstruction_calls': 1, 'world_model_generation_calls': 0,
        'human_verdict': None, 'failure_ledger_delta': 'none',
    }
    save(OUT/'result.json', result)
    print(json.dumps({'status': result['status'], 'geometry': geometry, 'states': states}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
