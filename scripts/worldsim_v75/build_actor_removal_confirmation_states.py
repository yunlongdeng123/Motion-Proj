"""把独立 DVGT 读出冻结成三个状态臂及同一对象移除编辑。"""
import copy
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from prepare_argoverse import CORNERS
from readout_natural import bbox, read_center


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
OUT = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01/20260921-r1'
DIMS = np.array([3.9, 1.6, 1.56])
EVENT_FRAME = 5


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def project(center_world, rotation_world, dimensions, camera_world, K):
    center = (center_world-camera_world[:3, 3])@camera_world[:3, :3]
    rotation = camera_world[:3, :3].T@rotation_world
    return bbox(center, rotation, dimensions, K)


def fit_class(observed, K, rotation):
    center = (observed[:2]+observed[2:])/2
    ray = np.linalg.inv(K)@np.r_[center, 1.]
    distance = K[1, 1]*DIMS[2]/(observed[3]-observed[1])
    fit = least_squares(lambda c: bbox(c, rotation, DIMS, K)-observed, ray*distance,
                        bounds=([-200, -200, 2], [200, 200, 200]),
                        loss='soft_l1', f_scale=1)
    assert fit.success and np.isfinite(fit.x).all()
    return fit.x, fit.fun


def modify(scene, actor_id, center_world, dimensions=None, rotation_world=None):
    output = copy.deepcopy(scene)
    target = next(t for t in output['tracks'] if t['id'] == actor_id and 0 in t['frames'])
    original_center = np.asarray(target['centers'][0])
    target['centers'] = (np.asarray(target['centers'])-original_center+center_world).tolist()
    if dimensions is not None:
        target['dimensions'] = np.asarray(dimensions).tolist()
    if rotation_world is not None:
        original = Rotation.from_quat(target['quaternions']).as_matrix()
        initial = original[target['frames'].index(0)]
        target['quaternions'] = Rotation.from_matrix(original@initial.T@rotation_world).as_quat().tolist()
    return output


def remove(scene, actor_id):
    output = copy.deepcopy(scene)
    target = next(t for t in output['tracks'] if t['id'] == actor_id and 0 in t['frames'])
    keep = [i for i, f in enumerate(target['frames']) if f < EVENT_FRAME]
    assert keep and target['frames'][keep[-1]] == EVENT_FRAME-1
    for key in ['frames', 'centers', 'quaternions']:
        target[key] = [target[key][i] for i in keep]
    return output


def main():
    prep = json.loads((OUT/'protocol.json').read_text())
    recon = OUT/'reconstruction'
    inference = json.loads((recon/'inference_result.json').read_text())
    readout = json.loads((recon/'readout_ray_control_result.json').read_text())
    assert inference['status'] == 'complete' and readout['status'] == 'complete'
    assert readout['generation_admitted'], readout['stop_reasons']
    p = json.loads((recon/'protocol.json').read_text())
    base = Path(prep['base']); scene = json.loads((base/'scene.json').read_text())
    actor_id, behind_id = prep['actor'], prep['behind']
    target = next(t for t in scene['tracks'] if t['id'] == actor_id and 0 in t['frames'])
    assert any(t['id'] == behind_id and 0 in t['frames'] for t in scene['tracks'])
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
        'dvgt_metric': modify(scene, actor_id, dvgt_center),
        'class_prior': modify(scene, actor_id, class_center, DIMS, class_rotation_world),
    }
    states = {}
    for name, state in arms.items():
        source = OUT/f'condition-{name}.json'; edited = OUT/f'condition-{name}-removed-after-{EVENT_FRAME:03d}.json'
        save(source, state); save(edited, remove(state, actor_id))
        t = next(t for t in state['tracks'] if t['id'] == actor_id and 0 in t['frames'])
        states[name] = {
            'source': str(source), 'edited': str(edited),
            'initial_center_world_m': t['centers'][t['frames'].index(0)],
            'target_dimensions_m': t['dimensions'],
            'target_rotation_world': Rotation.from_quat(t['quaternions'][t['frames'].index(0)]).as_matrix().tolist(),
            'projected_bounds': project(np.asarray(t['centers'][t['frames'].index(0)]),
                                        Rotation.from_quat(t['quaternions'][t['frames'].index(0)]).as_matrix(),
                                        np.asarray(t['dimensions']), camera, K).tolist(),
            'retained_target_frames': list(range(EVENT_FRAME)),
            'other_tracks_exact_reference': True, 'map_exact_reference': True,
        }
    geometry = {
        'truth_center_world_m': truth_center.tolist(), 'truth_dimensions_m': truth_dims.tolist(),
        'truth_rotation_world': truth_rotation.tolist(), 'observed_bbox_network': observed.tolist(),
        'dvgt_center_error_m': float(np.linalg.norm(dvgt_center-truth_center)),
        'class_center_error_m': float(np.linalg.norm(class_center-truth_center)),
        'class_dimension_error_m': (DIMS-truth_dims).tolist(),
        'class_yaw_error_deg': float(np.rad2deg(Rotation.from_matrix(class_rotation_world.T@truth_rotation).magnitude())),
        'ordinary_class_bbox_residual_px': residual.tolist(),
        'class_readout': class_read,
        'initial_bbox_fit': {name: {
            'max_bound_error_vs_detector_px': float(np.max(np.abs(np.asarray(info['projected_bounds'])-observed))),
            'iou_inputs_saved_for_report': True,
        } for name, info in states.items()},
    }
    result = {
        'status': 'passed_pending_raster',
        'selected': {'log': prep['log'], 'actor': actor_id, 'behind': behind_id,
                     'event_frame': EVENT_FRAME, 'state_inputs': states,
                     'conditioning': str(OUT/'conditioning'), 'base': str(base)},
        'geometry': geometry, 'real_gate_passed': True,
        'generation_admitted': False, 'raster_probe_required': True,
        'new_reconstruction_calls': 1, 'world_model_generation_calls': 0,
        'human_verdict': None, 'failure_ledger_delta': 'none',
    }
    save(OUT/'result.json', result)
    print(json.dumps({'status': result['status'], 'geometry': geometry, 'states': states}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
