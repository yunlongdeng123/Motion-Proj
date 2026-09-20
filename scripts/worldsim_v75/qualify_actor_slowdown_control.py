"""在既有小误差独立源上冻结同一未来减速，作为状态误差对照。"""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from prepare_argoverse import project_track


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
SOURCE = ROOT/'WS-V75-ACTOR-REMOVAL-CONFIRM-QUALIFY-01/20260921-r3'
OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONTROL-QUALIFY-01/20260921-r1'
LATE = [45, 61, 85, 109, 116]
EVENT = 5


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


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
    assert not OUT.exists(); OUT.mkdir(parents=True)
    source = json.loads((SOURCE/'result.json').read_text()); s = source['selected']; base = Path(s['base'])
    assert source['status'] == 'qualified'
    tr = np.load(base/'trajectory.npz'); states = {}; separations = {}
    for arm, info in s['state_inputs'].items():
        scene = json.loads(Path(info['source']).read_text()); edited = slow(scene, s['actor'])
        source_path = OUT/f'condition-{arm}.json'; edited_path = OUT/f'condition-{arm}-slowed-after-005.json'
        save(source_path, scene); save(edited_path, edited)
        ot = next(t for t in scene['tracks'] if t['id'] == s['actor'] and 0 in t['frames'])
        et = next(t for t in edited['tracks'] if (t['id'], t['segment']) == (ot['id'], ot['segment']))
        arm_sep = []
        for frame in LATE:
            op = project_track(ot, frame, tr['camera_world'], tr['K']); ep = project_track(et, frame, tr['camera_world'], tr['K'])
            assert op is not None and ep is not None
            delta = np.linalg.norm((np.asarray(ep['bounds'][:2])+ep['bounds'][2:])/2-(np.asarray(op['bounds'][:2])+op['bounds'][2:])/2)
            arm_sep.append({'frame': frame, 'center_separation_px': float(delta)})
        separations[arm] = arm_sep
        states[arm] = {'source': str(source_path), 'edited': str(edited_path),
                       'initial_center_world_m': info['initial_center_world_m'],
                       'target_dimensions_m': info['target_dimensions_m'],
                       'target_rotation_world': info['target_rotation_world'],
                       'exact_through_frame': 4, 'other_tracks_exact_reference': True, 'map_exact_reference': True}
    assert min(x['center_separation_px'] for x in separations['reference']) >= 10
    protocol = {
        'task_id': 'WS-V75-ACTOR-SLOWDOWN-CONTROL-QUALIFY-01', 'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': '7a5f22d2',
        'source_qualification': str(SOURCE), 'log': s['log'], 'actor': s['actor'],
        'role': 'independent small-reconstruction-error control fixed before slowdown generations',
        'geometry_errors_m': source['geometry'],
        'intervention': 'same event frame5 and factor0.5 as the large-error source; all states exact through frame4',
        'late_frames': LATE, 'input_separations': separations,
        'reference_gate': 'unedited original match >=4/5; edited slowed match and >=5px preference >=4/5',
        'followup': 'reference gate first; only if it passes, generate DVGT/class-prior edited sequences and reuse their frozen unedited sequences',
        'stop': 'one source/factor/event/seed; no threshold search; OOM/error stops',
        'human_verdict': None, 'failure_ledger_delta': 'none',
    }
    save(OUT/'protocol.json', protocol)
    result = {'status': 'passed_pending_raster',
              'selected': {'log': s['log'], 'actor': s['actor'], 'behind': s['behind'], 'event_frame': EVENT,
                           'state_inputs': states, 'conditioning': s['conditioning'], 'base': str(base)},
              'geometry': source['geometry'], 'generation_admitted': False, 'raster_probe_required': True,
              'new_reconstruction_calls': 0, 'world_model_generation_calls': 0,
              'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'result.json', result)
    print(json.dumps({'status': result['status'], 'separations': separations,
                      'dvgt_center_error_m': source['geometry']['dvgt_center_error_m'],
                      'class_center_error_m': source['geometry']['class_center_error_m']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
