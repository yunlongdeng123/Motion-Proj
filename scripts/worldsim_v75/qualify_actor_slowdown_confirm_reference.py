"""为新冻结来源构造 reference 未来减速并验证条件输入。"""
from datetime import datetime, timezone
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import torch

from closed_loop_bridge import upload_scene
from prepare_argoverse import project_track
from render_argoverse import render


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
SCREEN = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONFIRM-SCREEN-01/20260921-r2'
OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONFIRM-QUALIFY-01/20260921-r1'
FRAMES = [0, 4, 5, 45, 61, 85, 109, 116]
LATE = [45, 61, 85, 109, 116]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    assert not OUT.exists(); OUT.mkdir(parents=True)
    screen = json.loads((SCREEN/'result.json').read_text()); s = screen['selected']; assert screen['status'] == 'qualified'
    base = Path(s['base']); scene = json.loads((base/'scene.json').read_text()); edited = copy.deepcopy(scene)
    target = next(t for t in edited['tracks'] if t['id'] == s['actor'] and 0 in t['frames'])
    original = next(t for t in scene['tracks'] if (t['id'], t['segment']) == (target['id'], target['segment']))
    frames = np.asarray(target['frames'], float); tau = np.where(frames >= 5, 4+0.5*(frames-4), frames); centers = np.asarray(target['centers'])
    target['centers'] = np.stack([np.interp(tau, frames, centers[:, i]) for i in range(3)], 1).tolist()
    target['quaternions'] = Slerp(frames, Rotation.from_quat(target['quaternions']))(tau).as_quat().tolist()
    source_path = OUT/'condition-reference.json'; edited_path = OUT/'condition-reference-slowed-after-005.json'
    save(source_path, scene); save(edited_path, edited)
    tr = np.load(base/'trajectory.npz'); separations = []
    for frame in LATE:
        op = project_track(original, frame, tr['camera_world'], tr['K']); ep = project_track(target, frame, tr['camera_world'], tr['K'])
        assert op is not None and ep is not None
        oc = (np.asarray(op['bounds'][:2])+op['bounds'][2:])/2; ec = (np.asarray(ep['bounds'][:2])+ep['bounds'][2:])/2
        separations.append({'frame': frame, 'center_separation_px': float(np.linalg.norm(oc-ec)),
                            'original_bounds': op['bounds'], 'edited_bounds': ep['bounds']})
    protocol = {
        'task_id': 'WS-V75-ACTOR-SLOWDOWN-CONFIRM-QUALIFY-01', 'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': 'e4c895cb',
        'source_screen': str(SCREEN/'result.json'), 'log': s['log'], 'actor': s['actor'],
        'role': 'new independent source selected before reconstruction output',
        'intervention': 'exact through frame4; from frame5 target follows tau=4+0.5*(frame-4)',
        'late_frames': LATE, 'separations': separations,
        'reference_gate': 'unedited original match >=4/5; edited slowed match and >=5px preference >=4/5',
        'followup': 'only a passing reference G1 admits one official DVGT readout',
        'stop': 'one source, two reference sequences, no source/factor/event/seed/threshold replacement',
        'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'protocol.json', protocol)
    states = {'reference': {'source': str(source_path), 'edited': str(edited_path),
                            'initial_center_world_m': original['centers'][original['frames'].index(0)],
                            'target_dimensions_m': original['dimensions'], 'other_tracks_exact_reference': True, 'map_exact_reference': True}}
    result = {'status': 'passed_pending_raster',
              'selected': {'log': s['log'], 'actor': s['actor'], 'behind': None, 'event_frame': 5,
                           'state_inputs': states, 'conditioning': str(OUT/'conditioning'), 'base': str(base)},
              'generation_admitted': False, 'raster_probe_required': True, 'new_reconstruction_calls': 0,
              'world_model_generation_calls': 0, 'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'result.json', result); began = time.monotonic(); terminal = {'status': 'started', 'human_verdict': None, 'failure_ledger_delta': 'none'}
    try:
        images = {}
        for variant, state in [('unedited', scene), ('edited', edited)]:
            ctx, sid, fit = upload_scene(state, tr['timestamps_us'], tr['K'])
            images[variant] = render(ctx, sid, tr['timestamps_us'][FRAMES], tr['camera_world'][FRAMES]); del ctx; torch.cuda.empty_cache()
        rows = []
        for i, frame in enumerate(FRAMES):
            mask = np.any(images['unedited'][i] != images['edited'][i], axis=-1)
            rows.append({'frame': frame, 'changed_pixels': int(mask.sum()), 'exactly_equal': bool(not mask.any())})
        passed = all(next(x for x in rows if x['frame'] == f)['exactly_equal'] for f in [0, 4]) and all(next(x for x in rows if x['frame'] == f)['changed_pixels'] >= 50 for f in LATE)
        terminal.update(status='passed' if passed else 'failed', rows=rows, generation_admitted=passed,
                        condition_render_calls=2, rendered_condition_frames=len(FRAMES)*2)
    except BaseException as exc:
        terminal.update(status='oom_stopped' if isinstance(exc, torch.OutOfMemoryError) else 'failed_stopped',
                        error_type=type(exc).__name__, error=str(exc), generation_admitted=False); raise
    finally:
        terminal.update(wall_s=time.monotonic()-began, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        world_model_generation_calls=0); save(OUT/'raster_result.json', terminal)
    result.update(status='qualified' if terminal['generation_admitted'] else terminal['status'],
                  generation_admitted=terminal['generation_admitted'], raster_result=str(OUT/'raster_result.json'))
    save(OUT/'result.json', result); print(json.dumps({'result': result, 'raster': terminal}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
