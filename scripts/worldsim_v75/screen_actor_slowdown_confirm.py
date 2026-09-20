"""冻结新八日志，按输入与reference轨迹可分性选未来减速独立确认源。"""
from datetime import datetime, timezone
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation, Slerp

from evaluate_localization import model, predict
from prepare_argoverse import project_track
from qualify_actor_removal import area, match_vehicle


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONFIRM-SCREEN-01/20260921-r1'
LOGS = ['95bf6003-7068-3a78-a0c0-9e470a06e60f', '9a448a80-0e9a-3bf0-90f3-21750dfef55a',
        '9bb1f857-8b61-369f-a537-484c1323ae32', '9f871fb4-3b8e-34b3-9161-ed961e71a6da',
        'a33a44fb-6008-3dc2-b7c5-2d27b70741e8', 'a91d4c7b-bf55-3a0e-9eba-1a43577bcca8',
        'adf9a841-e0db-30ab-b5b3-bf0b61658e1e', 'b19f3c1a-a84a-3a2d-8d1b-8a4ae201020b']
LATE = [45, 61, 85, 109, 116]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def slowed(track):
    output = copy.deepcopy(track); frames = np.asarray(output['frames'], float)
    tau = np.where(frames >= 5, 4+0.5*(frames-4), frames); centers = np.asarray(output['centers'])
    output['centers'] = np.stack([np.interp(tau, frames, centers[:, i]) for i in range(3)], 1).tolist()
    output['quaternions'] = Slerp(frames, Rotation.from_quat(output['quaternions']))(tau).as_quat().tolist()
    return output


def center(bounds):
    bounds = np.asarray(bounds); return (bounds[:2]+bounds[2:])/2


def main():
    global OUT
    parser = argparse.ArgumentParser(); parser.add_argument('--run-id', default='20260921-r1')
    args = parser.parse_args(); OUT = ROOT/'WS-V75-ACTOR-SLOWDOWN-CONFIRM-SCREEN-01'/args.run_id
    assert not OUT.exists(); OUT.mkdir(parents=True)
    protocol = {
        'task_id': 'WS-V75-ACTOR-SLOWDOWN-CONFIRM-SCREEN-01', 'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source_revision': 'e4c895cb', 'logs': LOGS,
        'engineering_recovery_of': ('20260921-r1' if args.run_id == '20260921-r2' else None),
        'source_selection': 'next8 lexicographic locally complete AV2 val logs after the closed far-removal window; no reconstruction output read',
        'window': 'one fixed 7.9s window starting at +6.5s per log',
        'order': 'log then actor UUID; first input-qualified and real-detected actor; at most one',
        'actor_gate': 'regular/large vehicle, fully inside frames0,4,45,61,85,109,116; initial depth15..65m; area2000..15000px2',
        'edit_gate': 'same event frame5 and factor0.5; original and slowed projections valid; center separation >=10px at all five late frames',
        'real_gate': 'same COCO detector at frame0; vehicle score>=0.5 and IoU>=0.3',
        'followup': 'reference unedited/edited only; DVGT forbidden until reference G1 passes',
        'stop': 'eight logs, first eligible actor, no offset/source/threshold/factor/event/seed replacement',
        'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'protocol.json', protocol)
    detector = None; records = []; selected = None; calls = 0
    for log in LOGS:
        base = OUT/'cases'/log/'base'; base.parent.mkdir(parents=True)
        cmd = [sys.executable, str(Path(__file__).with_name('prepare_argoverse.py')), '--output', str(base), '--log-id', log,
               '--start-offset-seconds', '6.5', '--task-id', protocol['task_id'], '--run-id', OUT.name]
        with (base.parent/'prepare.log').open('w') as stream:
            subprocess.run(cmd, stdout=stream, stderr=subprocess.STDOUT, check=True)
        tracks = json.loads((base/'scene.json').read_text())['tracks']; tr = np.load(base/'trajectory.npz'); candidates = []
        prediction = None
        for actor in sorted(tracks, key=lambda x: (x['id'], x['segment'])):
            if actor['category'] not in ['REGULAR_VEHICLE', 'LARGE_VEHICLE'] or 0 not in actor['frames']:
                continue
            op = {f: project_track(actor, f, tr['camera_world'], tr['K']) for f in [0, 4]+LATE}
            if any(op[f] is None or not op[f]['fully_inside'] for f in op):
                continue
            a0 = op[0]; a = area(a0['bounds'])
            if not (15 <= a0['depth_m'] <= 65 and 2000 <= a <= 15000):
                continue
            edit = slowed(actor); ep = {f: project_track(edit, f, tr['camera_world'], tr['K']) for f in LATE}
            if any(ep[f] is None or not ep[f]['fully_inside'] for f in LATE):
                continue
            sep = [{'frame': f, 'center_separation_px': float(np.linalg.norm(center(ep[f]['bounds'])-center(op[f]['bounds'])))} for f in LATE]
            if min(x['center_separation_px'] for x in sep) < 10:
                continue
            if prediction is None:
                if detector is None: detector = model()
                prediction = predict(detector, Image.open(base/'reference-000.png').convert('RGB')); calls += 1
            match = match_vehicle(prediction, a0['bounds']); passed = bool(match and match['iou'] >= .3)
            row = {'actor': actor['id'], 'category': actor['category'], 'actor_depth_m': a0['depth_m'],
                   'actor_area_px2': a, 'initial_bounds': a0['bounds'], 'late_separations': sep,
                   'frame0_match': match, 'real_gate_passed': passed}
            candidates.append(row)
            if passed:
                selected = {'log': log, 'base': str(base), **row}; break
        records.append({'log': log, 'base': str(base), 'candidates': candidates})
        save(OUT/'progress.json', {'logs': records, 'selected': selected, 'new_detector_calls': calls})
        print(json.dumps({'log': log, 'candidates': len(candidates), 'selected': None if selected is None else selected['actor']}), flush=True)
        if selected: break
    if selected:
        image = Image.open(Path(selected['base'])/'reference-000.png').convert('RGB'); draw = ImageDraw.Draw(image)
        draw.rectangle(selected['initial_bounds'], outline='#ff5148', width=4)
        if selected['frame0_match']: draw.rectangle(selected['frame0_match']['box'], outline='#ffe86b', width=3)
        draw.text((10, 10), f"{selected['log']} | {selected['actor']} | depth={selected['actor_depth_m']:.1f}m", fill='white', stroke_width=2, stroke_fill='black')
        image.save(OUT/'selected-input.jpg', quality=94)
    result = {'status': 'qualified' if selected else 'no_qualified_source', 'screened_logs': len(records), 'logs': records,
              'selected': selected, 'new_detector_calls': calls, 'new_reconstruction_calls': 0,
              'world_model_generation_calls': 0, 'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(OUT/'result.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'logs'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
