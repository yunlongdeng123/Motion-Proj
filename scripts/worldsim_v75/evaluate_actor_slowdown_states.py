"""比较 reference、DVGT 与 class-prior 状态对同一未来减速编辑的响应。"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from evaluate_localization import model, predict
from prepare_argoverse import project_track
from qualify_actor_removal import match_vehicle


ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
QUAL = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-QUALIFY-01/20260921-r2'
SOURCE = ROOT/'WS-V75-ACTOR-SLOWDOWN-STATE-GENERATION-01/20260921-r1'
BASE = SOURCE
REF_EDIT = ROOT/'WS-V75-ACTOR-SLOWDOWN-G1-GENERATION-01/20260921-r1/reference-edited'
REF_BASE = ROOT/'WS-V75-ACTOR-REMOVAL-G1-GENERATION-01/20260921-r1/reference-unedited'
ARMS = ['reference', 'dvgt_metric', 'class_prior']
LATE = [45, 61, 85, 109, 116]


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def center(box):
    box = np.asarray(box)
    return (box[:2]+box[2:])/2


def median(values):
    return None if not values else float(np.median(values))


def main():
    global QUAL,SOURCE,BASE,REF_EDIT,REF_BASE
    parser=argparse.ArgumentParser();parser.add_argument('--qualification',type=Path,default=QUAL)
    parser.add_argument('--output',type=Path,default=SOURCE);parser.add_argument('--base-generation',type=Path,default=BASE)
    parser.add_argument('--reference-edited',type=Path,default=REF_EDIT);parser.add_argument('--reference-unedited',type=Path,default=REF_BASE)
    args=parser.parse_args();QUAL,SOURCE,BASE=args.qualification,args.output,args.base_generation
    REF_EDIT,REF_BASE=args.reference_edited,args.reference_unedited
    assert json.loads((SOURCE/'queue_result.json').read_text())['status'] == 'complete'
    assert not (SOURCE/'evaluation.json').exists()
    protocol = json.loads((SOURCE/'protocol.json').read_text())
    q = json.loads((QUAL/'result.json').read_text()); selected = q['selected']; base = Path(selected['base'])
    tr = np.load(base/'trajectory.npz')
    arrays = {
        'reference': {'unedited': np.load(REF_BASE/'generated.npy', mmap_mode='r'),
                      'edited': np.load(REF_EDIT/'generated.npy', mmap_mode='r')},
        'dvgt_metric': {'unedited': np.load(BASE/'dvgt_metric-unedited/generated.npy', mmap_mode='r'),
                        'edited': np.load(SOURCE/'dvgt_metric-edited/generated.npy', mmap_mode='r')},
        'class_prior': {'unedited': np.load(BASE/'class_prior-unedited/generated.npy', mmap_mode='r'),
                        'edited': np.load(SOURCE/'class_prior-edited/generated.npy', mmap_mode='r')},
    }
    for arm in ARMS:
        assert np.array_equal(arrays[arm]['unedited'][:5], arrays[arm]['edited'][:5])
    reference_info = selected['state_inputs']['reference']
    reference_tracks = {}
    for variant, key in [('unedited', 'source'), ('edited', 'edited')]:
        scene = json.loads(Path(reference_info[key]).read_text())
        reference_tracks[variant] = next(t for t in scene['tracks'] if t['id'] == selected['actor'] and 0 in t['frames'])
    reference_centers = {(variant, frame): center(project_track(track, frame, tr['camera_world'], tr['K'])['bounds'])
                         for variant, track in reference_tracks.items() for frame in LATE}
    detector = model(); began = time.monotonic(); rows = []; summaries = {}
    for arm in ARMS:
        info = selected['state_inputs'][arm]
        original_scene = json.loads(Path(info['source']).read_text())
        edited_scene = json.loads(Path(info['edited']).read_text())
        ot = next(t for t in original_scene['tracks'] if t['id'] == selected['actor'] and 0 in t['frames'])
        et = next(t for t in edited_scene['tracks'] if (t['id'], t['segment']) == (ot['id'], ot['segment']))
        arm_rows = []
        for frame in LATE:
            op = project_track(ot, frame, tr['camera_world'], tr['K'])
            ep = project_track(et, frame, tr['camera_world'], tr['K'])
            detections = {}
            for variant, projection in [('unedited', op), ('edited', ep)]:
                pred = predict(detector, arrays[arm][variant][frame])
                own_match = None if projection is None else match_vehicle(pred, projection['bounds'])
                reference_projection = project_track(reference_tracks[variant], frame, tr['camera_world'], tr['K'])
                reference_match = match_vehicle(pred, reference_projection['bounds'])
                observed_match = own_match if own_match is not None else reference_match
                detection_center = None if observed_match is None else center(observed_match['box'])
                detections[variant] = {'match': own_match, 'reference_match': reference_match, 'center': detection_center}
            uc, ec = detections['unedited']['center'], detections['edited']['center']
            requested = None if op is None or ep is None else center(ep['bounds'])-center(op['bounds'])
            response = None if requested is None or uc is None or ec is None else ec-uc
            response_error = None if response is None else float(np.linalg.norm(response-requested))
            request_norm = None if requested is None else float(np.linalg.norm(requested))
            gain = None if response is None or request_norm < 1e-6 else float(np.dot(response, requested)/(request_norm**2))
            unedited_ok = detections['unedited']['match'] is not None and detections['unedited']['match']['iou'] >= .3
            edited_ok = detections['edited']['match'] is not None and detections['edited']['match']['iou'] >= .3
            preference = None if ec is None or op is None or ep is None else float(np.linalg.norm(ec-center(op['bounds']))-np.linalg.norm(ec-center(ep['bounds'])))
            source_unedited_error = None if uc is None else float(np.linalg.norm(uc-reference_centers['unedited', frame]))
            source_edited_error = None if ec is None else float(np.linalg.norm(ec-reference_centers['edited', frame]))
            row = {'arm': arm, 'frame': frame, 'original_projection': op, 'edited_projection': ep,
                   'unedited_match': detections['unedited']['match'], 'edited_match': detections['edited']['match'],
                   'unedited_reference_match': detections['unedited']['reference_match'],
                   'edited_reference_match': detections['edited']['reference_match'],
                   'unedited_detection_center': None if uc is None else uc.tolist(),
                   'edited_detection_center': None if ec is None else ec.tolist(),
                   'requested_delta_px': None if requested is None else requested.tolist(),
                   'response_delta_px': None if response is None else response.tolist(),
                   'pair_response_error_px': response_error, 'response_gain': gain,
                   'unedited_own_state_success': bool(unedited_ok),
                   'edited_own_state_success': bool(edited_ok and preference is not None and preference >= 5),
                   'edited_preference_px': preference,
                   'source_state_unedited_error_px': source_unedited_error,
                   'source_state_edited_error_px': source_edited_error}
            rows.append(row); arm_rows.append(row)
        summaries[arm] = {
            'unedited_own_state_successes': sum(x['unedited_own_state_success'] for x in arm_rows),
            'edited_own_state_successes': sum(x['edited_own_state_success'] for x in arm_rows),
            'median_pair_response_error_px': median([x['pair_response_error_px'] for x in arm_rows if x['pair_response_error_px'] is not None]),
            'median_response_gain': median([x['response_gain'] for x in arm_rows if x['response_gain'] is not None]),
            'median_source_state_unedited_error_px': median([x['source_state_unedited_error_px'] for x in arm_rows if x['source_state_unedited_error_px'] is not None]),
            'median_source_state_edited_error_px': median([x['source_state_edited_error_px'] for x in arm_rows if x['source_state_edited_error_px'] is not None]),
        }
    reference = summaries['reference']; effects = {}
    for arm in ['dvgt_metric', 'class_prior']:
        arm_error = summaries[arm]['median_pair_response_error_px']
        reference_error = reference['median_pair_response_error_px']
        response_increase = None if arm_error is None or reference_error is None else arm_error-reference_error
        success_loss = reference['edited_own_state_successes']-summaries[arm]['edited_own_state_successes']
        effects[arm] = {'response_error_increase_vs_reference_px': None if response_increase is None else float(response_increase),
                        'edited_success_loss_vs_reference': int(success_loss),
                        'materially_worse': bool(response_increase is not None and response_increase >= 10 or success_loss >= 2)}
    material = any(x['materially_worse'] for x in effects.values())
    sheet = Image.new('RGB', (len(LATE)*366, len(ARMS)*2*230), '#13202e')
    draw = ImageDraw.Draw(sheet)
    for ai, arm in enumerate(ARMS):
        for vi, variant in enumerate(['unedited', 'edited']):
            y = (ai*2+vi)*230
            draw.text((5, y+4), f'{arm} / {variant} | red=own original cyan=own slowed yellow=detection', fill='white')
            for ci, frame in enumerate(LATE):
                row = next(x for x in rows if x['arm'] == arm and x['frame'] == frame)
                image = Image.fromarray(np.asarray(arrays[arm][variant][frame])).copy(); d = ImageDraw.Draw(image)
                if row['original_projection'] is not None:
                    d.rectangle(row['original_projection']['bounds'], outline='#ff5148', width=3)
                if row['edited_projection'] is not None:
                    d.rectangle(row['edited_projection']['bounds'], outline='#00e6cf', width=3)
                match = row[f'{variant}_match']
                if match is None:
                    match = row[f'{variant}_reference_match']
                if match is not None and match['iou'] >= .3:
                    d.rectangle(match['box'], outline='#ffe86b', width=3)
                d.text((8, 8), f'f={frame}', fill='white', stroke_width=2, stroke_fill='black')
                sheet.paste(image.resize((366, 206)), (ci*366, y+24))
    sheet.save(SOURCE/'evaluation-review.jpg', quality=94)
    result = {'status': 'complete', 'late_frames': LATE, 'rows': rows, 'summaries': summaries,
              'material_effects': effects, 'material_reconstruction_state_effect': bool(material),
              'scientific_readout': ('state_errors_matter_on_this_source' if material else 'tested_state_errors_absorbed_on_this_source'),
              'new_detector_calls': len(rows)*2, 'new_world_model_calls': 0,
              'wall_s': time.monotonic()-began, 'cuda_initialized': torch.cuda.is_initialized(),
              'human_verdict': None, 'failure_ledger_delta': 'none'}
    save(SOURCE/'evaluation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
