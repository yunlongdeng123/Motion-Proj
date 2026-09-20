"""仅按真实任务冻结前车与来源；不读取DVGT误差或生成结果来选任务。"""
from datetime import datetime, timezone
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from following_geometry import BASES, OUT, FRAMES, Route, scene, reference_lead, ground_plane
from prepare_argoverse import project_track, crop_image, CAMERA


def main():
    global BASES, OUT
    parser = argparse.ArgumentParser(); parser.add_argument('--source-window', choices=['visible-two', 'natural-four'], default='visible-two')
    args = parser.parse_args()
    if args.source_window == 'natural-four':
        root = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATURAL-SOURCES-01/20260920-r1/cases')
        BASES = [root/log/'base' for log in ['05fa5048-f355-3274-b565-c0ddc547b315', '0bae3b5e-417d-3b03-abaa-806b433233b8',
                                          '0c3bad78-9f1e-395d-a376-2eb7499229fd', '0fb7276f-ecb5-3e5b-87a8-cc74c709c715']]
        OUT = OUT.parent/'20260920-natural4'
    assert not (OUT/'task_sources.json').exists(), '不覆盖已完成任务筛查'
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'task_id': 'WS-V75-FOLLOWING-BASELINE-01', 'run_id': OUT.name,
                'frozen_utc': datetime.now(timezone.utc).isoformat(), 'role': 'exposed_development_sources',
                'bases': list(map(str, BASES)), 'frames': FRAMES,
                'selection': ('reuse two complete visible development sources; first route leader present at >=3/5 times, no reconstruction-error ranking'
                              if args.source_window == 'visible-two' else 'four already prepared natural development logs, frozen order; initial route leader must persist at >=3/5 times; first real-policy-qualified task; no further replacement window'),
                'virtual_ego': {'length_m': 4.8, 'width_m': 2.0, 'reference': 'center, official demo default; not claimed to be the recorded AV2 vehicle'},
                'task': 'follow/brake on provided route; non-interactive actors; first2s then fixed4s if real baseline passes',
                'policy': 'official nuPlan IDM + ordinary calibrated ground-contact RGB perception + route tracking',
                'idm': {'target_velocity': 10., 'min_gap': 1., 'headway': 1.5, 'accel_max': 1., 'decel_max': 3.},
                'detector': {'model': 'official FasterRCNN ResNet50 FPN v2 COCO_V1 cached', 'score': .5, 'classes': [3, 6, 8]},
                'input': ['current RGB', 'ego state and camera calibration', 'provided route', 'metric local map ground plane'],
                'reference_only': ['GT actors and trajectories', 'recorded future images for real-input baseline'],
                'admission': {'correct_leader_fraction': .8, 'median_abs_gap_error_m': 3., 'median_abs_idm_accel_difference_mps2': .5,
                              'max_false_braking_difference_mps2': 2., 'plane_p90_m': .2},
                'stop': 'no threshold or source replacement; real baseline failure stops error-arm generation; any OOM stops without retry',
                'human_verdict': None, 'failure_ledger_refs': ['V74-H2-F22'], 'failure_ledger_delta': 'none'}
    if (OUT/'protocol.json').exists():
        frozen = json.loads((OUT/'protocol.json').read_text())
        assert {k:v for k,v in frozen.items() if k != 'frozen_utc'} == {k:v for k,v in protocol.items() if k != 'frozen_utc'}
    else:
        (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    rows = []
    for base in BASES:
        traj = np.load(base/'trajectory.npz'); data = scene(base); route = Route(traj['ego_world'][:, :3, 3])
        manifest = json.loads((base/'input_manifest.json').read_text())
        images = sorted((Path(manifest['raw_path'])/'sensors/cameras'/CAMERA).glob('*.jpg'))
        times = np.array([int(x.stem) for x in images], np.int64)
        leads = [reference_lead(data, traj, route, f) for f in FRAMES]
        coef, plane = ground_plane(data, traj['ego_world'][0, :3, 3])
        counts = {x['id']: sum(y is not None and y['id'] == x['id'] for y in leads) for x in leads if x}
        initial_id = leads[0]['id'] if leads[0] else None
        target = initial_id if initial_id and counts[initial_id] >= 3 else None
        case = OUT/base.parent.name; case.mkdir(exist_ok=True)
        sheet = Image.new('RGB', (1280, 5*378), '#122335'); draw = ImageDraw.Draw(sheet)
        bounds = []
        for row, (f, lead) in enumerate(zip(FRAMES, leads)):
            idx = int(abs(times-traj['timestamps_ns'][f]).argmin())
            assert abs(times[idx]-traj['timestamps_ns'][f]) <= 1000
            rgb = crop_image(images[idx], manifest['crop_xyxy'])
            rgb.save(case/f'real-{f:03d}.png')
            overlay = rgb.copy(); d = ImageDraw.Draw(overlay)
            if lead:
                track = next(t for t in data['tracks'] if t['id'] == lead['id'] and t['segment'] == lead['segment'])
                proj = project_track(track, f, traj['camera_world'], traj['K'])
                if proj:
                    d.rectangle(proj['bounds'], outline='yellow', width=3)
                bounds.append(proj)
            else:
                bounds.append(None)
            y = row*378
            draw.text((8, y+5), f'{base.parent.name[:8]} | t={f/30:.1f}s | leader={lead}', fill='white')
            sheet.paste(rgb.resize((640, 352)), (0, y+26)); sheet.paste(overlay.resize((640, 352)), (640, y+26))
        sheet.save(case/'real-task-review.jpg', quality=94)
        rows.append({'log_id': base.parent.name, 'base': str(base), 'leaders': leads, 'projections': bounds,
                     'target': target, 'target_times': 0 if target is None else counts[target],
                     'ground_plane': coef.tolist(), 'ground_plane_check': plane})
    (OUT/'task_sources.json').write_text(json.dumps({'status': 'complete', 'cases': rows, 'human_verdict': None}, indent=2)+'\n')
    print(json.dumps([{'log': r['log_id'], 'target': r['target'], 'gaps': [x['gap_m'] if x else None for x in r['leaders']],
                       'plane_p90_m': r['ground_plane_check']['p90_residual_m']} for r in rows]), flush=True)


if __name__ == '__main__': main()
