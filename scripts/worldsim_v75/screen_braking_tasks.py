"""按记录减速和可见前车冻结有限任务窗口，不读重建或生成输出。"""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from prepare_argoverse import ROOT, CAMERA, poses, project_track, crop_image
from following_geometry import Route, scene, reference_lead, ground_plane, FRAMES
from evaluate_following_baseline import logged_state, oracle_lead
from rgb_idm_policy import IDMPolicy, RGBIDMPolicy

OUT = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-BRAKING-TASKS-01/20260920-r1')


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    assert not OUT.exists(), '不覆盖或换窗口重试'
    OUT.mkdir(parents=True)
    logs = sorted(p for p in ROOT.iterdir() if p.is_dir() and (p/'annotations.feather').exists()
                  and (p/'city_SE3_egovehicle.feather').exists()
                  and len(list((p/'sensors/cameras'/CAMERA).glob('*.jpg'))) >= 200)[:12]
    protocol = {'task_id': 'WS-V75-BRAKING-TASKS-01', 'run_id': OUT.name,
        'frozen_utc': datetime.now(timezone.utc).isoformat(), 'logs': [p.name for p in logs],
        'start_offsets_seconds': [.5, 2.5, 4.5, 6.5], 'measurement_frames': FRAMES,
        'boundary': 'development task selection from locally complete AV2 logs; exposure not independent',
        'target_selection': 'first twelve lexicographic locally complete logs, fixed four starts; recorded speed drop then persistent visible route leader; no reconstruction-error ranking',
        'motion_gate': {'initial_speed_min_mps': 2., 'speed_drop_over_2s_min_mps': 1.,
                        'speed_estimation': 'backward 0.5s position displacement projected on current ego forward'},
        'task_gate': {'initial_leader_present_times_min': 3, 'fully_visible_all_leader_times': True,
                      'initial_bbox_min_wh': [48,32], 'oracle_leader_braking_times_min': 2,
                      'oracle_acceleration_max_mps2': -.5, 'leader_effect_vs_free_acceleration_max_mps2': -.5},
        'admission': {'correct_leader_fraction': .8, 'median_abs_gap_error_m': 3.,
                      'median_abs_idm_accel_difference_mps2': .5, 'max_false_braking_difference_mps2': 2., 'plane_p90_m': .2},
        'max_real_baseline_tasks': 2, 'selection_order': 'first qualifying start per log; first two distinct logs for real RGB policy; first qualified for GT feedback',
        'generation': 'fixed117 frames seed42; GT feedback then native DVGT / ordinary global LiDAR scale / target LiDAR if baseline and references pass',
        'stop': 'finite12x4 window; do not replace failed windows, tune thresholds or select by reconstruction error; any OOM immediately stops',
        'failure_ledger_refs': ['V74-H2-F22'], 'failure_ledger_delta': 'none', 'human_verdict': None}
    save(OUT/'protocol.json', protocol)
    # 只需要IDM算式，不实例化检测器、不使用GPU。
    oracle_policy = object.__new__(RGBIDMPolicy)
    oracle_policy.idm = IDMPolicy(target_velocity=10., min_gap_to_lead_agent=1., headway_time=1.5, accel_max=1., decel_max=3.)
    rows, cases = [], []
    for raw in logs:
        images = sorted((raw/'sensors/cameras'/CAMERA).glob('*.jpg'))
        times = np.array([int(p.stem) for p in images], np.int64)
        ego = pd.read_feather(raw/'city_SE3_egovehicle.feather')
        qualifying = []
        for offset in protocol['start_offsets_seconds']:
            start = int(times[np.searchsorted(times, times[0]+round(offset*1e9))])
            row = {'log_id': raw.name, 'offset_seconds': offset, 'start_ns': start}
            if start-500_000_000 < ego.timestamp_ns.min() or start+7_866_666_667 > min(times[-1], ego.timestamp_ns.max()):
                row['reason'] = 'insufficient_fixed_window_or_speed_history'; rows.append(row); continue
            stamps = start+np.array([-.5,0,1.5,2])*1e9
            E = poses(ego, stamps.astype(np.int64))
            speeds = [float((E[b,:3,3]-E[a,:3,3])@E[b,:3,0]/.5) for a,b in [(0,1),(2,3)]]
            row.update(recorded_speed_mps=speeds, recorded_speed_drop_mps=speeds[0]-speeds[1])
            if speeds[0] < 2 or speeds[0]-speeds[1] < 1:
                row['reason'] = 'no_recorded_braking_demand'; rows.append(row); continue
            base = OUT/'cases'/raw.name/f'start-{offset:.1f}'/'base'
            command = [sys.executable, str(Path(__file__).with_name('prepare_argoverse.py')), '--output', str(base),
                       '--log-id', raw.name, '--start-offset-seconds', str(offset), '--task-id', protocol['task_id']]
            base.parent.mkdir(parents=True)
            with (base.parent/'prepare.log').open('w') as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
            data = scene(base); traj = np.load(base/'trajectory.npz'); route = Route(traj['ego_world'][:,:3,3])
            leads = [reference_lead(data,traj,route,f) for f in FRAMES]
            target = leads[0]['id'] if leads[0] else None
            projections, actions = [], []
            for f, lead in zip(FRAMES, leads):
                state = logged_state(base,traj,f)
                ref = oracle_lead(lead,data,traj,f,state['yaw_rad'],state['speed_mps'])
                accel = oracle_policy.acceleration(state['speed_mps'],ref)
                free = oracle_policy.acceleration(state['speed_mps'],None)
                actions.append({'frame':f,'oracle_acceleration_mps2':accel,'free_acceleration_mps2':free,
                                'leader_effect_mps2':accel-free,'recorded_speed_mps':state['speed_mps']})
                track = next((t for t in data['tracks'] if lead and t['id']==lead['id'] and t['segment']==lead['segment']),None)
                projections.append(project_track(track,f,traj['camera_world'],traj['K']) if track else None)
            count = sum(bool(x and x['id']==target) for x in leads) if target else 0
            visible = all(p and p['fully_inside'] for lead,p in zip(leads,projections) if lead)
            box = projections[0]['bounds'] if projections[0] else [0,0,0,0]
            braking = sum(a['oracle_acceleration_mps2']<=-.5 and a['leader_effect_mps2']<=-.5 for a in actions)
            qualified = count>=3 and visible and box[2]-box[0]>=48 and box[3]-box[1]>=32 and braking>=2
            coef, plane = ground_plane(data,traj['ego_world'][0,:3,3])
            row.update(base=str(base), target=target, leaders=leads, projections=projections, oracle_actions=actions,
                       target_times=count, fully_visible=visible, oracle_braking_times=braking,
                       ground_plane=coef.tolist(), ground_plane_check=plane,
                       qualified=bool(qualified), reason='task_geometry_qualified' if qualified else 'no_visible_leader_braking_task')
            rows.append(row)
            if qualified: qualifying.append(row)
        if qualifying: cases.append(qualifying[0])
        save(OUT/'screening_progress.json', {'rows': rows, 'qualified_logs': [x['log_id'] for x in cases]})
        print(json.dumps({'log':raw.name,'qualified_starts':len(qualifying)}),flush=True)
    selected = cases[:2]
    for case in selected:
        base = Path(case['base']); manifest = json.loads((base/'input_manifest.json').read_text())
        traj = np.load(base/'trajectory.npz'); folder = OUT/case['log_id']; folder.mkdir()
        images = sorted((Path(manifest['raw_path'])/'sensors/cameras'/CAMERA).glob('*.jpg'))
        stamps = np.array([int(p.stem) for p in images], np.int64)
        sheet = Image.new('RGB',(1280,5*378),'#122335')
        for i,f in enumerate(FRAMES):
            index = abs(stamps-traj['timestamps_ns'][f]).argmin()
            assert abs(stamps[index]-traj['timestamps_ns'][f])<=1000
            rgb = crop_image(images[index],manifest['crop_xyxy']); rgb.save(folder/f'real-{f:03d}.png')
            overlay = rgb.copy(); p = case['projections'][i]
            if p: ImageDraw.Draw(overlay).rectangle(p['bounds'],outline='yellow',width=3)
            sheet.paste(rgb.resize((640,352)),(0,i*378+26)); sheet.paste(overlay.resize((640,352)),(640,i*378+26))
            ImageDraw.Draw(sheet).text((8,i*378+5),f'{case["log_id"][:8]} +{case["offset_seconds"]}s | t={f/30:.1f}s | GT leader, real RGB',fill='white')
        sheet.save(folder/'real-task-review.jpg',quality=94)
    result = {'status':'complete','window_count':len(rows),'qualified_log_count':len(cases),
              'rows':rows,'cases':selected,'human_verdict':None,'failure_ledger_delta':'none'}
    save(OUT/'task_sources.json',result)
    print(json.dumps({'status':'complete','windows':len(rows),'motion_windows':sum('base' in r for r in rows),
                      'qualified_logs':len(cases),'selected':[x['log_id'] for x in selected]}),flush=True)


if __name__ == '__main__': main()
