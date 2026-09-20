"""真实RGB策略基线：参考车辆只进入评价/独立oracle，不传入RGB策略。"""
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
from scipy.spatial.transform import Rotation
from PIL import Image, ImageDraw
import torch
from closed_loop_bridge import VehicleState, initial_state
from evaluate_localization import iou
from following_geometry import OUT as ROOT_OUT, FRAMES, Route, scene, reference_lead
from prepare_argoverse import project_track
from rgb_idm_policy import RGBIDMPolicy

OUT = ROOT_OUT.parent/'20260920-natural4'


def logged_state(base, trajectory, frame):
    if frame == 0: return asdict(initial_state(base, trajectory)[0])
    poses = trajectory['ego_world']; pose = poses[frame]
    roll, pitch, yaw = Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')
    dt = (trajectory['timestamps_us'][frame]-trajectory['timestamps_us'][frame-1])/1e6
    velocity = (pose[:3, 3]-poses[frame-1, :3, 3])/dt
    return asdict(VehicleState(*map(float, pose[:3, 3]), float(yaw), float(velocity@pose[:3, 0]), 0.,
                              pitch_rad=float(pitch), roll_rad=float(roll),
                              velocity_x_mps=float(velocity[0]), velocity_y_mps=float(velocity[1])))


def oracle_lead(reference, data, trajectory, frame, yaw, ego_speed):
    if reference is None: return None
    track = next(t for t in data['tracks'] if t['id'] == reference['id'] and t['segment'] == reference['segment'])
    idx = track['frames'].index(frame)
    if idx:
        dt = (trajectory['timestamps_us'][frame]-trajectory['timestamps_us'][track['frames'][idx-1]])/1e6
        velocity = (np.array(track['centers'][idx])-track['centers'][idx-1])/dt
        lead_speed = float(velocity[:2]@np.array([np.cos(yaw), np.sin(yaw)]))
    else:
        # 参考初帧也不读取未来速度；与RGB的零相对速度初始化一致。
        lead_speed = ego_speed
    return {**reference, 'lead_speed_mps': float(np.clip(lead_speed, 0, 35))}


def main():
    path = OUT/'real_policy_result.json'; assert not path.exists()
    source = json.loads((OUT/'task_sources.json').read_text())
    protocol = json.loads((OUT/'protocol.json').read_text()); gate = protocol['admission']
    result = {'status': 'started', 'cases': [], 'selected_log': None, 'human_verdict': None,
              'failure_ledger_delta': 'none', 'generated_policy_rollouts': 0}
    started = time.monotonic()
    try:
        detector = None
        for case in source['cases']:
            if not case['target']:
                result['cases'].append({'log': case['log_id'], 'admitted': False, 'reason': 'no_persistent_initial_leader'}); continue
            base = Path(case['base']); folder = OUT/case['log_id']; traj = np.load(base/'trajectory.npz'); data = scene(base)
            extrinsic = np.linalg.inv(traj['ego_world'][0])@traj['camera_world'][0]
            policy = RGBIDMPolicy(traj['ego_world'][:, :3, 3], traj['K'], extrinsic, case['ground_plane'], detector)
            detector = policy.detector
            rows = []; sheet = Image.new('RGB', (1280, 5*410), '#122335'); draw = ImageDraw.Draw(sheet)
            for row_index, f in enumerate(FRAMES):
                rgb = np.array(Image.open(folder/f'real-{f:03d}.png'))
                state = logged_state(base, traj, f)
                policy.step(rgb=rgb, frame_index=f, timestamp_us=int(traj['timestamps_us'][f]), ego_state=state,
                            camera_override=traj['camera_world'][f])
                pred = policy.last['lead']; gt = reference_lead(data, traj, policy.route, f)
                oracle = oracle_lead(gt, data, traj, f, state['yaw_rad'], state['speed_mps'])
                oracle_acc = policy.acceleration(state['speed_mps'], oracle)
                bounds = None
                if gt:
                    track = next(t for t in data['tracks'] if t['id'] == gt['id'] and t['segment'] == gt['segment'])
                    projection = project_track(track, f, traj['camera_world'], traj['K'])
                    bounds = None if projection is None else projection['bounds']
                correct = pred is None and gt is None or (pred is not None and bounds is not None and iou(pred['box'], bounds) >= .3)
                gap_error = abs(pred['gap_m']-gt['gap_m']) if correct and gt is not None else None
                row = {'frame': f, 'reference_lead': gt, 'reference_bounds': bounds, 'policy': policy.last,
                       'correct_leader': bool(correct), 'gap_error_m': gap_error,
                       'oracle_acceleration_mps2': oracle_acc,
                       'acceleration_difference_mps2': policy.last['acceleration_mps2']-oracle_acc}
                rows.append(row)
                image = Image.fromarray(rgb); d = ImageDraw.Draw(image)
                if bounds: d.rectangle(bounds, outline='yellow', width=3)
                if pred: d.rectangle(pred['box'], outline='#22e6b3', width=3)
                y = row_index*410
                label = f't={f/30:.1f}s | GT gap={gt["gap_m"] if gt else None} | RGB gap={pred["gap_m"] if pred else None} | correct={correct}'
                draw.text((8, y+5), label, fill='white')
                draw.text((8, y+24), f'IDM acceleration: RGB={policy.last["acceleration_mps2"]:.3f}, oracle={oracle_acc:.3f} m/s^2 | yellow=GT / green=RGB policy', fill='white')
                sheet.paste(image.resize((640, 352)), (0, y+54))
                # 右侧固定前方ROI，包含所有当前检测，不只裁选中的前车。
                crop = image.crop((320, 240, 1120, 620)); sheet.paste(crop.resize((640, 304)), (640, y+54))
            sheet.save(folder/'real-policy-review.jpg', quality=94)
            correct_fraction = np.mean([r['correct_leader'] for r in rows])
            gap_errors = [r['gap_error_m'] for r in rows if r['gap_error_m'] is not None]
            gap = float(np.median(gap_errors)) if gap_errors else None
            acc = float(np.median([abs(r['acceleration_difference_mps2']) for r in rows]))
            false_brake = float(max(max(0, -r['acceleration_difference_mps2']) for r in rows))
            admitted = bool(correct_fraction >= gate['correct_leader_fraction'] and gap is not None and gap <= gate['median_abs_gap_error_m']
                            and acc <= gate['median_abs_idm_accel_difference_mps2'] and false_brake <= gate['max_false_braking_difference_mps2']
                            and case['ground_plane_check']['p90_residual_m'] <= gate['plane_p90_m'])
            summary = {'log': case['log_id'], 'admitted': admitted, 'correct_leader_fraction': float(correct_fraction),
                       'median_abs_gap_error_m': gap, 'median_abs_acceleration_difference_mps2': acc,
                       'max_excess_braking_mps2': false_brake, 'rows': rows}
            (folder/'real_policy_result.json').write_text(json.dumps(summary, indent=2)+'\n')
            result['cases'].append(summary)
            if admitted and result['selected_log'] is None: result['selected_log'] = case['log_id']
            print(json.dumps({k:v for k,v in summary.items() if k != 'rows'}), flush=True)
        result['status'] = 'complete'
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc, torch.OutOfMemoryError) else 'failed_stopped',
                      error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        result.update(wall_s=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        path.write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps({k:v for k,v in result.items() if k != 'cases'}), flush=True)


if __name__ == '__main__': main()
