"""四份冻结场景的CPU直接状态反馈；先复现旧动力学，再做普通强控制。"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from closed_loop_bridge import initial_state, VehicleConfig, DriverCommand, rig_pose_from_state, integrate_vehicle
from interactive_drive.simulation.ground_snap import GroundSnapper
from evaluate_following_baseline import oracle_lead
from following_geometry import scene, reference_lead
from raster_ground import RasterGround
from rgb_idm_policy import RGBIDMPolicy
from run_following_closed_loop import footprint, reference_clearance

SOURCE = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2')
OUT = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-APPROACH-STATE-CONTROL-01/20260920-r1')
ARMS = ['gt_clean', 'dvgt_metric', 'dvgt_lidar_scaled', 'reference_lidar']


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False)+'\n')


def state_action(data, tr, policy, frame, state):
    ref = reference_lead(data, tr, policy.route, frame, [state['x_m'], state['y_m']])
    lead = oracle_lead(ref, data, tr, frame, state['yaw_rad'], state['speed_mps'])
    return lead, policy.acceleration(state['speed_mps'], lead)


def target_clearance(target, frame, state):
    i = target['frames'].index(frame)
    yaw = Rotation.from_quat(target['quaternions'][i]).as_euler('xyz')[2]
    a = footprint([state['x_m'], state['y_m']], state['yaw_rad'], 4.8, 2.)
    b = footprint(target['centers'][i][:2], yaw, *target['dimensions'][:2])
    return float(a.distance(b))


def rollout(base, tr, policy, terrain, condition, reference, target, source_rows, replay):
    state, _ = initial_state(base, tr); config = VehicleConfig()
    snapper = GroundSnapper(*terrain); snapper.snap(state, config)
    extrinsic = np.linalg.inv(tr['ego_world'][0])@tr['camera_world'][0]
    frames = []; decisions = []; cameras = []; cursor = 0
    for block in range(15):
        observed = 0 if block == 0 else cursor-1
        before = asdict(state)
        lead, direct_acc = state_action(condition, tr, policy, observed, before)
        if replay or block == 0:
            command = DriverCommand(**source_rows[block]['command'])
            acceleration = source_rows[block]['policy']['acceleration_mps2']
        else:
            acceleration = direct_acc
            command, _ = policy.command_for_acceleration(before, acceleration)
        end = cursor+(5 if block == 0 else 8)
        decisions.append({'block': block, 'observation_frame': observed,
                          'observation_source': 'saved_rgb_commands' if replay else ('shared_initial_rgb_action' if block == 0 else 'condition_state_at_current_time'),
                          'first_condition_frame': cursor, 'last_condition_frame': end-1,
                          'ego_state_before': before, 'condition_lead': lead,
                          'direct_state_acceleration_mps2': direct_acc,
                          'applied_acceleration_mps2': acceleration, 'command': asdict(command)})
        for f in range(cursor, end):
            if f > 0:
                dt = float(tr['timestamps_us'][f]-tr['timestamps_us'][f-1])/1e6
                state = snapper.snap(integrate_vehicle(state, command, dt, config), config)
            s = asdict(state)
            camera = rig_pose_from_state(state.x_m, state.y_m, state.z_m, state.yaw_rad,
                                        state.pitch_rad+state.suspension_pitch_rad,
                                        state.roll_rad+state.suspension_roll_rad)@extrinsic
            cameras.append(camera)
            frames.append({'frame': f, 'time_s': float(tr['timestamps_us'][f]-tr['timestamps_us'][0])/1e6,
                           'ego_state': s, 'route_progress_lateral': list(policy.route.project([state.x_m, state.y_m])),
                           'target_reference_clearance_m': target_clearance(target, f, s),
                           'reference_clearance': reference_clearance(reference, f, s)})
        cursor = end
    assert cursor == 117 and len(decisions) == 15
    result = {'status': 'complete', 'mode': 'replay' if replay else 'direct_state_feedback',
              'frames': frames, 'decisions': decisions,
              'progress_m': frames[-1]['route_progress_lateral'][0]-frames[0]['route_progress_lateral'][0],
              'final_speed_mps': state.speed_mps,
              'final_target_reference_clearance_m': frames[-1]['target_reference_clearance_m'],
              'overlap_frames': [r['frame'] for r in frames if r['reference_clearance'] and r['reference_clearance']['overlap_area_m2'] > 0],
              'human_verdict': None}
    return result, np.array(cameras)


def main():
    global SOURCE,OUT
    parser=argparse.ArgumentParser();parser.add_argument('--source-run',type=Path,default=SOURCE)
    parser.add_argument('--run-dir',type=Path,default=OUT);parser.add_argument('--task-id',default='WS-V75-APPROACH-STATE-CONTROL-01')
    args=parser.parse_args();SOURCE,OUT=args.source_run,args.run_dir
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == '', '本控制必须明确禁用GPU'
    assert not OUT.exists(); OUT.mkdir(parents=True)
    source = json.loads((SOURCE/'protocol.json').read_text()); base = Path(source['base'])
    assert json.loads((SOURCE/'queue_result.json').read_text())['status'] == 'complete'
    tr = np.load(base/'trajectory.npz'); reference = scene(base)
    target = next(t for t in reference['tracks'] if t['id'] == source['target'] and 0 in t['frames'])
    protocol = {'task_id': args.task_id, 'run_id': OUT.name,
                'frozen_utc': datetime.now(timezone.utc).isoformat(), 'source': str(SOURCE), 'base': str(base),
                'source_log': source['source_log'], 'target': source['target'], 'arms': ARMS,
                'frames': 117, 'decisions': 15, 'seed': None,
                'question': 'Which execution differences already arise from geometry and ordinary control, before generated observations?',
                'intervention': 'same initial RGB-derived command; from frame4 read each frozen condition scene at current time; no future-state lookahead',
                'velocity': 'same oracle helper as previous evaluation: backward difference of current and previous actor positions; initial command shared',
                'same_components': ['initial ego', '117 original timestamps', '5 then8 frame zero-order hold', 'nuPlan IDM parameters',
                                    '40m lead range', 'provided route steering and actuator mapping', 'official integrate_vehicle', 'official GroundSnapper'],
                'extra_information': ['full current condition actor positions, dimensions, yaw and exact backward velocities',
                                      'provided route and metric ground', 'shared nonreactive actor trajectories'],
                'role': 'post hoc strong diagnostic on one exposed development scene; not same-information visual policy ranking',
                'verification': {'replay_all_four_camera_matrices_max_abs': 1e-7, 'saved_action_acceleration_max_abs': 1e-10},
                'analysis': ['four actual state feedback rollouts', 'offline intended-state action at each saved generated ego state',
                             'offline geometry action at common GT-generated ego states, excluding shared startup'],
                'stop': 'four frozen arms only; no new generation, detector, source, seed, horizon, thresholds, or policy tuning',
                'human_verdict': None, 'failure_ledger_delta': 'none', 'failure_ledger_refs': ['V74-H2-F22']}
    save(OUT/'protocol.json', protocol)
    start = time.monotonic(); result = {'status': 'running', 'human_verdict': None, 'failure_ledger_delta': 'none'}
    try:
        extrinsic = np.linalg.inv(tr['ego_world'][0])@tr['camera_world'][0]
        policy = RGBIDMPolicy(tr['ego_world'][:, :3, 3], tr['K'], extrinsic, [0.,0.,0.], detector=False)
        terrain = RasterGround(base).mesh_for_route(tr['ego_world'][:,:2,3])
        runs = {}; recorded = {}; conditions = {}; checks = []
        for arm in ARMS:
            recorded[arm] = json.loads((SOURCE/arm/'decisions.json').read_text())
            conditions[arm] = json.loads((SOURCE/arm/'condition_scene.json').read_text())
            assert recorded[arm][0]['command'] == recorded['gt_clean'][0]['command']
            for row in recorded[arm]:
                state = row['policy']['ego_state']
                acc = policy.acceleration(state['speed_mps'], row['policy']['lead'])
                command, steer = policy.command_for_acceleration(state, acc)
                assert abs(acc-row['policy']['acceleration_mps2']) < 1e-10
                assert asdict(command) == row['command'] and steer == row['policy']['steer_rad']
            replay, cameras = rollout(base, tr, policy, terrain, conditions[arm], reference, target, recorded[arm], True)
            old_cameras = np.load(SOURCE/arm/'camera_trajectory.npz')['camera_world']
            error = float(np.max(abs(cameras-old_cameras)))
            assert error <= protocol['verification']['replay_all_four_camera_matrices_max_abs'], error
            state_error = max(abs(replay['frames'][row['last_condition_frame']]['ego_state'][key]-row['ego_state'][key])
                              for row in recorded[arm] for key in ['x_m','y_m','z_m','speed_mps','yaw_rad'])
            assert state_error <= 1e-10
            checks.append({'arm':arm, 'camera_max_abs':error, 'boundary_state_max_abs':state_error,
                           'frames_checked':117, 'saved_commands_checked':15, 'status':'passed'})
        save(OUT/'replay_verification.json', {'status':'passed','arms':checks,'new_model_calls':0})
        print(json.dumps({'stage':'four_command_replays_passed','checks':checks}), flush=True)
        for arm in ARMS:
            runs[arm], cameras = rollout(base, tr, policy, terrain, conditions[arm], reference, target, recorded[arm], False)
            save(OUT/f'{arm}.json', runs[arm])
            np.savez(OUT/f'{arm}_trajectory.npz', camera_world=cameras, timestamps_us=tr['timestamps_us'][:117])
        offline = []
        for arm in ARMS:
            for i, row in enumerate(recorded[arm]):
                f = row['observation_frame']; s = row['policy']['ego_state']
                lead, acc = state_action(conditions[arm], tr, policy, f, s)
                truth, true_acc = state_action(reference, tr, policy, f, s)
                common_s = recorded['gt_clean'][i]['policy']['ego_state']
                _, common_acc = state_action(conditions[arm], tr, policy, f, common_s)
                _, common_gt = state_action(reference, tr, policy, f, common_s)
                pred = row['policy']['lead']
                offline.append({'arm':arm,'frame':f,'shared_startup':i==0,'condition_lead':lead,'physical_reference_lead':truth,
                                'actual_rgb_acceleration_mps2':row['policy']['acceleration_mps2'],
                                'condition_acceleration_at_own_ego_mps2':acc,'reference_acceleration_at_own_ego_mps2':true_acc,
                                'rgb_minus_condition_action_mps2':row['policy']['acceleration_mps2']-acc,
                                'condition_minus_reference_action_mps2':acc-true_acc,
                                'geometry_action_change_at_common_gt_ego_mps2':common_acc-common_gt,
                                'rgb_lead_present':pred is not None,
                                'rgb_minus_condition_gap_m':pred['gap_m']-lead['gap_m'] if pred and lead else None,
                                'rgb_minus_condition_speed_mps':pred['lead_speed_mps']-lead['lead_speed_mps'] if pred and lead else None,
                                'gap_speed_warning':'selected leads not identity-matched; action comparison does not need correspondence'})
        save(OUT/'offline_action_comparison.json', {'status':'complete','rows':offline,'role':'saved-ego one-step diagnostic, not new generated closed loops'})
        old_summary = {r['arm']:r for r in json.loads((SOURCE/'review/comparison.json').read_text())['cases']}
        baseline = runs['gt_clean']; summary = []
        for arm in ARMS:
            r = runs[arm]; x = [v for v in offline if v['arm']==arm and not v['shared_startup']]
            actions = np.array([a['applied_acceleration_mps2'] for a in r['decisions']])
            ref_actions = np.array([a['applied_acceleration_mps2'] for a in baseline['decisions']])
            delta = r['progress_m']-baseline['progress_m']
            actual_delta = old_summary[arm]['progress_change_vs_gt_m']
            summary.append({'arm':arm, 'initial_geometry_residual_m':old_summary[arm]['initial_target_center_residual_m'],
                            'direct_progress_m':r['progress_m'],'direct_progress_change_vs_gt_m':delta,
                            'rgb_progress_change_vs_gt_m':actual_delta,
                            'difference_between_branch_contrasts_m':actual_delta-delta,
                            'direct_mean_abs_action_change_vs_gt_mps2':float(np.mean(abs(actions-ref_actions))),
                            'rgb_mean_abs_action_change_vs_gt_mps2':old_summary[arm]['mean_abs_acceleration_change_vs_gt_mps2'],
                            'mean_abs_rgb_minus_condition_action_at_own_ego_mps2':float(np.mean([abs(v['rgb_minus_condition_action_mps2']) for v in x])),
                            'mean_abs_geometry_action_change_at_common_gt_ego_mps2':float(np.mean([abs(v['geometry_action_change_at_common_gt_ego_mps2']) for v in x])),
                            'direct_final_speed_mps':r['final_speed_mps'],
                            'direct_final_target_reference_clearance_m':r['final_target_reference_clearance_m'],
                            'direct_overlap_frames':r['overlap_frames']})
        assert not torch.cuda.is_initialized()
        result.update(status='complete',cases=summary, new_gpu_calls=0, direct_control_frames=468, replay_verified_frames=468,
                      saved_commands_verified=60, offline_current_state_comparisons=60,
                      cuda_initialized=False, contrast_boundary='difference of two policy-interface branch contrasts; not a pure generative causal effect',
                      failure_ledger_delta='none')
    except BaseException as exc:
        result.update(status='failed_stopped', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        result['wall_s'] = time.monotonic()-start
        save(OUT/'result.json', result); print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__': main()
