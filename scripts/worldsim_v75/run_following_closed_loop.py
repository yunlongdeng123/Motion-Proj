"""单卡真实RGB策略反馈生成：每段生成后才计算下一动作，OOM直接停止。"""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import av
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from shapely.geometry import Polygon
import torch
from closed_loop_bridge import FeedbackBridge, run_feedback
from common import CAMERA, config
from evaluate_following_baseline import oracle_lead
from following_geometry import OUT as SOURCE, Route, scene, reference_lead
from rgb_idm_policy import RGBIDMPolicy

SOURCE = SOURCE.parent/'20260920-natural4'
OUT = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-FOLLOWING-CLOSEDLOOP-01/20260920-r1')
ARMS = ['gt_clean', 'dvgt_metric', 'dvgt_lidar_scaled', 'reference_lidar']


def footprint(xy, yaw, length, width):
    local = np.array([[length/2, width/2], [-length/2, width/2], [-length/2, -width/2], [length/2, -width/2]])
    R = np.array([[np.cos(yaw), -np.sin(yaw)], [np.sin(yaw), np.cos(yaw)]])
    return Polygon(local@R.T+np.asarray(xy))


def reference_clearance(data, frame, state):
    ego = footprint([state['x_m'], state['y_m']], state['yaw_rad'], 4.8, 2.)
    values = []
    for track in data['tracks']:
        if frame not in track['frames']: continue
        i = track['frames'].index(frame)
        yaw = Rotation.from_quat(track['quaternions'][i]).as_euler('xyz')[2]
        poly = footprint(track['centers'][i][:2], yaw, *track['dimensions'][:2])
        values.append({'id': track['id'], 'category': track['category'], 'distance_m': float(ego.distance(poly)),
                       'overlap_area_m2': float(ego.intersection(poly).area)})
    return min(values, key=lambda x: (x['distance_m'], -x['overlap_area_m2'])) if values else None


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', choices=ARMS, default='gt_clean')
    args = parser.parse_args(); baseline = json.loads((SOURCE/'real_policy_result.json').read_text())
    assert baseline['status'] == 'complete' and baseline['selected_log']
    source = next(x for x in json.loads((SOURCE/'task_sources.json').read_text())['cases'] if x['log_id'] == baseline['selected_log'])
    base = Path(source['base']); reference = scene(base)
    read = base.parent/'reconstruction/readout_ray_control_result.json'
    readings = json.loads(read.read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    protocol = {'task_id': 'WS-V75-FOLLOWING-CLOSEDLOOP-01', 'run_id': OUT.name, 'source_run': str(SOURCE),
                'base': str(base), 'source_log': source['log_id'], 'target': source['target'],
                'role': 'exposed_development_task; not independent confirmation', 'seed': 42,
                'frames': 117, 'fps': 30, 'blocks': 15, 'arms': ARMS,
                'change': 'target initial translation readout only, same offset across track; physical reference unchanged',
                'interface': 'generated RGB -> fixed RGB+IDM -> official integrate_vehicle -> camera -> Ludus -> cached OmniDreams',
                'extra_information': ['provided recorded route', 'known camera and ego state', 'known map ground plane',
                                      'GT dimensions/yaw and future actor trajectories shared', 'LiDAR arms add metric observations'],
                'first_gate': 'real RGB baseline admitted; GT generated feedback before error arms',
                'gt_feedback_gate': {'max_route_lateral_m': 1., 'no_reference_overlap': True,
                                     'median_abs_online_oracle_acceleration_difference_mps2': .5,
                                     'max_excess_braking_mps2': 2.},
                'stop': 'GT task baseline failure stops error arms; any OOM stops immediately without retry; no seed/threshold search',
                'human_verdict': None, 'failure_ledger_delta': 'none'}
    if (OUT/'protocol.json').exists():
        assert json.loads((OUT/'protocol.json').read_text()) == protocol
    else:
        assert args.arm == 'gt_clean'
        (OUT/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
        (OUT/'frozen_utc.txt').write_text(datetime.now(timezone.utc).isoformat()+'\n')
    if args.arm != 'gt_clean':
        gate = json.loads((OUT/'gt_clean/result.json').read_text())
        assert gate['status'] == 'complete' and gate['baseline_admitted']
        assert json.loads((OUT/'gt_clean/dense_reference_result.json').read_text())['status'] == 'passed'
    dest = OUT/args.arm; assert not dest.exists(); dest.mkdir()
    result = {'status': 'started', 'arm': args.arm, 'human_verdict': None}
    started = time.monotonic()
    try:
        assert source['target'] == readings['target']
        condition = copy.deepcopy(reference)
        target = next(t for t in condition['tracks'] if t['id'] == source['target'])
        offset = np.zeros(3) if args.arm == 'gt_clean' else np.array(readings['readouts'][args.arm]['offset_world_m'])
        target['centers'] = (np.array(target['centers'])+offset).tolist()
        assert all(a == b for a, b in zip(reference['tracks'], condition['tracks']) if a['id'] != source['target'])
        assert reference['lines'] == condition['lines'] and reference['crossings'] == condition['crossings']
        (dest/'condition_scene.json').write_text(json.dumps(condition, indent=2)+'\n')
        bridge = FeedbackBridge(base, condition)
        trajectory = bridge.trajectory
        policy = RGBIDMPolicy(trajectory['ego_world'][:, :3, 3], trajectory['K'], bridge.extrinsic, source['ground_plane'])
        cfg = config(); cfg.text_encoder = None; cfg.image_encoder = None; cfg.diffusion_model.seed = 42
        pipeline = cfg.setup().to('cuda').eval()
        # 文本与真实初帧的既有编码；四个分支完全共享，未使用未来RGB编码。
        embedding_path = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-NATURAL-ROLLOUT-01/20260920-r1/gt_clean/embeddings.pt')
        original_rgb = Image.open(base/'initial_rgb.png')
        assert np.array_equal(np.array(original_rgb), np.array(Image.open(embedding_path.parent/'initial_rgb.png')))
        embeddings = torch.load(embedding_path, weights_only=True, map_location='cpu')
        cache = pipeline.initialize_cache_from_embeddings(**embeddings, view_names=[CAMERA])
        frames_out = np.lib.format.open_memmap(dest/'generated.npy', mode='w+', dtype=np.uint8, shape=(117, 704, 1280, 3))
        condition_out = np.lib.format.open_memmap(dest/'conditions.npy', mode='w+', dtype=np.uint8, shape=(117, 704, 1280, 3))
        poses = []; decisions = []; route = Route(trajectory['ego_world'][:, :3, 3])
        with av.open(str(dest/'generated.mp4'), 'w') as writer:
            stream = writer.add_stream('libx264', rate=30); stream.width=1280; stream.height=704; stream.pix_fmt='yuv420p'; stream.options={'crf':'18'}
            def consume(row, record, generated):
                indices = record['indices']; frames_out[indices] = generated; condition_out[indices] = record['conditions']; poses.extend(record['camera_world'])
                observed = row['observation_frame']; state = policy.last['ego_state']
                gt = reference_lead(reference, trajectory, route, observed, [state['x_m'], state['y_m']])
                oracle = oracle_lead(gt, reference, trajectory, observed, state['yaw_rad'], state['speed_mps'])
                oracle_acc = policy.acceleration(state['speed_mps'], oracle)
                row.update(policy=copy.deepcopy(policy.last), oracle_lead=oracle, oracle_acceleration_mps2=oracle_acc,
                           acceleration_difference_mps2=policy.last['acceleration_mps2']-oracle_acc,
                           clearance=reference_clearance(reference, row['last_condition_frame'], row['ego_state']),
                           route_lateral_m=route.project([row['ego_state']['x_m'], row['ego_state']['y_m']])[1])
                decisions.append(row)
                for f in generated:
                    for packet in stream.encode(av.VideoFrame.from_ndarray(f, format='rgb24')): writer.mux(packet)
                Image.fromarray(generated[-1]).save(dest/f'generated-{indices[-1]:03d}.jpg', quality=94)
                (dest/'decisions.json').write_text(json.dumps(decisions, indent=2)+'\n')
                print(json.dumps({'arm': args.arm, 'block': row['block'], 'frames': row['last_condition_frame']+1,
                                  'observed_frame': observed, 'acceleration': policy.last['acceleration_mps2'],
                                  'lead_gap': policy.last['lead']['gap_m'] if policy.last['lead'] else None}), flush=True)
            run_feedback(pipeline, cache, bridge, policy, np.array(original_rgb), 15, consume)
            for packet in stream.encode(): writer.mux(packet)
        frames_out.flush(); condition_out.flush(); np.savez(dest/'camera_trajectory.npz', camera_world=np.array(poses), timestamps_us=trajectory['timestamps_us'][:117])
        assert reference == scene(base)
        with av.open(str(dest/'generated.mp4')) as video: decoded = sum(1 for _ in video.decode(video=0))
        assert decoded == 117
        max_lat = max(abs(r['route_lateral_m']) for r in decisions)
        abs_acc = float(np.median([abs(r['acceleration_difference_mps2']) for r in decisions]))
        max_brake = max(max(0, -r['acceleration_difference_mps2']) for r in decisions)
        overlaps = sum(r['clearance'] is not None and r['clearance']['overlap_area_m2'] > 0 for r in decisions)
        gate = protocol['gt_feedback_gate']
        admitted = max_lat <= gate['max_route_lateral_m'] and not overlaps and abs_acc <= gate['median_abs_online_oracle_acceleration_difference_mps2'] and max_brake <= gate['max_excess_braking_mps2']
        result.update(status='complete', baseline_admitted=bool(admitted) if args.arm == 'gt_clean' else None,
                      frames=117, decoded_frames=decoded, actual_policy_feedback=True, decisions=len(decisions),
                      max_route_lateral_m=max_lat, median_abs_online_oracle_acceleration_difference_mps2=abs_acc,
                      max_excess_braking_mps2=max_brake, reference_overlap_at_chunk_boundaries=overlaps,
                      min_reference_boundary_clearance_m=min(r['clearance']['distance_m'] for r in decisions if r['clearance']),
                      progress_m=route.project([bridge.state.x_m, bridge.state.y_m])[0], final_speed_mps=bridge.state.speed_mps,
                      offset_world_m=offset.tolist(), embedding_path=str(embedding_path), fixed_reference_unchanged=True)
    except BaseException as exc:
        result.update(status='oom_stopped' if isinstance(exc, torch.OutOfMemoryError) else 'failed_stopped', error_type=type(exc).__name__, error=str(exc)); raise
    finally:
        result.update(wall_s=time.monotonic()-started, peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
        (dest/'result.json').write_text(json.dumps(result, indent=2)+'\n')
        print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
