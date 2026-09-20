"""独立逐帧参考评价和动作时序审计；不能只查chunk边界排除碰撞。"""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from following_geometry import Route, scene
from run_following_closed_loop import OUT, reference_clearance


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--arm', required=True); args = parser.parse_args()
    folder = OUT/args.arm; path = folder/'dense_reference_result.json'; assert not path.exists()
    run = json.loads((folder/'result.json').read_text()); assert run['status'] == 'complete'
    protocol = json.loads((OUT/'protocol.json').read_text()); base = Path(protocol['base']); reference = scene(base)
    original = np.load(base/'trajectory.npz'); trajectory = np.load(folder/'camera_trajectory.npz')
    extrinsic = np.linalg.inv(original['ego_world'][0])@original['camera_world'][0]
    rig = trajectory['camera_world']@np.linalg.inv(extrinsic)
    route = Route(original['ego_world'][:, :3, 3]); rows = []
    for f, pose in enumerate(rig):
        yaw = float(Rotation.from_matrix(pose[:3, :3]).as_euler('xyz')[2])
        state = {'x_m': float(pose[0, 3]), 'y_m': float(pose[1, 3]), 'yaw_rad': yaw}
        rows.append({'frame': f, 'time_s': float((trajectory['timestamps_us'][f]-trajectory['timestamps_us'][0])/1e6),
                     'ego_xy_yaw': [state['x_m'], state['y_m'], yaw],
                     'route_progress_lateral': list(route.project(pose[:2, 3])),
                     'clearance': reference_clearance(reference, f, state)})
    assert len(rows) == 117 and np.array_equal(trajectory['timestamps_us'], original['timestamps_us'][:117])
    decisions = json.loads((folder/'decisions.json').read_text())
    for i, row in enumerate(decisions):
        expected = 0 if i == 0 else 4+8*(i-1)
        assert row['observation_frame'] == expected and row['policy']['frame'] == expected
        assert row['policy']['timestamp_us'] == int(original['timestamps_us'][expected])
        assert not row['policy']['uses_gt_actor']
        assert row['observation_source'] == ('initial_real_rgb' if i == 0 else 'previous_generated_chunk')
        if i: assert expected < row['first_condition_frame']
    overlaps = [r['frame'] for r in rows if r['clearance'] and r['clearance']['overlap_area_m2'] > 0]
    lateral = max(abs(r['route_progress_lateral'][1]) for r in rows)
    limit = protocol['gt_feedback_gate']['max_route_lateral_m']
    result = {'status': 'passed' if not overlaps and lateral <= limit else 'failed_baseline',
              'frames': len(rows), 'decisions_checked': len(decisions), 'causal_feedback_indices_passed': True,
              'overlap_frames': overlaps, 'max_route_lateral_m': lateral,
              'minimum_reference_clearance_m': min(r['clearance']['distance_m'] for r in rows if r['clearance']),
              'progress_m': rows[-1]['route_progress_lateral'][0]-rows[0]['route_progress_lateral'][0],
              'rows': rows, 'boundary': 'virtual ego footprint against recorded non-reactive actor footprints; not real-world counterfactual pixels',
              'human_verdict': None}
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k != 'rows'}), flush=True)


if __name__ == '__main__': main()
