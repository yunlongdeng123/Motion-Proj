"""仅诊断r23拒绝的输入依据；不改变原规划或准入阈值。"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
import reveal_factory as f
from geometry_factory import transform, footprint
from scipy.spatial import cKDTree
from shapely.geometry import Point
from pyquaternion import Quaternion
import numpy as np


def full_support(g, sid):
    c = g.sources[sid]; start, end = c['frames'][0]['timestamp'], c['frames'][-1]['timestamp']
    cams = np.array([r['camera_to_world'] for r in c['frames']])[:, :3, 3]
    plane = np.array(g.ground[sid]['plane']); clouds = []; records = []
    for ctx in g.context[sid]['frames']:
        d = ctx['sensors']['LIDAR_TOP']
        if not start <= d['timestamp'] <= end: continue
        path = f.ROOT/'rgb'/d['filename']
        if not path.is_file(): continue
        pts = np.fromfile(path, np.float32).reshape(-1, 5)[:, :3]
        cal, ego = d['calibrated_sensor'], d['ego_pose']
        m = transform(ego['translation'], ego['rotation'])@transform(cal['translation'], cal['rotation'])
        pts = pts@m[:3, :3].T+m[:3, 3]
        dist = cKDTree(cams[:, :2]).query(pts[:, :2])[0]
        height = pts[:, 2]-np.c_[pts[:, :2], np.ones(len(pts))]@plane
        pts = pts[(dist <= 40)&(abs(height) <= .08)]
        for a in ctx['annotations']:
            loc = (pts-np.array(a['translation']))@Quaternion(a['rotation']).rotation_matrix
            w, length, h = a['size']; inside = np.all(abs(loc) < np.array([length/2+.2, w/2+.2, h/2+.2]), 1)
            pts = pts[~inside]
        pts = pts[[g.road.covers(Point(p[:2])) for p in pts]]
        clouds.append(pts); records.append({'file': d['filename'], 'timestamp': d['timestamp'], 'ground_points': len(pts)})
    points = np.concatenate(clouds) if clouds else np.empty((0, 3))
    return points, records


def main():
    f.legacy.O = f.O; f.legacy.ROOT = f.ROOT; g = f.legacy.geometry()
    map_api = f.NuScenesMap(dataroot=str(f.ROOT), map_name='boston-seaport'); rows = []
    for sid in ['N054', 'N065', 'N152']:
        c = g.sources[sid]; g.prepare(sid); f.legacy.support(g, sid)
        points, scans = full_support(g, sid); tree = cKDTree(points[:, :2]); oldtree = g.ground[sid]['_tree']
        asset = f.POLICY['split_shape'][c['source_split']]; size = [1.85,4.5,1.5] if asset == 'sedan' else [1.9,4.6,1.7]
        samples = []
        for q in f.centers(g, map_api, c, size):
            for speed in f.POLICY['speeds_mps']:
                oldtraj, oldwhy = f.trajectory(g, c, q, speed, asset)
                if oldwhy != 'ground_support_gap': continue
                g.ground[sid]['_tree'] = tree
                tr, why = f.trajectory(g, c, q, speed, asset)
                g.ground[sid]['_tree'] = oldtree
                samples.append({'lane_token': q['token'], 's': q['s'], 'speed_mps': speed,
                                'old_reject': oldwhy, 'full_window_result': 'geometry_pass' if tr else why})
        rows.append({'source_id': sid, 'old_support_points': oldtree.n, 'full_observed_points': len(points),
                     'in_window_scans': scans, 'trajectories': samples,
                     'unchanged_thresholds': {'plane_band_m': .08, 'max_corner_distance_m': 2.5}})
    f.dump(f.O/'ground_support_audit.json', {'stage': 'diagnostic_only', 'sources': rows,
        'question': '原支持点是否被B周围16m裁剪，误把已观测道路当无支持？', 'training_admission': 0})
    print([(r['source_id'],len(r['trajectories']),sum(v['full_window_result']=='geometry_pass' for v in r['trajectories'])) for r in rows])


if __name__ == '__main__': main()
