"""前车跟随任务的路线坐标与参考查询；感知器不接收参考actors。"""
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from prepare_argoverse import CORNERS

BASES = [Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-VISIBLE-DEV-01/20260920-r1/cases')/log/'base'
         for log in ['24642607-2a51-384a-90a7-228067956d05', '29a00842-ead2-3050-b587-c5ef507e4125']]
OUT = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-FOLLOWING-BASELINE-01/20260920-r1')
FRAMES = [0, 15, 30, 45, 60]
VEHICLES = {'REGULAR_VEHICLE', 'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'TRUCK_CAB', 'SCHOOL_BUS'}


class Route:
    def __init__(self, xyz):
        xy = np.array(xyz)[:, :2]
        keep = np.r_[True, np.linalg.norm(np.diff(xy, axis=0), axis=1) > .01]
        self.xy = xy[keep]
        self.segments = np.diff(self.xy, axis=0)
        self.lengths = np.linalg.norm(self.segments, axis=1)
        assert len(self.lengths) and np.min(self.lengths) > 0
        self.arc = np.r_[0, np.cumsum(self.lengths)]

    def project(self, xy):
        xy = np.asarray(xy)[:2]
        alpha = np.clip(np.einsum('ij,ij->i', xy-self.xy[:-1], self.segments)/self.lengths**2, 0, 1)
        nearest = self.xy[:-1]+alpha[:, None]*self.segments
        idx = int(np.linalg.norm(xy-nearest, axis=1).argmin())
        tangent = self.segments[idx]/self.lengths[idx]
        delta = xy-nearest[idx]
        lateral = tangent[0]*delta[1]-tangent[1]*delta[0]
        progress = self.arc[idx]+alpha[idx]*self.lengths[idx]
        # 超出路线端点的对象不能被截断到端点，误当作40m内障碍物。
        longitudinal = float(delta @ tangent)
        if idx == 0 and alpha[idx] == 0 and longitudinal < 0:
            progress += longitudinal
        if idx == len(self.segments)-1 and alpha[idx] == 1 and longitudinal > 0:
            progress += longitudinal
        return float(progress), float(lateral)

    def at(self, progress):
        return np.array([np.interp(progress, self.arc, self.xy[:, i]) for i in range(2)])


def scene(base):
    return json.loads((Path(base)/'scene.json').read_text())


def reference_lead(data, trajectory, route, frame, ego_xy=None):
    ego = trajectory['ego_world'][frame, :2, 3] if ego_xy is None else np.asarray(ego_xy)
    ego_s, _ = route.project(ego)
    candidates = []
    for track in data['tracks']:
        if track['category'] not in VEHICLES or frame not in track['frames']:
            continue
        i = track['frames'].index(frame)
        rotation = Rotation.from_quat(track['quaternions'][i]).as_matrix()
        points = CORNERS*np.array(track['dimensions']) @ rotation.T + track['centers'][i]
        coords = np.array([route.project(p) for p in points])
        if coords[:, 1].min() > 1 or coords[:, 1].max() < -1:
            continue
        gap = float(coords[:, 0].min()-ego_s-2.4)
        if not (0 < gap <= 40):
            continue
        candidates.append({'id': track['id'], 'segment': track['segment'], 'gap_m': gap,
                           'center_world': track['centers'][i], 'route_lateral_center_m': route.project(track['centers'][i])[1]})
    return min(candidates, key=lambda x: x['gap_m']) if candidates else None


def ground_plane(data, origin):
    points = np.concatenate([x['xyz'] for x in data['lines']])
    points = points[np.linalg.norm(points[:, :2]-origin[:2], axis=1) < 40]
    points = np.unique(points, axis=0)
    assert len(points) >= 20
    A = np.c_[points[:, :2], np.ones(len(points))]
    keep = np.ones(len(points), bool)
    for _ in range(3):
        coef = np.linalg.lstsq(A[keep], points[keep, 2], rcond=None)[0]
        keep = abs(A@coef-points[:, 2]) <= .2
    residual = abs(A@coef-points[:, 2])
    return coef, {'source': 'known map road polyline elevations within 40m; extra metric map input',
                  'points': len(points), 'retained': int(keep.sum()),
                  'median_residual_m': float(np.median(residual)), 'p90_residual_m': float(np.percentile(residual, 90))}
