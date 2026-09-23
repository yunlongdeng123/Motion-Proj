"""CPU-only geometry shortlist for the human case-review page."""
import json
from pathlib import Path
import sys

import numpy as np
from shapely.geometry import Polygon, box
from shapely.ops import unary_union

sys.path.insert(0, '/root/autodl-tmp/motion_proj')
from motion_proj.cfbench.geometry import project_box, shifted_pose

DATA = Path('/root/autodl-tmp/data')
V5 = DATA/'worldsim_v5/drivestudio_processed_10Hz/trainval'
OLD = DATA/'dynamic_editing_v2/drivestudio_processed_10Hz/trainval'
MAP = DATA/'worldsim_v75_downstream_bench/map_metadata_driverq11'

CASES = [('350', '0', 'singapore-queenstown', 60, V5),
         ('350', '0', 'singapore-queenstown', 70, V5),
         ('350', '0', 'singapore-queenstown', 80, V5),
         ('350', '0', 'singapore-queenstown', 90, V5),
         ('350', '9', 'singapore-queenstown', 0, V5),
         ('204', '49', 'boston-seaport', 40, OLD),
         ('756', '14', 'singapore-queenstown', 50, V5),
         ('756', '14', 'singapore-queenstown', 60, V5),
         ('756', '5', 'singapore-queenstown', 90, V5)]


def footprint(pose, size):
    length, width = map(float, size[:2])
    return Polygon([tuple((pose @ np.array([x, y, 0, 1]))[:2])
                    for x, y in [(-length/2, -width/2), (length/2, -width/2),
                                 (length/2, width/2), (-length/2, width/2)]])


def local_road(location, region):
    data = json.loads((MAP/f'{location}.json').read_text())
    nodes = {v['token']: (v['x'], v['y']) for v in data['node']}
    polys = {v['token']: v for v in data['polygon']}
    found = []
    for area in data['drivable_area']:
        for token in area['polygon_tokens']:
            p = polys[token]
            try:
                q = Polygon([nodes[n] for n in p['exterior_node_tokens']],
                            [[nodes[n] for n in h['node_tokens']] for h in p['holes']])
            except (KeyError, ValueError):
                continue
            if q.is_valid and q.intersects(region):
                found.append(q)
    return unary_union(found)


def evaluate(scene, key, location, start, base):
    root = base/scene
    info = json.loads((root/'instances/instances_info.json').read_text())
    ann = info[key]['frame_annotations']
    poses = dict(zip(ann['frame_idx'], map(np.asarray, ann['obj_to_world'])))
    sizes = dict(zip(ann['frame_idx'], ann['box_size']))
    frames = range(start, start+100, 5)
    if any(f not in poses for f in frames):
        return {'scene': scene, 'key': key, 'start': start, 'error': 'track_gap'}
    xy = np.array([poses[f][:2, 3] for f in frames])
    road = local_road(location, box(xy[:,0].min()-15, xy[:,1].min()-15,
                                    xy[:,0].max()+15, xy[:,1].max()+15))
    result = {'scene': scene, 'key': key, 'start': start}
    for sign in [-3.5, 3.5]:
        valid = visible = 0
        collisions = []
        for f in frames:
            pose = shifted_pose(poses[f], [0, sign, 0])
            area = footprint(pose, sizes[f])
            valid += road.covers(area)
            visible += project_box(root, f, 0, pose, sizes[f]) is not None
            for other_key, item in info.items():
                if other_key == key:
                    continue
                other = item['frame_annotations']
                if f not in other['frame_idx']:
                    continue
                j = other['frame_idx'].index(f)
                if area.intersects(footprint(np.asarray(other['obj_to_world'][j]), other['box_size'][j])):
                    collisions.append((f, other_key))
        result[str(sign)] = {'road_footprint_2hz': valid, 'visible_2hz': visible,
                             'collisions_2hz': collisions}
    return result


if __name__ == '__main__':
    for args in CASES:
        print(json.dumps(evaluate(*args)), flush=True)
