"""Input-only 3D box/camera conflict audit, without declaring free road space.

Uses real measured labels at the output world time. Camera-center nonintersection
is not ego-body collision clearance; absence of box collisions is not road GT.
"""
import os
for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from benchmark_geometry import ROOT, read
from prepare_motion_suite import apply, rz
from resource_control import require_start


def intersects(a, Ra, size_a, b, Rb, size_b):
    axes = [*Ra.T, *Rb.T]
    axes += [np.cross(u, v) for u in Ra.T for v in Rb.T]
    delta = b-a
    for axis in axes:
        if np.linalg.norm(axis) < 1e-10:
            continue
        radius = np.sum(np.abs(Ra.T@axis)*size_a/2)+np.sum(np.abs(Rb.T@axis)*size_b/2)
        if abs(delta@axis) > radius+1e-8:
            return False
    return True


class InputTracks:
    def __init__(self, context):
        self.context = context
        self.source = ROOT/'motion_scene_inputs_v1'/context
        self.frames = read(self.source/'sensor_preparation.json')['frames']
        self.times = np.arange(49)*.02-.48
        self.boxes = []
        self.interpolators = {}
        for row in self.frames:
            pose = np.loadtxt(self.source/'pose'/(row['frame_id']+'.txt'))
            labels = read(self.source/'annotations'/(row['frame_id']+'.json'))['laser_labels']
            boxes = {}
            for label in labels:
                if label.get('type') not in [1, 'TYPE_VEHICLE']:
                    continue
                b = label['box']
                center = apply(pose, np.array([[b['center_x'], b['center_y'], b['center_z']]]))[0]
                boxes[label['id']] = (center, pose[:3, :3]@rz(float(b['heading'])), np.array([b['length'], b['width'], b['height']]))
            self.boxes.append(boxes)

    def at(self, track, query):
        assert self.times[0] <= query <= self.times[-1]
        if track not in self.interpolators:
            rows = [x[track] for x in self.boxes]
            self.interpolators[track] = (np.array([x[0] for x in rows]), Slerp(self.times, Rotation.from_matrix([x[1] for x in rows])), np.array([x[2] for x in rows]))
        centers, angles, sizes = self.interpolators[track]
        return (np.array([np.interp(query, self.times, centers[:, j]) for j in range(3)]),
                angles(query).as_matrix(), np.array([np.interp(query, self.times, sizes[:, j]) for j in range(3)]))


def main():
    require_start()
    # Analytic 3D checks include overlap, strict separation, touching and tilt.
    I = np.eye(3)
    assert intersects(np.zeros(3), I, np.ones(3)*2, np.array([1., 0, 0]), I, np.ones(3)*2)
    assert intersects(np.zeros(3), I, np.ones(3)*2, np.array([2., 0, 0]), I, np.ones(3)*2)
    assert not intersects(np.zeros(3), I, np.ones(3)*2, np.array([2.01, 0, 0]), I, np.ones(3)*2)
    assert not intersects(np.zeros(3), I, np.ones(3)*2, np.array([0., 0, 5]), Rotation.from_euler('xyz', [.2, .1, .3]).as_matrix(), np.ones(3)*2)
    cache = {}
    def tracks(context):
        if context not in cache:
            cache[context] = InputTracks(context)
        return cache[context]
    cases, aggregate = [], {}
    for folder in sorted((ROOT/'benchmark_cases_v1/scenes').iterdir()):
        require_start()
        context = folder.name
        own = tracks(context)
        for p in sorted((folder/'requests').glob('*.json')):
            request = read(p)
            family = request['family']
            summary = dict(case_id=request['case_id'], context=context, family=family, split=request['split'],
                           input_screen_eligible=request['input_screen_eligible'], source_request=str(p),
                           rows=[], changed_input_eligibility=False)
            if not request['input_screen_eligible']:
                summary['audit_status'] = 'input_review_not_reclassified'
                cases.append(summary)
                continue
            if family == 'vehicle_removal':
                summary['audit_status'] = 'collision_not_required_for_deletion'
                cases.append(summary)
                continue
            source = None
            if family == 'vehicle_insertion':
                source = tracks(request['source_asset_context'])
                source_track = read(source.source/'native_preparation/selection_reviewed.json')['track']
                assert source.context != own.context
            for row in request['frames']:
                current = own.boxes[row['camera_index']]
                query = row['query_model_time']
                collisions = []
                if family.startswith('ego_'):
                    camera = np.array(row['camera_c2w_world'])[:3, 3]
                    for track, (center, axes, size) in current.items():
                        if np.all(np.abs((camera-center)@axes) <= size/2):
                            collisions.append(track)
                    kind = 'camera_center_inside_annotated_vehicle_only_not_ego_body_clearance'
                else:
                    _, axes, size = own.at(request['target_track'], query)
                    center = np.array(row['target_requested_center_world'])
                    axes = np.array(row['target_rotation_delta_world'])@axes
                    if source is not None:
                        _, _, source_size = source.at(source_track, query)
                        center += axes[:, 2]*(source_size[2]-size[2])/2
                        size = source_size
                    for track, (other_center, other_axes, other_size) in current.items():
                        if track == request['target_track'] and family != 'vehicle_insertion':
                            continue
                        if intersects(center, axes, size, other_center, other_axes, other_size):
                            collisions.append(track)
                    kind = 'full_3D_measured_vehicle_boxes; fixed_world_other_actor_times'
                summary['rows'].append(dict(raw=row['raw'], conflict_tracks=collisions, audit_kind=kind))
            summary['audit_status'] = 'input_conflicts_measured_free_space_not_certified'
            summary['conflict_frames'] = sum(bool(x['conflict_tracks']) for x in summary['rows'])
            d = aggregate.setdefault(family, dict(cases=0, cases_with_conflict=0, audited_frames=0, conflict_frames=0))
            d['cases'] += 1
            d['cases_with_conflict'] += bool(summary['conflict_frames'])
            d['audited_frames'] += len(summary['rows'])
            d['conflict_frames'] += summary['conflict_frames']
            cases.append(summary)
    result = dict(status='input_only_3D_request_conflict_audit_complete', cases=cases, by_family=aggregate,
                  contexts=len(cache), consumes_edited_images=False, changes_input_screen_eligibility=False,
                  uses_actual_source_asset_dimensions=True, ego_speed_semantics='camera_path_only; world_time_fixed',
                  camera_center_test_not_vehicle_footprint=True, drivable_space_or_ground_support_certified=False,
                  hidden_road_ground_truth=False, boxes_used_to_filter_training_points=False,
                  case_count_increase=0, quality_accepted=False, human_verdict=None)
    output = ROOT/'benchmark_request_space_audit_v1.json'
    assert not output.exists()
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'cases'}))


if __name__ == '__main__':
    main()
