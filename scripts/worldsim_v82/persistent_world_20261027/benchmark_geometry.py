"""Metric camera/asset transforms for the fixed eight-family diagnostic suite.

Labels condition object placement, not hidden road truth or training filters.
No aspect-ratio deformation and no clamping of unsupported time queries.
"""
from pathlib import Path
import json
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from prepare_motion_suite import apply, rz, footprint, overlap

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/persistent_world_20261027')


def read(path):
    return json.loads(Path(path).read_text())


def edit_clocks(row, family, ego_speed_clock='global_retime_control'):
    assert ego_speed_clock in ['fixed_world', 'global_retime_control']
    world = float(row['model_time'])
    actor = float(row['query_model_time']) if family == 'vehicle_speed' else world
    camera_path = float(row['query_model_time']) if family == 'ego_speed' else world
    if family == 'ego_speed' and ego_speed_clock == 'global_retime_control':
        world = actor = camera_path
    return dict(world_model_time=world, dynamic_query_model_time=actor,
                camera_path_query_model_time=camera_path)


def model_camera(T, world):
    T, world = np.asarray(T), np.asarray(world)
    scale = np.linalg.norm(T[:3, 0])
    result = T @ world
    result[:3, :3] /= scale
    assert np.allclose(result[:3, :3].T @ result[:3, :3], np.eye(3), atol=1e-8)
    return result


def sensor_camera(source, sensor_row, context):
    calibration = sensor_row.get('front_camera_calibration')
    if calibration:
        pose = np.loadtxt(source/'pose'/(sensor_row['frame_id']+'.txt'))
        ext = np.asarray(calibration['extrinsic']).reshape(4, 4)
        optical = np.eye(4)
        optical[:3, :3] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
        return pose @ ext @ np.linalg.inv(optical)
    rows = read(ROOT/'evaluation_admission_v2/sensors'/(context+'.json'))['frames']
    return np.asarray(next(x for x in rows if x['raw'] == sensor_row['raw'])['camera_world_from_optical'])


class ActorTrack:
    def __init__(self, context):
        self.context = context
        self.source = ROOT/'motion_scene_inputs_v1'/context
        selected = read(self.source/'native_preparation/selection_reviewed.json')
        assert selected['status'] == 'agent_visually_reviewed'
        self.track = selected['track']
        self.frames = read(self.source/'sensor_preparation.json')['frames']
        self.times = np.arange(49)*.02-.48
        self.seconds = (np.asarray([x['timestamp_micros'] for x in self.frames])-
                        self.frames[0]['timestamp_micros'])/1e6
        centers, axes, sizes = [], [], []
        for row in self.frames:
            pose = np.loadtxt(self.source/'pose'/(row['frame_id']+'.txt'))
            labels = read(self.source/'annotations'/(row['frame_id']+'.json'))['laser_labels']
            b = next(x['box'] for x in labels if x['id'] == self.track)
            centers.append(apply(pose, np.array([[b['center_x'], b['center_y'], b['center_z']]]))[0])
            axes.append(pose[:3, :3] @ rz(float(b['heading'])))
            sizes.append([b['length'], b['width'], b['height']])
        self.centers = np.asarray(centers)
        self.sizes = np.asarray(sizes)
        self.rotations = Slerp(self.times, Rotation.from_matrix(axes))

    def at(self, query):
        assert self.times[0] <= query <= self.times[-1], 'No unsupported asset extrapolation'
        center = np.array([np.interp(query, self.times, self.centers[:, k]) for k in range(3)])
        size = np.array([np.interp(query, self.times, self.sizes[:, k]) for k in range(3)])
        return center, self.rotations(query).as_matrix(), size


def asset_transfer(source_track, destination_track, row, T_source, T_destination):
    """Retarget one *different* source asset, preserving metric source size.

    Bottom registration uses the destination annotated vehicle box as a coarse
    placement reference. It is explicitly not certified road support.
    """
    assert source_track.context != destination_track.context
    query = row['query_model_time']
    source_center, source_axes, source_size = source_track.at(query)
    _, destination_axes, destination_size = destination_track.at(query)
    target = np.asarray(row['target_requested_center_world'], dtype=float).copy()
    target += destination_axes[:, 2]*(source_size[2]-destination_size[2])/2
    R = destination_axes @ source_axes.T
    W = np.eye(4)
    W[:3, :3] = R
    W[:3, 3] = target - R @ source_center
    A = T_destination @ W @ np.linalg.inv(T_source)
    ratio = np.linalg.norm(A[:3, 0])
    rotation = A[:3, :3]/ratio
    assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-8)
    assert np.isclose(np.linalg.det(rotation), 1, atol=1e-8)
    actual = apply(A, apply(T_source, source_center[None]))[0]
    expected = apply(T_destination, target[None])[0]
    assert np.max(np.abs(actual-expected)) < 1e-9
    sensor = destination_track.frames[row['camera_index']]
    pose = np.loadtxt(destination_track.source/'pose'/(sensor['frame_id']+'.txt'))
    labels = read(destination_track.source/'annotations'/(sensor['frame_id']+'.json'))['laser_labels']
    collisions = []
    polygon = footprint(target, source_size, destination_axes)
    for label in labels:
        if label.get('type') not in (1, 'TYPE_VEHICLE'):
            continue
        b = label['box']
        center = apply(pose, np.array([[b['center_x'], b['center_y'], b['center_z']]]))[0]
        if abs(center[2]-target[2]) > (source_size[2]+float(b['height']))/2:
            continue
        if overlap(polygon, footprint(center, [b['length'], b['width'], b['height']], pose[:3, :3]@rz(float(b['heading'])))):
            collisions.append(label['id'])
    audit = dict(source_context=source_track.context, source_track=source_track.track,
                 destination_track=destination_track.track, source_query_model_time=query,
                 source_query_physical_seconds=float(np.interp(query, source_track.times, source_track.seconds)),
                 target_center_world=target.tolist(), source_size_m=source_size.tolist(),
                 shape_aspect_ratio_preserved=True, model_unit_scale_ratio=ratio,
                 annotated_collision_tracks=collisions,
                 road_support_status='coarse_destination_box_bottom_only_not_measured_road_GT',
                 training_points_filtered_by_boxes=False)
    return A, rotation, ratio, audit
