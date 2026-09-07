"""仅使用 build 观测，显式记录相机、时间、米制坐标与 Actor 归属。"""
from collections import defaultdict
import json
from pathlib import Path

import ijson
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp
import torch

from motion_proj.worldsim_v72.data.nuscenes_camera import NuScenesCameraIndex, _transform


def interpolate_pose(rows, time_us):
    times = np.array([r[0] for r in rows], dtype=np.float64)
    if time_us < times[0] or time_us > times[-1]:
        return None
    right = min(max(int(np.searchsorted(times, time_us)), 1), len(times) - 1)
    left = right - 1
    alpha = (time_us - times[left]) / max(times[right] - times[left], 1)
    pose = np.eye(4)
    pose[:3, 3] = (1-alpha)*rows[left][1][:3, 3] + alpha*rows[right][1][:3, 3]
    pose[:3, :3] = Slerp([0, 1], Rotation.from_matrix(np.stack([
        rows[left][1][:3, :3], rows[right][1][:3, :3]])))([alpha]).as_matrix()[0]
    return pose


def load_scene_inputs(config, progress):
    index = NuScenesCameraIndex(Path(config['dataset_root']))
    split = json.loads(Path(config['visual_split']).read_text())
    selected = [(scene, role) for role, key in [('fit', 'fit_scenes'), ('development', 'development_scenes')]
                for scene in split[key]]
    if config.get('fit_scene_limit'):
        selected = [(s, r) for s, r in selected if r != 'fit' or s in split['fit_scenes'][:config['fit_scene_limit']]]
    if config.get('development_scene_limit'):
        selected = [(s, r) for s, r in selected if r != 'development' or s in split['development_scenes'][:config['development_scene_limit']]]
    scene_by_name = {r['name']: r for r in index.scenes}
    samples_by_scene = defaultdict(list)
    for sample in index.samples:
        samples_by_scene[sample['scene_token']].append(sample)
    wanted_samples = {row['token'] for s, _ in selected for row in samples_by_scene[scene_by_name[s]['token']]}
    categories = {r['token']: r['name'] for r in json.loads((index.metadata_root/'category.json').read_text())}
    instances = {r['token']: categories[r['category_token']] for r in json.loads((index.metadata_root/'instance.json').read_text())}
    tracks = defaultdict(list)
    with (index.metadata_root/'sample_annotation.json').open('rb') as handle:
        for row in ijson.items(handle, 'item'):
            if row['sample_token'] not in wanted_samples:
                continue
            sample = index.sample_by_token[row['sample_token']]
            tracks[row['instance_token']].append((int(sample['timestamp']),
                _transform(row['translation'], row['rotation']), np.array(row['size'], dtype=float)[[1,0,2]], sample['scene_token']))
    for rows in tracks.values():
        rows.sort(key=lambda x: x[0])
    tracks_by_scene = defaultdict(dict)
    for track, rows in tracks.items():
        if len(rows) >= 2:
            tracks_by_scene[rows[0][3]][track] = rows
    del tracks
    scenes = []
    for scene, role in selected:
        scene_info = scene_by_name[scene]
        rows = sorted(samples_by_scene[scene_info['token']], key=lambda r: int(r['timestamp']))
        available = []
        for sample_index, sample in enumerate(rows):
            # 沿用每三帧留一帧的 build/held-out 含义，保留相邻位姿用于只读插值。
            if sample_index % 3 == 2 or sample_index == 0 or sample_index == len(rows)-1:
                continue
            sensor_rows = [index.sample_data.get(index.data_by_sample_channel.get((sample['token'], c), ''))
                           for c in [*config['camera_channels'], 'LIDAR_TOP']]
            if all(r is not None and (index.dataset_root/r['filename']).is_file() for r in sensor_rows):
                available.append((sample, sensor_rows))
            if len(available) == config['build_times_per_scene']:
                break
        record = {'scene_id': scene, 'log_id': scene_info['log_token'], 'role': role,
                  'requested_times': config['build_times_per_scene'], 'available_times': len(available),
                  'actor_count': sum(instances.get(t, '') in config['rigid_categories']
                                     for t in tracks_by_scene[scene_info['token']]), 'views': [],
                  'actor_count_scope':'all annotated rigid tracks in scene; not a window-visible denominator',
                  'point_ownership':'annotation box +0.1m, overlapping membership excluded; proxy, not instance segmentation'}
        build_stamps=[int(sensor_rows[-1]['timestamp']) for _,sensor_rows in available]
        record['window_actor_count']=sum(instances.get(track,'') in config['rigid_categories'] and
            any(interpolate_pose(trajectory,stamp) is not None for stamp in build_stamps)
            for track,trajectory in tracks_by_scene[scene_info['token']].items())
        lidar_supported_actors=set()
        if not available:
            scenes.append(record)
            progress({'phase':'data', 'scene':scene, 'views':0})
            continue
        for sample, sensor_rows in available:
            lidar_row = sensor_rows[-1]
            world_points = index.sensor_points_world(sample['token']).astype(np.float64)
            owners = np.full(len(world_points), '', dtype='U32')
            canonical = np.zeros_like(world_points)
            scene_tracks = tracks_by_scene[scene_info['token']]
            for track, trajectory in scene_tracks.items():
                pose = interpolate_pose(trajectory, int(lidar_row['timestamp']))
                if pose is None:
                    continue
                local = (world_points - pose[:3,3]) @ pose[:3,:3]
                closest = min(trajectory, key=lambda r: abs(r[0]-int(lidar_row['timestamp'])))
                inside = np.all(np.abs(local) <= closest[2]/2 + .10, axis=1)
                # 重叠框归属记为 ambiguous，不把同一点训练为两个 Actor。
                overlap = inside & (owners != '')
                assign = inside & (owners == '')
                owners[assign] = track
                canonical[assign] = local[assign]
                owners[overlap] = 'ambiguous'
            lidar_supported_actors.update(o for o in np.unique(owners)
                                          if instances.get(o,'') in config['rigid_categories'])
            for channel, camera in zip(config['camera_channels'], sensor_rows[:-1]):
                calibration = index.calibrated[camera['calibrated_sensor_token']]
                ego = index.ego_poses[camera['ego_pose_token']]
                world_from_camera = _transform(ego['translation'], ego['rotation']) @ _transform(calibration['translation'], calibration['rotation'])
                moved = world_points.copy()
                valid = owners != 'ambiguous'
                actor_mask = np.array([instances.get(o, '') in config['rigid_categories'] for o in owners])
                actor_poses = {}
                for track in np.unique(owners[owners != '']):
                    if track == 'ambiguous':
                        continue
                    take = owners == track
                    pose = interpolate_pose(scene_tracks[track], int(camera['timestamp']))
                    if pose is None:
                        valid[take] = False
                        continue
                    moved[take] = canonical[take] @ pose[:3,:3].T + pose[:3,3]
                    actor_poses[track] = pose
                    if not instances.get(track, '').startswith('vehicle.'):
                        valid[take] = False
                cam_points = (moved - world_from_camera[:3,3]) @ world_from_camera[:3,:3]
                height, width = config['image_hw']
                intrinsics = np.array(calibration['camera_intrinsic'], dtype=np.float64)
                intrinsics = np.diag([width/int(camera['width']), height/int(camera['height']), 1]) @ intrinsics
                uvz = cam_points @ intrinsics.T
                uv = uvz[:,:2]/np.maximum(uvz[:,2:], 1e-8)
                valid &= (cam_points[:,2] > .5) & (uv[:,0]>=0) & (uv[:,0]<width-1) & (uv[:,1]>=0) & (uv[:,1]<height-1)
                ids = np.flatnonzero(valid)
                pixel = np.rint(uv[ids]).astype(int)
                flat = pixel[:,1]*width + pixel[:,0]
                order = np.argsort(cam_points[ids,2], kind='stable')
                _, unique = np.unique(flat[order], return_index=True)
                ids = ids[order[unique]]
                # 按输入点序号留出1/5 build点作插值诊断，仍属于开发数据，不作为最终确认。
                diagnostic = ids % 5 == 0
                with Image.open(index.dataset_root/camera['filename']) as image:
                    rgb = np.asarray(image.convert('RGB').resize((width,height), Image.Resampling.LANCZOS), dtype=np.float32)/255
                record['views'].append({'image':torch.from_numpy(rgb).permute(2,0,1).contiguous(),
                    'uv':torch.tensor(uv[ids], dtype=torch.float32), 'z_m':torch.tensor(cam_points[ids,2], dtype=torch.float32),
                    'actor_mask':torch.tensor(actor_mask[ids]), 'diagnostic_mask':torch.tensor(diagnostic),
                    'owners':owners[ids].tolist(), 'points_actor_m':torch.tensor(canonical[ids], dtype=torch.float32),
                    'world_from_actor':actor_poses, 'world_from_camera':world_from_camera, 'intrinsics':intrinsics,
                    'sample_id':sample['token'], 'camera_id':channel, 'camera_time_us':int(camera['timestamp']),
                    'lidar_time_us':int(lidar_row['timestamp']), 'image_path':str(index.dataset_root/camera['filename'])})
        record['observed_actor_count'] = len({o for v in record['views'] for o, m in zip(v['owners'],v['actor_mask']) if m})
        record['lidar_supported_actor_count']=len(lidar_supported_actors)
        scenes.append(record)
        progress({'phase':'data', 'scene':scene, 'views':len(record['views']), 'actors':record['observed_actor_count']})
    return scenes


def sample_depth(depth, uv):
    h,w = depth.shape[-2:]
    grid = torch.stack((2*uv[:,0]/(w-1)-1, 2*uv[:,1]/(h-1)-1), dim=-1)[None,None]
    return torch.nn.functional.grid_sample(depth[None,None].float(), grid, align_corners=True)[0,0,0]
