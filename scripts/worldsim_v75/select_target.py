"""只根据原始观测/标定筛选可见目标；不读取模型生成视频。"""
import argparse
import json
from itertools import product
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation
from interactive_drive.camera import FThetaCameraModel
from interactive_drive.config import RasterConfig
from interactive_drive.scene_loader import load_scene_bundle
from common import CAMERA, RUN, SCENE

P1 = Path('/root/autodl-tmp/runs/worldsim_v75/WS-V75-LOCALIZE-01/20260920-r1')
SAMPLE_FRAMES = [0, 5, 13, 21, 29, 36, 37, 53, 69, 85, 101, 117, 149, 181, 213, 236]

def box_projection(track, frame, trajectory, camera, offset=None):
    timestamp = int(trajectory['timestamps_us'][frame])
    # 选目标时只使用真实轨迹时间范围，不依赖官方允许的半秒外推。
    if timestamp < track.timestamps_us[0] or timestamp > track.timestamps_us[-1]:
        return None
    center, dims, quat = track.interpolate_at_timestamp(timestamp)
    if offset is not None:
        center = center + np.asarray(offset, dtype=np.float32)
    corners = np.asarray(list(product([-0.5, 0.5], repeat=3)), dtype=np.float32) * dims
    world = Rotation.from_quat(quat).apply(corners) + center
    uv, depth, valid = camera.project_world(world.astype(np.float32), trajectory['rig_poses_world'][frame])
    if not valid.all():
        return None
    bounds = [float(uv[:, 0].min()), float(uv[:, 1].min()), float(uv[:, 0].max()), float(uv[:, 1].max())]
    w, h = bounds[2] - bounds[0], bounds[3] - bounds[1]
    fully_visible = bounds[0] >= 4 and bounds[1] >= 4 and bounds[2] < 1276 and bounds[3] < 700
    return {'frame': int(frame), 'bounds': bounds, 'center_uv': uv.mean(axis=0).tolist(),
            'width': w, 'height': h, 'area': w * h, 'depth_m': float(depth.mean()),
            'fully_in_image': bool(fully_visible), 'world_center': center.tolist()}

def scene_and_camera():
    scene = load_scene_bundle(SCENE, CAMERA, 'default', None, RasterConfig())
    trajectory = np.load(RUN / 'trajectory.npz')
    camera = FThetaCameraModel(scene.selected_camera, output_width=1280, output_height=704)
    return scene, trajectory, camera

def main():
    P1.mkdir(parents=True, exist_ok=True)
    scene, trajectory, camera = scene_and_camera()
    rows = []
    for track in scene.vehicle_bbox_tracks:
        first = box_projection(track, 0, trajectory, camera)
        if first is None or not first['fully_in_image'] or first['depth_m'] < 3 or first['depth_m'] > 80:
            continue
        if first['width'] < 16 or first['height'] < 12:
            continue
        projections = [box_projection(track, f, trajectory, camera) for f in SAMPLE_FRAMES]
        first_period = [p for f, p in zip(SAMPLE_FRAMES, projections) if f <= 101]
        eligible = all(p is not None and p['fully_in_image'] for p in first_period)
        rows.append({'track_id': track.track_id, 'object_type': track.object_type,
                     'initial_area': first['area'], 'eligible_duration': eligible,
                     'projections': projections})
    rows.sort(key=lambda r: (-r['initial_area'], r['track_id']))
    # 用较近包围框的覆盖作为排除提示；最终仍检查真实初帧，不能把投影当成可见性真值。
    for row in rows:
        target = row['projections'][0]
        x0, y0, x1, y1 = target['bounds']
        overlaps = []
        for other in rows:
            if other['track_id'] == row['track_id']:
                continue
            p = other['projections'][0]
            if p['depth_m'] >= target['depth_m']:
                continue
            a, b, c, d = p['bounds']
            overlap = max(0, min(x1, c) - max(x0, a)) * max(0, min(y1, d) - max(y0, b))
            overlaps.append(overlap / target['area'])
        row['max_nearer_bbox_overlap_fraction'] = max(overlaps, default=0.0)
        row['eligible'] = row['eligible_duration'] and row['max_nearer_bbox_overlap_fraction'] < 0.1
    ranked = [r for r in rows if r['eligible']]
    payload = {'selection_inputs': ['original RGB', 'recorded actor tracks', 'calibration', 'ego trajectory'],
               'generated_video_used_for_selection': False,
               'rule': 'initial >=16x12 px; full box inside image at all prespecified samples 0..101; depth 3..80m; initial nearer-box overlap <10%; largest initial projected area, track ID breaks ties',
               'scene': str(SCENE), 'initial_candidates': len(rows), 'eligible_candidates': len(ranked),
               'ranked_ids': [r['track_id'] for r in ranked], 'candidates': rows, 'human_verdict': None}
    (P1 / 'target_candidates.json').write_text(json.dumps(payload, indent=2) + '\n')
    canvas = Image.fromarray(scene.initial_rgb)
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(ranked[:10]):
        b = row['projections'][0]['bounds']
        draw.rectangle(b, outline='yellow', width=2)
        draw.text((b[0], b[1]-14), f'{index+1}: {row["track_id"]}', fill='yellow')
    canvas.save(P1 / 'target_candidates.png')
    print(json.dumps({'initial_candidates':len(rows), 'eligible_candidates':len(ranked),
                      'ranked':[{k:r[k] for k in ['track_id','object_type','initial_area']} for r in ranked[:10]]}), flush=True)

if __name__ == '__main__':
    main()
