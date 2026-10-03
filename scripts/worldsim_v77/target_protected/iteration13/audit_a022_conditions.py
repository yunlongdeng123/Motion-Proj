"""CPU只读追踪A022已有条件，逐层统计；不重建或修改训练输入。"""
from common import *
import numpy as np
import cv2
from PIL import Image, ImageDraw
from geometry_factory import transform
from actor_state import Observation, cuboid_front_depth, project_world

cv2.setNumThreads(1)


def raster(points, ob):
    uv, z, good = project_world(points, ob)
    ids = np.flatnonzero(good)
    return uv, z, ids


def main():
    plan = read(O/'manifest.json')
    c = next(c for c in plan['cases'] if c['case_id'] == 'A022')
    x = images(c, 'rgb')
    holes = images(c, 'hole') > 0
    obs = [Observation(np.where(h[..., None], 127, im).astype('uint8'), h,
                       np.array(fr['camera_to_world']), np.array(fr['intrinsics_1024']), fr['timestamp'])
           for im, h, fr in zip(x, holes, c['frames'])]
    selected = sorted({0, 5, 9} | {
        min(range(len(obs)), key=lambda i: abs(obs[i].timestamp-s['timestamp'])) for s in c['scans']})
    retained = [[a for a in f['actors'] if a['instance_token'] != c['target_token']] for f in c['frames']]
    front = {i: cuboid_front_depth(obs[i], retained[i]) for i in selected}
    allfront = {i: cuboid_front_depth(obs[i], c['frames'][i]['actors']) for i in selected}
    background, source_rows = [], []
    for scan in c['scans']:
        assert scan['path'] and Path(scan['path']).is_file()
        raw = np.fromfile(scan['path'], np.float32).reshape(-1, 5)[:, :3]
        cal, ego = scan['calibrated_sensor'], scan['ego_pose']
        m = transform(ego['translation'], ego['rotation']) @ transform(cal['translation'], cal['rotation'])
        world = raw @ m[:3, :3].T + m[:3, 3]
        i = min(range(len(obs)), key=lambda i: abs(obs[i].timestamp-scan['timestamp']))
        age = abs(obs[i].timestamp-scan['timestamp'])
        assert age <= 100000 and c['frames'][i]['geometry_available']
        pts = world[np.linalg.norm(world-obs[i].camera_to_world[:3, 3], axis=1) < 40]
        uv, z, ids = raster(pts, obs[i])
        u, v = uv[ids].T
        hit_h = holes[i][v, u]
        blocked = cv2.dilate(np.isfinite(allfront[i]).astype('uint8'), np.ones((5, 5), 'uint8')) > 0
        hit_actor = blocked[v, u]
        kept = ids[(~hit_h) & (~hit_actor)]
        order = np.argsort(z[kept], kind='stable')
        kept = kept[order]
        flat = uv[kept, 1]*1024+uv[kept, 0]
        _, first = np.unique(flat, return_index=True)
        kept = kept[first]
        row = dict(filename=scan['filename'], source_frame=i, source_age_us=age,
                   raw_points=len(raw), within_40m=len(pts), in_camera=len(ids),
                   hole_reject_points=int(hit_h.sum()), actor_reject_points=int(hit_actor.sum()),
                   hole_only_reject_points=int((hit_h & ~hit_actor).sum()),
                   kept_unique_points=len(kept), destinations=[])
        for f in [0, 5, 9]:
            raw_uv, raw_z, raw_ids = raster(pts, obs[f])
            raw_h = holes[f][raw_uv[raw_ids, 1], raw_uv[raw_ids, 0]]
            kept_uv, kept_z, kept_ids = raster(pts[kept], obs[f])
            dest_block = cv2.dilate(np.isfinite(front[f]).astype('uint8'), np.ones((5, 5), 'uint8')) > 0
            dest_ids = kept_ids[~dest_block[kept_uv[kept_ids, 1], kept_uv[kept_ids, 0]]]
            nh = np.zeros_like(holes[f])
            nh[kept_uv[dest_ids, 1], kept_uv[dest_ids, 0]] = True
            row['destinations'].append(dict(frame=f, raw_projected_into_H=int(raw_h.sum()),
                                            kept_N_pixels_H=int((nh & holes[f]).sum())))
        source_rows.append(row)
        background.append((pts[kept], scan['timestamp']))
    dest = O/'a022_condition_audit'
    dest.mkdir(exist_ok=True)
    frames = []
    for f in range(len(obs)):
        state = dict(np.load(O/'conditions/A022'/f'{f:05}.npz'))
        h = holes[f]
        row = dict(frame=f, H=int(h.sum()), O_all=int(state['O'].sum()), N_all=int(state['N'].sum()),
                   O_H=int((state['O'] & h).sum()), N_H=int((state['N'] & h).sum()),
                   U_H=int((state['U'] & h).sum()), Q_H_max=float(state['Q'][h].max()),
                   actor_labels=len(c['frames'][f]['actors']), geometry_available=c['frames'][f]['geometry_available'])
        if f in [0, 5, 9]:
            # 按同一过滤计算N，确认显示/保存没有丢点。原图仅供审核，不进入条件。
            n = np.zeros_like(h)
            blocked = cv2.dilate(np.isfinite(front[f]).astype('uint8'), np.ones((5, 5), 'uint8')) > 0
            for pts, stamp in background:
                uv, z, ids = raster(pts, obs[f])
                ids = ids[~blocked[uv[ids, 1], uv[ids, 0]]]
                n[uv[ids, 1], uv[ids, 0]] = True
            n &= ~state['O']
            assert np.array_equal(n, state['N']), '过滤后N与既有tensor不一致'
            projections = []
            for a in c['frames'][f]['actors']:
                d = cuboid_front_depth(obs[f], [a])
                count = int(np.isfinite(d).sum())
                if count:
                    projections.append(dict(instance_token=a['instance_token'], category=a['category'],
                                            target=a['instance_token']==c['target_token'], envelope_pixels=count,
                                            envelope_pixels_H=int((np.isfinite(d)&h).sum())))
            row['projected_instances'] = projections
            rgb = x[f].copy()
            contours, _ = cv2.findContours(h.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(rgb, contours, -1, (255, 195, 45), 2)
            # 审核点放大，实际N仍1px；可辨认稀疏N，并不扩充条件。
            enlarged = cv2.dilate(state['N'].astype('uint8'), np.ones((3, 3), 'uint8')) > 0
            visible = np.full_like(rgb, 40)
            visible[enlarged] = [45, 130, 250]
            visible[state['O']] = [45, 210, 110]
            cv2.drawContours(visible, contours, -1, (255, 195, 45), 2)
            picture = Image.fromarray(np.concatenate([rgb, visible], axis=1))
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 0, 2048, 27), fill=(17, 24, 33))
            draw.text((8, 5), f'A022 f{f:02} | ORIGINAL RGB / yellow H | RIGHT: N display 3px ONLY; tensor remains 1px', fill='white')
            picture.save(dest/f'frame_{f:02}.jpg', quality=94)
        frames.append(row)
    motion = np.array([ob.camera_to_world[:3, 3] for ob in obs])
    rotations = [obs[0].camera_to_world[:3, :3].T @ ob.camera_to_world[:3, :3] for ob in obs]
    max_rotation = max(float(np.degrees(np.arccos(np.clip((np.trace(r)-1)/2, -1, 1)))) for r in rotations)
    reveal = [dict(frame=f, f00_H_pixels_outside_current_H=int((holes[0] & ~holes[f]).sum()),
                   fraction=float((holes[0] & ~holes[f]).sum()/holes[0].sum())) for f in range(len(obs))]
    # 同像素对照仅用于近乎静止相机的显露诊断，不把它当已验证3D warp或训练标签。
    panels = []
    for f in [0, 9]:
        panel = obs[f].rgb.copy()
        contours, _ = cv2.findContours(holes[0].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(panel, contours, -1, (255, 195, 45), 2)
        current, _ = cv2.findContours(holes[f].astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if f:
            cv2.drawContours(panel, current, -1, (200, 90, 230), 2)
        panels.append(panel)
    picture = Image.fromarray(np.concatenate(panels, axis=1))
    draw = ImageDraw.Draw(picture)
    draw.rectangle((0, 0, 2048, 27), fill=(17, 24, 33))
    draw.text((8, 5), 'LEFT: f00 masked input | RIGHT: f09 masked input | yellow = f00 H; purple = f09 H', fill='white')
    picture.save(dest/'temporal_reveal.jpg', quality=94)
    targets = [next(a for a in fr['actors'] if a['instance_token']==c['target_token']) for fr in c['frames']]
    target_pos = np.array([a['translation'] for a in targets])
    total_h = sum(r['H'] for r in frames)
    result = dict(task_id=plan['task_id'], run_id=O.name, case_id='A022', scene=c['scene'],
                  evidence_scope='read-only existing input/condition trace; original RGB is audit display only',
                  camera_window_seconds=(obs[-1].timestamp-obs[0].timestamp)/1e6,
                  camera_path_m=float(np.linalg.norm(np.diff(motion, axis=0), axis=1).sum()),
                  camera_max_rotation_degrees=max_rotation,
                  same_pixel_reveal_diagnostic=reveal,
                  reveal_scope='mask overlap in almost stationary view; not certified world surface correspondences',
                  target_displacement_m=float(np.linalg.norm(target_pos[-1]-target_pos[0])),
                  missing_lidar=0, scans=len(source_rows), source_points=sum(r['kept_unique_points'] for r in source_rows),
                  cumulative_N_H_fraction=sum(r['N_H'] for r in frames)/total_h,
                  frames=frames, source_trace=source_rows,
                  existing_N_tensor_exact_match_frames=[0, 5, 9], inputs_or_tensors_modified=False,
                  new_GPU_jobs=0, human_verdict=None)
    dump(dest/'trace.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
