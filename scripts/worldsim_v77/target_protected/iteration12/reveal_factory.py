"""有界车道轨迹规划；真实Y/SAM只做数据质检，不能作为模型条件。"""
from pathlib import Path
import os, sys, math, time, argparse
from collections import Counter, defaultdict
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
S = Path('/root/autodl-tmp/motion_proj_v77/scripts/worldsim_v77/target_protected')
sys.path[:0] = [str(S), str(S/'iteration8'), str(S/'iteration9'), str(S/'iteration11')]
import numpy as np
import cv2
from pyquaternion import Quaternion
from nuscenes.map_expansion.map_api import NuScenesMap
import long_factory as legacy
from geometry_factory import read, dump, footprint, projection, ground_orientation, wrap
from temporal_metrics import process
from iteration5.planning import normalized_masks
sys.path.insert(0, str(S.parent/'delete_audit'))
from mask_contract import prepare_masks

T = Path('/root/autodl-tmp/runs/worldsim_v77/WS-V77-TARGET-PROTECTED-20260929')
O = T/'r23'
ROOT = O/'factory'
POLICY = {
    'id': 'lane_reveal_v1', 'max_midpoint_positions': 24, 'separation_m': 3.,
    'speeds_mps': [0., 3., 6.], 'lane_sample_m': 1., 'map_radius_m': 40.,
    'max_candidates_per_family_per_source': 2,
    'visible_box_fraction_min': .60, 'GT_clearance_m': .3, 'support_distance_m': 2.5,
    'revealed_surface_support_min': .50, 'minimum_occlusion_range': .15,
    'model_mask': 'sam_full_v2 rectangle around silhouette; write silhouette separate',
    'split_shape': {'train': 'sedan', 'validation': 'suv'},
    'quality_only': True, 'condition_builder_can_read_Y': False,
    'upstream': 'https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py',
}


def register():
    p = O/'geometry_policy.json'
    if p.exists():
        assert read(p) == POLICY, '既有run规则不一致，不能静默变更'
    else:
        dump(p, POLICY)
    plan = '''# r23有界道路轨迹工厂（lane_reveal_v1）

先固定规则再跑。使用nuScenes官方车道/连接车道中心线，间距1m离散；在相机40m内选择最多24个相隔3m的中点：12个优先与真实保护车有投影交叠、12个在有效视野内均匀覆盖。每中点仅试0/3/6m/s，沿同一车道连续行驶，不外推缺失车道。最多72条/来源，不反复追加速度或位置网格。

保留地图支持、LiDAR贴地、GT包络至少0.3m间距、ego底部512像素保护、位置/尺度/yaw连续。允许左右截边，但投影框至少60%可见；上边及ego边不放宽。逐帧遮B可超过85%，改为检查整个30帧窗口：实际遮挡峰值>=30%、>=3帧有遮挡、遮挡比例变化>=15%或扫过跨度>=20%；同身份框面对应代理在其他帧的平均可见支持>=50%。该代理只作Y/SAM质检，不证明条件中有足够像素；后续必须在遮后RGB重跑SAM和合法状态构建。

同一个synthetic actor的真实网格轮廓是影响区；model H统一使用新真实DELETE的sam_full_v2，避免旧合成小洞与真实矩形洞混淆。H覆盖到的额外B也必须在质量统计里计入，不能只统计轮廓。真实Y永不改动。几何规划、完整视频SAM、GT用于离线质量检查，不流入条件API。

背景样本分密集交通（至少2个真实车辆在画面内）及普通背景，H不能覆盖未纳入检查的动态对象。每来源/家族最多2条保留候选；最终按预定30/13/7配比取50条、40train/10DEV-val。数量不足如实记录，不改变空间/显露门槛凑数。所有候选先隔离，人工0/1/2留空。

这是来源/合成工厂，不是模型效果实验。mask来源train=sedan、validation=suv，有类别差异且只两种旧DEV形状；不是最终泛化集。技术合同通过后仍需独立抽帧QA。
'''
    p = O/'geometry_plan.md'
    if not p.exists(): p.write_text(plan)


def actor_on_lane(path, distances, s, size, plane):
    if s < 0 or s > distances[-1]: return None
    x, y, yaw = [float(np.interp(s, distances, path[:, k])) for k in range(3)]
    R = ground_orientation(yaw, plane)
    center = np.array([x, y, np.dot([x, y, 1], plane)]) + R[:, 2]*size[2]/2
    return {'translation': center.tolist(), 'rotation': Quaternion(matrix=R).elements.tolist(), 'size': size}


def valid_projection(pr):
    if pr is None: return False
    bb = pr['box']; wh = bb[2:]-bb[:2]
    visible_wh = np.minimum(bb[2:], [1024, 576])-np.maximum(bb[:2], [0, 0])
    fraction = np.prod(np.maximum(visible_wh, 0))/max(1., np.prod(wh))
    return (bb[1] >= 8 and bb[3]+32 < 512 and fraction >= .60
            and wh[0] >= 72 and wh[1] >= 40 and .006 <= np.prod(wh)/(1024*576) <= .18)


def centers(g, map_api, c, size):
    f = c['frames'][15]; cam = np.array(f['camera_to_world'])[:3, 3]
    nearby = map_api.get_records_in_radius(*cam[:2], 40., ['lane', 'lane_connector'])
    paths = map_api.discretize_lanes(sorted(nearby['lane']+nearby['lane_connector']), 1.)
    options = []
    for token, raw in paths.items():
        p = np.asarray(raw)
        if len(p) < 2: continue
        p[:, 2] = np.unwrap(p[:, 2])
        ss = np.r_[0., np.cumsum(np.linalg.norm(np.diff(p[:, :2], axis=0), axis=1))]
        good = np.r_[True, np.diff(ss) > 1e-6]; p, ss = p[good], ss[good]
        for i in range(0, len(p), 3):
            a = actor_on_lane(p, ss, ss[i], size, g.ground[c['source_id']]['plane'])
            pr = projection(a, f)
            if not valid_projection(pr): continue
            foot = footprint(a)
            if not g.road.covers(foot): continue
            if any(foot.distance(o['_foot']) < .3 for o in g.obstacles[c['source_id']][15]): continue
            ab = pr['box']; score = 0.
            for b in f['actors']:
                bb = np.array(b['projection']['box_xyxy'])
                overlap = np.maximum(0., np.minimum(ab[2:], bb[2:])-np.maximum(ab[:2], bb[:2]))
                score += float(np.prod(overlap)/max(1., np.prod(bb[2:]-bb[:2])))
            options.append({'token': token, 's': float(ss[i]), 'xy': p[i, :2], 'path': p, 'distances': ss,
                            'overlap': score, 'image_x': float(np.mean(ab[[0, 2]]))})
    selected = []
    def add(q):
        if all(np.linalg.norm(q['xy']-r['xy']) >= 3. for r in selected): selected.append(q)
    for q in sorted(options, key=lambda v: (-v['overlap'], v['token'], v['s'])):
        if len(selected) >= 12: break
        add(q)
    # 第二半沿水平画面轮流取点，不用生成模型结果挑位置。
    remaining = sorted(options, key=lambda v: (v['image_x'], v['token'], v['s']))
    for stride in [max(1, len(remaining)//24), 1]:
        for q in remaining[::stride]:
            if len(selected) >= 24: break
            add(q)
    return selected


def trajectory(g, c, q, speed, asset):
    size = [1.85, 4.5, 1.5] if asset == 'sedan' else [1.90, 4.6, 1.7]
    sid = c['source_id']; plane = g.ground[sid]['plane']; rows = []; prev = None
    min_gap = 1e9; max_support = 0.
    for f, obs in zip(c['frames'], g.obstacles[sid]):
        dt = (f['timestamp']-c['frames'][15]['timestamp'])/1e6
        a = actor_on_lane(q['path'], q['distances'], q['s']+speed*dt, size, plane)
        if a is None: return None, 'lane_window_exhausted'
        foot = footprint(a)
        if not g.road.covers(foot): return None, 'outside_mapped_area'
        support = float(g.ground[sid]['_tree'].query(np.array(foot.exterior.coords)[:4])[0].max())
        max_support = max(max_support, support)
        if support > 2.5: return None, 'ground_support_gap'
        pr = projection(a, f)
        if not valid_projection(pr): return None, 'size_border_or_ego'
        for ob in obs:
            gap = foot.distance(ob['_foot']); min_gap = min(min_gap, gap)
            if gap < .3: return None, 'collision_clearance'
        R = Quaternion(a['rotation']).rotation_matrix; yaw = math.atan2(R[1, 0], R[0, 0])
        wh = pr['box'][2:]-pr['box'][:2]; center = np.array(a['translation'])
        if prev:
            factor = 100000/(f['timestamp']-prev['timestamp']); ratio = (wh/prev['wh'])**factor
            if (np.linalg.norm(center-prev['center'])*factor > 2 or
                    abs(wrap(math.degrees(yaw-prev['yaw'])))*factor > 5 or ratio.min() < .85 or ratio.max() > 1.18):
                return None, 'trajectory_continuity'
        rows.append({'frame': f['frame'], 'timestamp': f['timestamp'], 'actor': a, 'box': pr['box'].tolist(),
                     'depth_interval': [pr['near_depth'], pr['far_depth']]})
        prev = {'center': center, 'wh': wh, 'yaw': yaw, 'timestamp': f['timestamp']}
    return {'source_id': sid, 'scene': c['scene'], 'source_split': c['source_split'], 'asset': asset, 'frames': rows,
            'lane_token': q['token'], 'lane_midpoint_distance_m': q['s'], 'speed_mps': speed,
            'trajectory_policy': 'world_static' if speed == 0 else 'constant_speed_along_mapped_lane',
            'min_GT_clearance_m': min_gap, 'max_ground_support_distance_m': max_support}, None


def exact(g, c, tr, alphas, pm, reveal_policy='all_affected'):
    if reveal_policy not in {'all_affected','primary_plus_preserved'}:
        raise ValueError('未知显露任务准入规则')
    holes = [prepare_masks(a)[0]['model_mask'] for a in alphas]
    ratios = {t: [] for t in pm}; affected = set(); static = set()
    for i, (a, h, p, obs) in enumerate(zip(alphas, holes, tr['frames'], g.obstacles[c['source_id']])):
        yy, xx = np.where(a)
        if len(xx) < 200 or np.ptp(xx)+1 < 72 or np.ptp(yy)+1 < 40: return None, 'actual_size'
        if h[512:].any(): return None, 'ego_band'
        for tok, masks in pm.items(): ratios[tok].append(float((h&masks[i]).sum()/max(1, masks[i].sum())))
        for ob in obs:
            pr = ob['_projection']; tok = ob['instance_token']
            if pr is None: continue
            bb = pr['box']; x0, y0 = np.maximum(np.floor(bb[:2]).astype(int), 0)
            x1, y1 = np.minimum(np.ceil(bb[2:]).astype(int), [1024, 576])
            if x1 <= x0 or y1 <= y0 or h[y0:y1, x0:x1].sum() <= max(12, .01*h.sum()): continue
            if tok in pm:
                if ratios[tok][-1] > .01:
                    affected.add(tok)
                    if p['depth_interval'][1]+.3 >= pr['near_depth']: return None, 'protected_depth_order'
            elif ob['category'] in {'movable_object.barrier', 'movable_object.trafficcone', 'static_object.bicycle_rack'}:
                if ob.get('interpolation_uncertain') or p['depth_interval'][1]+.3 >= pr['near_depth']: return None, 'static_foreground'
                static.add(tok)
            else: return None, 'unreviewed_dynamic_envelope'
    stats = normalized_masks(alphas, [p['box'] for p in tr['frames']])
    if any(s['normalized_iou'] < .8 or not .85 <= s['normalized_area_ratio'] <= 1.15 for s in stats):
        return None, 'mask_continuity'
    active = sorted(affected)
    bactors = {t: [next(a for a in f['actors'] if a['instance_token'] == t) for f in c['frames']] for t in active}
    proc = process(c['frames'], [p['actor'] for p in tr['frames']], holes, {t: pm[t] for t in active}, bactors)
    roles={}
    if active:
        if reveal_policy=='all_affected':
            # 保留旧实验的精确行为，以便复现原拒绝及有界对照。
            for t in active:
                v = np.array(ratios[t]); d = proc['protected'][t]
                if max(v) < .30 or sum(v > .05) < 3: return None, 'insufficient_actual_occlusion'
                if not (d['visibility_transition'] or d['sweep_over_B']): return None, 'no_reveal_process'
                if (d['approx_other_frame_support_mean'] or 0) < .5: return None, 'insufficient_other_frame_evidence'
        else:
            from reveal_roles import primary_and_preserved
            roles,why=primary_and_preserved({t:ratios[t] for t in active},proc)
            if roles is None:return None,why
        family = 'protected_reveal'; kind = 'single_actor' if len(active) == 1 else 'dense_actors'
    else:
        family = 'dense_known_background' if len(c['actors']) >= 2 else 'ordinary_background'; kind = 'background'
    return {'type': kind, 'data_family': family, 'protected_instances': active,
            'model_H_occlusion_fractions': ratios, 'temporal_process': proc, 'silhouette_stats': stats,
            'static_background_annotations_behind_A': sorted(static), **roles}, None


def main(shard, count):
    cv2.setNumThreads(1); register()
    assert read(O/'segmentation_state.json')['stage'] == 'complete_quarantined_pending_synthetic_QA'
    legacy.O = O; legacy.ROOT = ROOT
    g = legacy.geometry(); map_api = NuScenesMap(dataroot=str(ROOT), map_name='boston-seaport')
    assets = {n: dict(np.load(T/'r8/assets'/f'{n}.npz')) for n in ['sedan', 'suv']}
    out = O/'lane_candidates'; out.mkdir(exist_ok=True)
    state = {'pid': os.getpid(), 'shard': shard, 'stage': 'running', 'completed': [], 'training_admission': 0}
    for sid in sorted(g.sources)[shard::count]:
        dest = out/(sid+'.json')
        if dest.exists(): state['completed'].append(sid); continue
        start = time.time(); c = g.sources[sid]; ground = g.prepare(sid); reject = Counter(); chosen = defaultdict(list)
        pm = legacy.protections(g, sid); attempts = 0; midpoint_count = 0
        if ground['pass'] and pm:
            legacy.support(g, sid)
            asset = POLICY['split_shape'][c['source_split']]
            size = [1.85, 4.5, 1.5] if asset == 'sedan' else [1.90, 4.6, 1.7]
            mids = centers(g, map_api, c, size); midpoint_count = len(mids)
            for q in mids:
                for speed in POLICY['speeds_mps']:
                    attempts += 1; tr, why = trajectory(g, c, q, speed, asset)
                    if tr is None: reject[why] += 1; continue
                    mesh = assets[asset]
                    alphas = [legacy.old.silhouette(mesh['vertices'], mesh['faces'], p['actor'], f) for p, f in zip(tr['frames'], c['frames'])]
                    quality, why = exact(g, c, tr, alphas, pm)
                    if quality is None: reject[why] += 1; continue
                    family = quality['data_family']
                    if len(chosen[family]) >= 2: reject['family_quota_full'] += 1; continue
                    chosen[family].append(tr | quality | {'human_verdict': None,
                        'ground': {k: v for k, v in g.ground[sid].items() if not k.startswith('_')}})
        else: reject['ground_or_stable_mask_unavailable'] += 1
        rows = [r for family in chosen.values() for r in family]
        dump(dest, {'source_id': sid, 'scene': c['scene'], 'source_split': c['source_split'], 'policy': POLICY['id'],
                    'midpoints': midpoint_count, 'attempts': attempts, 'candidates': rows,
                    'rejects': dict(reject), 'seconds': time.time()-start, 'ground_pass': ground['pass'],
                    'available_protected_tracks': len(pm), 'training_admission': 0})
        state['completed'].append(sid); dump(O/f'lane_state_{shard}.json', state)
        print('LANE_FACTORY', sid, len(rows), dict(reject), round(time.time()-start, 1), flush=True)
    state['stage'] = 'complete_pending_selection_and_QA'; dump(O/f'lane_state_{shard}.json', state)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--shard', type=int, default=0); p.add_argument('--count', type=int, default=1)
    a = p.parse_args(); main(a.shard, a.count)
