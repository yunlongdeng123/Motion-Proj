"""两个已验证制动任务的时间接口、过去观测和普通运动控制；不调用新模型。"""
from datetime import datetime, timezone
from pathlib import Path
import copy
import json
import os
import time
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation
import torch
from prepare_argoverse import POS, QUAT, CAMERA, poses, project_track, crop_image
from evaluate_localization import iou
from following_geometry import scene
from rgb_idm_policy import RGBIDMPolicy
from raster_ground import RasterGround
from run_approach_state_control import rollout

ROOT = Path('/root/autodl-tmp/runs/worldsim_v75')
OUT = ROOT/'WS-V75-TEMPORAL-STATE-AUDIT-01/20260920-r1'
CASES = [
    ('WS-V75-APPROACH-CLOSEDLOOP-01/20260920-association-r2',
     'WS-V75-APPROACH-STATE-CONTROL-01/20260920-r1'),
    ('WS-V75-APPROACH-CLOSEDLOOP-02/20260920-r1',
     'WS-V75-APPROACH-STATE-CONTROL-02/20260920-r1'),
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')


def velocity(times, xy):
    t = np.asarray(times, float)
    x = np.asarray(xy, float)
    return np.linalg.lstsq(np.c_[np.ones(len(t)), t-t.mean()], x, rcond=None)[0][1]


def reference_rows(raw, target_id, origin):
    ann = pd.read_feather(raw/'annotations.feather')
    ann = ann[ann.track_uuid == target_id].sort_values('timestamp_ns').drop_duplicates('timestamp_ns')
    ego = pd.read_feather(raw/'city_SE3_egovehicle.feather')
    timestamps = ann.timestamp_ns.to_numpy(np.int64)
    local = np.repeat(np.eye(4)[None], len(ann), axis=0)
    local[:, :3, :3] = Rotation.from_quat(ann[QUAT].to_numpy()).as_matrix()
    local[:, :3, 3] = ann[POS].to_numpy()
    world = poses(ego, timestamps) @ local
    world[:, :3, 3] -= origin
    packed = pd.DataFrame({'timestamp_ns': timestamps})
    packed[POS] = world[:, :3, 3]
    packed[QUAT] = Rotation.from_matrix(world[:, :3, :3]).as_quat()
    return packed, ego


def inspect_case(source_path, state_path):
    source = json.loads((source_path/'protocol.json').read_text())
    base = Path(source['base']); real = Path(source['source_run'])
    real_protocol = json.loads((real/'protocol.json').read_text())
    real_result = json.loads((real/'result.json').read_text())
    manifest = json.loads((base/'input_manifest.json').read_text())
    reference = scene(base); tr = np.load(base/'trajectory.npz')
    target = next(t for t in reference['tracks'] if t['id'] == source['target'] and 0 in t['frames'])
    cutoff = int(manifest['first_timestamp_ns']); raw = Path(manifest['raw_path'])
    out = OUT/source['source_log']; out.mkdir()
    packed, ego = reference_rows(raw, source['target'], np.array(manifest['city_origin']))
    past = packed[(packed.timestamp_ns <= cutoff) & (packed.timestamp_ns >= cutoff-1_000_000_000)]
    assert len(past) >= 2
    past_time = (past.timestamp_ns.to_numpy(np.int64)-cutoff)/1e9
    assert (past_time <= 0).all() and np.diff(past.timestamp_ns).max() <= 150_000_000
    past_velocity = velocity(past_time, past[POS].to_numpy())
    # 未来标注只用于评价；匀速控制的速度只由截止时刻之前的标注得到。
    first_center = np.array(target['centers'][target['frames'].index(0)])
    fidx = [target['frames'].index(f) for f in range(117)]
    future = np.array(target['centers'])[fidx]
    future_time = (tr['timestamps_ns'][:117]-cutoff)/1e9
    predicted_cv = first_center + future_time[:, None]*past_velocity
    max_displacement = float(np.linalg.norm(future[:, :2]-first_center[:2], axis=1).max())
    future_cv_error = np.linalg.norm(predicted_cv[:, :2]-future[:, :2], axis=1)
    # 复核旧四组只改位置，时间相关差异不能被包装成DVGT原生运动输出。
    old_contract = []
    for arm in ['gt_clean', 'dvgt_metric', 'dvgt_lidar_scaled', 'reference_lidar']:
        condition = json.loads((source_path/arm/'condition_scene.json').read_text())
        modified = next(t for t in condition['tracks'] if t['id'] == target['id'] and t['segment'] == target['segment'])
        delta = np.array(modified['centers'])-np.array(target['centers'])
        row = {'arm': arm, 'center_offset_m': delta[0].tolist(),
               'max_offset_variation_m': float(np.max(abs(delta-delta[0]))),
               'same_quaternions': modified['quaternions'] == target['quaternions'],
               'same_frames': modified['frames'] == target['frames'],
               'same_dimensions': modified['dimensions'] == target['dimensions'],
               'other_tracks_unchanged': all(t == next(x for x in condition['tracks'] if (x['id'],x['segment']) == (t['id'],t['segment']))
                                             for t in reference['tracks'] if t != target)}
        assert row['max_offset_variation_m'] < 1e-10 and all(row[k] for k in ['same_quaternions','same_frames','same_dimensions','other_tracks_unchanged'])
        old_contract.append(row)
    # 只审计已经保存的20过去帧+起点，不重跑检测；GT框只做评价对应，不反馈给模型。
    captures = real_result['warmup'] + [{'timestamp_ns': real_result['rows'][0]['actual_timestamp_ns'],
                                       'image': str(raw/'sensors/cameras'/CAMERA/f'{cutoff}.jpg'),
                                       'policy': real_result['rows'][0]['policy']}]
    capture_ns = np.array([r['timestamp_ns'] for r in captures], np.int64)
    assert len(captures) == 21 and (capture_ns <= cutoff).all()
    target_poses = poses(packed, capture_ns)
    target_capture = {'frames': list(range(21)), 'centers': target_poses[:, :3, 3].tolist(),
                      'quaternions': Rotation.from_matrix(target_poses[:, :3, :3]).as_quat().tolist(),
                      'dimensions': target['dimensions']}
    e = poses(ego, capture_ns); e[:, :3, 3] -= np.array(manifest['city_origin'])
    extrinsic = np.linalg.inv(tr['ego_world'][0]) @ tr['camera_world'][0]
    camera = e @ extrinsic
    matches = []; observations = []
    for i, capture in enumerate(captures):
        projection = project_track(target_capture, i, camera, tr['K'])
        candidates = capture['policy']['detections']
        overlaps = [iou(d['box'], projection['bounds']) for d in candidates] if projection else []
        best = int(np.argmax(overlaps)) if overlaps else None
        detected = candidates[best] if best is not None and overlaps[best] >= .3 else None
        row = {'timestamp_ns': int(capture_ns[i]), 'time_s': float((capture_ns[i]-cutoff)/1e9),
               'image': capture['image'], 'gt_projection': projection,
               'best_iou': overlaps[best] if best is not None else None,
               'matched_detection': detected, 'reference_center': target_capture['centers'][i]}
        observations.append(row)
        if detected:
            matches.append(row)
    fit = {'matched_count': len(matches), 'total_captures': 21,
           'warning': 'GT-assisted evaluation correspondence; contact midpoint is not a cuboid center; tracker velocity is not a reconstruction-model output'}
    if len(matches) >= 4:
        t = np.array([r['time_s'] for r in matches]); x = np.array([r['matched_detection']['world_center'] for r in matches])
        v = velocity(t, x); a = t <= -.5; b = t > -.5
        even = velocity(t[::2], x[::2]); odd = velocity(t[1::2], x[1::2])
        half_difference = (float(np.linalg.norm(velocity(t[a], x[a])-velocity(t[b], x[b])))
                           if a.sum() >= 2 and b.sum() >= 2 else None)
        fit.update(span_s=float(t[-1]-t[0]), newest_age_s=float(-t[-1]),
                   fitted_contact_velocity_mps=v.tolist(), fitted_speed_mps=float(np.linalg.norm(v)),
                   reference_xy_velocity_mps=past_velocity[:2].tolist(),
                   velocity_error_mps=float(np.linalg.norm(v-past_velocity[:2])),
                   half_velocity_difference_mps=half_difference,
                   interleaved_velocity_difference_mps=float(np.linalg.norm(even-odd)),
                   residual_rms_m=float(np.sqrt(np.mean((x-x.mean(0)-(t-t.mean())[:,None]*v)**2))),
                   tracker_ids=sorted(set(r['matched_detection']['track_id'] for r in matches)),
                   final_tracker_velocity_mps=matches[-1]['matched_detection']['world_velocity'])
    save(out/'past_observations.json', observations)
    save(out/'past_reference.json', {'role':'extra reference annotations, never model input',
         'timestamps_ns':past.timestamp_ns.tolist(),'times_s':past_time.tolist(),
         'centers_world':past[POS].to_numpy().tolist(),'fitted_velocity_mps':past_velocity.tolist(),
         'initial_center_role':'shared reference center at t0 for isolating motion, may interpolate annotation after t0; not causal reconstruction'})
    # 官方2Hz时间序列输入清单，仅检查存在与截止时间，不申请新GPU推理。
    temporal_manifest = []
    for offset in [-1., -.5, 0.]:
        query = cutoff + round(offset*1e9); views = []
        for folder in sorted((raw/'sensors/cameras').glob('ring_*')):
            candidates = [p for p in folder.glob('*.jpg') if int(p.stem) <= query]
            image = max(candidates, key=lambda p:int(p.stem)); timestamp = int(image.stem)
            assert timestamp <= cutoff and query-timestamp < 50_001_000
            views.append({'camera':folder.name,'image':str(image),'timestamp_ns':timestamp,'delay_to_requested_ms':(query-timestamp)/1e6})
        assert len(views) == 7
        temporal_manifest.append({'requested_offset_s':offset,'requested_timestamp_ns':query,'views':views})
    save(out/'available_temporal_rgb.json', temporal_manifest)
    policy = RGBIDMPolicy(tr['ego_world'][:, :3, 3], tr['K'], extrinsic, [0.,0.,0.], detector=False)
    terrain = RasterGround(base).mesh_for_route(tr['ego_world'][:, :2, 3])
    source_rows = json.loads((source_path/'gt_clean/decisions.json').read_text())
    controls = {}
    for arm in ['recorded_trajectory', 'target_static', 'target_past_cv']:
        condition = copy.deepcopy(reference)
        modified = next(t for t in condition['tracks'] if (t['id'],t['segment']) == (target['id'],target['segment']))
        if arm != 'recorded_trajectory':
            times = (tr['timestamps_ns'][modified['frames']]-cutoff)/1e9
            modified['centers'] = (np.repeat(first_center[None], len(times), 0) +
                                   times[:, None]*(past_velocity if arm == 'target_past_cv' else np.zeros(3))).tolist()
            modified['quaternions'] = [target['quaternions'][0]]*len(times)
        value, cameras = rollout(base, tr, policy, terrain, condition, reference, target, source_rows, False)
        save(out/f'{arm}.json', value)
        controls[arm] = {k:value[k] for k in ['progress_m','final_speed_mps','final_target_reference_clearance_m','overlap_frames']}
        controls[arm]['actions_mps2'] = [r['applied_acceleration_mps2'] for r in value['decisions']]
        if arm == 'recorded_trajectory':
            old = json.loads((state_path/'gt_clean.json').read_text())
            errors = [abs(a['ego_state'][key]-b['ego_state'][key]) for a,b in zip(value['frames'],old['frames'])
                      for key in ['x_m','y_m','z_m','speed_mps','yaw_rad']]
            err = float(max(errors)); assert err < 1e-10
            controls[arm]['old_117_frames_max_state_difference'] = err
    baseline = controls['recorded_trajectory']
    for value in controls.values():
        value['progress_change_vs_recorded_m'] = value['progress_m']-baseline['progress_m']
        value['mean_abs_action_change_vs_recorded_mps2'] = float(np.mean(abs(np.array(value['actions_mps2'])-baseline['actions_mps2'])))
    # 显示真实输入，而非凭拟合参数制作运动示意假图。
    sheet = Image.new('RGB', (1280, 3*396), '#132334'); draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
    for k, i in enumerate([0,10,20]):
        row = observations[i]; image = crop_image(Path(row['image']),manifest['crop_xyxy'])
        d = ImageDraw.Draw(image)
        if row['gt_projection']: d.rectangle(row['gt_projection']['bounds'],outline='yellow',width=3)
        if row['matched_detection']: d.rectangle(row['matched_detection']['box'],outline='#12d0ac',width=3)
        draw.text((10,k*396+5),f"{source['source_log'][:8]} | actual RGB t={row['time_s']:+.2f}s | yellow: reference projection; green: matched saved detection",font=font,fill='white')
        sheet.paste(image.resize((640,352)),(0,k*396+35))
        if row['gt_projection']:
            x0,y0,x1,y1=row['gt_projection']['bounds']; cx,cy=(x0+x1)/2,(y0+y1)/2
            sheet.paste(image.crop((int(cx-160),int(cy-88),int(cx+160),int(cy+88))).resize((640,352)),(640,k*396+35))
    sheet.save(out/'past-input-review.jpg',quality=93)
    result = {'log_id':source['source_log'],'target':target['id'],'base':str(base),
              'reference_past_annotations':len(past),'reference_past_velocity_mps':past_velocity.tolist(),
              'reference_past_speed_xy_mps':float(np.linalg.norm(past_velocity[:2])),
              'reference_future_max_displacement_xy_m':max_displacement,
              'past_cv_max_future_error_xy_m':float(future_cv_error.max()),
              'readout':fit,'controls':controls,'old_condition_contract':old_contract,
              'future_times_s':future_time.tolist(),'reference_future_centers':future.tolist(),
              'cutoff_ns':cutoff,'human_verdict':None}
    save(out/'result.json',result)
    print(json.dumps({k:result[k] for k in ['log_id','reference_past_speed_xy_mps','reference_future_max_displacement_xy_m','past_cv_max_future_error_xy_m','readout','controls']}),flush=True)
    return result


def main():
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
    assert not OUT.exists(), '不覆盖原审计'
    OUT.mkdir(parents=True)
    protocol = {'task_id':'WS-V75-TEMPORAL-STATE-AUDIT-01','run_id':OUT.name,
                'frozen_utc':datetime.now(timezone.utc).isoformat(),
                'cases':[{'generated_run':str(ROOT/a),'state_control_run':str(ROOT/b)} for a,b in CASES],
                'role':'post hoc observability and strong-control audit on both exposed braking development tasks',
                'past_input':'exact saved20 pre-start captures and current capture; no detector rerun',
                'reference':'AV2 annotations are evaluation/extra-information controls, not pure-visual model inputs',
                'ordinary_readout':'least-squares ground-contact world position slope; fixed halves and interleaved consistency checks, no tuning',
                'association':'maximum GT-projection IoU >=0.3 only for evaluation; report all missing observations',
                'controls':['recorded trajectory','target static at common GT initial pose','target CV from past1s reference annotations at common GT initial pose'],
                'control_information':'oracle t0 center/size/yaw and other actors future; past-only speed; not a runnable visual reconstruction ranking',
                'feedback':'same15 decisions117 frames/official dynamics; same first RGB action; direct state, no new RGB generation',
                'verification':'reproduce two prior GT direct-state runs at all117 frames before interpreting new control differences',
                'stop':'two existing tasks only, no thresholds/seeds/horizon/source replacement, no new reconstruction or generation calls; report feasibility, not scientific failure attribution',
                'human_verdict':None,'failure_ledger_refs':['V74-H2-F22'],'failure_ledger_delta':'none'}
    save(OUT/'protocol.json',protocol)
    start = time.monotonic(); result={'status':'running','human_verdict':None,'failure_ledger_delta':'none'}
    try:
        cases=[inspect_case(ROOT/a,ROOT/b) for a,b in CASES]
        assert not torch.cuda.is_initialized()
        result.update(status='complete',cases=cases,new_model_calls=0,cuda_initialized=False,
                      direct_state_frames=6*117,reproduced_prior_frames=2*117,new_control_frames=4*117)
    except BaseException as exc:
        result.update(status='failed_stopped',error_type=type(exc).__name__,error=str(exc));raise
    finally:
        result['wall_s']=time.monotonic()-start;save(OUT/'result.json',result)
        print(json.dumps({k:v for k,v in result.items() if k!='cases'}),flush=True)


if __name__ == '__main__':
    main()
