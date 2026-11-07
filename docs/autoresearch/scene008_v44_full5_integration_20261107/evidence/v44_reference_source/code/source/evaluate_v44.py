"""V44：冻结当前车体，成对验证 RCR-1 背景接入；不训练、不管理电源。"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

for _key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_key] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

CAM = Path('/root/autodl-tmp/drivingeditor_repro/campaign')
ART = CAM/'scene008_a6_insert_artifacts_20261105'
CONTROLS = CAM/'scene008_a6_asset_controls_20261105'
CANONICAL = ART/'surface_fit/combined_v1/observed_fit/formal/complete_a6.npz'
POSTERIOR = ART/'surface_fit/masked_depth_v1/posterior.npz'
FIXED = CAM/'scene008_a6_a37_dynamic_20261104_r1/inputs/a6_probe_reference.npz'
PILOT = (105, 113, 130, 137)
HELD = (89, 95, 101, 105, 113, 121, 125, 130, 137)
BLUE_ROIS = {113: (276, 175, 309, 220), 137: (276, 175, 309, 220)}
SILVER_ROIS = {130: (202, 192, 298, 244), 137: (210, 213, 290, 233)}
SO_SHA = '0eaff4d7a9b8ef6e711627f4c19824bf427c4387f7c6453fcb769a2c919c920e'
ASSET_SHA = '0b1f69e13d0e54328734325c8f118bc27dc7b75080869f2d52f55038afc0b20d'
POST_SHA = 'b98a0d932730fbf1462e555eb7d5eb931030b51fe68848990f073b9752879e06'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False)+'\n', encoding='utf-8')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mask(array):
    import numpy as np
    result = np.asarray(array).astype(bool).squeeze()
    if result.shape != (320, 480):
        raise ValueError(f'固定评价掩码尺寸错误：{result.shape}')
    return result


def summarize(error, domain):
    import numpy as np
    n = int(domain.sum())
    total = float(np.asarray(error, dtype=np.float64)[domain].sum())
    return dict(pixels=n, sum=total, mean=total/n if n else None)


def roi_mask(box):
    import numpy as np
    result = np.zeros((320, 480), dtype=bool)
    x0, y0, x1, y1 = box
    result[y0:y1, x0:x1] = True
    return result


def row_hash(rows):
    digest = hashlib.sha256()
    for key, value in sorted(rows.items()):
        digest.update(key.encode())
        if value is not None:
            value = value.detach().cpu().contiguous().numpy()
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
    return digest.hexdigest()


def worker(args):
    import numpy as np
    import torch
    from PIL import Image
    if not torch.cuda.is_available():
        raise RuntimeError('GPU不可用：不渲染、不自动排队，也不操作电源。')
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(.70)
    assert sha(CANONICAL) == ASSET_SHA and sha(POSTERIOR) == POST_SHA
    start = time.time()
    if args.worker == 'v44':
        from v44_runtime import V44Runtime
        rt = V44Runtime(args.contract)
        asset = rt.asset
    else:
        sys.path[:0] = [str(CONTROLS/'source'), str(ART/'source/scale')]
        from scene_runtime import Runtime
        from sh_asset import SHAsset
        from surface_lod_asset import SurfaceLODAsset
        rt = Runtime()
        asset = SurfaceLODAsset(SHAsset.from_geometry(CANONICAL).cuda(), POSTERIOR)
    from diff_gaussian_rasterization import _C
    assert sha(_C.__file__) == SO_SHA
    dest = args.output/args.worker
    dest.mkdir(exist_ok=False)
    raws = PILOT if args.mode == 'pilot' else tuple(range(89, 138))
    with np.load(FIXED, allow_pickle=False) as fixed:
        np.testing.assert_array_equal(fixed['raw_ids'], np.arange(89, 138))
        assert fixed['layer_names'].tolist()[3] == 'ground_beneath_a37_proxy'
        ground = fixed['depth_m'][:, 3] > 0
    records = []
    with torch.no_grad():
        for raw in raws:
            data = rt.raw(raw)
            obs = data['obs']
            camera = data['cam']
            assert (camera.image_width, camera.image_height) == (480, 320)
            gt = np.asarray(obs['rgb'], dtype=np.float32)
            if gt.max() > 1:
                gt = gt/255
            gt = gt.transpose(2, 0, 1)
            a6 = mask(obs['visible'][0])
            a37 = mask(obs['visible'][1])
            measured = mask(obs['depth_valid'][0])
            target_z = np.asarray(obs['depth_m'][0]).squeeze()
            road_proxy = np.asarray(obs['road_edit_rgb'], dtype=np.float32)
            if road_proxy.max() > 1:
                road_proxy = road_proxy/255
            road_proxy = road_proxy.transpose(2, 0, 1)
            frame = dest/f'{raw:03d}'
            frame.mkdir()
            camera_binding = dict(width=int(camera.image_width), height=int(camera.image_height),
                **{k: float(getattr(camera, k)) for k in ('fx', 'fy', 'cx', 'cy', 'FoVx', 'FoVy', 'timestamp')},
                world_view=camera.world_view_transform.detach().cpu().numpy().tolist(),
                full_projection=camera.full_proj_transform.detach().cpu().numpy().tolist(),
                object_pose=data['pose'].detach().cpu().numpy().tolist())
            record = dict(raw=raw, split='HELD9_internal_development' if raw in HELD else 'BUILD40',
                          fixed_domains_from_original_observations=True,
                          actual_camera_pose=camera_binding,
                          camera_pose_identity=hashlib.sha256(b''.join(
                              np.ascontiguousarray(obs[k]).tobytes() for k in
                              ('K', 'model_to_camera', 'model_time', 'model_from_local'))).hexdigest(),
                          states={})
            keep_asset_rows = None
            for state in ('keep', 'delete'):
                delete = state == 'delete'
                if args.worker == 'current':
                    scene = rt.scene(raw, asset, delete=delete)
                    rows = scene['rows']
                    owner = rows['features'][:, 2] > .5
                    ah = row_hash({k: None if v is None else v[owner] for k, v in rows.items()})
                    output = rt.render_rows(raw, scene)
                    meta = output['meta']
                else:
                    output = rt.render_scene(raw, delete=delete)
                    meta = output['meta']
                    ah = meta['native_asset_row_hash']
                assert meta['native_a6_rows_input'] == 0
                if delete:
                    assert meta['native_a37_packed_rows'] == 0
                    assert ah == keep_asset_rows, 'keep/delete资产行必须一致'
                else:
                    keep_asset_rows = ah
                rgb = output['rgb'].detach().cpu().numpy()
                alpha = output['alpha'].detach().cpu().numpy()[0]
                depth_sum = output['depth_sum'].detach().cpu().numpy()[0]
                weights = output['component_alpha_T'].detach().cpu().numpy()
                assert rgb.shape == (3, 320, 480) and weights.shape == (4, 320, 480)
                assert np.isfinite(rgb).all() and np.isfinite(depth_sum).all()
                conservation = float(abs(weights.sum(0)-alpha).max())
                assert conservation < 2e-5
                depth = depth_sum/np.maximum(alpha, 1e-8)/rt.scale
                error = abs(rgb-gt).mean(0)
                road = mask(obs['road_edit']) & mask(ground[raw-89]) if delete else mask(obs['road'])
                road_target = road_proxy if delete else gt
                protected = mask(obs['protect']) & ~a6 & ~a37
                item = dict(meta=meta, a6_RGB=summarize(error, a6),
                            a6_depth_m=summarize(abs(depth-target_z), measured),
                            road_RGB=summarize(abs(rgb-road_target).mean(0), road),
                            protected_RGB=summarize(error, protected),
                            a6_alpha=summarize(weights[0], a6),
                            a6_alpha_below09=int((weights[0][a6] < .9).sum()),
                            a6_alpha_below05=int((weights[0][a6] < .5).sum()),
                            group_alpha_conservation_maxabs=conservation,
                            keep_delete_asset_rows=ah)
                if not delete:
                    item['a37_RGB'] = summarize(error, a37)
                if raw in BLUE_ROIS:
                    domain = roi_mask(BLUE_ROIS[raw]) & mask(obs['road_edit']) & mask(ground[raw-89]) & ~a6 & ~a37
                    residual = rgb-road_proxy
                    blue = residual[2]-.5*(residual[0]+residual[1])
                    item['blue_fixed_ROI'] = dict(xyxy=BLUE_ROIS[raw],
                        absolute_excess=summarize(abs(blue), domain),
                        positive_excess_gt005=int((blue[domain] > .05).sum()),
                        semantics='原始观测道路候选域；非历史RCR分母；target是重观测proxy')
                if raw in SILVER_ROIS:
                    domain = roi_mask(SILVER_ROIS[raw])
                    item['silver_fixed_ROI'] = dict(xyxy=SILVER_ROIS[raw],
                        brightness=summarize(rgb.mean(0), domain),
                        group_alpha_mean=[summarize(w, domain)['mean'] for w in weights],
                        semantics='整个预登记ROI的描述统计；无删后隐藏真值，不按预测亮度挑点')
                png = frame/(state+'.png')
                Image.fromarray(np.uint8(np.clip(rgb.transpose(1, 2, 0), 0, 1)*255+.5)).save(png)
                item['PNG'] = dict(path=png.relative_to(args.output).as_posix(), sha256=sha(png))
                np.savez_compressed(frame/(state+'_float.npz'), rgb=rgb, alpha=alpha,
                                    depth_m=depth, group_alpha_T=weights)
                if args.worker == 'v44' and raw in PILOT and delete:
                    parts = output  # 同一次 forward 已携带真实联合排序下的 RGB/alpha 归因。
                    group = parts['group_rgb_T'].detach().cpu().numpy()
                    env = parts['env_rgb_T'].detach().cpu().numpy()
                    np.savez_compressed(frame/'delete_components.npz', group_rgb_T=group,
                                        group_alpha_T=parts['group_alpha_T'].detach().cpu().numpy(),
                                        env_rgb_T=env,
                                        constant_background_rgb_T=parts['constant_background_rgb_T'].detach().cpu().numpy())
                    item['layer_decomposition'] = dict(maxabs=parts['color_reconstruction_maxabs'],
                                                      groups=['asset', 'a37', 'road', 'native_static'])
                    if raw in SILVER_ROIS:
                        domain = roi_mask(SILVER_ROIS[raw])
                        item['silver_fixed_ROI']['actual_group_RGB_mean'] = [g[:, domain].mean(1).tolist() for g in group]
                        item['silver_fixed_ROI']['env_RGB_mean'] = env[:, domain].mean(1).tolist()
                record['states'][state] = item
                del output
            Image.fromarray(np.uint8(np.clip(gt.transpose(1, 2, 0), 0, 1)*255+.5)).save(frame/'observed_a37_present.png')
            assert asset.assert_persistent_unchanged()
            if args.worker == 'current':
                assert rt.modelunchanged()
            else:
                rt.assert_unchanged()
            records.append(record)
            write(frame/'metrics.json', record)
            write(dest/'progress.json', dict(completed=[r['raw'] for r in records]))
            print(json.dumps(dict(variant=args.worker, raw=raw, completed=len(records))), flush=True)
            rt._data.clear()
            torch.cuda.empty_cache()
    assert sha(CANONICAL) == ASSET_SHA and sha(POSTERIOR) == POST_SHA
    receipt = dict(status='RENDERED_NOT_HUMAN_ACCEPTED', method='v44', variant=args.worker,
                   mode=args.mode, raws=list(raws), frames=records, seconds=time.time()-start,
                   persistent_SF_hash=asset.persistent_SF_hash, asset_sha256=ASSET_SHA,
                   posterior_sha256=POST_SHA, kernel_sha256=SO_SHA,
                   neural_weight_updates=0, geometry_updates=0, colour_fit_updates=0,
                   native_or_road_training_updates=0, shutdown_performed=False,
                   independent_generalization_claim=False, quality_accepted=False)
    if args.worker == 'v44':
        receipt['runtime_receipt'] = rt.receipt()
    else:
        receipt['frozen_background_binding'] = rt.meta
    write(dest/'receipt.json', receipt)


def orchestrate(args):
    # 子进程隔离两套 native module import，禁止旧 gate 行号串入新模型。
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('需要带GPU模式开机后再运行；本入口不排队、不启动实例、不关机。')
    args.output.mkdir(parents=True, exist_ok=False)
    for variant in ('current', 'v44'):
        command = [sys.executable, str(Path(__file__).resolve()), '--contract', str(args.contract),
                   '--output', str(args.output), '--mode', args.mode, '--worker', variant]
        subprocess.run(command, check=True)
    current = read(args.output/'current/receipt.json')
    candidate = read(args.output/'v44/receipt.json')
    assert current['persistent_SF_hash'] == candidate['persistent_SF_hash']
    assert current['raws'] == candidate['raws']
    comparisons = []
    for before, after in zip(current['frames'], candidate['frames']):
        assert before['raw'] == after['raw']
        assert before['camera_pose_identity'] == after['camera_pose_identity']
        assert before['actual_camera_pose'] == after['actual_camera_pose'], '两条运行链相机/轨迹不一致'
        for state in ('keep', 'delete'):
            b, a = before['states'][state], after['states'][state]
            changes = {}
            for key in ('a6_RGB', 'a6_depth_m', 'road_RGB', 'a37_RGB', 'protected_RGB'):
                if key not in b:
                    continue
                assert b[key]['pixels'] == a[key]['pixels'], (before['raw'], state, key)
                changes[key] = dict(pixels=b[key]['pixels'], before=b[key]['mean'], after=a[key]['mean'],
                                    delta=None if b[key]['mean'] is None else a[key]['mean']-b[key]['mean'])
            import numpy as np
            with np.load(CAM/f'scene008_instances_20261105_r1/observations/frames/{before["raw"]:03d}.npz') as obs:
                # 不使用预测alpha决定评价域；车辆amodal框范围也保守排除。
                outside = mask(obs['protect']) & ~np.asarray(obs['visible_reviewed']).any(0)
                outside &= ~np.asarray(obs['box_candidate']).any(0)
            float_name = f'{before["raw"]:03d}/{state}_float.npz'
            with np.load(args.output/'current'/float_name) as cb, np.load(args.output/'v44'/float_name) as ca:
                delta = abs(ca['rgb']-cb['rgb'])
                outside_change = dict(mean_channel_absolute=summarize(delta.mean(0), outside),
                    pixels_max_channel_change_gt002=int((delta.max(0)[outside] > .02).sum()),
                    max_channel_absolute=float(delta[:, outside].max()) if outside.any() else None,
                    semantics='观测protect减两车reviewed与候选框；变化不是无隐藏真值区域的准确率')
            comparisons.append(dict(raw=before['raw'], state=state, changes=changes,
                                    protected_outside_change=outside_change))
    pooled = []
    # 像素加权汇总：使用浮点误差sum/count，而不是逐帧MAE的平均。
    for state in ('keep', 'delete'):
        for split in ('ALL', 'BUILD40', 'HELD9_internal_development'):
            selected = [(b, a) for b, a in zip(current['frames'], candidate['frames'])
                        if split == 'ALL' or b['split'] == split]
            if not selected:
                continue
            for key in ('a6_RGB', 'a6_depth_m', 'road_RGB', 'a37_RGB', 'protected_RGB'):
                if key not in selected[0][0]['states'][state]:
                    continue
                count = sum(b['states'][state][key]['pixels'] for b, _ in selected)
                base_sum = sum(b['states'][state][key]['sum'] for b, _ in selected)
                new_sum = sum(a['states'][state][key]['sum'] for _, a in selected)
                pooled.append(dict(state=state, split=split, metric=key, frames=len(selected),
                                   pixels=count, before=base_sum/count if count else None,
                                   after=new_sum/count if count else None,
                                   delta=(new_sum-base_sum)/count if count else None))
    write(args.output/'comparison.json', dict(status='PAIRED_RESULTS_REQUIRE_VISUAL_REVIEW',
        method='v44', mode=args.mode, variants=['current', 'v44'], comparisons=comparisons,
        pixel_weighted_pooled=pooled,
        frozen_current_asset=True, road_field_mode='frozen_transfer_from_RCR1_M2_not_recalibrated',
        silver_repair_claim=False, quality_accepted=False, shutdown_performed=False))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--contract', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--mode', choices=('pilot', 'full49'), default='pilot')
    ap.add_argument('--worker', choices=('current', 'v44'))
    args = ap.parse_args()
    (worker if args.worker else orchestrate)(args)


if __name__ == '__main__':
    main()
