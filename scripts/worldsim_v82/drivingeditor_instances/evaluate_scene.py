"""真实49观测视角与固定相机49模型时间的只读评估；默认CPU预检。

--run 才取得共享GPU锁并恢复checkpoint。固定相机只copy.copy后更改timestamp，
没有反事实GT。逐ID可见RGB通过完整几何的feature alpha*T颜色通道导出；
孤立全部车辆raw RGB/alpha是另一种量，绝不能替代joint可见mask指标。
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

if __package__:
    from .runtime import validate_spec, load_models, masks_for_frame
    from .priority import require_gpu_slot, inspect_priority
else:
    from runtime import validate_spec, load_models, masks_for_frame
    from priority import require_gpu_slot, inspect_priority


VISIBLE_SEMANTICS = 'full_scene_transmittance_weighted_alpha'
RGB_SEMANTICS = 'full_scene_sum_alpha_times_transmittance_times_object_local_SH_RGB'


def _write_json(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def read_mask_labels(path, width, height, instance_count):
    """与runtime一致的nearest缩放；标签零不推断未知mask已就绪。"""
    import numpy as np
    from PIL import Image
    with Image.open(path) as image:
        labels = np.asarray(image.resize((width, height), Image.Resampling.NEAREST))
    if (labels.shape != (height, width) or labels.dtype.kind not in 'iu'
            or (labels < 0).any() or (labels > instance_count).any()):
        raise ValueError('真实实例mask必须是合法的二维类别索引: ' + str(path))
    return labels


def cpu_preflight(args):
    """只读真实绑定与完整mask表；不导入Torch，不解析Torch checkpoint。"""
    import numpy as np
    spec, report = validate_spec(args.spec)
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError('真实实例checkpoint不存在: ' + str(args.checkpoint))
    if args.expected_step < 1 or args.max_features < 7 or min(args.width, args.height) < 1:
        raise ValueError('expected-step/尺寸必须为正；逐ID RGB至少需要7 feature channels')
    if args.output and Path(args.output).exists():
        raise FileExistsError('评估目录已存在，拒绝覆盖: ' + str(args.output))
    prepared = Path(spec['prepared'])
    with np.load(prepared / 'instances.npz', allow_pickle=False) as bundle:
        ids = bundle['instance_ids'].tolist()
        frames = bundle['frame_ids'].tolist()
        times = bundle['timestamps'].astype(float).tolist()
        valid = bundle['valid'].copy()
        if not bool(bundle['mask_known'].all()):
            raise ValueError('观测质量评估需要全部49xK真实已知mask，未知不能当空body')
        normalization = bundle['normalization'].tolist()
        physical = bundle['physical_timestamps'].astype(float).tolist() if 'physical_timestamps' in bundle else None
        raw_micros = bundle['raw_timestamp_micros'].astype(int).tolist() if 'raw_timestamp_micros' in bundle else None
    if args.fixed_camera_frame not in frames:
        raise ValueError('固定相机必须是实际49相机中的frame ID')
    mask_rows = []
    for frame, frame_id in enumerate(frames):
        labels = read_mask_labels(prepared / 'instance_masks' / (frame_id + '.png'), args.width, args.height, len(ids))
        counts = [int((labels == index + 1).sum()) for index in range(len(ids))]
        if any(count and not valid[frame, index] for index, count in enumerate(counts)):
            raise ValueError('真实正body mask不能绑定无效轨迹: ' + frame_id)
        mask_rows.append(counts)
    counts = np.asarray(mask_rows)
    receipt = {
        'status': 'CPU_EVALUATION_SPEC_VALIDATED_GPU_NOT_STARTED',
        'scene_id': spec['scene_id'], 'checkpoint_path': str(args.checkpoint),
        'checkpoint_exists': True, 'checkpoint_deserialized': False,
        'checkpoint_binding_and_step_verification': 'deferred_until_locked_run',
        'expected_checkpoint_step': args.expected_step,
        'instance_ids': ids, 'frame_ids': frames, 'model_times': times,
        'physical_timestamps': physical, 'raw_timestamp_micros': raw_micros,
        'normalization': normalization, 'units': report.get('units', {}),
        'native_size': [args.width, args.height], 'mask_known_pairs': 49 * len(ids),
        'body_pixels_by_frame_and_ID': mask_rows,
        'nonempty_body_frame_denominators': (counts > 0).sum(0).astype(int).tolist(),
        'full_frame_metric_denominator': 49,
        'fixed_camera_frame_id': args.fixed_camera_frame, 'planned_FP32_frame_files': 98,
        'fixed_camera_has_GT': False, 'GPU_started': False, 'Torch_imported': False,
    }
    return spec, receipt


def fixed_camera_at(camera, timestamp):
    """真实相机浅拷贝；保留pose/projection/intrinsics/grid/directions所用对象。"""
    result = copy.copy(camera)
    result.timestamp = float(timestamp)
    if result is camera:
        raise AssertionError('固定相机必须是独立浅拷贝')
    for name, value in vars(camera).items():
        if name != 'timestamp' and vars(result).get(name) is not value:
            raise AssertionError('固定相机修改了timestamp之外的字段: ' + name)
    return result


def render_evaluation_frame(camera, static, instances, config, environment, *, max_features=10, edits=None, render_fn=None):
    """逐ID颜色通道使用相同完整场景；不删除任何遮挡几何。"""
    import torch
    if __package__:
        from .renderer import render, evaluate_sh_rgb
    else:
        from renderer import render, evaluate_sh_rgb
    render_fn = render if render_fn is None else render_fn
    ids = tuple(instances.instance_ids)
    background = static.get_xyz.new_zeros(3)
    options = dict(env_map=environment, max_features=max_features, instance_ids=ids, is_training=False, edits=edits)
    base = render_fn(camera, static, instances, config, background, **options)
    if tuple(base['instance_ids']) != ids or base['instance_visibility_semantics'] != VISIBLE_SEMANTICS:
        raise ValueError('联合贡献的真实身份/合成语义不符')
    if base.get('dynamic_alpha_semantics') != 'isolated_all_instances_raw_union_body_opacity':
        raise ValueError('孤立全部车辆alpha语义不符')
    world = instances.evaluate(camera.timestamp, edits=edits)
    # 此入口是原场景重建，不读取编辑后的checkpoint；替换会复用row_ids，不能
    # 把owner颜色写进同一个packed source行。明确拒绝而不是混淆单车颜色。
    if (world.row_ids.numel() != torch.unique(world.row_ids).numel()
            or not torch.equal(world.point_ids, instances.point_ids[world.row_ids])):
        raise ValueError('重建评估要求未替换的唯一packed行与owner身份')
    colors = evaluate_sh_rgb(world.features, world.local_view_directions(camera.camera_center), int(world.active_sh_degree))
    visible_rgb = []
    for index, track in enumerate(ids):
        packed = instances.local_xyz.new_zeros((len(instances.local_xyz), 3))
        packed[world.row_ids] = colors * (world.point_ids == index)[:, None]
        # 3 RGB + depth/alpha/union + 1身份 = 7；其他实例仍保留完整几何。
        package = render_fn(camera, static, instances, config, background,
                            env_map=environment, other=static.get_xyz.new_zeros((len(static.get_xyz), 3)),
                            dynamic_other=packed, instance_ids=[track], max_features=max_features,
                            render_separate=False, is_training=False, edits=edits)
        if not torch.equal(base['render'], package['render']) or not torch.equal(base['alpha'], package['alpha']):
            raise AssertionError('逐ID RGB feature pass改变了完整场景RGB/alpha')
        torch.testing.assert_close(package['instance_visible_contribution'][0],
                                   base['instance_visible_contribution'][index], rtol=0, atol=1e-6)
        visible_rgb.append(package['feature'])
    torch.testing.assert_close(base['instance_visible_contribution'].sum(0, keepdim=True),
                               base['dynamic_visible_alpha'], rtol=0, atol=2e-6)
    return {
        'composite': base['render'], 'static_rgb': base['render_s'], 'static_raw_rgb': base['raw_render_s'],
        'all_vehicles_raw_rgb': base['raw_render_d'],
        'scene_alpha': base['alpha'], 'dynamic_visible_alpha': base['dynamic_visible_alpha'],
        'static_visible_alpha': base['static_visible_alpha'],
        'all_vehicles_isolated_raw_alpha': base['dynamic_alpha'],
        'instance_visible_contribution': base['instance_visible_contribution'],
        'instance_visible_rgb': torch.stack(visible_rgb),
    }


def validate_arrays(arrays, instance_ids, height, width, *, observed):
    import numpy as np
    k = len(instance_ids)
    required = {'composite': (3, height, width), 'static_rgb': (3, height, width),
                'static_raw_rgb': (3, height, width), 'all_vehicles_raw_rgb': (3, height, width),
                'scene_alpha': (1, height, width), 'dynamic_visible_alpha': (1, height, width),
                'static_visible_alpha': (1, height, width), 'all_vehicles_isolated_raw_alpha': (1, height, width),
                'instance_visible_contribution': (k, height, width), 'instance_visible_rgb': (k, 3, height, width)}
    if observed:
        required.update(gt=(3, height, width), observed_masks=(k, height, width))
    if set(arrays) != set(required):
        raise ValueError('FP32观测/固定相机字段不符，固定相机不能带GT')
    for name, shape in required.items():
        array = arrays[name]
        if array.dtype != np.float32 or array.shape != shape or not np.isfinite(array).all():
            raise ValueError('导出必须为finite真实FP32且尺寸完整: ' + name)
        if 'alpha' in name or name in ('instance_visible_contribution', 'observed_masks'):
            if float(array.min()) < -2e-6 or float(array.max()) > 1 + 2e-6:
                raise ValueError('alpha/mask越出合法范围: ' + name)
    if observed and (not np.isin(arrays['observed_masks'], [0, 1]).all()
                     or (arrays['gt'] < 0).any() or (arrays['gt'] > 1).any()):
        raise ValueError('真实GT需0..1，真实mask需二值')


def observed_metrics(arrays, instance_ids):
    """full49均权；body只对真实非空mask取均值，孤立raw alpha不进入指标。"""
    import numpy as np
    gt = arrays['gt'].astype(np.float64)
    rgb_error = (arrays['composite'].astype(np.float64) - gt) ** 2
    clipped_error = (arrays['composite'].clip(0, 1).astype(np.float64) - gt) ** 2
    mse = float(clipped_error.mean())
    metrics = {'RGB_MSE_raw': float(rgb_error.mean()), 'RGB_MSE_clipped': mse,
               'RGB_PSNR_clipped_db': -10 * math.log10(mse) if mse > 0 else None,
               'RGB_PSNR_zero_error': mse == 0, 'instances': []}
    for index, track in enumerate(instance_ids):
        mask = arrays['observed_masks'][index].astype(bool)
        alpha = arrays['instance_visible_contribution'][index].astype(np.float64)
        error = (alpha - mask) ** 2
        metrics['instances'].append({
            'index': index, 'track_id': track, 'body_pixels': int(mask.sum()),
            'joint_visible_alpha_body_mean': float(alpha[mask].mean()) if mask.any() else None,
            'joint_visible_alpha_outside_mean': float(alpha[~mask].mean()) if (~mask).any() else None,
            'joint_mask_MSE_full_HxW': float(error.mean()),
            'joint_mask_MSE_inside': float(error[mask].mean()) if mask.any() else None,
            'joint_mask_MSE_outside': float(error[~mask].mean()) if (~mask).any() else None,
            'composite_RGB_MSE_body_clipped': float(clipped_error[:, mask].mean()) if mask.any() else None,
        })
    return metrics


def aggregate_observed(records, instance_ids):
    if len(records) != 49 or [row['frame_index'] for row in records] != list(range(49)):
        raise ValueError('观测指标必须完整49帧，不能将部分导出当最终结果')
    metrics = [row['metrics'] for row in records]
    average = lambda values: sum(values) / len(values) if values else None
    result = {'frame_denominator': 49, 'instances': []}
    for key in ('RGB_MSE_raw', 'RGB_MSE_clipped', 'RGB_PSNR_clipped_db', 'RGB_SSIM_clipped'):
        values = [row[key] for row in metrics if row.get(key) is not None]
        result[key + '_frame_mean'] = average(values)
        result[key + '_finite_frame_denominator'] = len(values)
    for index, track in enumerate(instance_ids):
        rows = [row['instances'][index] for row in metrics]
        if any(row['track_id'] != track or row['index'] != index for row in rows):
            raise ValueError('逐帧mask度量身份顺序不一致')
        body = [row for row in rows if row['body_pixels'] > 0]
        values = {'index': index, 'track_id': track, 'full_frame_denominator': 49,
                  'nonempty_body_frame_denominator': len(body), 'empty_body_frames': 49 - len(body),
                  'nonempty_outside_frame_denominator': sum(row['joint_mask_MSE_outside'] is not None for row in rows)}
        for key in ('joint_mask_MSE_full_HxW', 'joint_visible_alpha_outside_mean', 'joint_mask_MSE_outside'):
            values[key + '_frame_mean'] = average([row[key] for row in rows if row[key] is not None])
        for key in ('joint_visible_alpha_body_mean', 'joint_mask_MSE_inside', 'composite_RGB_MSE_body_clipped'):
            values[key + '_nonempty_frame_mean'] = average([row[key] for row in body])
        result['instances'].append(values)
    return result


def camera_record(camera):
    def ordinary(value):
        if hasattr(value, 'detach'):
            value = value.detach().cpu()
        return value.tolist() if hasattr(value, 'tolist') else value
    keys = ('R', 'T', 'world_view_transform', 'projection_matrix', 'full_proj_transform',
            'camera_center', 'cx', 'cy', 'fx', 'fy', 'FoVx', 'FoVy', 'trans', 'scale', 'c2w')
    result = {name: ordinary(getattr(camera, name)) for name in keys if hasattr(camera, name)}
    result.update(width=int(camera.image_width), height=int(camera.image_height))
    return result


def export_frame(output, stream, prefix, arrays, instance_ids, *, observed):
    """FP32原值不clamp；PNG仅作0..1显示，无resize或contrast stretch。"""
    import numpy as np
    from PIL import Image
    height, width = arrays['composite'].shape[-2:]
    validate_arrays(arrays, instance_ids, height, width, observed=observed)
    directory = Path(output) / stream
    directory.mkdir(exist_ok=True)
    npz = directory / (prefix + '.npz')
    if npz.exists():
        raise FileExistsError('拒绝覆盖逐帧资产: ' + str(npz))
    np.savez_compressed(npz, **arrays)
    def save(name, array):
        path = directory / (prefix + '_' + name + '.png')
        if path.exists():
            raise FileExistsError('拒绝覆盖PNG: ' + str(path))
        image = array[0] if array.shape[0] == 1 else array.transpose(1, 2, 0)
        Image.fromarray((image.clip(0, 1) * 255).round().astype(np.uint8)).save(path)
        return path.relative_to(output).as_posix()
    keys = ('composite', 'static_rgb', 'static_raw_rgb', 'all_vehicles_raw_rgb',
            'scene_alpha', 'dynamic_visible_alpha', 'static_visible_alpha', 'all_vehicles_isolated_raw_alpha')
    images = {name: save(name, arrays[name]) for name in keys}
    if observed:
        images['gt'] = save('gt', arrays['gt'])
    instances = []
    for index, track in enumerate(instance_ids):
        names = {'visible_rgb': save('instance%02d_visible_rgb' % index, arrays['instance_visible_rgb'][index]),
                 'visible_alpha': save('instance%02d_visible_alpha' % index, arrays['instance_visible_contribution'][index:index+1])}
        if observed:
            names['body_mask'] = save('instance%02d_body_mask' % index, arrays['observed_masks'][index:index+1])
        instances.append({'index': index, 'track_id': track, 'images': names})
    return {'npz': npz.relative_to(output).as_posix(), 'images': images, 'instances': instances}


def run(args, spec, preflight):
    slot, admission = require_gpu_slot(args.gpu_lock)
    output = Path(args.output)
    monitor, created, failure, final_status = None, False, None, 'failed'
    try:
        if output.exists():
            raise FileExistsError('评估目录已存在，不覆盖')
        output.mkdir(parents=True)
        created = True
        import numpy as np
        import torch
        if __package__:
            from .restore_contract import assert_exact_state
        else:
            from restore_contract import assert_exact_state
        sys.path.insert(0, str(Path(spec['matrix_tools'])))
        from memory_monitor import MemoryMonitor
        monitor = MemoryMonitor(output, torch_module=torch, expected_iterations=98)
        monitor.start()
        with monitor.phase('checkpoint_and_model_restore'):
            restored = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
            if (restored.get('schema') != 'drivingeditor_rigid_scene_v1' or restored.get('spec') != spec
                    or restored.get('step') != args.expected_step):
                raise ValueError('评估checkpoint真实场景/step绑定不符')
            config, scene, static, instances, optimizer, environment, exclusion = load_models(spec, output, restore=restored)
            del restored
            golden = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
            restored_counts = {}
            for name, value in (('background', static.capture()), ('instances', instances.capture(optimizer)),
                                ('environment', environment.capture())):
                restored_counts[name] = assert_exact_state(golden[name], value, path=name)
            del value
            if tuple(instances.instance_ids) != tuple(preflight['instance_ids']):
                raise ValueError('恢复checkpoint的instance IDs与真实prepared不符')
            if not np.allclose(instances.timestamps.detach().cpu().numpy(), preflight['model_times'], atol=1e-7, rtol=0):
                raise ValueError('恢复checkpoint的模型时间与真实49相机不符')
            if (bool(instances.deleted.any()) or bool((instances.replacement_source >= 0).any())
                    or bool((instances.edit_translation != 0).any())
                    or not torch.equal(instances.edit_rotation, instances.edit_rotation.new_tensor([1, 0, 0, 0]).expand_as(instances.edit_rotation))):
                raise ValueError('原场景评估拒绝含持久单车编辑的checkpoint')
        from utils.loss_utils import ssim
        cameras = list(scene.getTrainCameras())
        ids = tuple(instances.instance_ids)
        if len(cameras) != 49 or [Path(c.image_name).stem for c in cameras] != preflight['frame_ids']:
            raise ValueError('真实相机身份/顺序不符')
        if any((c.image_width, c.image_height) != (args.width, args.height) for c in cameras):
            raise ValueError('真实相机native尺寸与预检mask尺寸不符')
        fixed_index = preflight['frame_ids'].index(args.fixed_camera_frame)
        fixed_source = cameras[fixed_index]
        reference_static = None
        streams = {'observed': {'role': 'same49_training_reconstruction_observed_GT', 'GT_available': True, 'frames': []},
                   'fixed_camera': {'role': 'counterfactual_fixed_pose_time_diagnostic_without_GT', 'GT_available': False,
                                    'source_camera_frame_id': args.fixed_camera_frame, 'frames': []}}
        with torch.no_grad():
            for stream, observed in (('observed', True), ('fixed_camera', False)):
                with monitor.phase(stream):
                    for frame, observed_camera in enumerate(cameras):
                        priority = inspect_priority()
                        if not priority['allowed']:
                            _write_json(output / 'priority_yield.json', {'status': 'YIELDED_TO_PRIORITY_WORK', 'priority': priority,
                                        'completed_frames': {name: len(value['frames']) for name, value in streams.items()},
                                        'partial_assets_preserved': True, 'resume_requires_new_directory': True})
                            raise RuntimeError('评估让出StreetCrafter及其他GPU任务')
                        with monitor.phase('render_' + stream, iteration=frame + (0 if observed else 49)):
                            camera = observed_camera if observed else fixed_camera_at(fixed_source, observed_camera.timestamp)
                            values = render_evaluation_frame(camera, static, instances, config, environment, max_features=args.max_features)
                        metrics = None
                        if observed:
                            masks, known = masks_for_frame(spec, instances, observed_camera)
                            if not bool(known.all()):
                                raise ValueError('执行期间真实mask就绪合同变化')
                            values.update(gt=observed_camera.original_image.to(device=values['composite'].device), observed_masks=masks)
                        arrays = {key: value.detach().cpu().float().numpy() for key, value in values.items()}
                        validate_arrays(arrays, ids, args.height, args.width, observed=observed)
                        if observed:
                            metrics = observed_metrics(arrays, ids)
                            metrics['RGB_SSIM_clipped'] = float(ssim(values['composite'].clamp(0, 1), values['gt']))
                            if not math.isfinite(metrics['RGB_SSIM_clipped']):
                                raise ValueError('观测SSIM不是finite')
                            if [row['body_pixels'] for row in metrics['instances']] != preflight['body_pixels_by_frame_and_ID'][frame]:
                                raise ValueError('执行期间真实body mask内容或尺寸变化')
                            if frame == fixed_index:
                                reference_static = arrays['static_rgb'].copy()
                        record = export_frame(output, stream, '%03d_%s' % (frame, preflight['frame_ids'][frame]), arrays, ids, observed=observed)
                        record.update(frame_index=frame, frame_id=preflight['frame_ids'][frame], model_time=float(camera.timestamp),
                                      physical_timestamp=preflight['physical_timestamps'][frame] if preflight['physical_timestamps'] else None,
                                      raw_timestamp_micros=preflight['raw_timestamp_micros'][frame] if preflight['raw_timestamp_micros'] else None,
                                      camera_frame_id=Path((observed_camera if observed else fixed_source).image_name).stem,
                                      camera_original_timestamp=float((observed_camera if observed else fixed_source).timestamp),
                                      source_image_name=str((observed_camera if observed else fixed_source).image_name),
                                      camera=camera_record(camera), all_FP32_finite=True)
                        if observed:
                            record['metrics'] = metrics
                            for index, item in enumerate(record['instances']):
                                item['metrics'] = metrics['instances'][index]
                        else:
                            if frame == fixed_index and not np.array_equal(arrays['static_rgb'], reference_static):
                                raise AssertionError('固定相机anchor时刻静态RGB与观测同相机不一致')
                            record['diagnostic'] = {'static_RGB_MAE_vs_original_camera_time': float(np.abs(arrays['static_rgb'].astype(np.float64)-reference_static).mean()),
                                                    'instance_visible_alpha_sum': arrays['instance_visible_contribution'].sum((1, 2), dtype=np.float64).tolist(),
                                                    'counterfactual_GT_available': False}
                        streams[stream]['frames'].append(record)
                        del arrays, values
        metrics = aggregate_observed(streams['observed']['frames'], ids)
        if [row['nonempty_body_frame_denominator'] for row in metrics['instances']] != preflight['nonempty_body_frame_denominators']:
            raise AssertionError('完整49body分母不符')
        if len(streams['fixed_camera']['frames']) != 49:
            raise AssertionError('固定相机必须覆盖全部49模型时间')
        for stream in streams.values():
            for record in stream['frames']:
                paths = [record['npz'], *record['images'].values()]
                paths.extend(path for item in record['instances'] for path in item['images'].values())
                if any(not (output / path).is_file() for path in paths):
                    raise AssertionError('98帧最终manifest引用不完整')
        with monitor.phase('model_readonly_verification'):
            for name, value in (('background', static.capture()), ('instances', instances.capture(optimizer)),
                                ('environment', environment.capture())):
                assert_exact_state(golden[name], value, path=name)
            del value, golden
        manifest = {
            'schema': 'drivingeditor_rigid_scene_evaluation_v1', 'status': 'SCENE_EVALUATION_EXPORTED',
            'completed_utc': datetime.now(timezone.utc).isoformat(), 'spec': spec,
            'checkpoint': {'path': str(args.checkpoint), 'step': args.expected_step},
            'instance_ids': list(ids), 'units': preflight['units'], 'normalization': preflight['normalization'],
            'background_exclusion': exclusion, 'admission': admission, 'arguments': vars(args),
            'semantics': {'composite': 'full_scene_RGB_with_inherited_environment_fill',
                          'static_rgb': 'isolated_static_RGB_with_inherited_environment_fill',
                          'static_raw_rgb': 'isolated_static_RGB_black_background_without_environment_fill',
                          'all_vehicles_raw_rgb': 'isolated_all_instances_RGB_black_background_without_static_or_environment',
                          'all_vehicles_isolated_raw_alpha': 'isolated_all_instances_raw_union_body_opacity_not_visible_supervision',
                          'instance_visible_contribution': VISIBLE_SEMANTICS,
                          'instance_visible_rgb': RGB_SEMANTICS + '; premultiplied_black_canvas_no_environment',
                          'fixed_camera': 'only_timestamp_changed; no_counterfactual_GT_or_PSNR_or_mask_supervision'},
            'contracts': {'observed_frames': 49, 'fixed_camera_frames': 49, 'FP32_frame_files': 98,
                          'all_FP32_finite': True, 'identity_and_native_size_verified': True,
                          'every_frame_priority_checked': True, 'MemoryMonitor_started': True,
                          'model_and_optimizer_restored_exactly': True, 'restore_comparison_counts': restored_counts,
                          'restore_scope': 'background_instances_environment_models_and_Adam; evaluation_does_not_restore_or_claim_training_RNG_parity',
                          'model_and_optimizer_unmodified_after_evaluation': True, 'optimizer_updates': 0,
                          'fixed_camera_fields_except_timestamp_share_actual_objects': True,
                          'fixed_camera_anchor_static_bitwise_matches_observed': True,
                          'per_ID_RGB_full_scene_composite_alpha_unchanged': True,
                          'per_ID_RGB_identity_contributions_match_base_pass': True,
                          'mask_metrics_use_joint_visible_contribution': True,
                          'PNG_display_clipped_FP32_not_clipped': True,
                          'same49_train_GT_not_heldout': True},
            'preflight': preflight, 'observed_metrics': metrics, 'streams': streams,
            'fixed_camera_diagnostic': {'camera_frame_id': args.fixed_camera_frame,
                'model_times': preflight['model_times'], 'GT_available': False,
                'static_RGB_MAE_vs_original_camera_time_frame_mean': sum(row['diagnostic']['static_RGB_MAE_vs_original_camera_time'] for row in streams['fixed_camera']['frames']) / 49},
        }
        _write_json(output / 'scene_evaluation_manifest.json', manifest)
        final_status = 'complete'
        print(json.dumps({'status': manifest['status'], 'manifest': str(output / 'scene_evaluation_manifest.json'),
                          'checkpoint_step': args.expected_step, 'FP32_frame_files': 98}, ensure_ascii=False))
    except BaseException as error:
        failure = error
        if created:
            _write_json(output / 'failure.json', {'error': repr(error), 'partial_assets_preserved': True,
                                                  'complete_evaluation': False, 'resume_requires_new_directory': True})
        raise
    finally:
        try:
            if monitor is not None:
                monitor.close(status=final_status, error=failure)
        finally:
            slot.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-step', type=int, default=30000)
    parser.add_argument('--fixed-camera-frame', default='0008033')
    parser.add_argument('--width', type=int, default=480)
    parser.add_argument('--height', type=int, default=320)
    parser.add_argument('--max-features', type=int, default=10)
    parser.add_argument('--output')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--gpu-lock', default='/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
    args = parser.parse_args(argv)
    spec, preflight = cpu_preflight(args)
    if not args.run:
        print(json.dumps(preflight, ensure_ascii=False, indent=2, allow_nan=False))
        return
    if not args.output:
        parser.error('--run需要新的--output目录')
    run(args, spec, preflight)


if __name__ == '__main__':
    main()
