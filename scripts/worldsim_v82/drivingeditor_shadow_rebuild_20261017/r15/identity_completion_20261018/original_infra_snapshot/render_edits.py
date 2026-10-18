"""按真实track渲染49观测帧的单实例删除/平移/局部yaw/场景内资产替换。

默认只读验证。物理平移单位为米，方向是车辆局部 x(前)/y(左)/z(上)；
旋转绕车辆局部z及当前实例中心，Gaussian covariance/SH方向同步变换。
"""
import argparse
import json
import math
from pathlib import Path
import sys

if __package__:
    from .runtime import validate_spec, load_models
    from .priority import require_gpu_slot, inspect_priority
else:
    from runtime import validate_spec, load_models
    from priority import require_gpu_slot, inspect_priority


def verify_checkpoint_binding(restored, spec, expected_step=None):
    """读取实际保存步数；文件名不能替代场景与step内容验证。"""
    if restored.get('schema') != 'drivingeditor_rigid_scene_v1' or restored.get('spec') != spec:
        raise ValueError('编辑checkpoint场景绑定不符')
    step = restored.get('step')
    if isinstance(step, bool) or not isinstance(step, int) or step < 1:
        raise ValueError('编辑checkpoint缺少真实有效保存步数')
    if expected_step is not None and step != expected_step:
        raise ValueError('编辑checkpoint真实step与显式expected-step不符')
    return step


def frame_edits(model, timestamp, track, operation, *, translation_m=None, angle_degrees=0, source=None, scale=1):
    import torch
    if __package__:
        from .rigid_model import quaternion_to_matrix, quaternion_multiply
    else:
        from rigid_model import quaternion_to_matrix, quaternion_multiply
    index = model.instance_ids.index(track)
    position, orientation, valid = model.trajectory_at(timestamp)
    effective = quaternion_multiply(model.edit_rotation[index], orientation[index])
    rotation = quaternion_to_matrix(effective)
    if operation == 'delete':
        value = {'delete': True}
    elif operation == 'translate':
        delta = rotation @ torch.as_tensor(translation_m, device=rotation.device, dtype=rotation.dtype) * scale
        value = {'translation': delta}
    elif operation == 'rotate':
        half_angle = math.radians(angle_degrees) / 2
        axis = rotation[:, 2]
        value = {'quaternion': torch.cat((rotation.new_tensor([math.cos(half_angle)]), axis * math.sin(half_angle)))}
    elif operation == 'replace':
        value = {'replacement': source}
    else:
        raise ValueError('未知编辑操作')
    return {track: value}, bool(valid[index] & ~model.deleted[index])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--spec', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--expected-step', type=int, help='执行时检查实际checkpoint保存步数，不能只凭文件名')
    parser.add_argument('--track', required=True)
    parser.add_argument('--operation', required=True, choices=['delete', 'translate', 'rotate', 'replace'])
    parser.add_argument('--translate-local-m', nargs=3, type=float)
    parser.add_argument('--rotate-degrees', type=float, default=30)
    parser.add_argument('--replacement-track')
    parser.add_argument('--frame-index', type=int, help='不带此项导出全部49观察帧')
    parser.add_argument('--output')
    parser.add_argument('--max-features', type=int, default=10)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--gpu-lock', default='/root/autodl-tmp/drivingeditor_repro/campaign/gpu_slot.lock')
    args = parser.parse_args()
    if args.expected_step is not None and args.expected_step < 1: parser.error('expected-step必须为正')
    spec, report = validate_spec(args.spec)
    import numpy as np
    with np.load(Path(spec['prepared']) / 'instances.npz', allow_pickle=False) as bundle:
        identities = list(bundle['instance_ids'])
        scale = float(bundle['world_scale'])
    if args.track not in identities: parser.error('track不属于此真实场景')
    if not Path(args.checkpoint).is_file(): parser.error('真实实例checkpoint不存在')
    if args.operation == 'translate' and args.translate_local_m is None: parser.error('平移需要--translate-local-m dx dy dz')
    if args.operation == 'replace' and args.replacement_track not in identities: parser.error('替换需要同场景有效--replacement-track')
    if args.frame_index is not None and not 0 <= args.frame_index < 49: parser.error('frame-index必须0..48')
    if not args.run:
        print(json.dumps({'status': 'EDIT_SPEC_VALIDATED_GPU_NOT_STARTED', 'scene': spec['scene_id'], 'arguments': vars(args)}, ensure_ascii=False, indent=2))
        return
    if not args.output: parser.error('--run需要新的--output')
    slot, admission = require_gpu_slot(args.gpu_lock)
    monitor = None
    final_status, failure = 'failed', None
    output = Path(args.output)
    output_created = False
    try:
        if output.exists(): raise FileExistsError('编辑导出目录已存在，不覆盖')
        output.mkdir(parents=True)
        output_created = True
        import torch
        from PIL import Image
        if __package__:
            from .renderer import render
        else:
            from renderer import render
        sys.path.insert(0, str(Path(spec['matrix_tools'])))
        from memory_monitor import MemoryMonitor
        monitor = MemoryMonitor(output, torch_module=torch, expected_iterations=49)
        monitor.start()
        restored = torch.load(args.checkpoint, map_location='cuda', weights_only=False)
        checkpoint_step = verify_checkpoint_binding(restored, spec, args.expected_step)
        with monitor.phase('initialization'):
            config, scene, static, instances, _, environment, exclusion = load_models(spec, output, restore=restored)
        records = []
        with torch.no_grad(), monitor.phase('edit_evaluation'):
            for frame, camera in enumerate(scene.getTrainCameras()):
                if args.frame_index is not None and frame != args.frame_index: continue
                priority = inspect_priority()
                if not priority['allowed']: raise RuntimeError('编辑导出暂停，让出StreetCrafter: ' + json.dumps(priority))
                edits, valid = frame_edits(instances, camera.timestamp, args.track, args.operation,
                                          translation_m=args.translate_local_m, angle_degrees=args.rotate_degrees,
                                          source=args.replacement_track, scale=scale)
                options = dict(env_map=environment, other=[static.get_scaling_t.clamp_max(2), static.get_inst_velocity],
                               instance_ids=[args.track], max_features=args.max_features)
                background = torch.zeros(3, device='cuda')
                before = render(camera, static, instances, config, background, **options)
                after = render(camera, static, instances, config, background, edits=edits, **options)
                before_world = instances.evaluate(camera.timestamp)
                after_world = instances.evaluate(camera.timestamp, edits=edits)
                target_index = instances.instance_ids.index(args.track)
                other_before = before_world.point_ids != target_index
                other_after = after_world.point_ids != target_index
                other_unchanged = torch.equal(before_world.row_ids[other_before], after_world.row_ids[other_after]) and all(
                    torch.equal(getattr(before_world, key)[other_before], getattr(after_world, key)[other_after])
                    for key in ('xyz', 'rotation', 'scaling', 'features', 'opacity'))
                if not other_unchanged: raise AssertionError('单实例编辑修改了其他实例Gaussian')
                data = {'gt': camera.original_image, 'before': before['render'], 'after': after['render'],
                        'static': before['render_s'], 'after_static': after['render_s'], 'before_vehicles': before['raw_render_d'], 'after_vehicles': after['raw_render_d'],
                        'before_target_visible': before['instance_visible_contribution'], 'after_target_visible': after['instance_visible_contribution']}
                prefix = '%03d_%s' % (frame, Path(camera.image_name).stem)
                arrays = {key: value.detach().cpu().float().numpy() for key, value in data.items()}
                np.savez_compressed(output / (prefix + '.npz'), **arrays)
                for name, array in arrays.items():
                    if array.shape[0] == 1:
                        image = (array[0].clip(0, 1) * 255).round().astype(np.uint8)
                    else:
                        image = (array.transpose(1, 2, 0).clip(0, 1) * 255).round().astype(np.uint8)
                    Image.fromarray(image).save(output / (prefix + '_' + name + '.png'))
                unchanged = bool(torch.equal(before['render_s'], after['render_s']))
                if not unchanged: raise AssertionError('单实例编辑改变了独立静态渲染')
                records.append({'frame': frame, 'frame_id': camera.image_name, 'timestamp': camera.timestamp,
                                'target_track_valid': valid, 'static_render_bitwise_unchanged': unchanged,
                                'other_instance_Gaussians_bitwise_unchanged': other_unchanged,
                                'pixel_RGB_MAE': float((before['render']-after['render']).abs().mean()),
                                'other_visible_pixels_may_change_due_to_occlusion': True,
                                'paired_edited_GT_available': False})
        receipt = {'status': 'EDIT_RENDER_EXPORTED_VISUAL_REVIEW_PENDING', 'spec': spec, 'arguments': vars(args),
                   'actual_checkpoint_step': checkpoint_step,
                   'admission': admission, 'frames': records, 'background_exclusion': exclusion,
                   'no_model_parameters_mutated_by_edits': True, 'human_quality_verdict': None,
                   'deletion_background_holes_or_shadow_verified': False, 'relighting': False}
        (output / 'edit_receipt.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
        final_status = 'complete'
    except BaseException as error:
        failure = error
        if output_created:
            (output / 'failure.json').write_text(json.dumps({'error': repr(error), 'assets_preserved': True}), encoding='utf-8')
        raise
    finally:
        if monitor is not None: monitor.close(status=final_status, error=failure)
        slot.close()


if __name__ == '__main__': main()
