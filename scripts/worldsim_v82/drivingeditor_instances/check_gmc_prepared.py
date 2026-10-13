"""实际GMC双实例输入的CPU编辑独立性检查；不是训练/照片渲染质量评价。"""
import argparse
import json
import os
from pathlib import Path
import time


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cpu-threads', type=int, default=4)
    args = parser.parse_args()
    import numpy as np
    import torch
    if __package__:
        from .rigid_model import RigidInstanceGaussianModel
        from .render_edits import frame_edits
    else:
        from rigid_model import RigidInstanceGaussianModel
        from render_edits import frame_edits
    if not 1 <= args.cpu_threads <= 4: raise ValueError('CPU仅允许1..4线程')
    torch.set_num_threads(args.cpu_threads)
    started = time.monotonic()
    with np.load(args.prepared / 'instances.npz', allow_pickle=False) as bundle:
        if len(bundle['instance_ids']) != 2 or not bundle['mask_known'].all(): raise ValueError('必须实际GMC2x49已知mask')
        scale = float(bundle['world_scale'])
    model = RigidInstanceGaussianModel.from_npz(args.prepared / 'instances.npz', device='cpu', sh_degree=0)
    before_parameters = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    frames = []
    with torch.no_grad():
        for frame, timestamp in enumerate(model.timestamps):
            base = model.evaluate(timestamp)
            target = model.instance_ids[0]
            other = base.point_ids == 1
            translated, valid = frame_edits(model, timestamp, target, 'translate', translation_m=[.5, 0, 0], scale=scale)
            rotated, _ = frame_edits(model, timestamp, target, 'rotate', angle_degrees=30, scale=scale)
            shifted, turned, removed = model.evaluate(timestamp, translated), model.evaluate(timestamp, rotated), model.evaluate(timestamp, {target: {'delete': True}})
            invariant = all(torch.equal(getattr(base, name)[other], getattr(shifted, name)[other])
                            and torch.equal(getattr(base, name)[other], getattr(turned, name)[other])
                            for name in ('xyz', 'rotation', 'scaling', 'features', 'opacity'))
            deletion_ok = not bool((removed.point_ids == 0).any()) and torch.equal(removed.xyz, base.xyz[other])
            delta = (shifted.xyz[~other] - base.xyz[~other]).norm(dim=-1)
            displacement_error = float((delta - .5 * scale).abs().max())
            if not invariant or not deletion_ok or displacement_error > 2e-5:
                raise AssertionError('实际GMC单实例编辑改变了另一实例或平移单位错误')
            frames.append({'frame': frame, 'model_time': float(timestamp), 'target_valid': valid,
                           'other_instance_parameters_unchanged': invariant, 'delete_only_target': deletion_ok,
                           'translation_0_5m_max_length_error_model_units': displacement_error})
    if any(not torch.equal(before_parameters[name], parameter) for name, parameter in model.named_parameters()):
        raise AssertionError('查询编辑写回了模型参数')
    restored = RigidInstanceGaussianModel.from_capture(model.capture(), device='cpu')
    if any(not torch.equal(value, restored.state_dict()[key]) for key, value in model.state_dict().items()):
        raise AssertionError('实际刚体模型CPU状态恢复不精确')
    receipt = {'status': 'PASS_REAL_GMC49_CPU_INSTANCE_EDIT_CONTRACT', 'prepared': str(args.prepared),
               'instance_ids': model.instance_ids, 'real_points': model.num_points, 'frames': frames,
               'model_parameters_mutated': False, 'core_CPU_state_restore_exact': True,
               'full_scene_GPU_model_Adam_RNG_restore_verified': False, 'photoreal_edit_quality_verified': False,
               'CUDA_initialized': torch.cuda.is_initialized(), 'cpu_threads': args.cpu_threads,
               'wall_seconds': time.monotonic()-started}
    with args.output.open('x', encoding='utf-8') as stream: json.dump(receipt, stream, ensure_ascii=False, indent=2)
    print(json.dumps({key: value for key, value in receipt.items() if key != 'frames'}, ensure_ascii=False))


if __name__ == '__main__': main()
