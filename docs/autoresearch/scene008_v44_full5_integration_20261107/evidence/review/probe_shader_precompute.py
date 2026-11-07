"""raw130 同 Full5 联合渲染：原生 SH 与私有 prepare RGB 预计算控制，0 更新。"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time
import types


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    repo = Path('/root/autodl-tmp/motion_proj_v82_gn11_v42')
    model = Path('/root/autodl-tmp/drivingeditor_repro/campaign/scene008_full_dynamic_edit_20261107_r1/inference10k')
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from PIL import Image
    from scripts.worldsim_v82.drivingeditor_dynamic_edit.scene_io import (
        load_native_models, make_camera, camera_headers, model_tensors)
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    pipe, models, renderer = load_native_models(model, 10000)
    from utils.sh_utils import eval_sh
    assert bool(pipe.including_dynamic) and not bool(pipe.convert_SHs_python)
    assert models['static'].active_sh_degree == models['dynamic'].active_sh_degree == 3
    camera = make_camera(camera_headers(model)[130])
    background = torch.ones(3, device='cuda') if bool(pipe.white_background) else torch.zeros(3, device='cuda')

    def fingerprints():
        hashes = {}
        for name, tensor in sorted(model_tensors(models).items()):
            value = tensor.detach().cpu().contiguous().numpy()
            digest = hashlib.sha256(str((str(value.dtype), value.shape)).encode())
            digest.update(value.tobytes())
            hashes[name] = digest.hexdigest()
        scalars = {group: {key: getattr(models[group], key) for key in
                   ['active_sh_degree', 'spatial_lr_scale', 'T', 'velocity_decay']}
                   for group in ['static', 'dynamic']}
        return hashes, scalars

    before = fingerprints()
    native = renderer.render
    original_prepare = renderer.prepare_rasterization
    native_rasterizer = native.__globals__['GaussianRasterizer']
    counters = dict(native_SH=0, precomputed_RGB=0)
    prepares = dict(native_SH=[], precomputed_RGB=[])
    phase = ['native_SH']

    class CountedRasterizer:
        def __init__(self, *values, **kwargs):
            self.inner = native_rasterizer(*values, **kwargs)

        def __call__(self, *values, **kwargs):
            counters[phase[0]] += 1
            return self.inner(*values, **kwargs)

    def prepare(*values, **kwargs):
        rows = original_prepare(*values, **kwargs)
        point_model = values[1]
        prepares[phase[0]].append('dynamic' if point_model is models['dynamic'] else 'static')
        if phase[0] == 'precomputed_RGB':
            assert rows['colors_precomp'] is None and rows['shs'] is not None
            direction = rows['means3D'] - values[0].camera_center
            direction = direction / direction.norm(dim=-1, keepdim=True)
            colour = (eval_sh(point_model.active_sh_degree,
                      rows['shs'].transpose(1, 2), direction) + .5).clamp_min(0)
            rows = dict(rows, shs=None, colors_precomp=colour)
        return rows

    # 独立 globals；不替换已导入 renderer 或类的共享方法。
    namespace = dict(native.__globals__, prepare_rasterization=prepare,
                     GaussianRasterizer=CountedRasterizer)
    private = types.FunctionType(native.__code__, namespace, native.__name__,
                                 native.__defaults__, native.__closure__)
    private.__kwdefaults__ = native.__kwdefaults__
    images = {}
    with torch.no_grad():
        for name in ['native_SH', 'precomputed_RGB']:
            phase[0] = name
            output = private(camera, models['static'], models['dynamic'], pipe, background,
                             env_map=models['environment'])
            images[name] = {key: output[key].detach().cpu().numpy().copy()
                            for key in ['render', 'alpha']}
            (args.output / 'progress.json').write_text(json.dumps(dict(
                completed_phase=name, native_forwards=dict(counters), prepare_calls=prepares), indent=2) + '\n')
            assert counters[name] == 3 and sorted(prepares[name]) == ['dynamic', 'static', 'static']
            rgb = images[name]['render']
            Image.fromarray(np.uint8(np.clip(rgb.transpose(1, 2, 0), 0, 1) * 255 + .5)).save(args.output / (name + '.png'))
            del output
    assert renderer.render is native and renderer.prepare_rasterization is original_prepare
    after = fingerprints()
    assert before == after
    difference = {key: float(abs(images['native_SH'][key] - images['precomputed_RGB'][key]).max())
                  for key in ['render', 'alpha']}
    tolerance = dict(render=2e-5, alpha=0.)
    passed = all(difference[key] <= tolerance[key] for key in difference)
    np.savez_compressed(args.output / 'joint_float.npz',
        native_rgb=images['native_SH']['render'], precomputed_rgb=images['precomputed_RGB']['render'],
        native_alpha=images['native_SH']['alpha'], precomputed_alpha=images['precomputed_RGB']['alpha'])
    receipt = dict(status='PASS_NATIVE_SH_PRECOMPUTED_RGB_BRIDGE' if passed else 'FAIL_NATIVE_SH_PRECOMPUTED_RGB_BRIDGE',
        raw=130, model=str(model), iteration=10000, static_active_sh_degree=3, dynamic_active_sh_degree=3,
        max_abs_difference=difference, tolerance=tolerance, native_forwards=counters,
        total_native_forwards=sum(counters.values()), prepare_calls=prepares,
        full5_parameters_identity_and_environment_unchanged=True, shared_renderer_globals_unchanged=True,
        no_V44_background_or_asset_in_this_control=True, optimizer_created=False, optimization_updates=0,
        model_tensor_fingerprints_before=before[0], source_scalars=before[1],
        seconds=time.time() - started, mixed_representation_compatibility_verdict=None)
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({key: value for key, value in receipt.items() if key != 'model_tensor_fingerprints_before'}, indent=2))
    if not passed:
        raise RuntimeError('SH bridge mismatch; do not attribute the dark car to mixed representation')


if __name__ == '__main__':
    main()
