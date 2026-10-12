"""实际生成源码/四臂配置的CPU集成检查；不导入train或CUDA扩展。"""
import argparse
import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = os.environ['MKL_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)
import prepare_cpu
import train_patch
from runtime import CameraSampler, preserve_rng, validate_config
from metrics import region_metrics


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_sources(root, source):
    import sgcm_patch
    target = root / 'frozen_repository'
    functions = {'train.py': train_patch.patch_train_source,
                 'utils/paper_checkpoint.py': train_patch.patch_checkpoint_source,
                 'scene/paper_sgcm.py': sgcm_patch.patch_sgcm_source,
                 'scene/gaussian_model.py': sgcm_patch.patch_gaussian_source}
    for relative, patch in functions.items():
        old = (source / relative).read_text(encoding='utf-8').replace('\r\n', '\n')
        expected = patch(old)
        assert (target / relative).read_text(encoding='utf-8') == expected, relative
    for path in target.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'), feature_version=(3, 9))
    tree = ast.parse((target / 'train.py').read_text())
    training = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'training')
    hybrid = [n for n in ast.walk(training) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'compose_hybrid_loss']
    assert len(hybrid) == 1
    branch = (target / 'train.py').read_text()
    assert branch.index('hybrid_result = compose_hybrid_loss') < branch.index('loss.backward()')
    assert branch.index('"dynamic_alpha": render_pkg["dynamic_alpha"]') < branch.index('dynamic_alpha = dynamic_alpha * dynamic_mask')
    assert 'legacy_static_feature = render_pkg["static_feature"] / alpha.clamp_min(EPS)' in branch
    assert 'static_feature = render_pkg[\'static_feature\'] / (static_alpha if paper_method else alpha)' in branch
    assert 'iteration < args.densify_until_iter and (iteration % args.opacity_reset_interval' in branch
    assert 'hybrid_sampler' in (target / 'utils/paper_checkpoint.py').read_text()
    assert (target / 'utils/paper_losses.py').read_bytes() == (source / 'utils/paper_losses.py').read_bytes()


def fake_cameras():
    return [SimpleNamespace(image_path='/images/image_0/0017%03d.png' % i,
                            image_name='0017%03d' % i, colmap_id=i, timestamp=i / 50,
                            image_width=480, image_height=320) for i in range(49)]


def check_sampler():
    cameras = fake_cameras()
    a, b = CameraSampler(0, cameras), CameraSampler(0, list(reversed(cameras)))
    rows = []
    for _ in range(149):
        cam, shift = a.select(cameras, .5, .02)
        # 模拟各臂不同的拓扑、评价及随机消耗，不得影响独立采样器。
        random.random(); np.random.random(); torch.rand(5)
        second, second_shift = b.select(cameras, .5, .02)
        assert cam.image_name == second.image_name and shift == second_shift
        rows.append(cam.image_name)
    assert len(set(rows[:49])) == 49 and len(set(rows[49:98])) == 49
    capture = a.capture()
    for bad in (True, -1, 1.5, '149'):
        try:
            CameraSampler(0, cameras).restore(dict(capture, steps=bad))
        except ValueError:
            pass
        else:
            raise AssertionError('不合法采样步数未拒绝')
    try:
        CameraSampler(0, cameras).restore(dict(capture, stack=[]))
    except ValueError:
        pass
    else:
        raise AssertionError('步数与栈长度不一致未拒绝')
    restored = CameraSampler(0, cameras)
    restored.restore(capture)
    for _ in range(49):
        x, sx = a.select(cameras, .5, .02)
        y, sy = restored.select(cameras, .5, .02)
        assert x.image_name == y.image_name and sx == sy


def check_rng():
    python, numpy, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with preserve_rng():
        random.random(); np.random.random(); torch.rand(8)
    assert random.getstate() == python
    current = np.random.get_state()
    assert current[0] == numpy[0] and np.array_equal(current[1], numpy[1]) and current[2:] == numpy[2:]
    assert torch.equal(cpu, torch.get_rng_state())


def check_checkpoint_capture(root):
    sys.path.insert(0, str(root / 'frozen_repository'))
    checkpoint = load(root / 'frozen_repository/utils/paper_checkpoint.py', 'hybrid_checkpoint_cpu')
    cameras = fake_cameras()
    sampler = CameraSampler(0, cameras)
    sampler.select(cameras, .5, .02)
    scene = SimpleNamespace(train_cameras={4: cameras}, resolution_scales=[4], scale_index=0, hybrid_sampler=sampler)
    cfg = json.loads((root / 'configs/C00.json').read_text())
    state = checkpoint.capture_training_state(1, scene, [], {}, cfg, ema={'loss': 1.0})
    assert state['hybrid_sampler']['steps'] == state['iteration'] == 1
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    restored = torch.load(buffer, map_location='cpu', weights_only=False)
    probe = CameraSampler(0, cameras)
    probe.restore(restored['hybrid_sampler'])
    x, sx = sampler.select(cameras, .5, .02)
    y, sy = probe.select(cameras, .5, .02)
    assert x.image_name == y.image_name and sx == sy
    # 此处只验证小型sidecar序列化；GPU模型/Adam恢复还需实际GPU。


def check_metrics():
    actor = np.zeros((13, 17), dtype=bool); actor[3:10, 5:12] = True
    sky = np.zeros_like(actor)
    arrays = {name: np.zeros((3, 13, 17), dtype=np.float32) for name in ('gt', 'composite', 'static', 'dynamic')}
    arrays['dynamic'][:] = .4
    arrays['alpha_d'] = np.full((1, 13, 17), .8, dtype=np.float32)
    arrays['alpha_s'] = np.ones((1, 13, 17), dtype=np.float32)
    result = region_metrics(arrays, actor, sky)
    assert result['non_sky_outside_actor']['alpha_d_mean'] > .79
    assert result['non_sky_outside_actor']['dynamic_RGB_support_8bit_gt2'] == 1.0
    assert result['actor']['pixel_count'] == 49 and result['full']['composite_mse'] == 0


def check_static_time():
    from static_time import run_static_time_check
    cameras = fake_cameras()
    model = SimpleNamespace(_velocity=torch.ones(3, 3), get_scaling_t=torch.ones(3, 1), T=.2)
    with tempfile.TemporaryDirectory() as directory:
        scene = SimpleNamespace(getTrainCameras=lambda: cameras, model_path=directory,
                                gaussians=model, dynamic_gaussians=None)
        before = cameras[10].timestamp
        def renderer(camera, *_args, **_kwargs):
            return {'render_s': torch.full((3, 2, 3), camera.timestamp), 'static_alpha': torch.ones(1, 2, 3)}
        result = run_static_time_check(scene, renderer, (), None, 10000)
        assert cameras[10].timestamp == before and len(result['rows']) == 49
        assert result['rows'][10]['static_rgb_MAE_from_anchor'] == 0
        assert result['rows'][48]['static_rgb_MAE_from_anchor'] > 0
        # 异常路径也必须恢复相机timestamp。
        def failure(camera, *_args, **_kwargs):
            if camera.timestamp != before:
                raise RuntimeError('fixture')
            return renderer(camera)
        try:
            run_static_time_check(scene, failure, (), None, 20000)
        except RuntimeError:
            assert cameras[10].timestamp == before
        else:
            raise AssertionError('异常路径未执行')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=prepare_cpu.ROOT)
    parser.add_argument('--source', type=Path, default=prepare_cpu.SOURCE)
    parser.add_argument('--output', type=Path)
    cli = parser.parse_args()
    check_sources(cli.root, cli.source)
    configs = {arm: json.loads((cli.root / 'configs' / (arm + '.json')).read_text()) for arm in ('C00', 'G10', 'N01', 'GN11')}
    for cfg in configs.values():
        validate_config(cfg)
    ignored = {'hybrid_arm', 'model_path', 'paper_sgcm_enabled', 'hybrid_nps_enabled'}
    common = [{k: v for k, v in cfg.items() if k not in ignored} for cfg in configs.values()]
    assert all(common[0] == row for row in common)
    assert [(c['paper_sgcm_enabled'], c['hybrid_nps_enabled']) for c in configs.values()] == [(False, False), (True, False), (False, True), (True, True)]
    check_sampler(); check_rng(); check_checkpoint_capture(cli.root); check_metrics(); check_static_time()
    assert not torch.cuda.is_initialized()
    result = {'status': 'PASS_CPU_INTEGRATION',
              'checks': ['actual_patched_source_equal_and_python39_syntax', 'four_arm_only_module_and_output_differences',
                         'independent_sampling_exact_and_restore', 'evaluation_RNG_preserved',
                         'actual_sidecar_capture_and_small_serialization', 'raw_alpha_and_RGB_leakage_metrics',
                         'fixed_pose_timestamp_restored_success_and_failure'],
              'GPU_used': False, 'checkpoint_loaded': False, 'small_fixture_sidecar_only': True,
              'GPU_first_backward_and_model_Adam_restore_verified': False,
              'torch_version': torch.__version__, 'threads': 1}
    if cli.output:
        cli.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
