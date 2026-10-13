"""只测CPU契约/真实renderer数学参考；不启动CUDA，不冒充实际98帧评估。"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from PIL import Image

from scripts.worldsim_v82.drivingeditor_instances.tests.test_training_render_contract import spec_fixture
from scripts.worldsim_v82.drivingeditor_instances import evaluate_scene as evaluator


ENTRY = Path(evaluator.__file__).resolve()


def evaluation_fixture(tmp_path):
    spec_path, spec = spec_fixture(tmp_path, complete=True)
    prepared = Path(spec['prepared'])
    masks = prepared / 'instance_masks'
    masks.mkdir()
    for frame in range(49):
        labels = np.zeros((2, 4), dtype=np.uint8)
        if frame < 7:
            labels[0, 0] = 1
        if frame < 39:
            labels[0, 1] = 2
        Image.fromarray(labels).save(masks / ('fixture_%04d.png' % frame))
    checkpoint = tmp_path / 'unit_fixture_not_a_real_model.pt'
    checkpoint.write_bytes(b'CPU preflight only; never deserialize this unit fixture')
    argv = [str(ENTRY), '--spec', str(spec_path), '--checkpoint', str(checkpoint),
            '--fixed-camera-frame', 'fixture_0010', '--width', '4', '--height', '2']
    return argv, spec


def fresh_entry(code, tmp_path):
    prefix = f"""
import builtins, json, pathlib, runpy, sys
sys.path.insert(0, {str(ENTRY.parent)!r})
original_import = builtins.__import__
def deny_model_import(name, *args, **kwargs):
    if name.split('.')[0] in ('torch', 'scene', 'diff_gaussian_rasterization', 'kornia', 'omegaconf'):
        raise AssertionError('CPU preflight imported a model/GPU module: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = deny_model_import
"""
    result = subprocess.run([sys.executable, '-I', '-c', prefix + code], cwd=tmp_path,
                            env=dict(os.environ, CUDA_VISIBLE_DEVICES='', PYTHONDONTWRITEBYTECODE='1'),
                            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr
    return json.loads(result.stdout)


def test_default_cpu_validation_no_torch_no_gpu_no_output_and_true_mask_denominators(tmp_path):
    argv, _ = evaluation_fixture(tmp_path)
    output = tmp_path / 'must_not_create'
    result = fresh_entry(f"""
import priority
priority.require_gpu_slot = lambda *args: (_ for _ in ()).throw(AssertionError('default validation requested GPU'))
priority.inspect_priority = lambda *args: (_ for _ in ()).throw(AssertionError('default validation queried GPU'))
sys.argv = {argv + ['--output', str(output)]!r}
runpy.run_path({str(ENTRY)!r}, run_name='__main__')
assert 'torch' not in sys.modules
assert not pathlib.Path({str(output)!r}).exists()
""", tmp_path)
    assert result['status'] == 'CPU_EVALUATION_SPEC_VALIDATED_GPU_NOT_STARTED'
    assert result['nonempty_body_frame_denominators'] == [7, 39]
    assert result['full_frame_metric_denominator'] == 49
    assert result['checkpoint_deserialized'] is False
    assert result['planned_FP32_frame_files'] == 98 and not result['fixed_camera_has_GT']


def test_blocked_admission_and_existing_output_preserve_assets_before_torch(tmp_path):
    argv, _ = evaluation_fixture(tmp_path)
    output = tmp_path / 'new_blocked'
    result = fresh_entry(f"""
import priority
def busy_slot(*args):
    raise RuntimeError('unit fixture StreetCrafter preparation owns GPU admission')
priority.require_gpu_slot = busy_slot
sys.argv = {argv + ['--run', '--output', str(output)]!r}
try:
    runpy.run_path({str(ENTRY)!r}, run_name='__main__')
except RuntimeError as error:
    assert 'StreetCrafter' in str(error)
    assert 'torch' not in sys.modules
    assert not pathlib.Path({str(output)!r}).exists()
    print(json.dumps({{'blocked': True}}))
else:
    raise AssertionError('busy GPU admission did not reject evaluation')
""", tmp_path)
    assert result['blocked']
    output.mkdir()
    sentinel = output / 'existing_experiment_receipt.json'
    sentinel.write_text('preserve actual old receipt', encoding='utf-8')
    result = fresh_entry(f"""
import priority
priority.require_gpu_slot = lambda *args: (_ for _ in ()).throw(AssertionError('old directory attempted GPU admission'))
sys.argv = {argv + ['--run', '--output', str(output)]!r}
try:
    runpy.run_path({str(ENTRY)!r}, run_name='__main__')
except FileExistsError:
    assert 'torch' not in sys.modules
    assert pathlib.Path({str(sentinel)!r}).read_text(encoding='utf-8') == 'preserve actual old receipt'
    assert [p.name for p in pathlib.Path({str(output)!r}).iterdir()] == ['existing_experiment_receipt.json']
    print(json.dumps({{'old_assets_preserved': True}}))
else:
    raise AssertionError('existing experiment directory was accepted')
""", tmp_path)
    assert result['old_assets_preserved']


def test_per_ID_RGB_uses_full_scene_occlusion_and_mask_metrics_ignore_isolated_alpha():
    import torch
    from scripts.worldsim_v82.drivingeditor_instances.rigid_model import RigidInstanceGaussianModel
    from scripts.worldsim_v82.drivingeditor_instances import renderer
    from scripts.worldsim_v82.drivingeditor_instances.tests.test_renderer_contract import StaticPVG, camera, pipe, reference_factory
    assert not torch.cuda.is_initialized()
    colors = torch.tensor([[.8, .2, .1], [.1, .7, .3]])
    model = RigidInstanceGaussianModel(
        torch.tensor([[0., 0., 4.], [0., 0., 2.]]), colors, torch.ones(2, 3),
        torch.tensor([0, 1]), ('fixture_A', 'fixture_B'), [0.],
        torch.eye(4).expand(1, 2, 4, 4).clone(), torch.ones(1, 2, dtype=torch.bool), sh_degree=0)
    with torch.no_grad():
        model._opacity.copy_(torch.logit(torch.tensor([[.8], [.9]])))
    calls = []
    def reference_render(*args, **kwargs):
        return renderer.render(*args, **kwargs, rasterizer_factory=reference_factory(calls))
    values = evaluator.render_evaluation_frame(camera(), StaticPVG(depth=1., opacity=.6), model, pipe(), None,
                                                render_fn=reference_render, max_features=10)
    torch.testing.assert_close(values['instance_visible_contribution'][:, 0, 0], torch.tensor([.032, .36]))
    torch.testing.assert_close(values['instance_visible_rgb'][:, :, 0, 0], colors * torch.tensor([.032, .36])[:, None])
    torch.testing.assert_close(values['all_vehicles_isolated_raw_alpha'], torch.tensor([[[.98]]]))
    # base联合+两个独立raw分支+两个per-ID完整场景pass；末两次都含静态和两车。
    assert len(calls) == 5
    assert all(len(call['means3D']) == 3 for call in calls[-2:])
    assert all(call['features'].shape[1] == 7 for call in calls[-2:])
    arrays = {key: value.detach().numpy().astype(np.float32) for key, value in values.items()}
    arrays.update(gt=np.zeros((3, 1, 1), dtype=np.float32), observed_masks=np.array([[[0.]], [[1.]]], dtype=np.float32))
    metrics = evaluator.observed_metrics(arrays, model.instance_ids)
    assert metrics['instances'][0]['joint_mask_MSE_full_HxW'] == pytest.approx(.032**2)
    assert metrics['instances'][0]['joint_visible_alpha_body_mean'] is None
    assert metrics['instances'][1]['joint_mask_MSE_full_HxW'] == pytest.approx((1-.36)**2)
    arrays['all_vehicles_isolated_raw_alpha'].fill(0)
    assert evaluator.observed_metrics(arrays, model.instance_ids) == metrics
    perfect = evaluator.observed_metrics({**arrays, 'gt': arrays['composite'].clip(0, 1).copy()}, model.instance_ids)
    assert perfect['RGB_MSE_clipped'] == 0
    assert perfect['RGB_PSNR_clipped_db'] is None and perfect['RGB_PSNR_zero_error'] is True
    rows = []
    for frame in range(49):
        arrays['observed_masks'][:] = np.array([frame < 7, frame < 39], dtype=np.float32)[:, None, None]
        rows.append({'frame_index': frame, 'metrics': evaluator.observed_metrics(arrays, model.instance_ids)})
    aggregate = evaluator.aggregate_observed(rows, model.instance_ids)
    assert [row['nonempty_body_frame_denominator'] for row in aggregate['instances']] == [7, 39]
    assert all(row['full_frame_denominator'] == 49 for row in aggregate['instances'])
    with pytest.raises(ValueError, match='完整49'):
        evaluator.aggregate_observed(rows[:-1], model.instance_ids)
    assert not torch.cuda.is_initialized()


def test_fixed_camera_changes_only_timestamp_and_export_rejects_GT_nonfinite_or_bad_dtype(tmp_path):
    from types import SimpleNamespace
    original = SimpleNamespace(timestamp=-.28, image_name='0008033', R=np.eye(3), T=np.zeros(3),
                               world_view_transform=np.eye(4), full_proj_transform=np.eye(4),
                               projection_matrix=np.eye(4), camera_center=np.zeros(3),
                               cx=2., cy=1., fx=3., fy=3., image_width=4, image_height=2,
                               grid=np.zeros((2, 4, 2)), direction_state=object())
    for timestamp in np.arange(49)*.02-.48:
        fixed = evaluator.fixed_camera_at(original, timestamp)
        assert fixed.timestamp == float(timestamp) and original.timestamp == -.28
        assert fixed is not original
        assert all(getattr(fixed, key) is value for key, value in vars(original).items() if key != 'timestamp')
    ids = ('fixture_A', 'fixture_B')
    arrays = {name: np.zeros((3, 2, 4), dtype=np.float32) for name in ('composite','static_rgb','static_raw_rgb','all_vehicles_raw_rgb')}
    arrays.update({name: np.zeros((1, 2, 4), dtype=np.float32) for name in ('scene_alpha','dynamic_visible_alpha','static_visible_alpha','all_vehicles_isolated_raw_alpha')})
    arrays['instance_visible_contribution'] = np.zeros((2, 2, 4), dtype=np.float32)
    arrays['instance_visible_rgb'] = np.zeros((2, 3, 2, 4), dtype=np.float32)
    evaluator.validate_arrays(arrays, ids, 2, 4, observed=False)
    arrays['gt'] = np.zeros((3, 2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match='固定相机不能带GT'):
        evaluator.validate_arrays(arrays, ids, 2, 4, observed=False)
    del arrays['gt']
    arrays['composite'] = arrays['composite'].astype(np.float64)
    with pytest.raises(ValueError, match='FP32'):
        evaluator.validate_arrays(arrays, ids, 2, 4, observed=False)
    arrays['composite'] = arrays['composite'].astype(np.float32)
    arrays['composite'][0, 0, 0] = np.nan
    with pytest.raises(ValueError, match='finite'):
        evaluator.validate_arrays(arrays, ids, 2, 4, observed=False)
    arrays['composite'][0, 0, 0] = 1.25
    record = evaluator.export_frame(tmp_path, 'fixed_camera', '000_unit_fixture', arrays, ids, observed=False)
    assert 'gt' not in record['images']
    with np.load(tmp_path / record['npz']) as exported:
        assert exported['composite'][0, 0, 0] == 1.25
        assert 'gt' not in exported
    with Image.open(tmp_path / record['images']['composite']) as image:
        assert image.size == (4, 2) and np.asarray(image)[0, 0, 0] == 255
    with pytest.raises(FileExistsError):
        evaluator.export_frame(tmp_path, 'fixed_camera', '000_unit_fixture', arrays, ids, observed=False)
