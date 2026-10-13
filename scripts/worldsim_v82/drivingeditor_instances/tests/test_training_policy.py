"""只验证固定续训策略/预算/字段；CPU fixture不认证真实GPU恢复。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import subprocess
import sys

import numpy as np
import pytest
import torch

from scripts.worldsim_v82.drivingeditor_instances.rigid_model import RigidInstanceGaussianModel
from scripts.worldsim_v82.drivingeditor_instances.training_policy import (
    POLICY_FIELD,
    resolve_training_policy,
    validate_resume_checkpoint,
    validate_training_policy,
)


def arguments(**changes):
    result = dict(seed=0, joint_background=False, refine_poses_after=0, pose_translation_limit_m=.5,
                box_boundary_weight=1.0, pose_prior_weight=.01, max_features=10, geometry_only_smoke=False,
                iterations=300, output='/new/smoke', resume=None, spec='/actual/scene.json',
                gpu_lock='/shared/slot.lock', run=True)
    result.update(changes)
    return result


def reference():
    return dict(iterations=30000, source_path='/actual/source', model_path='/old/static_prior',
                position_lr_init=.00016, position_lr_final=.0000016, position_lr_delay_mult=.01,
                hybrid_sgcm_enabled=False, hybrid_nps_enabled=False,
                hybrid_lambda_static_rgb=1.0, hybrid_lambda_dynamic_opacity=100.0,
                loss_config={'sky': .001, 'depth': .01, 'opacity_domain': 'full'})


@pytest.fixture
def full_cpu_checkpoint():
    # 真正core和Adam数据，旧背景/环境/CUDA边界仅为本策略测试的显式替代。
    poses = np.broadcast_to(np.eye(4), (2, 2, 4, 4)).copy()
    poses[:, 1, 0, 3] = 2
    model = RigidInstanceGaussianModel([[.1, 0, 0], [0, .1, 0]], [[.9, .1, .1], [.1, .1, .9]],
                                         [[.01]*3]*2, [0, 1], ['GMC', 'Corolla'], [-.02, .02], poses,
                                         np.ones((2, 2), dtype=bool), sh_degree=1, box_size=np.ones((2, 2, 3)))
    optimizer = model.make_optimizer()
    model.evaluate(-.02).xyz.square().sum().backward()
    optimizer.step()
    return {'schema': 'drivingeditor_rigid_scene_v1', 'step': 300,
            'spec': {'scene_id': 'scene0008_023_071', 'prepared': '/actual/two_actor'},
            'background': ('GPU_background_capture_boundary',), 'instances': model.capture(optimizer),
            'environment': ('GPU_environment_capture_boundary',), 'background_exclusion': {'remaining_points': 613405},
            'sampler': random.Random(0).getstate(), 'python_rng': random.Random(0).getstate(),
            'numpy_rng': np.random.RandomState(0).get_state(), 'torch_rng': torch.get_rng_state(),
            'cuda_rng': [torch.tensor([1, 2], dtype=torch.uint8)], 'restoration_verified': False}


def test_policy_records_actual_values_and_operational_changes_are_allowed():
    original_args, original_reference = arguments(), reference()
    saved = resolve_training_policy(original_args, original_reference)
    assert saved['reference_config'] == original_reference
    assert saved['adapter_constants']['xyz_lr_horizon'] == 30000
    current_args = dict(original_args, iterations=30000, output='/new/continuation', resume='/old/checkpoint_000300.pt',
                        spec='/relocated/scene.json', gpu_lock='/relocated/slot.lock', run=False)
    current_reference = dict(original_reference, source_path='/relocated/source', model_path='/relocated/prior')
    validate_training_policy(saved, resolve_training_policy(current_args, current_reference))
    original_reference['loss_config']['sky'] = 3
    assert saved['reference_config']['loss_config']['sky'] == .001


@pytest.mark.parametrize('key,value', [('seed', 1), ('joint_background', True), ('refine_poses_after', 301),
                                    ('pose_translation_limit_m', .4), ('box_boundary_weight', 2),
                                    ('pose_prior_weight', .02), ('max_features', 9)])
def test_resume_rejects_each_training_argument_change(key, value):
    saved = resolve_training_policy(arguments(), reference())
    changed = dict(arguments(), **{key: value})
    with pytest.raises(ValueError, match=key):
        validate_training_policy(saved, resolve_training_policy(changed, reference()))


@pytest.mark.parametrize('key,value', [('iterations', 50000), ('hybrid_sgcm_enabled', True), ('hybrid_nps_enabled', True),
                                    ('hybrid_lambda_static_rgb', 2), ('position_lr_init', .0002)])
def test_resume_rejects_reference_horizon_mechanism_and_loss_changes(key, value):
    saved = resolve_training_policy(arguments(), reference())
    changed = dict(reference(), **{key: value})
    with pytest.raises(ValueError, match=key):
        validate_training_policy(saved, resolve_training_policy(arguments(), changed))


def test_nested_loss_changes_cannot_hide_in_raw_reference():
    saved = resolve_training_policy(arguments(), reference())
    changed = reference()
    changed['loss_config']['opacity_domain'] = 'foreground_only'
    with pytest.raises(ValueError, match='loss_config.opacity_domain'):
        validate_training_policy(saved, resolve_training_policy(arguments(), changed))


def test_future_checkpoint_policy_allows_only_301_to_total30000(full_cpu_checkpoint):
    saved = resolve_training_policy(arguments(), reference())
    full_cpu_checkpoint[POLICY_FIELD] = saved
    current = resolve_training_policy(dict(arguments(), iterations=30000), reference())
    receipt = validate_resume_checkpoint(full_cpu_checkpoint, current, 30000, expected_step=300)
    assert receipt['step'] == 300 and receipt['first_step'] == 301
    assert receipt[POLICY_FIELD] == saved
    assert receipt['policy_evidence']['historical_config_in_checkpoint']
    assert receipt['restoration_verified'] is False


@pytest.mark.parametrize('target', [1, 299, 300, 30001])
def test_empty_or_overbudget_continuation_is_rejected(full_cpu_checkpoint, target):
    saved = resolve_training_policy(arguments(), reference())
    full_cpu_checkpoint[POLICY_FIELD] = saved
    with pytest.raises(ValueError, match='总步数'):
        validate_resume_checkpoint(full_cpu_checkpoint, saved, target)


def test_wrong_expected_short_step_is_rejected(full_cpu_checkpoint):
    saved = resolve_training_policy(arguments(), reference())
    full_cpu_checkpoint[POLICY_FIELD] = saved
    with pytest.raises(ValueError, match='短测终点'):
        validate_resume_checkpoint(full_cpu_checkpoint, saved, 30000, expected_step=299)


def test_legacy_reconstruction_uses_actual_parent_files_and_records_limit(full_cpu_checkpoint, tmp_path):
    effective = {'spec': full_cpu_checkpoint['spec'], 'arguments': arguments(), 'seed': 0}
    effective_path = tmp_path / 'effective_config.json'
    reference_path = tmp_path / 'actual_smoke_reference.json'
    effective_path.write_text(json.dumps(effective), encoding='utf-8')
    reference_path.write_text(json.dumps(reference()), encoding='utf-8')
    current = resolve_training_policy(dict(arguments(), iterations=30000), reference())
    receipt = validate_resume_checkpoint(full_cpu_checkpoint, current, 30000, expected_step=300,
                                         legacy_effective_config=effective_path, legacy_reference_config=reference_path,
                                         legacy_reference_provenance={'role': 'supplied actual smoke reference file'})
    evidence = receipt['policy_evidence']
    assert evidence['effective_config']['path'] == str(effective_path.resolve())
    assert evidence['reference_config']['path'] == str(reference_path.resolve())
    assert evidence['historical_config_in_checkpoint'] is False
    assert evidence['historical_reference_contents_certified_by_this_check'] is False
    assert receipt['restoration_verified'] is False
    assert receipt['first_step'] == 301


def test_legacy_requires_reference_provenance_and_correct_run_binding(full_cpu_checkpoint):
    current = resolve_training_policy(arguments(), reference())
    effective = {'spec': full_cpu_checkpoint['spec'], 'arguments': arguments()}
    with pytest.raises(ValueError, match='reference_config'):
        validate_resume_checkpoint(full_cpu_checkpoint, current, 30000, legacy_effective_config=effective)
    effective['spec'] = {'scene_id': 'another_scene'}
    with pytest.raises(ValueError, match='场景绑定'):
        validate_resume_checkpoint(full_cpu_checkpoint, current, 30000, legacy_effective_config=effective,
                                     legacy_reference_config=reference())


@pytest.mark.parametrize('field', ['background', 'sampler', 'cuda_rng', 'numpy_rng'])
def test_incomplete_fullscene_checkpoint_is_rejected(full_cpu_checkpoint, field):
    full_cpu_checkpoint[POLICY_FIELD] = resolve_training_policy(arguments(), reference())
    del full_cpu_checkpoint[field]
    with pytest.raises(ValueError, match='缺少状态'):
        validate_resume_checkpoint(full_cpu_checkpoint, full_cpu_checkpoint[POLICY_FIELD], 30000)


def test_missing_adam_and_pose_table_are_rejected(full_cpu_checkpoint):
    saved = resolve_training_policy(arguments(), reference())
    full_cpu_checkpoint[POLICY_FIELD] = saved
    no_optimizer = copy.deepcopy(full_cpu_checkpoint)
    no_optimizer['instances']['optimizer'] = None
    with pytest.raises(ValueError, match='Adam'):
        validate_resume_checkpoint(no_optimizer, saved, 30000)
    del full_cpu_checkpoint['instances']['state_dict']['base_poses']
    with pytest.raises(ValueError, match='实际模型'):
        validate_resume_checkpoint(full_cpu_checkpoint, saved, 30000)


def test_policy_module_import_does_not_load_torch_or_launch_process():
    root = Path(__file__).resolve().parents[4]
    script = """import builtins,subprocess,sys
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','tensorflow','cupy','jax'}: raise AssertionError(name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
def denied(*args,**kwargs): raise AssertionError('process started')
subprocess.run=denied
subprocess.Popen=denied
from scripts.worldsim_v82.drivingeditor_instances.training_policy import resolve_training_policy
resolve_training_policy({'seed':0,'joint_background':False,'refine_poses_after':0,'pose_translation_limit_m':.5,
 'box_boundary_weight':1,'pose_prior_weight':.01,'max_features':10},{'iterations':30000})
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script], cwd=root, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
