"""CPU精确恢复合同；CUDA RNG只替代边界，不生成GPU恢复认证。"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import random
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.worldsim_v82.drivingeditor_instances.restore_contract import (
    RestoreContractError,
    assert_exact_state,
    verify_runtime_restore,
)
from scripts.worldsim_v82.drivingeditor_instances.rigid_model import RigidInstanceGaussianModel


def test_recursive_tensor_adam_numpy_and_metadata_exactness():
    state = {'parameter': torch.tensor([[1., 2.]]), 'optimizer': {'state': {0: {
        'step': torch.tensor(300.), 'exp_avg': torch.tensor([[.1, .2]]), 'exp_avg_sq': torch.tensor([[.01, .04]])}},
        'param_groups': [{'lr': .001, 'betas': (.9, .999), 'params': [0]}]},
        'lineage': np.array([7, 9], dtype=np.int64), 'metadata': (False, None, 'fixed')}
    counts = assert_exact_state(state, copy.deepcopy(state), path='background')
    assert counts['tensor_leaves'] == 4 and counts['numpy_leaves'] == 1
    bad = copy.deepcopy(state)
    bad['optimizer']['state'][0]['exp_avg'][0, 1] += 1e-6
    with pytest.raises(RestoreContractError, match=r'background.optimizer.state\[0\].exp_avg'):
        assert_exact_state(state, bad, path='background')


@pytest.mark.parametrize('left,right', [
    ({'lineage_version': 1}, {}),
    ({'mode': False}, {'mode': 0}),
    ((1, 2), [1, 2]),
    (torch.tensor([1], dtype=torch.float32), torch.tensor([1], dtype=torch.float64)),
    (torch.tensor([float('nan')]), torch.tensor([float('nan')])),
    (np.array([1], dtype=np.int32), np.array([1], dtype=np.int64)),
    (np.array([1.]), np.array([1. + 1e-12])),
])
def test_metadata_type_precision_dtype_and_nan_changes_fail_closed(left, right):
    with pytest.raises(RestoreContractError):
        assert_exact_state(left, right)


def test_requires_grad_is_not_silently_normalized():
    expected = torch.nn.Parameter(torch.tensor([1.]), requires_grad=False)
    changed = torch.nn.Parameter(torch.tensor([1.]), requires_grad=True)
    with pytest.raises(RestoreContractError, match='requires_grad'):
        assert_exact_state(expected, changed, path='background[1]')


@pytest.fixture
def runtime_pair():
    model = RigidInstanceGaussianModel([[.1, 0, 0]], [[.9, .1, .1]], [[.01]*3], [0], ['GMC'], [0.0],
                                         np.eye(4)[None, None], np.ones((1, 1), dtype=bool),
                                         sh_degree=1, box_size=np.ones((1, 1, 3)))
    optimizer = model.make_optimizer()
    model.evaluate(0.0).xyz.square().sum().backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    saved_instances = model.capture(optimizer)
    restored = RigidInstanceGaussianModel.from_capture(saved_instances)
    background = (torch.nn.Parameter(torch.tensor([.5]), requires_grad=False),
                  {'optimizer': {'state': {0: {'exp_avg': torch.tensor([.2]), 'step': torch.tensor(300.)}},
                                 'param_groups': [{'lr': .0001, 'params': [0]}]}, 'lineage': np.array([1])})
    environment = (torch.nn.Parameter(torch.tensor([.7]), requires_grad=False), {'lr': .01})
    sampler = random.Random(4)
    saved_cuda_rng = [torch.tensor([3, 4, 5], dtype=torch.uint8)]
    checkpoint = {'background': copy.deepcopy(background), 'instances': saved_instances,
                  'environment': copy.deepcopy(environment), 'sampler': sampler.getstate(),
                  'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
                  'torch_rng': torch.get_rng_state(), 'cuda_rng': copy.deepcopy(saved_cuda_rng),
                  'restoration_verified': False}
    # 保留真实Tensor/equal/CPU RNG，只替代本测试不能调用的CUDA RNG边界。
    torch_boundary = SimpleNamespace(Tensor=torch.Tensor, equal=torch.equal, get_rng_state=torch.get_rng_state,
                                     cuda=SimpleNamespace(get_rng_state_all=lambda: copy.deepcopy(saved_cuda_rng)))
    return (checkpoint, SimpleNamespace(capture=lambda: copy.deepcopy(background)), restored, restored.optimizer,
            SimpleNamespace(capture=lambda: copy.deepcopy(environment)), sampler, torch_boundary)


def test_runtime_proof_checks_core_adam_and_rng_without_modifying_checkpoint(runtime_pair):
    checkpoint, static, model, optimizer, environment, sampler, torch_boundary = runtime_pair
    old = copy.deepcopy(checkpoint)
    proof = verify_runtime_restore(checkpoint, static, model, optimizer, environment, sampler, torch_module=torch_boundary)
    assert proof['runtime_restoration_verified']
    assert proof['execution_device'] == 'cpu'
    assert proof['deterministic_next_step_parity_verified'] is False
    assert proof['checkpoint_restoration_verified_claim_unchanged'] is False
    assert set(proof['checked_fields']) == {'background', 'instances', 'environment', 'sampler', 'python_rng',
                                          'numpy_rng', 'torch_rng', 'cuda_rng'}
    assert_exact_state(old, checkpoint)
    assert not torch.cuda.is_initialized()
    json.dumps(proof)


def test_runtime_proof_reports_real_core_parameter_difference(runtime_pair):
    checkpoint, static, model, optimizer, environment, sampler, torch_boundary = runtime_pair
    with torch.no_grad():
        model._xyz[0, 0] += .001
    with pytest.raises(RestoreContractError, match=r'instances.state_dict._xyz'):
        verify_runtime_restore(checkpoint, static, model, optimizer, environment, sampler, torch_module=torch_boundary)


def test_runtime_proof_reports_real_adam_moment_difference(runtime_pair):
    checkpoint, static, model, optimizer, environment, sampler, torch_boundary = runtime_pair
    optimizer.state[model._xyz]['exp_avg'][0, 0] += .001
    with pytest.raises(RestoreContractError, match='exp_avg'):
        verify_runtime_restore(checkpoint, static, model, optimizer, environment, sampler, torch_module=torch_boundary)


@pytest.mark.parametrize('field', ['sampler', 'python_rng', 'numpy_rng', 'torch_rng', 'cuda_rng'])
def test_each_rng_stream_mismatch_is_detected(runtime_pair, field):
    checkpoint, static, model, optimizer, environment, sampler, torch_boundary = runtime_pair
    if field in ('sampler', 'python_rng'):
        checkpoint[field] = random.Random(99).getstate()
    elif field == 'numpy_rng':
        checkpoint[field] = np.random.RandomState(99).get_state()
    elif field == 'torch_rng':
        checkpoint[field] = checkpoint[field].clone()
        checkpoint[field][0] ^= 1
    else:
        checkpoint[field][0][0] ^= 1
    with pytest.raises(RestoreContractError, match=field):
        verify_runtime_restore(checkpoint, static, model, optimizer, environment, sampler, torch_module=torch_boundary)


def test_restore_contract_module_import_does_not_import_torch():
    script = """import builtins,sys
old=builtins.__import__
def guard(name,*args,**kwargs):
 if name.split('.')[0] in {'torch','tensorflow','cupy','jax'}: raise AssertionError(name)
 return old(name,*args,**kwargs)
builtins.__import__=guard
from scripts.worldsim_v82.drivingeditor_instances.restore_contract import verify_runtime_restore
assert 'torch' not in sys.modules
"""
    result = subprocess.run([sys.executable, '-X', 'utf8', '-c', script], cwd=Path(__file__).resolve().parents[4],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
