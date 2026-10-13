"""首次更新前逐元素检查完整运行恢复；不推断下一步CUDA计算确定性。"""
from __future__ import annotations

from collections.abc import Mapping
import random


class RestoreContractError(RuntimeError):
    def __init__(self, path, reason):
        self.path, self.reason = path, reason
        super().__init__('完整恢复合同不符: ' + path + ' (' + reason + ')')


def _libraries(torch_module, numpy_module):
    if torch_module is None:
        import torch
        torch_module = torch
    if numpy_module is None:
        import numpy
        numpy_module = numpy
    return torch_module, numpy_module


def _child_path(path, key):
    return path + '.' + key if isinstance(key, str) and key.isidentifier() else path + '[' + repr(key) + ']'


def assert_exact_state(expected, actual, *, path='state', torch_module=None, numpy_module=None):
    """检查每个字段；仅tensor的存储设备可因checkpoint搬运而不同。

    参数、Adam和RNG的dtype、shape、layout及值必须完全相同。比较使用CPU
    副本，不修改原数据；NaN也会失败。metadata差异不得静默忽略。
    """
    torch, np = _libraries(torch_module, numpy_module)
    counts = {'tensor_leaves': 0, 'numpy_leaves': 0, 'scalar_leaves': 0,
              'mapping_nodes': 0, 'sequence_nodes': 0}

    def fail(where, reason):
        raise RestoreContractError(where, reason)

    def compare(left, right, where):
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
                fail(where, 'tensor/type mismatch')
            if left.dtype != right.dtype or left.shape != right.shape or left.layout != right.layout:
                fail(where, 'tensor dtype/shape/layout mismatch')
            if left.requires_grad != right.requires_grad:
                fail(where, 'tensor requires_grad mismatch')
            if not torch.equal(left.detach().cpu(), right.detach().cpu()):
                fail(where, 'tensor values differ')
            counts['tensor_leaves'] += 1
            return
        if isinstance(left, (np.ndarray, np.generic)) or isinstance(right, (np.ndarray, np.generic)):
            if type(left) is not type(right) or left.dtype != right.dtype or left.shape != right.shape:
                fail(where, 'NumPy type/dtype/shape mismatch')
            if not np.array_equal(left, right):
                fail(where, 'NumPy values differ')
            counts['numpy_leaves'] += 1
            return
        if isinstance(left, Mapping) or isinstance(right, Mapping):
            if type(left) is not type(right):
                fail(where, 'mapping type mismatch')
            if set(left) != set(right):
                missing = sorted(set(left) - set(right), key=repr)
                extra = sorted(set(right) - set(left), key=repr)
                fail(where, 'mapping keys differ; missing=' + repr(missing) + ', extra=' + repr(extra))
            counts['mapping_nodes'] += 1
            for key in left:
                compare(left[key], right[key], _child_path(where, key))
            return
        if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
            if type(left) is not type(right) or len(left) != len(right):
                fail(where, 'sequence type/length mismatch')
            counts['sequence_nodes'] += 1
            for index, (a, b) in enumerate(zip(left, right)):
                compare(a, b, where + '[' + str(index) + ']')
            return
        if type(left) is not type(right) or left != right:
            fail(where, 'scalar type/value mismatch')
        counts['scalar_leaves'] += 1

    compare(expected, actual, path)
    return counts


def verify_runtime_restore(checkpoint, static, instances, optimizer, environment, sampler, *,
                           torch_module=None, numpy_module=None):
    """在load_models及所有RNG恢复后、LR调整和第一次backward之前调用。"""
    torch, np = _libraries(torch_module, numpy_module)
    required = ('background', 'instances', 'environment', 'sampler', 'python_rng',
                'numpy_rng', 'torch_rng', 'cuda_rng')
    for key in required:
        if key not in checkpoint:
            raise RestoreContractError('checkpoint.' + key, 'required state missing')
    runtime = {'background': static.capture(), 'instances': instances.capture(optimizer),
               'environment': environment.capture(), 'sampler': sampler.getstate(),
               'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
               'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all()}
    counts = {}
    for key in required:
        counts[key] = assert_exact_state(checkpoint[key], runtime[key], path=key,
                                        torch_module=torch, numpy_module=np)
    return {'runtime_restoration_verified': True,
            'scope': 'runtime_model_Adam_and_RNG_tensors_restored_before_first_update',
            'execution_device': str(instances.local_xyz.device),
            'comparison': 'exact_keys_types_dtype_shape_layout_requires_grad_and_values; tensor_device_transport_allowed',
            'checked_fields': list(required), 'comparison_counts': counts,
            'checkpoint_restoration_verified_claim_unchanged': checkpoint.get('restoration_verified', False),
            'deterministic_next_step_parity_verified': False,
            'render_quality_or_edit_acceptance_verified': False}
