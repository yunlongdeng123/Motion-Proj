"""Runtime observers preserve genuine training math and reject altered restore."""
import inspect
import math
from pathlib import Path
import random

class ReadOnlyLoaderNumpy:
    def __init__(self, numpy_module, source):
        self.module, self.source = numpy_module, Path(source).resolve()
        self.checked = []

    def __getattr__(self, name):
        return getattr(self.module, name)

    def savetxt(self, filename, values, *args, **kwargs):
        path = Path(filename).resolve()
        if path != self.source / 'transform_matrix.txt' or not path.is_file():
            raise RuntimeError('Unexpected source-data write during resume: ' + str(path))
        existing = self.module.loadtxt(path)
        if not self.module.allclose(values, existing, rtol=1e-12, atol=1e-12):
            raise RuntimeError('Recomputed genuine loader normalization differs from original input matrix')
        self.checked.append(str(path))

def optimizer_receipt(models):
    result = {}
    for branch, model in models.items():
        if model is None:
            continue
        groups = []
        for group in model.optimizer.param_groups:
            parameter = group['params'][0]
            state = model.optimizer.state[parameter]
            step = float(state['step'])
            if not math.isfinite(step):
                raise ValueError('Nonfinite conditional Adam step: ' + branch)
            groups.append({'name': group.get('name', 'base'), 'step': step, 'lr': float(group['lr']),
                           'parameter_shape': list(parameter.shape), 'parameter_dtype': str(parameter.dtype),
                           'parameter_device': str(parameter.device), 'step_device': str(state['step'].device),
                           'runtime_options': {k: v for k, v in group.items() if k not in ('params', 'name')}})
        result[branch] = groups
    return result

def install_restore_observer(guard, state, receipt, torch, np, expected_iteration):
    original = guard.restore_training_state

    def restore(path, iteration, scene, models, args, sgcm=None):
        caller = inspect.currentframe().f_back
        stack, ema = original(path, iteration, scene, models, args, sgcm)
        if iteration != expected_iteration or stack != state['viewpoint_stack'] or ema != state['ema']:
            raise ValueError('Real continuation sampler/EMA differs from saved requested-iteration state')
        if scene.scale_index != state['scale_index']:
            raise ValueError('Real scale differs after resume')
        for order in state['camera_orders']:
            if [guard.camera_id(c) for c in scene.train_cameras[order['scale']]] != order['ids']:
                raise ValueError('Real training-camera order differs after resume')
        if random.getstate() != state['python_rng']:
            raise ValueError('Real Python RNG restore differs')
        actual_np = np.random.get_state()
        expected_np = state['numpy_rng']
        if actual_np[0] != expected_np[0] or not np.array_equal(actual_np[1], expected_np[1]) or actual_np[2:] != expected_np[2:]:
            raise ValueError('Real NumPy RNG restore differs')
        if not torch.equal(torch.get_rng_state(), state['torch_rng']):
            raise ValueError('Real Torch CPU RNG restore differs')
        actual_cuda = torch.cuda.get_rng_state_all()
        if len(actual_cuda) != len(state['cuda_rng']) or any(not torch.equal(a, b) for a, b in zip(actual_cuda, state['cuda_rng'])):
            raise ValueError('Real CUDA RNG restore differs')
        for branch, attributes in state['model_runtime'].items():
            if any(getattr(models[branch], key) != value for key, value in attributes.items()):
                raise ValueError('Real model runtime flags differ after resume: ' + branch)
        for key in ('model_scale', 'plane_model', 'model_to_anchor', 'regions_anchor', 'metadata'):
            actual, expected = getattr(sgcm, key), state['sgcm'][key]
            equal = actual == expected if key in ('model_scale', 'metadata') else np.array_equal(actual, expected)
            if not equal:
                raise ValueError('Real saved SGCM continuation differs: ' + key)
        captures = {'static': caller.f_locals['model_params'], 'dynamic': caller.f_locals['dynamic_params'],
                    'environment': caller.f_locals['light_params']}
        gaussian_fields = [(1, '_xyz'), (2, '_features_dc'), (3, '_features_rest'), (4, '_scaling'),
                           (5, '_rotation'), (6, '_opacity'), (7, '_t'), (8, '_scaling_t'), (9, '_velocity'),
                           (10, 'max_radii2D'), (11, 'xyz_gradient_accum'), (12, 't_gradient_accum'), (13, 'denom')]
        for branch, capture in captures.items():
            model = models[branch]
            if branch == 'environment':
                if model.base is not capture[0]:
                    raise ValueError('Environment restore replaced saved parameter')
                saved_optimizer = capture[1]
            else:
                if any(getattr(model, key) is not capture[index] for index, key in gaussian_fields):
                    raise ValueError('Gaussian restore lost saved tensors: ' + branch)
                saved_optimizer = capture[14]
            if len(saved_optimizer['param_groups']) != len(model.optimizer.param_groups):
                raise ValueError('Adam group count differs: ' + branch)
            for group, saved_group in zip(model.optimizer.param_groups, saved_optimizer['param_groups']):
                if len(group['params']) != 1:
                    raise ValueError('Unexpected Adam group layout')
                parameter = group['params'][0]
                restored = model.optimizer.state[parameter]
                saved = saved_optimizer['state'][saved_group['params'][0]]
                if set(restored) != set(saved) or any(restored[k].dtype != saved[k].dtype or restored[k].device != saved[k].device or not torch.equal(restored[k], saved[k]) for k in saved):
                    raise ValueError('Real Adam conditional steps/moments/device placement changed: ' + branch)
                current_options = {k: v for k, v in group.items() if k != 'params'}
                saved_options = {k: v for k, v in saved_group.items() if k != 'params'}
                if current_options != saved_options or parameter.dtype != torch.float32:
                    raise ValueError('Real Adam runtime options or parameter precision changed: ' + branch)
            if branch == 'environment' and not any(p is model.base for g in model.optimizer.param_groups for p in g['params']):
                raise ValueError('Environment Adam optimizer is not bound to restored base')
        rng = random.Random()
        rng.setstate(state['python_rng'])
        pending = list(stack) or list(range(len(scene.getTrainCameras())))
        next_index = pending.pop(rng.randint(0, len(pending) - 1))
        receipt.update(status='actual_restore_pass', restored_iteration=iteration,
                       all_rng_restored_exactly=True, camera_sampler_scale_ema_restored_exactly=True,
                       model_runtime_sgcm_restored_exactly=True, conditional_adam_restored_exactly=True,
                       saved_optimizer_tensor_placement_preserved=True, optimizer_bound_to_saved_parameters=True,
                       optimizer_before_next_step=optimizer_receipt(models), next_camera_index=next_index,
                       next_camera_id=guard.camera_id(scene.getTrainCameras()[next_index]),
                       next_remaining_stack=pending, seed_after_restore=False)
        return stack, ema

    guard.restore_training_state = restore
    return original
