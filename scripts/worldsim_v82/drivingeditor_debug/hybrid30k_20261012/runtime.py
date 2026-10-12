"""配对相机采样、RNG 安全评价与轻量日志；不在导入时初始化 CUDA。"""
import contextlib
import json
from pathlib import Path
import random


def camera_id(camera):
    return Path(camera.image_path).parent.name + '/' + camera.image_name


class CameraSampler:
    """保留无放回相机与旧 time_shift 分布，独立于拓扑更新 RNG。"""
    def __init__(self, seed, cameras):
        import numpy as np
        self.ids = sorted(camera_id(c) for c in cameras)
        if len(self.ids) != 49 or len(set(self.ids)) != len(self.ids):
            raise ValueError('实验只允许固定49个FRONT观测视图')
        self.python = random.Random(seed)
        self.numpy = np.random.RandomState(seed)
        self.stack = []
        self.steps = 0

    def select(self, cameras, probability, time_interval):
        lookup = {camera_id(c): c for c in cameras}
        if sorted(lookup) != self.ids:
            raise ValueError('训练相机集合发生变化')
        if not self.stack:
            self.stack = list(self.ids)
        key = self.stack.pop(self.python.randrange(len(self.stack)))
        shift = (3 * (self.numpy.random_sample() - .5) * time_interval
                 if self.numpy.random_sample() < probability else None)
        self.steps += 1
        return lookup[key], shift

    def capture(self):
        return {'schema_version': 1, 'ids': self.ids, 'stack': list(self.stack),
                'steps': self.steps, 'python_rng': self.python.getstate(),
                'numpy_rng': self.numpy.get_state()}

    def restore(self, state):
        required = {'schema_version', 'ids', 'stack', 'steps', 'python_rng', 'numpy_rng'}
        if not isinstance(state, dict) or set(state) != required:
            raise ValueError('完整且精确的配对采样状态缺失')
        if type(state['schema_version']) is not int or state['schema_version'] != 1 or state['ids'] != self.ids:
            raise ValueError('配对采样器恢复合同不匹配')
        steps = state['steps']
        if type(steps) is not int or steps < 0:
            raise ValueError('采样步数必须是非负整数')
        if (not isinstance(state['stack'], list) or len(state['stack']) != len(set(state['stack']))
                or not set(state['stack']) <= set(self.ids) or len(state['stack']) != (-steps) % len(self.ids)):
            raise ValueError('相机采样栈无效')
        import numpy as np
        python = random.Random()
        numpy = np.random.RandomState()
        python.setstate(state['python_rng'])
        numpy.set_state(state['numpy_rng'])
        self.stack = list(state['stack'])
        self.steps = steps
        self.python, self.numpy = python, numpy


@contextlib.contextmanager
def preserve_rng():
    import numpy as np
    import torch
    python = random.getstate()
    numpy = np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    try:
        yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)
        torch.set_rng_state(cpu)
        if cuda is not None:
            torch.cuda.set_rng_state_all(cuda)


def append_json(path, record):
    path = Path(path)
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')


def record_step(args, iteration, camera, time_shift, gaussians, dynamic_gaussians, loss_log):
    learning_rates = {}
    for name, model in (('static', gaussians), ('dynamic', dynamic_gaussians)):
        if model is not None:
            learning_rates[name] = {g['name']: float(g['lr']) for g in model.optimizer.param_groups}
    append_json(Path(args.model_path) / 'paired_steps.jsonl', {
        'iteration': int(iteration), 'frame_id': camera_id(camera),
        'time_shift': None if time_shift is None else float(time_shift),
        'learning_rates': learning_rates, 'losses': loss_log})


def record_gradient(args, iteration, diagnostic):
    append_json(Path(args.model_path) / 'gradient_diagnostics.jsonl',
                {'iteration': int(iteration), 'diagnostic': diagnostic})


def validate_config(config):
    if config.get('hybrid_method') is not True or config.get('paper_method') is not True:
        raise ValueError('必须使用新hybrid目标与完整检查点基础设施')
    if config.get('iterations') != 30000 or config.get('position_lr_max_steps') != 30000:
        raise ValueError('训练与学习率预算必须均为30k')
    if config.get('start_checkpoint') is not None or config.get('seed') != 0:
        raise ValueError('本矩阵必须从零、seed0；不得复用旧30k/50k状态')
    for name in ('checkpoint_iterations', 'save_iterations', 'test_iterations'):
        if list(config[name]) != [10000, 20000, 30000]:
            raise ValueError(name + '必须为10k/20k/30k')
    if config.get('eval') is not False or list(config.get('resolution_scales', [])) != [4]:
        raise ValueError('固定49训练视图与scale4；不伪造独立测试划分')
    for name in ('hybrid_nps_enabled', 'paper_sgcm_enabled', 'sgcm_guard_enabled'):
        if type(config.get(name)) is not bool:
            raise ValueError(name + '必须是布尔值')
    if not config['sgcm_guard_enabled'] or config.get('lidar_decay') != 1.0:
        raise ValueError('修复后的SGCM及原深度衰减必须保留')
    if config.get('hybrid_opacity_reset_policy') != 'densification_only':
        raise ValueError('四臂共同reset修复未锁定')
    weights = {'hybrid_lambda_dynamic_rgb': 5.0, 'hybrid_lambda_sky_rgb': 5.0,
               'hybrid_lambda_static_rgb': 1.0, 'hybrid_lambda_dynamic_opacity': 100.0,
               'hybrid_time_reg_eps': 1e-8, 'paper_lambda_nps': .01,
               'paper_lambda_sky': .05, 'paper_lambda_depth': .1,
               'lambda_t_reg': .1, 'lambda_v_reg': .01, 'lambda_inv_depth': .1,
               'lambda_v_smooth': .1, 'lambda_opacity_entropy': .05, 'lambda_dssim': .2,
               'lidar_decay': 1.0}
    for name, value in weights.items():
        if type(config.get(name)) is not type(value) or config[name] != value:
            raise ValueError('固定目标数值/类型未锁定: ' + name)
