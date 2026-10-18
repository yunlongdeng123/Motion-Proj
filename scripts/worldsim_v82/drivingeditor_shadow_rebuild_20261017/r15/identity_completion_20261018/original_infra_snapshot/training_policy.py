"""续训合同：保存实际配置值，不导入Torch、不生成hash、不认证模型恢复。"""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
import random


POLICY_FIELD = 'training_policy'
POLICY_SCHEMA = 'drivingeditor_rigid_training_policy_v1'
CHECKPOINT_SCHEMA = 'drivingeditor_rigid_scene_v1'
_OPERATIONAL_ARGUMENTS = {'output', 'resume', 'spec', 'iterations', 'gpu_lock', 'run'}
_OPERATIONAL_REFERENCE_KEYS = {'source_path', 'model_path'}
_REQUIRED_POLICY_ARGUMENTS = {'seed', 'joint_background', 'refine_poses_after',
                              'pose_translation_limit_m', 'box_boundary_weight',
                              'pose_prior_weight', 'max_features'}
_CHECKPOINT_FIELDS = {'schema', 'step', 'spec', 'background', 'instances', 'environment',
                      'background_exclusion', 'sampler', 'python_rng', 'numpy_rng',
                      'torch_rng', 'cuda_rng'}
_RIGID_STATE_FIELDS = {'_xyz', '_features_dc', '_features_rest', '_scaling', '_rotation', '_opacity',
                        'point_ids', 'timestamps', 'base_poses', 'base_rotation', 'pose_valid', 'box_size',
                        'pose_translation_raw', 'pose_rotation_raw', 'edit_translation', 'edit_rotation',
                        'deleted', 'replacement_source'}


def _plain(value, label):
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(label + '必须是finite、可保存的实际JSON配置') from error


def _integer(value, label, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(label + '必须是合法整数')
    return value


def resolve_training_policy(arguments, raw_reference_config):
    """保留完整原始reference；只在比较视图中排除运行目录。

    iterations是停止预算，可以从300延长到30000。学习率horizon取原始
    reference.iterations，必须保持。以后新增非运行路径参数默认受合同约束。
    """
    args = dict(arguments) if isinstance(arguments, Mapping) else vars(arguments)
    missing = _REQUIRED_POLICY_ARGUMENTS - args.keys()
    if missing:
        raise ValueError('缺少实际训练参数: ' + ', '.join(sorted(missing)))
    if not isinstance(raw_reference_config, Mapping):
        raise ValueError('需要原始reference_config字典，不能使用重写路径后的运行配置')
    reference = _plain(dict(raw_reference_config), 'reference_config')
    _integer(reference.get('iterations'), 'reference学习率horizon', 1)
    policy_args = _plain({key: value for key, value in args.items() if key not in _OPERATIONAL_ARGUMENTS}, '训练参数')
    # 原短测CLI没有这个开关时，其实际默认也为False。
    policy_args.setdefault('geometry_only_smoke', False)
    _integer(policy_args['seed'], 'seed')
    _integer(policy_args['refine_poses_after'], 'refine_poses_after')
    _integer(policy_args['max_features'], 'max_features', 8)
    if policy_args['seed'] >= 2**32 or type(policy_args['joint_background']) is not bool:
        raise ValueError('seed/joint_background无效')
    for name in ('pose_translation_limit_m', 'box_boundary_weight', 'pose_prior_weight'):
        value = policy_args[name]
        if type(value) not in (int, float) or value < 0 or (name == 'pose_translation_limit_m' and value == 0):
            raise ValueError(name + '无效')
    return {'schema': POLICY_SCHEMA, 'arguments': policy_args, 'reference_config': reference,
            'adapter_constants': {'hybrid_lambda_dynamic_opacity': 0.0, 'hybrid_lambda_dynamic_rgb': 5.0,
                                  'visible_instance_opacity_weight': 100.0,
                                  'instance_opacity_domain': 'joint_visible_contribution_equal_known_instances_full_HxW',
                                  'xyz_lr_horizon': reference['iterations'], 'SH_raise_interval': 1000,
                                  'pose_optimizer_lr': 0.0001, 'vehicle_PVG_motion': False,
                                  'topology': 'fixed', 'new_SGCM_clones': False, 'anisotropic_split': False}}


def _comparison_view(policy):
    if not isinstance(policy, Mapping) or policy.get('schema') != POLICY_SCHEMA:
        raise ValueError('不支持的training_policy合同')
    for key in ('arguments', 'reference_config', 'adapter_constants'):
        if not isinstance(policy.get(key), Mapping):
            raise ValueError('training_policy缺少 ' + key)
    return _plain({'arguments': dict(policy['arguments']),
                   'reference_config': {key: value for key, value in policy['reference_config'].items()
                                        if key not in _OPERATIONAL_REFERENCE_KEYS},
                   'adapter_constants': dict(policy['adapter_constants'])}, 'training_policy')


def _different_fields(left, right, prefix=''):
    if isinstance(left, dict) and isinstance(right, dict):
        result = []
        for key in sorted(set(left) | set(right)):
            name = prefix + '.' + key if prefix else key
            if key not in left or key not in right:
                result.append(name)
            else:
                result.extend(_different_fields(left[key], right[key], name))
        return result
    return [] if left == right else [prefix]


def validate_training_policy(saved_policy, current_policy):
    changed = _different_fields(_comparison_view(saved_policy), _comparison_view(current_policy))
    if changed:
        raise ValueError('续训不能静默改变训练策略: ' + ', '.join(changed))


def _read_evidence(value, label):
    if isinstance(value, Mapping):
        return _plain(dict(value), label), {'kind': 'supplied_actual_mapping'}
    if value is None:
        raise ValueError('旧checkpoint缺少training_policy，需要实际 ' + label)
    path = Path(value)
    document = json.loads(path.read_text(encoding='utf-8-sig'))
    if not isinstance(document, dict):
        raise ValueError(label + '必须是实际JSON对象')
    return document, {'kind': 'actual_json_file', 'path': str(path.resolve())}


def validate_resume_checkpoint(checkpoint, current_policy, target_iterations, *, expected_step=None,
                               legacy_effective_config=None, legacy_reference_config=None,
                               legacy_reference_provenance=None, max_total_iterations=30000):
    """检查完整字段、预算及固定策略，返回可记录的receipt。

    旧短测没有原始配置快照：从其实际effective_config和指定reference文件
    重建策略，并明确记录这个证据边界。即使本函数通过，也不是GPU恢复认证。
    """
    if not isinstance(checkpoint, Mapping) or checkpoint.get('schema') != CHECKPOINT_SCHEMA:
        raise ValueError('不支持的完整场景checkpoint')
    missing = (_CHECKPOINT_FIELDS - checkpoint.keys()) | {key for key in _CHECKPOINT_FIELDS if key in checkpoint and checkpoint[key] is None}
    if missing:
        raise ValueError('完整checkpoint缺少状态: ' + ', '.join(sorted(missing)))
    step = _integer(checkpoint['step'], 'checkpoint step')
    target = _integer(target_iterations, '续训总步数', 1)
    maximum = _integer(max_total_iterations, '授权总步数上限', 1)
    if target <= step or target > maximum:
        raise ValueError('续训必须超过已保存step，且不得超过授权总步数上限')
    if expected_step is not None and step != _integer(expected_step, '预期恢复step'):
        raise ValueError('checkpoint step与预期短测终点不符')
    instances = checkpoint['instances']
    if (not isinstance(instances, Mapping) or not isinstance(instances.get('state_dict'), Mapping)
            or not isinstance(instances.get('optimizer'), Mapping) or instances.get('version') != 1
            or not isinstance(instances.get('config'), Mapping) or not instances.get('instance_ids')
            or _RIGID_STATE_FIELDS - instances['state_dict'].keys()):
        raise ValueError('实例checkpoint必须含实际模型和Adam状态')
    optimizer = instances['optimizer']
    if optimizer.get('class') not in ('Adam', 'AdamW') or not optimizer.get('groups') or not isinstance(optimizer.get('state_dict'), Mapping):
        raise ValueError('实例Adam状态不完整')
    if ('state' not in optimizer['state_dict'] or not optimizer['state_dict'].get('param_groups')
            or len(optimizer['state_dict']['param_groups']) != len(optimizer['groups'])):
        raise ValueError('实例Adam参数组/state不完整')
    if not isinstance(checkpoint['spec'], Mapping) or not isinstance(checkpoint['background_exclusion'], Mapping):
        raise ValueError('checkpoint真实场景/背景排除绑定不完整')
    if not isinstance(checkpoint['cuda_rng'], (list, tuple)) or (step > 0 and not checkpoint['cuda_rng']):
        raise ValueError('GPU训练checkpoint缺少CUDA RNG状态')
    if not isinstance(checkpoint['numpy_rng'], (list, tuple)) or len(checkpoint['numpy_rng']) != 5:
        raise ValueError('NumPy全局RNG状态不完整')
    for name in ('sampler', 'python_rng'):
        try:
            random.Random(0).setstate(checkpoint[name])
        except (TypeError, ValueError) as error:
            raise ValueError(name + '状态无效') from error

    if POLICY_FIELD in checkpoint:
        saved_policy = checkpoint[POLICY_FIELD]
        evidence = {'kind': 'policy_saved_in_checkpoint', 'historical_config_in_checkpoint': True}
    else:
        effective, effective_source = _read_evidence(legacy_effective_config, 'parent effective_config.json')
        if effective.get('spec') != checkpoint['spec'] or not isinstance(effective.get('arguments'), dict):
            raise ValueError('旧effective_config不属于这个真实checkpoint场景绑定')
        previous_budget = _integer(effective['arguments'].get('iterations'), '旧短测停止预算', 1)
        if step > previous_budget:
            raise ValueError('checkpoint step超过实际旧运行预算')
        if POLICY_FIELD in effective:
            saved_policy = effective[POLICY_FIELD]
            reference_source = {'kind': 'resolved_policy_in_parent_effective_config'}
        else:
            reference, reference_source = _read_evidence(legacy_reference_config, '旧运行reference_config')
            saved_policy = resolve_training_policy(effective['arguments'], reference)
        evidence = {'kind': 'legacy_effective_config_reconstruction', 'effective_config': effective_source,
                    'reference_config': reference_source, 'historical_config_in_checkpoint': False,
                    'historical_reference_contents_certified_by_this_check': False,
                    'provided_reference_provenance': _plain(legacy_reference_provenance or {}, '旧reference证据')}
    validate_training_policy(saved_policy, current_policy)
    return {'step': step, 'first_step': step + 1, 'target_iterations': target,
            POLICY_FIELD: _plain(dict(saved_policy), '已绑定training_policy'),
            'policy_evidence': evidence, 'restoration_verified': False,
            'verification_scope': 'saved_policy_and_checkpoint_fields_only_not_tensor_or_GPU_restore'}
