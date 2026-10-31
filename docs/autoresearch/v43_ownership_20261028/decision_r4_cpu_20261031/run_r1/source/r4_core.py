"""R4 CPU 数据合同：合法证据、固定小域、跨操作共享路由。

本模块不导入光栅器、不读取独立编辑参考、不产生教师标签。
"""
from copy import deepcopy
from hashlib import sha256
from itertools import product
import json
import math

OPERATIONS = ('none', 'delete', 'shift', 'rotate')
METRICS = ('f_geom', 'f_view', 'f_bg', 'f_motion')
LEGAL_ROLES = {'simulator_sensor_oracle', 'reviewed_measured_real_observations'}
FORBIDDEN_KEYS = {'semantic_id', 'surface_id', 'independent_gt', 'edited_rgb',
                  'student_output', 'student_prediction', 'capability_bin',
                  'evaluation_only_capability_bin', 'reference_loss'}


def canonical_hash(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def reject_privileged(value):
    if isinstance(value, dict):
        if FORBIDDEN_KEYS.intersection(value):
            raise ValueError('特权/学生字段不得进入合法证据')
        for child in value.values():
            reject_privileged(child)
    elif isinstance(value, list):
        for child in value:
            reject_privileged(child)


def validate_legal_evidence(evidence):
    reject_privileged(evidence)
    if evidence.get('role') not in LEGAL_ROLES:
        raise ValueError('未登记的合法观测角色')
    if set(evidence['metrics']) != set(METRICS):
        raise ValueError('资格指标集合不完整')
    if set(evidence.get('operation_view',{}))!=set(OPERATIONS) or set(evidence.get('operation_bg',{}))!=set(OPERATIONS):
        raise ValueError('逐操作合法指标集合不完整')
    rows=list(evidence['metrics'].items())
    rows.extend(('operation_view:'+q,row) for q,row in evidence['operation_view'].items())
    rows.extend(('operation_bg:'+q,row) for q,row in evidence['operation_bg'].items())
    for name, row in rows:
        denominator = row.get('denominator')
        numerator = row.get('numerator')
        if denominator is not None and (type(denominator) is not int or denominator < 0):
            raise ValueError('分母必须是非负整数或null')
        if numerator is not None and (type(numerator) not in (int,float) or not math.isfinite(numerator) or numerator < 0):
            raise ValueError('分子无效')
        if numerator is not None and denominator is not None and numerator > denominator:
            raise ValueError('分子超过分母')
    if not evidence['legal_observation_ids']:
        raise ValueError('缺少合法观测身份')


def metric_value(row):
    if row.get('numerator') is None or not row.get('denominator'):
        return None
    return row['numerator'] / row['denominator']


def eligibility(evidence, operation, thresholds):
    """资格只读白名单证据；未校准或缺测不能默认通过。"""
    if operation not in OPERATIONS:
        raise ValueError('非法用户操作')
    validate_legal_evidence(evidence)
    if set(thresholds) != set(METRICS):
        raise ValueError('阈值集合不完整')
    features = deepcopy(evidence['metrics'])
    features['f_view'] = deepcopy(evidence['operation_view'][operation])
    features['f_bg'] = deepcopy(evidence['operation_bg'][operation])
    reasons, outcomes = [], []
    for name, row in features.items():
        value = metric_value(row)
        row['value'] = value
        threshold = thresholds[name]
        if threshold is not None and (type(threshold) not in (int,float) or not math.isfinite(threshold) or not 0<=threshold<=1):
            raise ValueError('资格阈值必须在[0,1]内或null')
        if threshold is None:
            outcomes.append(None)
            reasons.append(name + ':threshold_uncalibrated')
        elif value is None:
            outcomes.append(None)
            reasons.append(name + ':measurement_unknown_or_zero_denominator')
        else:
            result = value >= threshold
            outcomes.append(result)
            if not result:
                reasons.append(name + ':below_locked_threshold')
    passed = False if False in outcomes else None if None in outcomes else True
    return dict(version='eligibility_r4_cpu_draft_v1', preedit_only=True,
                passed=passed, features=features, thresholds=deepcopy(thresholds),
                exclusion_reasons=reasons, calibration_status='not_calibrated',
                observation_role=evidence['role'],
                legal_observation_ids=evidence['legal_observation_ids'],
                missing_lidar_policy='unknown_secondary_registration_only',
                native_full_scene_coverage_measured=False)


def validate_candidate(candidate):
    if candidate.get('schema') != 'r4_frozen_group_candidates_v1':
        raise ValueError('候选schema无效')
    if candidate.get('legal_input_role') not in LEGAL_ROLES:
        raise ValueError('候选不属于登记合法输入角色')
    n = candidate['static_row_count']
    seen, group_ids = set(), set()
    for group in candidate['groups']:
        gid = group['group_id']
        if gid in group_ids:
            raise ValueError('重复来源组')
        group_ids.add(gid)
        if group['original_binding'] != 'static' or group['rebind_target'] != candidate['target_id']:
            raise ValueError('当前小域只允许静态到登记目标的重绑定')
        rows = group['original_static_rows']
        if not rows or rows != sorted(rows) or len(set(rows)) != len(rows):
            raise ValueError('来源组必须有递增、唯一原始行')
        if any(type(i) is not int or i < 0 or i >= n or i in seen for i in rows):
            raise ValueError('原始行越界、重叠或类型无效')
        seen.update(rows)
    if len(candidate['groups']) > 3:
        raise ValueError('本次冻结域最多3来源组')
    return len(seen)


def freeze_family(candidate):
    count = validate_candidate(candidate)
    return dict(schema='r4_frozen_H0_group_binary_v1', candidate_hash=canonical_hash(candidate),
                target_id=candidate['target_id'],
                row_namespace=candidate['row_namespace'],
                scene_ancestor=candidate['scene_ancestor'],
                fixed_representation=True, group_ids=[g['group_id'] for g in candidate['groups']],
                choices=['keep', 'rebind'], candidate_row_count=count,
                possible_count=2 ** len(candidate['groups']),
                outside_candidate_rows='fixed_original_binding',
                native_dynamic_rows='fixed_original_binding',
                defer_role='query_layer_only',
                route_does_not_read_user_operation=True,
                rendering_backend='inherited_V42_native_pending_GPU_gate',
                scientific_scope='declared_local_group_domain_only_not_global_optimum')


def shared_route(candidate, choices):
    """纯场景路由不接受q；不得用逐操作选择替换共享状态。"""
    validate_candidate(candidate)
    group_ids = [g['group_id'] for g in candidate['groups']]
    if set(choices) != set(group_ids) or any(v not in ('keep', 'rebind') for v in choices.values()):
        raise ValueError('动作必须完整且仅含keep/rebind，defer不是Gaussian绑定')
    route = dict(schema='r4_shared_route_v1', candidate_hash=canonical_hash(candidate),
                 row_namespace=candidate['row_namespace'], target_id=candidate['target_id'],
                 assignments=[dict(group_id=gid, action=choices[gid],
                                   binding='static' if choices[gid]=='keep' else candidate['target_id'])
                              for gid in group_ids],
                 geometry_color_opacity_count_environment_frozen=True)
    route['route_assignment_hash'] = canonical_hash(route)
    return route


def enumerate_routes(candidate):
    family = freeze_family(candidate)
    gids = family['group_ids']
    return [shared_route(candidate, dict(zip(gids, values)))
            for values in product(('keep', 'rebind'), repeat=len(gids))]


def execution_plan(route, operation, *, defer=False):
    """仅执行器读用户操作；此CPU结果是计划，不是原生渲染收据。"""
    if operation not in OPERATIONS or type(defer) is not bool:
        raise ValueError('非法用户操作或defer类型')
    if route.get('schema') != 'r4_shared_route_v1':
        raise ValueError('非法共享路由')
    sealed = dict(route)
    route_hash = sealed.pop('route_assignment_hash')
    if canonical_hash(sealed) != route_hash:
        raise ValueError('共享路由已被修改')
    return dict(operation=operation, route_assignment_hash=route_hash,
                status='deferred' if defer else 'cpu_plan_native_execution_pending',
                rendered=False, task_completed=False, loss_vector=None,
                none_semantics='must_render_actual_modified_route_not_checkpoint_bypass',
                translation_world_m=[0,1.5,0] if operation=='shift' else None,
                yaw_delta_deg=30.0 if operation=='rotate' else None)


def register_tasks(scene_id, candidate, evidence, routes, checkpoint_hash, frozen_scene_context):
    family = freeze_family(candidate)
    if checkpoint_hash!=candidate['checkpoint_static_sha256']:
        raise ValueError('检查点与候选冻结域不一致')
    components=frozen_scene_context.get('component_sha256',{})
    if set(components)!={'static','dynamic','environment','training_state'} or components['static']!=checkpoint_hash:
        raise ValueError('场景冻结必须包含一致的静态/动态/环境/运行状态')
    expected_bundle=canonical_hash([dict(name=name,sha256=components[name]) for name in sorted(components)])
    if frozen_scene_context.get('checkpoint_bundle_sha256')!=expected_bundle:
        raise ValueError('完整场景bundle身份不匹配')
    if any(not frozen_scene_context.get(key) for key in ('event_sha256','gauge_sha256','camera_sha256','preparation_sha256')):
        raise ValueError('缺少配置/时间、gauge、相机或原准备合同的冻结身份')
    expected=enumerate_routes(candidate)
    if len(routes)!=family['possible_count'] or len({canonical_hash(r) for r in routes})!=len(routes):
        raise ValueError('声明完整枚举时不得缺动作或重复动作')
    if {canonical_hash(r) for r in routes}!={canonical_hash(r) for r in expected}:
        raise ValueError('路由不属于此checkpoint/candidate/target的完整冻结域')
    context_hash = canonical_hash(dict(candidate_hash=family['candidate_hash'],
                                       legal_evidence_hash=canonical_hash(evidence),
                                       frozen_scene_context=frozen_scene_context,
                                       family=family))
    rows = []
    for operation in OPERATIONS:
        action_rows=[]
        for i, route in enumerate(routes):
            plan = execution_plan(route, operation)
            action_rows.append(dict(action_id='h%03d'%i,
                route_assignment_hash=plan['route_assignment_hash'], rendered=False,
                loss_vector=None, valid_region_denominators={}, constraint_pass=None,
                runtime=None))
        rows.append(dict(scene_id=scene_id, scene_ancestor=candidate['scene_ancestor'],
            asset_id=candidate['target_id'], checkpoint_sha256=checkpoint_hash,
            checkpoint_bundle_sha256=expected_bundle,
            frozen_scene_context=deepcopy(frozen_scene_context),
            candidate_hash=family['candidate_hash'], shared_route_context_hash=context_hash,
            query_id=scene_id+':'+operation, edit_type=operation,
            split='development_existing_seen_ancestor',
            allowed_family='H0_fixed_representation_group_keep_rebind_query_defer',
            legal_feature_refs=evidence['legal_observation_ids'],
            eligibility=eligibility(evidence,operation,{m:None for m in METRICS}),
            action_evaluations=action_rows, reference_role='none',
            search=dict(exhaustive_over_frozen_H0=False, action_domain_enumeration_complete=True,
                enumerated_count=len(routes), evaluated_count=0, possible_count=family['possible_count'],
                best_found_action_id=None, oracle_status='safe_oracle_unavailable'),
            decision=dict(selected_action_id=None, deferred=None, decision_gap_kind='unavailable',
                decision_gap=None, representation_capability_eval_only='unknown'),
            representation_evaluation=dict(best_verified_residual=None,capability_bin='C?'),
            native_execution_status='E0_GPU_gate_pending', human_verdict=None))
    return rows
