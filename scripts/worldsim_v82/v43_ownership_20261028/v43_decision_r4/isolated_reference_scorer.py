"""R4 独立开发评分端：读取原生硬路由结果，绝不构建候选或训练模型。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                             allow_nan=False, separators=(',', ':')).encode()).hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')


def normalized_frame(value):
    text = str(value)
    return text.zfill(6) if text.isdigit() else text


def resolve_file(root, value):
    value = Path(value)
    return value if value.is_absolute() else root / value


def mean(values):
    return float(np.mean(values, dtype=np.float64))


def image_array(path):
    return np.asarray(Image.open(path).convert('RGB'), dtype=np.float32).transpose(2, 0, 1) / 255


def package(path):
    """原生浮点合成自洽检查；范围检查不会把 HDR RGB 上裁剪。"""
    with np.load(path, allow_pickle=False) as source:
        names = ['alpha', 'E', 'fill', 'final', 'white']
        values = {key: source[key].copy() for key in names}
        values['C'] = source['C_G'].copy() if 'C_G' in source else source['C'].copy()
    final = values['final']
    if final.dtype != np.float32 or final.ndim != 3 or final.shape[0] != 3:
        raise ValueError('要求原生 FP32 CHW RGB: ' + str(path))
    shape = final.shape
    alpha = values['alpha']
    if alpha.ndim == 2:
        alpha = alpha[None]
    values['alpha'] = alpha
    if alpha.shape != (1, shape[1], shape[2]):
        raise ValueError('alpha维度不符')
    if any(not np.isfinite(a).all() for a in values.values()):
        raise ValueError('原生分量出现非有限值')
    if any(values[k].shape != shape for k in ('C', 'E', 'fill', 'white')):
        raise ValueError('RGB分量形状不符')
    if float(alpha.min()) < -1e-6 or float(alpha.max()) > 1 + 1e-6:
        raise ValueError('alpha超出原生合成允许范围')
    errors = dict(fill=float(np.max(np.abs(values['fill'] - (1 - alpha) * values['E']))),
                  final=float(np.max(np.abs(final - values['C'] - values['fill']))),
                  white=float(np.max(np.abs(values['white'] - values['C'] - (1 - alpha)))))
    if max(errors.values()) > 3e-6:
        raise ValueError('原生合成身份不符: ' + str(errors))
    return final, errors


def legal_mask_semantic_gate(manifest, frozen):
    """独立于资格/候选实现检查PNG颜色语义；透明度不是车身标签。"""
    if not manifest['scene_id'].startswith('sim_'):
        return dict(status='not_applicable_sim_RGBA_guard_real_uses_reviewed_boolean_NPZ')
    event_path = Path(manifest['source_checkpoint_event'])
    event = read(event_path)
    if sha256(event_path.read_bytes()).hexdigest() != manifest['frozen_scene_context']['event_sha256']:
        raise ValueError('mask语义检查的event身份不符')
    prep = read(Path(event['source_path']) / 'preparation_manifest.json')
    obs = Path(prep['observations'])
    observations = read(obs / 'manifest.json')
    expected = {normalized_frame(r['id']): r['body_pixels'] for r in prep['frames']}
    context = read(frozen / 'camera_track_context.json')
    recorded = {r['frame_id']: r['original_body_pixels'] for r in context['frames']}
    checked = []
    for row in observations['frames']:
        frame = normalized_frame(row['id'])
        path = obs / row['body_mask']
        image = Image.open(path)
        values = np.asarray(image.convert('L'))
        if not set(np.unique(values).tolist()).issubset({0, 1, 255}):
            raise ValueError('合法body PNG并非已登记二值颜色语义')
        count = int((values > 0).sum())
        if count != expected[frame] or count != recorded[frame]:
            raise ValueError('INPUT_MASK_INVALID: ignore-alpha body count disagrees with original preparation/CPU context at ' + frame)
        checked.append(dict(frame=frame, mode=image.mode, body_pixels=count,
                            full_image_pixels=int(values.size), alpha_ignored=True))
    evidence = read(frozen / 'legal_preedit_evidence.json')
    if evidence['metrics']['f_geom']['denominator'] != sum(r['body_pixels'] for r in checked):
        raise ValueError('INPUT_MASK_INVALID: legal f_geom body denominator is inconsistent')
    return dict(status='PASS_INDEPENDENT_COLOR_CHANNEL_AND_PREPARATION_COUNT_CHECK',
                frames=len(checked), rows=checked, reference_GT_used=False)


def frozen_domain(manifest, frozen, contract):
    routes, tasks, candidate = (read(frozen / name) for name in
                                ('shared_routes.json', 'tasks.json', 'candidate_manifest.json'))
    if manifest.get('schema') != 'r4_native_shared_route_render_v1':
        raise ValueError('未知原生manifest schema')
    if manifest['scene_id'] != candidate['scene_id']:
        raise ValueError('场景身份不符')
    if manifest['frozen_scene_context'] != tasks[0]['frozen_scene_context']:
        raise ValueError('原生完整bundle/gauge/camera/runtime冻结上下文不符')
    for task in tasks:
        if task['candidate_hash'] != digest(candidate):
            raise ValueError('候选集合已改变')
        if task['frozen_scene_context'] != tasks[0]['frozen_scene_context']:
            raise ValueError('跨操作冻结上下文不一致')
        if task['shared_route_context_hash'] != tasks[0]['shared_route_context_hash']:
            raise ValueError('跨操作场景上下文不一致')
    route_by_hash, baseline = {}, None
    for index, route in enumerate(routes):
        sealed = {k: v for k, v in route.items() if k != 'route_assignment_hash'}
        rh = route['route_assignment_hash']
        if rh != digest(sealed) or route['candidate_hash'] != digest(candidate):
            raise ValueError('共享路由身份损坏')
        if rh in route_by_hash:
            raise ValueError('重复共享路由')
        route_by_hash[rh] = dict(action_id='h%03d' % index, route=route)
        if all(a['action'] == 'keep' for a in route['assignments']):
            baseline = rh
    if baseline is None or len(routes) != 2 ** len(candidate['groups']):
        raise ValueError('冻结有限域不完整或无allkeep')
    if manifest.get('action_count') != len(routes) or manifest.get('edits') != contract['operations']:
        raise ValueError('manifest动作数量或Q声明与实际冻结域不符')
    frames = contract['simulation_frames'] if candidate['scene_id'].startswith('sim_') else contract['real_frames']
    expected = {(h, f, q) for h in route_by_hash for f in frames for q in contract['operations']}
    actual = {}
    for row in manifest['rows']:
        key = (row['route_assignment_hash'], normalized_frame(row['frame']), row['edit'])
        if key in actual or key not in expected:
            raise ValueError('重复/域外原生评价: ' + str(key))
        if row.get('actual_shared_route_applied') is not True:
            raise ValueError('实际共享路由未验明，包括none不能旁路')
        if row.get('action_id') != route_by_hash[key[0]]['action_id']:
            raise ValueError('action ID与冻结路由顺序错接')
        actual[key] = row
    if set(actual) != expected:
        raise ValueError('原生有限域缺项: %d/%d，禁止给精确差距' % (len(actual), len(expected)))
    if manifest.get('none_uses_actual_route') is not True:
        raise ValueError('未声明实际修改路由none语义')
    parity = manifest.get('allkeep_native_none_parity_max_abs')
    if parity is None or not np.isfinite(parity) or parity > 2e-6:
        raise ValueError('allkeep原native none一致性门禁未通过')
    return routes, route_by_hash, baseline, frames, actual, tasks


def reference_domain(root, frames, operations):
    manifest = read(root / 'reference_manifest.json')
    if manifest.get('schema') != 'v43_sim_reference_v1':
        raise ValueError('独立参考schema不符')
    if [normalized_frame(v) for v in manifest['keyframes']] != frames:
        raise ValueError('独立编辑参考原有锚点与预注册子域不符')
    if manifest['edits']['shift']['translation_world_m'] != [0, 1.5, 0] or manifest['edits']['rotate']['yaw_delta_deg'] != 30:
        raise ValueError('独立参考与冻结操作参数不符')
    records = {}
    for row in manifest['references']:
        if row['vehicle_shadow'] is True:
            key = (normalized_frame(row['frame_id']), row['edit'])
            if key in records:
                raise ValueError('独立参考frame/edit重复')
            records[key] = row
    result = {}
    files = {root / 'reference_manifest.json'}
    for frame in frames:
        original = records[(frame, 'none')]
        with np.load(root / original['geometry'], allow_pickle=False) as g:
            body0 = g['semantic_id'] == 10
        for q in operations:
            row = records[(frame, q)]
            files.update((root / row['rgb'], root / row['geometry']))
            with np.load(root / row['geometry'], allow_pickle=False) as g:
                bodyq = g['semantic_id'] == 10
            rgb = image_array(root / row['rgb'])
            if bodyq.shape != body0.shape or body0.shape != rgb.shape[1:]:
                raise ValueError('独立参考区域和RGB尺寸错接')
            masks = dict(old_vacated=body0 & ~bodyq, new_body=bodyq,
                         keep=~(body0 | bodyq), full=np.ones_like(body0))
            result[(frame, q)] = dict(rgb=rgb, masks=masks,
                                      rgb_ref=str(root / row['rgb']), geometry_ref=str(root / row['geometry']))
    receipts = [dict(path=str(p), bytes=p.stat().st_size,
                     sha256=sha256(p.read_bytes()).hexdigest()) for p in sorted(files)]
    return result, receipts


def region_metrics(prediction, target, mask):
    pixels = int(mask.sum())
    if not pixels:
        return dict(pixels=0, channel_values=0, raw_mae=None, clamped_mae=None, status='empty_roi_unavailable')
    return dict(pixels=pixels, channel_values=3 * pixels,
                raw_mae=mean(np.abs(prediction[:, mask] - target[:, mask])),
                clamped_mae=mean(np.abs(np.clip(prediction[:, mask], 0, 1) - target[:, mask])), status='valid')


def constraints_pass(action, config, tolerance=None):
    tol = config['keep_mae_increase_max'] if tolerance is None else tolerance
    none_tol = config['none_raw_rgb_change_mae_max'] if tolerance is None else tolerance
    violations = []
    for row in action['observations']:
        if row['edit'] == 'none':
            if row['none_raw_rgb_change_mae'] > none_tol:
                violations.append(dict(frame=row['frame'], edit='none', kind='none_preservation',
                                       measured=row['none_raw_rgb_change_mae'], threshold=none_tol))
        elif row.get('keep_mae_increase') is not None:
            if row['keep_mae_increase'] > tol:
                violations.append(dict(frame=row['frame'], edit=row['edit'], kind='keep_mae_increase',
                                       measured=row['keep_mae_increase'], threshold=tol))
            if row['keep_worsened_fraction'] > config['keep_fraction_above_increase_threshold_max']:
                violations.append(dict(frame=row['frame'], edit=row['edit'], kind='keep_worsened_fraction',
                                       measured=row['keep_worsened_fraction'],
                                       threshold=config['keep_fraction_above_increase_threshold_max']))
        elif action['reference_available']:
            violations.append(dict(frame=row['frame'], edit=row['edit'], kind='keep_domain_unavailable'))
    return violations


def score(native_manifest, frozen, contract_path, output, reference_root=None):
    started = time.monotonic()
    if output.exists():
        raise FileExistsError('评分使用新目录，不覆盖旧收据')
    contract = read(contract_path)
    if contract['schema'] != 'v43_r4_isolated_development_evaluation_v1':
        raise ValueError('开发评价合同不符')
    manifest = read(native_manifest)
    routes, domain, baseline, frames, rows, tasks = frozen_domain(manifest, frozen, contract)
    mask_validation = legal_mask_semantic_gate(manifest, frozen)
    scene_id = manifest['scene_id']
    simulated = scene_id.startswith('sim_')
    if simulated and reference_root is None:
        raise ValueError('仿真评分需要显式隔离参考根目录')
    references, reference_receipts = reference_domain(reference_root, frames, contract['operations']) if simulated else ({}, [])
    native_root = native_manifest.parent
    baseline_images = {}
    for frame in frames:
        for q in contract['operations']:
            baseline_images[(frame, q)], _ = package(resolve_file(native_root, rows[(baseline, frame, q)]['fp32']))
    actions, max_compositor_error = [], 0.
    for rh, entry in domain.items():
        observations = []
        operation_targets = defaultdict(list)
        for frame in frames:
            for q in contract['operations']:
                row = rows[(rh, frame, q)]
                pred, consistency = package(resolve_file(native_root, row['fp32']))
                max_compositor_error = max(max_compositor_error, *consistency.values())
                base = baseline_images[(frame, q)]
                if pred.shape != base.shape:
                    raise ValueError('配对baseline图像尺寸不一致')
                item = dict(frame=frame, edit=q, route_assignment_hash=rh,
                            fp32=row['fp32'], compositor_max_errors=consistency,
                            full_frame_pixels=int(pred.shape[1] * pred.shape[2]),
                            raw_change_vs_allkeep_mae=mean(np.abs(pred - base)))
                if q == 'none':
                    item['none_raw_rgb_change_mae'] = item['raw_change_vs_allkeep_mae']
                if simulated:
                    ref = references[(frame, q)]
                    target = ref['rgb']
                    if target.shape != pred.shape:
                        raise ValueError('原生渲染和独立参考图像尺寸不符')
                    regions = {name: region_metrics(pred, target, mask) for name, mask in ref['masks'].items()}
                    item.update(regions=regions, independent_rgb_ref=ref['rgb_ref'],
                                reference_role='independent_existing_development_gt_isolated_scorer_only')
                    if q != 'none':
                        keep = ref['masks']['keep']
                        if keep.any():
                            error = np.abs(np.clip(pred, 0, 1) - target).mean(axis=0)
                            base_error = np.abs(np.clip(base, 0, 1) - target).mean(axis=0)
                            increase = error[keep] - base_error[keep]
                            item['keep_mae_increase'] = mean(increase)
                            item['keep_worsened_pixels'] = int((increase > contract['development_constraints']['keep_pixel_error_increase_threshold']).sum())
                            item['keep_worsened_fraction'] = item['keep_worsened_pixels'] / int(keep.sum())
                            item['keep_valid_pixel_denominator'] = int(keep.sum())
                        else:
                            item['keep_mae_increase'] = None
                            item['keep_worsened_fraction'] = None
                        valid = [regions[name]['clamped_mae'] for name in ('old_vacated', 'new_body')
                                 if regions[name]['pixels'] > 0]
                        item['normalized_target_loss'] = mean(valid) if valid else None
                        item['target_nonempty_roi_count'] = len(valid)
                        operation_targets[q].append(item['normalized_target_loss'])
                else:
                    item.update(reference_role='none_no_independent_real_counterfactual', regions=None,
                                normalized_target_loss=None, keep_mae_increase=None,
                                keep_worsened_fraction=None)
                observations.append(item)
        per_operation = {}
        for q in contract['scored_edit_operations']:
            values = operation_targets[q]
            per_operation[q] = mean(values) if len(values) == len(frames) and all(v is not None for v in values) else None
        joint = mean(list(per_operation.values())) if all(v is not None for v in per_operation.values()) else None
        action = dict(action_id=entry['action_id'], route_assignment_hash=rh,
                      assignments=entry['route']['assignments'], is_allkeep=rh == baseline,
                      reference_available=simulated, joint_target_loss=joint,
                      per_operation_target_losses=per_operation, observations=observations,
                      finite_decision_gap=None, depth_metric=None, temporal_metric=None)
        violations = constraints_pass(action, contract['development_constraints'])
        none_violations = [v for v in violations if v['kind'] == 'none_preservation']
        action.update(development_constraint_pass=(not violations) if simulated else None,
                      none_preservation_constraint_pass=not none_violations,
                      constraint_violations=violations,
                      counterfactual_safety_evaluated=simulated)
        actions.append(action)
    safe = [a for a in actions if a['development_constraint_pass'] and a['joint_target_loss'] is not None]
    best = min(safe, key=lambda a: (a['joint_target_loss'], a['action_id'])) if safe else None
    if best:
        for action in safe:
            gap = action['joint_target_loss'] - best['joint_target_loss']
            if gap < -1e-12:
                raise ValueError('完整有限域差距出现非法负值')
            action['finite_decision_gap'] = max(0., gap)
    sensitivity = []
    for tol in contract['constraint_sensitivity']['shared_tolerances'] if simulated else []:
        feasible = [a for a in actions if not constraints_pass(a, contract['development_constraints'], tol)
                    and a['joint_target_loss'] is not None]
        chosen = min(feasible, key=lambda a: (a['joint_target_loss'], a['action_id'])) if feasible else None
        sensitivity.append(dict(tolerance=tol, safe_count=len(feasible),
                                best_action_id=chosen['action_id'] if chosen else None,
                                best_joint_loss=chosen['joint_target_loss'] if chosen else None))
    baseline_action = next(a for a in actions if a['is_allkeep'])
    counterexamples = []
    if simulated:
        for action in actions:
            if action['joint_target_loss'] is not None and action['joint_target_loss'] < baseline_action['joint_target_loss'] and not action['development_constraint_pass']:
                counterexamples.append(dict(kind='target_gain_violates_preservation', action_id=action['action_id'],
                                             joint_gain=baseline_action['joint_target_loss'] - action['joint_target_loss'],
                                             violations=action['constraint_violations']))
        if best:
            for q in contract['scored_edit_operations']:
                delta = best['per_operation_target_losses'][q] - baseline_action['per_operation_target_losses'][q]
                if delta > 0:
                    counterexamples.append(dict(kind='best_joint_route_worsens_one_operation',
                                                 action_id=best['action_id'], edit=q, target_loss_increase=delta))
    result = dict(schema='v43_r4_isolated_development_scores_v1', scene_id=scene_id,
                  status='COMPLETE_NATIVE_DOMAIN_SCORED' if simulated else 'REAL_NATIVE_VALIDITY_AND_NONE_ONLY',
                  contract_sha256=sha256(contract_path.read_bytes()).hexdigest(),
                  contract_source_path=str(contract_path), contract_content_hash=digest(contract),
                  source_native_manifest=str(native_manifest), cpu_frozen_scene_dir=str(frozen),
                  shared_route_context_hash=tasks[0]['shared_route_context_hash'],
                  legal_mask_semantic_validation=mask_validation,
                  checkpoint_bundle_sha256=tasks[0]['checkpoint_bundle_sha256'],
                  reference_role='independent_existing_development_gt' if simulated else 'none',
                  isolated_reference_receipts=reference_receipts,
                  evaluated_anchor_frames=frames, registered_frame_denominator=contract['registered_frame_denominators'][scene_id],
                  evaluated_route_count=len(actions), expected_route_count=len(routes),
                  native_rendered_rows=len(rows), all_frozen_routes_native_evaluated=True,
                  evaluation_domain='3 pre-existing independent-GT anchors, one shared h across all q' if simulated else '3 registered real anchors, no counterfactual edit GT',
                  max_compositor_identity_error=max_compositor_error,
                  oracle_status='exact_finite_domain_under_development_constraints' if best else 'safe_oracle_unavailable',
                  decision_gap_kind='exact_finite_regret_development_3anchor_domain' if best else 'unavailable',
                  best_safe_shared_action_id=best['action_id'] if best else None,
                  best_verified_shared_residual=best['joint_target_loss'] if best else None,
                  best_shared_route_per_operation_losses=best['per_operation_target_losses'] if best else None,
                  capability_bin='C?', task_completion_threshold=None,
                  student_policy_evaluated=False, learned_generalization_result=None,
                  preedit_eligibility='unknown_not_assigned_from_evaluation_GT',
                  baseline_action_id=baseline_action['action_id'],
                  baseline_joint_target_loss=baseline_action['joint_target_loss'],
                  baseline_finite_decision_gap=baseline_action['finite_decision_gap'],
                  development_safe_action_count=len(safe) if simulated else None,
                  none_preservation_pass_count=sum(a['none_preservation_constraint_pass'] for a in actions),
                  constraint_sensitivity=sensitivity,
                  counterexamples=counterexamples, actions=actions,
                  scientific_claim_limits=['fixed group domain only; no row-level/global optimum',
                                           'existing development ancestry; no sealed-test or learned generalization claim',
                                           'no task completion/C0-C1 label without calibrated task threshold',
                                           'real-scene none check is not counterfactual edit safety'],
                  wall_seconds=time.monotonic() - started)
    output.mkdir(parents=True)
    write(output / 'scores.json', result)
    write(output / 'evaluation_contract.json', contract)
    compact = {k: v for k, v in result.items() if k != 'actions'}
    write(output / 'summary.json', compact)
    print(json.dumps(compact, ensure_ascii=False, allow_nan=False), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native-manifest', type=Path, required=True)
    parser.add_argument('--frozen-scene', type=Path, required=True)
    parser.add_argument('--contract', type=Path, required=True)
    parser.add_argument('--reference-root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    score(args.native_manifest, args.frozen_scene, args.contract, args.output, args.reference_root)
