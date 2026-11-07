"""从保存的 float 和原观测独立复算 v44 接入门；不加载模型、不使用 GPU。"""
from pathlib import Path
import argparse
import hashlib
import json
import numpy as np


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mask(value):
    value = np.asarray(value).astype(bool).squeeze()
    assert value.shape == (320, 480)
    return value


def stat(value, domain):
    count = int(domain.sum())
    total = float(np.asarray(value, dtype=np.float64)[domain].sum())
    return dict(pixels=count, sum=total, mean=total / count if count else None)


def arrays(path):
    with np.load(path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--candidate', default='candidate4')
    parser.add_argument('--baseline', default='baseline4/v44')
    parser.add_argument('--contract', default='contract/execution_contract.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = root / args.contract
    contract_bytes = contract_path.read_bytes()
    contract = json.loads(contract_bytes)
    limits = contract['numeric_gates']
    source = read(Path(contract['baseline']['input_contract']))['paths']
    observations = Path(source['observations'])
    candidate = root / args.candidate
    receipt = read(candidate / 'receipt.json')
    assert receipt['contract_sha256'] == hashlib.sha256(contract_bytes).hexdigest()
    runtime = receipt['runtime_receipt']
    for key in ['shared_native_class_methods_unchanged', 'full5_restore_all_parameters_and_current_owner_ids_exact',
                'all_original_v44_parameters_assets_road_gates_and_fine_preserved',
                'old_static_gates_applied_only_to_original_static']:
        assert runtime[key] is True
    for key in ['asset_all_named_tensors_bitwise_hash_unchanged',
                'frozen_native_road_gates_environment_unchanged', 'persistent_SF_unchanged']:
        assert runtime['frozen_v44'][key] is True
    assert runtime['added_actor_ids'] == contract['candidate']['retained_actor_ids']
    raws = receipt['raws']
    assert raws == contract['pilot_raws'] or raws == list(range(89, 138))
    baseline = root / args.baseline
    full_mask_sha, upper_sha, gt_sha = (hashlib.sha256() for _ in range(3))
    full_count = upper_count = 0
    for raw in range(89, 138):
        obs = arrays(observations / f'{raw:03d}.npz')
        fixed = obs['actor_labels'] == 1
        yx = np.column_stack(np.nonzero(fixed)).astype('<i2')
        upper = yx[:, 0] <= yx[:, 0].min() + .25 * (yx[:, 0].max() - yx[:, 0].min()) if len(yx) else np.zeros(0, bool)
        marker = np.asarray(raw, dtype='<i2').tobytes()
        full_mask_sha.update(marker + np.ascontiguousarray(fixed).tobytes())
        upper_sha.update(marker + yx[upper].tobytes())
        gt_sha.update(marker + np.ascontiguousarray(obs['rgb'][fixed], dtype='<f4').tobytes())
        full_count += len(yx)
        upper_count += int(upper.sum())
    reference = contract['domains']['full49_reference']
    assert [full_count, upper_count] == [reference['full_pixels'], reference['upper_pixels']]
    assert full_mask_sha.hexdigest() == reference['full_mask_sha256']
    assert upper_sha.hexdigest() == reference['upper_index_sha256']
    assert gt_sha.hexdigest() == reference['gt_sha256']
    with np.load(source['fixed_reference'], allow_pickle=False) as fixed:
        ground = fixed['depth_m'][:, 3] > 0
    comparisons, guards, gates = [], [], []
    max_partition = max_baseline_metric_error = 0.
    rows_by_raw = {row['raw']: row for row in receipt['frames']}
    for raw in raws:
        obs = arrays(observations / f'{raw:03d}.npz')
        target = np.asarray(obs['rgb'], dtype=np.float32).transpose(2, 0, 1)
        if target.max() > 1:
            target = target / 255
        proxy = np.asarray(obs['road_edit_rgb'], dtype=np.float32).transpose(2, 0, 1)
        if proxy.max() > 1:
            proxy = proxy / 255
        a5 = obs['actor_labels'] == 1
        yx = np.column_stack(np.nonzero(a5))
        upper = np.zeros_like(a5)
        if len(yx):
            selected = yx[:, 0] <= yx[:, 0].min() + .25 * (yx[:, 0].max() - yx[:, 0].min())
            upper[yx[selected, 0], yx[selected, 1]] = True
        if len(raws) == 4:
            assert int(a5.sum()) == contract['domains']['pilot_a5_pixels'][str(raw)]
            assert int(upper.sum()) == contract['domains']['pilot_upper_pixels'][str(raw)]
        base_metrics = read(baseline / f'{raw:03d}/metrics.json')['states']
        asset_hashes, added_hashes = [], []
        for state in ['keep', 'delete']:
            saved = arrays(candidate / f'frames/{raw:03d}/{state}_float.npz')
            before = arrays(baseline / f'{raw:03d}/{state}_float.npz')
            assert saved['rgb'].shape == (3, 320, 480)
            assert saved['group_alpha_T'].shape == (7, 320, 480)
            assert saved['added_owner_rgb_T'].shape == (3, 3, 320, 480)
            assert all(np.isfinite(value).all() for value in saved.values())
            groups = saved['group_alpha_T']
            for index, actor in enumerate([5, 18, 26]):
                if actor not in contract['candidate']['retained_actor_ids']:
                    assert float(abs(groups[4 + index]).max()) == 0
                    assert float(abs(saved['added_owner_rgb_T'][index]).max()) == 0
            partition = float(abs(groups.sum(0) - saved['alpha']).max())
            assert partition < limits['inherited_alpha_conservation_maxabs_strict_below']
            max_partition = max(max_partition, partition)
            if state == 'delete':
                assert float(abs(groups[1]).max()) == contract['exact_guards']['delete_a37_joint_group_max']
            gpu = rows_by_raw[raw]['states'][state]
            assert gpu['meta']['full5_added_actor_ids'] == contract['candidate']['retained_actor_ids']
            assert set(gpu['meta']['full5_added_rows']) == {str(actor) for actor in contract['candidate']['retained_actor_ids']}
            assert gpu['shared_physical_before_hash'] == gpu['shared_physical_after_hash']
            assert gpu['shared_env_before_hash'] == gpu['shared_env_after_hash']
            assert gpu['asset_row_hash'] == base_metrics[state]['meta']['native_asset_row_hash']
            assert gpu['meta']['frozen_v44_geometric_row_prefix_exact']
            assert gpu['meta']['old_static_gate_indices_not_transferred']
            assert gpu['meta']['full5_native_a6_rows_added'] == 0
            assert gpu['meta']['full5_native_a37_rows_added'] == 0
            asset_hashes.append(gpu['asset_row_hash'])
            added_hashes.append(gpu['added_physical_hash'])
            a6, a37 = mask(obs['visible'][0]), mask(obs['visible'][1])
            road = mask(obs['road']) if state == 'keep' else mask(obs['road_edit']) & mask(ground[raw - 89])
            domains = dict(a6_RGB=a6, a6_depth_m=mask(obs['depth_valid'][0]),
                           road_RGB=road, protected_RGB=mask(obs['protect']) & ~a6 & ~a37,
                           fixed_road_absolute_blue=road, a5_RGB=a5, a5_upper_RGB=upper)
            if state == 'keep':
                domains['a37_RGB'] = a37
            measured = {}
            for label, values in [('baseline', before), ('candidate', saved)]:
                rgb_error = abs(values['rgb'] - target).mean(0)
                residual = values['rgb'] - (target if state == 'keep' else proxy)
                errors = dict(a6_RGB=rgb_error, a6_depth_m=abs(values['depth_m'] - obs['depth_m'][0]),
                              road_RGB=abs(residual).mean(0), protected_RGB=rgb_error,
                              fixed_road_absolute_blue=abs(residual[2] - .5 * (residual[0] + residual[1])),
                              a5_RGB=rgb_error, a5_upper_RGB=rgb_error, a37_RGB=rgb_error)
                measured[label] = {name: stat(errors[name], domain) for name, domain in domains.items()}
            for name in ['a6_RGB', 'a6_depth_m', 'road_RGB', 'protected_RGB'] + (['a37_RGB'] if state == 'keep' else []):
                old = base_metrics[state][name]
                new = measured['baseline'][name]
                assert new['pixels'] == old['pixels']
                if old['mean'] is not None:
                    error = abs(old['mean'] - new['mean'])
                    assert error < 1e-10, (raw, state, name, error)
                    max_baseline_metric_error = max(max_baseline_metric_error, error)
            delta = {name: (measured['candidate'][name]['mean'] - measured['baseline'][name]['mean'])
                     if measured['baseline'][name]['mean'] is not None else None for name in domains}
            comparisons.append(dict(raw=raw, state=state, baseline=measured['baseline'],
                candidate=measured['candidate'], delta=delta,
                owner5_alphaT_a5=stat(groups[4], a5), owner5_alphaT_upper=stat(groups[4], upper),
                static_alphaT_a5=stat(groups[3], a5), other_actor_alphaT_a5=stat(groups[[0, 1, 5, 6]].sum(0), a5),
                new_a6_asset_alpha_below09=int(((groups[0] < .9) & (before['group_alpha_T'][0] >= .9) & a6).sum()),
                new_a6_asset_alpha_below05=int(((groups[0] < .5) & (before['group_alpha_T'][0] >= .5) & a6).sum())))
            guards.append(dict(raw=raw, state=state, alpha_partition_max_abs=partition,
                delete_a37_max=float(abs(groups[1]).max()) if state == 'delete' else None,
                prefix_physical_and_colours_hash_exact=True, environment_hash_exact=True,
                original_v44_asset_row_hash_exact=True))
        assert len(set(asset_hashes)) == len(set(added_hashes)) == 1

    def add_gate(name, value, threshold, strict=False, **context):
        gates.append(dict(name=name, value=value, threshold=threshold,
                          passed=value < threshold if strict else value <= threshold, **context))

    worst = dict(a6_RGB='a6_RGB_worst_frame_delta_max', a6_depth_m='a6_depth_m_worst_frame_delta_max',
                 a37_RGB='KEEP_a37_RGB_worst_frame_delta_max', road_RGB='road_RGB_worst_frame_delta_max',
                 protected_RGB='protected_RGB_worst_frame_delta_max',
                 fixed_road_absolute_blue='fixed_road_absolute_blue_worst_frame_delta_max')
    pooled = dict(road_RGB='road_RGB_pooled_delta_max', protected_RGB='protected_RGB_pooled_delta_max',
                  fixed_road_absolute_blue='fixed_road_absolute_blue_pooled_delta_max')
    for state in ['keep', 'delete']:
        rows = [row for row in comparisons if row['state'] == state]
        for metric, key in worst.items():
            available = [row for row in rows if row['delta'].get(metric) is not None]
            if available:
                row = max(available, key=lambda row: row['delta'][metric])
                add_gate(state + ':' + metric + ':worst_frame', row['delta'][metric], limits[key], raw=row['raw'])
        for metric, key in pooled.items():
            count = sum(row['baseline'][metric]['pixels'] for row in rows)
            delta = (sum(row['candidate'][metric]['sum'] for row in rows)
                     - sum(row['baseline'][metric]['sum'] for row in rows)) / count
            add_gate(state + ':' + metric + ':pooled', delta, limits[key], pixels=count)
        if 105 in raws:
            row = next(row for row in rows if row['raw'] == 105)
            add_gate(state + ':raw105_a5_RGB', row['delta']['a5_RGB'],
                     limits['raw105_a5_mask_MAE_delta_strictly_below'], strict=True)
    a5_summary = {}
    for state in ['keep', 'delete']:
        rows = [row for row in comparisons if row['state'] == state]
        visible = [row['raw'] for row in rows if row['baseline']['a5_RGB']['pixels']]
        summary = dict(visible_frame_count=len(visible), visible_raw_ids=visible)
        for name in ['a5_RGB', 'a5_upper_RGB']:
            count = sum(row['baseline'][name]['pixels'] for row in rows)
            values = {label: sum(row[label][name]['sum'] for row in rows) / count if count else None
                      for label in ['baseline', 'candidate']}
            summary[name] = dict(pixels=count, **values,
                                delta=values['candidate'] - values['baseline'] if count else None)
        a5_summary[state] = summary
    result = dict(status='PASS_SAVED_FLOAT_ENGINEERING_AND_NUMERIC_GATES' if all(g['passed'] for g in gates)
                  else 'ENGINEERING_PASS_NUMERIC_GATES_FAILED',
                  scope='pilot4' if len(raws) == 4 else 'all49', raw_ids=raws,
                  independent_CPU_only=True, CUDA_initialized=False, source_models_loaded=False,
                  candidate=str(candidate), baseline=str(baseline),
                  contract_sha256=hashlib.sha256(contract_bytes).hexdigest(),
                  fixed_original_domain_hashes_exact=True, fixed_all49_pixels=full_count,
                  fixed_all49_upper_pixels=upper_count, candidate_saved_float_max_partition_error=max_partition,
                  original_baseline_metrics_recomputed_max_abs_error=max_baseline_metric_error,
                  exact_guards=guards, comparisons=comparisons, numeric_gates=gates,
                  evaluated_a5_and_upper_summary=a5_summary,
                  runtime_source_model_asset_and_shared_class_guards_pass=True,
                  unselected_actor_alpha_and_RGB_slots_exact_zero=[actor for actor in [5, 18, 26]
                      if actor not in contract['candidate']['retained_actor_ids']],
                  failed_gates=[g for g in gates if not g['passed']],
                  physical_row_prefix_check_basis='GPU actual-row equality assertion and paired field hashes in receipt, not inversion from pixels',
                  alpha_or_ID_is_not_visual_acceptance=True, visual_review_required=True,
                  optimizer_updates=0, quality_accepted=False, human_verdict=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['status', 'scope', 'raw_ids', 'candidate_saved_float_max_partition_error',
          'original_baseline_metrics_recomputed_max_abs_error', 'failed_gates']}, indent=2))
    for row in comparisons:
        if row['baseline']['a5_RGB']['pixels']:
            print(json.dumps(row, indent=2))


if __name__ == '__main__':
    main()
