"""CPU-only consolidation of immutable native ideal-capacity results."""
from pathlib import Path
import hashlib
import json
import math
import time


GPU = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
CPU = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
ROOT = GPU / 'check2'
CASES = ('constant', 'lowfreq', 'marking_w120', 'marking_w240', 'stress_highfreq')
RUNS = {'primary': ROOT / 'native_primary_r1', 'relaxed': ROOT / 'native_relaxed_r1'}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def aggregate_metrics(frames, key):
    metrics = [r['soft_metrics'][key] for r in frames]
    count = sum(m['pixels'] for m in metrics)
    return dict(pixels=count,
                RGB_RMSE=math.sqrt(sum(m['pixels'] * m['RMSE'] ** 2 for m in metrics) / count),
                RGB_MAE=sum(m['pixels'] * m['MAE'] for m in metrics) / count)


def main():
    begin = time.monotonic()
    protocol = read(CPU / 'protocol_cpu_prepared.json')
    error = protocol['check2']['error_envelope']
    levels = error['registered_RMSE_levels']
    assert levels == [n / 255 for n in (1, 2, 4)]
    lease = read(GPU / 'gpu_lease.json')
    assert lease['owner'] == 'stage_qa' and not lease['active']
    endpoint = read(ROOT / 'endpoint_replay_r1/receipt.json')
    assert endpoint['native_forward_calls'] == 20 and endpoint['optimizer_updates'] == 0
    smoke = read(ROOT / 'smoke_r1/lifecycle/completed.json')
    assert smoke['extra_solver_updates'] == 3

    scopes = {}
    fingerprints = {}
    all_cases = []
    accounting = []
    preflights = {}
    for scope, run in RUNS.items():
        preflight = read(run / 'operator_preflight.json')
        preflights[scope] = dict(path=str(run / 'operator_preflight.json'), **preflight)
        binding = read(run / 'binding.json')
        for filename, key in [('runner_source.py', 'runner_sha256'), ('native_helper_source.py', 'helper_sha256')]:
            assert sha(run / filename) == binding[key]
        rows = []
        accepted = attempts = retries = public_clone_forwards = 0
        backward = preflight['native_backward_calls']
        for case in CASES:
            root = run / case
            completed = read(root / 'completed.json')
            evaluation = read(root / 'evaluation/evaluation.json')
            trace = read(root / 'trace.json')
            reload = read(root / 'public_reload.json')
            receipt = read(root / 'asset/receipt.json')
            certificate = evaluation['certificate']
            detail = evaluation['detail']
            n = sum(int(r['accepted']) for r in trace)
            retry = sum(r.get('retries', 0) for r in trace)
            assert n == len(trace) == completed['accepted_iterations'] == 500
            assert all(r['status'] == 'accepted' for r in trace) and retry == 0
            assert completed['solver_elapsed_seconds'] < 300 and completed['time_overrun_seconds'] == 0
            assert not certificate['negative_certificate_allowed']
            assert not certificate['strict_numeric_bound_established']
            assert evaluation['feasible_box']
            assert max(evaluation['alpha_actual_frozen_errors'].values()) == 0
            assert max(r['RGB_max'] for r in reload['rows']) == 0
            assert max(r['alpha_max'] for r in reload['rows']) == 0
            accepted += n
            attempts += len(trace)
            retries += retry
            backward += 5 + 5 * (n + retry)
            public_clone_forwards += len(reload['rows'])
            stress = case == 'stress_highfreq'
            tolerance_results = []
            for codes, tolerance in zip((1, 2, 4), levels):
                rgb_pass = certificate['RGB_RMSE'] <= tolerance
                witness = rgb_pass and detail['passed'] and not stress
                tolerance_results.append(dict(
                    RGB_code_values=codes, RGB_RMSE_tolerance=tolerance,
                    objective_tolerance=0.5 * tolerance ** 2,
                    RGB_pass=rgb_pass, detail_pass=None if stress else detail['passed'],
                    feasible_witness=witness,
                    combined_status=('diagnostic_stress_only' if stress else 'feasible_solution_witness'
                                     if witness else 'optimization_or_numeric_certificate_inconclusive'),
                    negative_certificate_allowed=False))
            floor = certificate['guarded_box_range_floor_numeric_proxy']
            strata = {key: aggregate_metrics(evaluation['frames'], key)
                      for key in ('D_full', 'D_exact', 'D_color', 'fallback')}
            assert [strata[k]['pixels'] for k in ('D_full', 'D_exact', 'D_color', 'fallback')] == [38416, 38198, 37206, 218]
            assert abs(strata['D_color']['RGB_RMSE'] - certificate['RGB_RMSE']) < 1e-12
            opaque_pixels = sum(r['opaque_target']['pixels'] for r in evaluation['frames'])
            opaque_RMSE = math.sqrt(sum(r['opaque_target']['pixels'] * r['opaque_target']['RMSE'] ** 2
                                        for r in evaluation['frames']) / opaque_pixels)
            opaque_floor = sum(r['opaque_target']['pixels'] * r['opaque_coverage_range_floor']
                               for r in evaluation['frames']) / opaque_pixels
            row = dict(case=case, scope=scope, RGB_RMSE=certificate['RGB_RMSE'],
                       RGB_RMSE_in_code_values=255 * certificate['RGB_RMSE'], objective=certificate['f'],
                       certificate=certificate, range_floor_numeric_proxy=floor,
                       range_floor_fraction_of_final_objective=floor / certificate['f'] if certificate['f'] else None,
                       strict_negative_certificate=False,
                       original_reference_2code_status=evaluation['capacity_status'],
                       registered_tolerances=tolerance_results, detail=detail, strata=strata,
                       opaque_target=dict(domain='D_full', pixels=opaque_pixels, RGB_RMSE=opaque_RMSE,
                                          coverage_range_floor_numeric_proxy=opaque_floor,
                                          role='separate opaque coverage objective; not soft-RGB capacity certificate'),
                       coverage=[dict(raw=r['raw'], D_full=r['soft_metrics']['D_full']['pixels'],
                                      gap=r['gap'], thin=r['thin'], supported=r['supported'])
                                 for r in evaluation['frames']],
                       frozen_alpha_max_error=0.0, K_exact=completed['K_exact'],
                       K_role='hard protected joint RGB' if scope == 'primary' else 'not claimed: virtual road-only diagnostic',
                       public_reload_RGB_max=0.0, public_reload_alpha_max=0.0,
                       public_reload_RGB8_max=max(r['RGB8_max'] for r in reload['rows']),
                       solver=dict(accepted_iterations=n, attempted_iterations=len(trace), retries=retry,
                                   elapsed_seconds=completed['solver_elapsed_seconds'], stop_reason=completed['stop_reason'],
                                   max_iterations=500, max_seconds=300, automatic_extension=False),
                       completed=str(root / 'completed.json'), evaluation=str(root / 'evaluation/evaluation.json'),
                       PNG_pattern=str(root / 'evaluation/{raw:03d}_{road|independent_teacher}.png'),
                       human_verdict=None, production_promoted=False)
            rows.append(row)
            all_cases.append(row)
            for rel in ('completed.json', 'evaluation/evaluation.json', 'trace.json'):
                p = root / rel
                fingerprints[str(p)] = sha(p)
        final_receipt = read(run / CASES[-1] / 'asset/receipt.json')
        # save_asset captures the cumulative context counter before the final five expected reload forwards.
        tracked_forwards = final_receipt['native_raster_calls_in_context'] + 5
        actual_forwards = tracked_forwards + public_clone_forwards
        accounting.append(dict(scope=scope, accepted_solver_iterations=accepted, solver_attempts=attempts,
                               retries=retries, native_backward_calls=backward,
                               cumulative_context_forward_calls=tracked_forwards,
                               separately_loaded_public_asset_forward_calls=public_clone_forwards,
                               actual_native_forward_calls=actual_forwards,
                               derivation='last asset cumulative context counter + final expected reload5 + all public clone reload forwards25'))
        scopes[scope] = dict(role='protected primary' if scope == 'primary' else 'root-authorized conditional all-active virtual diagnostic',
                             eligible_active_color_rows=preflight['eligible_rows'], all_active_rows=151096,
                             archived_Kguard_rows=38982 if scope == 'primary' else None,
                             initialization='exact same canonical oldS2 c0; each case independently restarts',
                             source_binding=str(run / 'binding.json'), cases=rows,
                             production_promoted=False)

    smoke_asset_receipt = read(ROOT / 'smoke_r1/lifecycle/asset/receipt.json')
    smoke_forward = smoke_asset_receipt['native_raster_calls_in_context'] + 5 + 5
    smoke_preflight = read(ROOT / 'smoke_r1/operator_preflight.json')
    smoke_backward = smoke_preflight['native_backward_calls'] + 5 + 5 * 3
    coverage = all_cases[0]['coverage']
    assert all(r['coverage'] == coverage for r in all_cases)
    solver_iterations = sum(r['accepted_solver_iterations'] for r in accounting)
    assert solver_iterations == 5000
    result = dict(
        schema='ideal_native_capacity_results.v1', role='five-view supervised capacity oracle, not generalization',
        canonical_asset=str(Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_asset_20261116_r1/phase2/formal_r1/asset')),
        protocol=str(CPU / 'protocol_cpu_prepared.json'), protocol_sha256=sha(CPU / 'protocol_cpu_prepared.json'),
        execution_contract=str(GPU / 'execution_contract.json'),
        independent_teacher='analytic world-continuous asphalt and flat-top markings ray-projected onto canonical mesh; GS render never used as RGB teacher',
        objective=dict(expression='0.5*sum(q*(P-alpha_fixed*T)^2)/(3*sum(q))', edge_weight=0,
                       D_full=38416, D_exact=38198, fallback=218, D_color=37206, q='one on fixed D_color',
                       RGB_denominator=111618, ci_box=[0, 1], alpha_stratum='exact G_s and frozen alpha>0.9',
                       fallback_role='full-domain description only; no negative representation inference'),
        fixed_fields=['world_xyz', 'normals', 'opacity', 'all_scales', 'active_mask', 'GS_density',
                      'cameras480x320', 'static', 'vehicles', 'environment', 'theta', 'z', 'eta'],
        numeric_certificate=dict(strict_numeric_error_bound_established=False,
                                 conservative_objective_plus_boxgradient_error_bound=None,
                                 required_error_bound=1e-5, negative_certificate_allowed=False,
                                 certified_lower_bound=None,
                                 FW_and_range_floor_role='numerical proxies only; spotchecks do not establish a conservative global bound',
                                 formula='G=sum(g*c-min(g,0)); numeric proxy max(0,f-G-1e-5), not certified'),
        error_envelope=error, scopes=scopes, operator_preflights=preflights,
        interpretation=[
            'Primary and relaxed results are both retained; relaxed cannot replace protected primary or claim K preservation.',
            'Primary constant/lowfreq satisfy4/255. Primary markings do not meet the combined registered levels.',
            'Relaxed constant/lowfreq are feasible witnesses at1/255; marking_w240 is a feasible witness at4/255, not2/255.',
            'Relaxed marking_w120 RGB meets4/255 but raw86 contrast retention0.8169 misses0.9; combined witness not established.',
            'High-frequency stress is diagnostic only; no stress failure is used as negative capacity evidence.',
            'Unsatisfied tolerances remain optimization or numerical-certificate inconclusive; no representation upper-limit claim.',
            'Fixed coverage and mesh-fallback limitations remain separately reported; soft fitting does not establish opaque coverage.'
        ],
        fixed_coverage=dict(D_full=38416, gap=sum(r['gap'] for r in coverage), thin=sum(r['thin'] for r in coverage),
                            supported=sum(r['supported'] for r in coverage), alpha_max_change=0.0, per_frame=coverage),
        lifecycle=dict(receipt=str(ROOT / 'smoke_r1/lifecycle/completed.json'), extra_solver_updates=3,
                       full_restore_exact=smoke['restore_tensor_state_exact'],
                       next_update_bitwise=smoke['next_update_bitwise'],
                       next_update_tensor_maxima=smoke['replay_tensor_maxima'],
                       next_update_eta_absolute_difference=smoke['replay_eta_difference'],
                       next_update_eta_relative_difference=smoke['replay_eta_difference'] / smoke['actions'][1]['next_eta'],
                       next_case_entry=smoke['next_case_entry'], K_exact=smoke['K_exact'], replay_pass=smoke['replay_pass']),
        engineering_accounting=dict(primary_solver_iterations=2500,
                                    authorized_conditional_diagnostic_solver_iterations=2500,
                                    formal_and_conditional_solver_iterations=solver_iterations,
                                    lifecycle_extra_solver_updates=3, total_solver_updates_including_engineering=5003,
                                    formal_runs=accounting,
                                    lifecycle_native_forward_calls=smoke_forward,
                                    lifecycle_native_backward_calls=smoke_backward,
                                    endpoint_postevaluation_native_forward_calls=20,
                                    endpoint_postevaluation_solver_updates=0,
                                    total_native_forward_calls=sum(r['actual_native_forward_calls'] for r in accounting) + smoke_forward + 20,
                                    total_native_backward_calls=sum(r['native_backward_calls'] for r in accounting) + smoke_backward,
                                    call_semantics='native raster invocations / autograd backward invocations, not CUDA kernel launches; includes saved-public-asset reload forwards',
                                    endpoint_receipt=str(ROOT / 'endpoint_replay_r1/receipt.json')),
        endpoint_replay=dict(receipt=str(ROOT / 'endpoint_replay_r1/receipt.json'),
                             array_pattern=str(ROOT / 'endpoint_replay_r1/{primary|relaxed}/{raw:03d}.npz'),
                             keys=['fixed_b', 'box_upper', 'fixed_alpha', 'D_color', 'q', 'eligible_world', 'active_world_rows'],
                             floor_max_absolute_replay_difference=max(r['absolute_difference'] for r in endpoint['floor_comparison']),
                             original_formal_results_modified=False),
        source_publication=dict(runner=str(ROOT / 'native_primary_r1/runner_source.py'),
                                metrics=str(ROOT / 'native_primary_r1/metrics_source.py'),
                                native_helper=str(ROOT / 'native_primary_r1/native_helper_source.py'),
                                endpoint_replay_code=str(GPU / 'code/replay_capacity_endpoints.py'),
                                consolidation_code=str(GPU / 'code/summarize_ideal_capacity.py'),
                                new_test_file=None),
        GPU=dict(executed=True, lease_active=False, keep_on=True, further_jobs_started=False),
        original_result_fingerprints=fingerprints,
        human_verdict=None, production_promoted=False, summary_CPU_seconds=time.monotonic() - begin)
    for path, digest in fingerprints.items():
        assert sha(path) == digest
    output = ROOT / 'results.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(results=str(output), accounting=result['engineering_accounting'],
                          cases=[dict(scope=r['scope'], case=r['case'], RMSE=r['RGB_RMSE'],
                                      detail=r['detail']['status'],
                                      levels=[dict(code=v['RGB_code_values'], RGB=v['RGB_pass'],
                                                   combined=v['combined_status']) for v in r['registered_tolerances']])
                                 for r in all_cases]), ensure_ascii=False))


if __name__ == '__main__':
    main()
