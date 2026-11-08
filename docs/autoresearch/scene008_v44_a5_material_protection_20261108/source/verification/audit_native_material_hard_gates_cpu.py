"""Independent native saved-RGB check of M's added hard gates, CPU only.

Run the saved-render original94/K6+106 audit first. This adds the sealed16
whole-domain MAE gates, six K-actual pool bounds, and exact A>0 pixel caps.
Positive fallback z is always reported separately from the strict main caps.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np


HERE = Path(__file__).resolve().parent
WS = HERE.parent.parent
OLD = WS / 'outputs/scene008_v44_a5_road_fusion_integration_20261107'
C49 = WS / 'outputs/scene008_v44_a5_c49_rgb_20261107'
RAWS = [94, 105, 128, 137]
STATES = ['keep', 'delete']


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def csv_read(p):
    with p.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def csv_write(p, rows):
    if not rows:
        return
    with p.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def chw(a):
    if a.shape == (320, 480, 3):
        a = a.transpose(2, 0, 1)
    assert a.shape == (3, 320, 480) and a.dtype == np.float32 and np.isfinite(a).all()
    return a


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, help='NAME=saved candidate directory')
    parser.add_argument('--saved-audit', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--solver-result', type=Path, required=True)
    parser.add_argument('--recovery-protocol', type=Path, help='Actual solver engineering protocol, if a recovery produced the candidate')
    parser.add_argument('--control-protocol', type=Path, help='Sealed finite diagnostic control protocol; original hard targets still come from --protocol/--registered-result')
    parser.add_argument('--registered-result', type=Path, help='Original r2 result containing the frozen domains and118 seam bounds')
    parser.add_argument('--active-mask-root', type=Path, default=WS / 'outputs/scene008_v44_a5_material_protection_20261108/evidence/operator_active_masks_cpu_v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists(), 'Prior independent audit stays immutable'
    name, folder = args.candidate.split('=', 1)
    folder = Path(folder)
    saved = read(args.saved_audit / 'summary.json')
    protocol, solve = read(args.protocol), read(args.solver_result)
    registered = read(args.registered_result) if args.registered_result else solve
    assert name in saved['results']
    assert Path(saved['variants'][name]).resolve() == folder.resolve()
    if args.control_protocol:
        assert args.registered_result and not args.recovery_protocol
        control = read(args.control_protocol)
        assert solve['protocol_sha256'] == sha(args.control_protocol)
        assert solve['source_sha256'] == control['source_sha256']
        assert control['immutable_input_source_sha256'] == protocol['source_sha256']
        assert control.get('parent_operator_manifest_sha256', control['operator_manifest_sha256']) == protocol['operator']['manifest_sha256']
        assert registered['protocol_sha256'] == sha(args.protocol)
        assert solve['diagnostic'] is True and solve['export_integrity_pass'] is True
    elif args.recovery_protocol:
        assert args.registered_result, 'Recovery checks still use the original registered hard targets'
        recovery = read(args.recovery_protocol)
        assert solve['protocol_sha256'] == sha(args.recovery_protocol)
        assert solve['source_sha256'] == recovery['source_sha256']
        assert recovery['r2_source_sha256'] == protocol['source_sha256']
        assert recovery['r2_result_sha256'] == sha(args.registered_result)
        assert registered['protocol_sha256'] == sha(args.protocol)
    else:
        assert solve['protocol_sha256'] == sha(args.protocol)
        assert solve['source_sha256'] == protocol['source_sha256']
    assert solve['status'] in ['MAIN_PROTECTION_CERTIFIED_PENDING_NATIVE_REVIEW', 'FALLBACK_MINIMAX_CERTIFIED_PENDING_NATIVE_REVIEW',
                              'FINITE_BOUNDED_SEAM_GUARDED_PALETTE_READY_NATIVE8F', 'FINITE_BOUNDED_CHILD_SEAM_GUARDED_PALETTE_READY_NATIVE8F',
                              'FINITE_BOUNDED_CHILD_ALL_PROTECTION_PALETTE_READY_NATIVE8F']
    assert 'candidate' in solve
    assert protocol['numerics']['original_photo_tolerance'] == 1e-6
    assert protocol['numerics']['pixel_tolerance'] == 1e-6
    assert protocol['validation']['quality_accepted'] is False
    assert solve['quality_accepted'] is False
    mask_receipt = read(args.active_mask_root / 'receipt.json')
    assert mask_receipt['operator_manifest_sha256'] == protocol['operator']['manifest_sha256']
    active_masks = load(args.active_mask_root / 'exact_A_positive_protection_masks.npz')
    all_metrics = csv_read(args.saved_audit / 'domain_metrics.csv')
    all_pools = csv_read(args.saved_audit / 'K_six_pools.csv')
    extra_pools = []
    for p in [r for r in all_pools if r['variant'] == name]:
        k = next(r for r in all_pools if r['variant'] == 'K' and (r['name'], r['state']) == (p['name'], p['state']))
        source_target = next(r['original_target'] for r in registered['seam_constraints']
                             if r['role'] == 'added_K_actual_pool_preservation' and (r['name'], r['state']) == (p['name'], p['state']))
        # The registered K-actual readout used float32 bin-score means. Our
        # independent K106 readout averages exact float32 jumps in float64.
        # Keep the registered bound exactly; precision comparison is diagnostic.
        precision_difference = float(k['actual_score']) + 1e-6 - source_target
        assert abs(precision_difference) < 1e-7
        target = source_target
        score = float(p['actual_score'])
        extra_pools.append(dict(name=p['name'], state=p['state'], bins=int(p['bins']),
                                actual_native_score=score, K_actual_score=float(k['actual_score']),
                                K_registered_actual_score=target-1e-6,
                                K_recomputed_minus_registered_precision_delta=precision_difference,
                                original_target=target, margin=target-score, passed=score <= target))
    assert len(extra_pools) == 6
    shared_z = float(solve.get('shared_pixel_slack', 0))
    fallback = bool(solve.get('fallback_used'))
    assert shared_z >= 0 and (fallback or shared_z == 0)
    whole, pixels, violations = [], [], []
    total_pixels = 0
    for raw in RAWS:
        obs = load(HERE / f'inputs/observations/{raw:03d}.npz')
        gt = chw(obs['rgb'])
        masks = dict(protected_nonroad_RGB=obs['protect'] & ~obs['visible'][0] & ~obs['visible'][1] & ~obs['road'] & ~obs['road_edit'],
                     original_road_RGB=obs['road'])
        for state in STATES:
            current = load(folder / f'frames/{raw:03d}/{state}_float.npz')
            warm = load(OLD / f'evidence/H2_source_warm/frames/{raw:03d}/{state}_float.npz')
            formal = load(C49 / f'evidence/H2_material2000_retry1/frames/{raw:03d}/{state}_float.npz')
            base = load(HERE / f'inputs/latest21K/frames/{raw:03d}/{state}_float.npz')
            c, w, f, b = [chw(v['rgb']) for v in [current, warm, formal, base]]
            c32, b32 = abs(c-gt).mean(0), abs(b-gt).mean(0)
            for domain, mask in masks.items():
                score = float(c32[mask].astype(np.float64).mean())
                baseline = float(b32[mask].astype(np.float64).mean())
                target = baseline + 1e-6
                cross = next(r for r in all_metrics if r['variant'] == name and int(r['raw']) == raw and r['state'] == state and r['domain'] == domain and r['scope'] == 'original')
                assert score == float(cross['MAE_to_observation'])
                source = next(r for r in registered['domains'] if r['raw'] == raw and r['state'] == state and r['name'] == ('protected_nonroad' if domain == 'protected_nonroad_RGB' else 'originalroad'))
                assert source['pixels'] == int(mask.sum())
                assert abs(source['original_target']-target) < 1e-14
                whole.append(dict(raw=raw, state=state, domain=domain, pixels=int(mask.sum()),
                                  latest21K_MAE=baseline, actual_native_MAE=score,
                                  original_target=target, margin=target-score, passed=score <= target))
            active = active_masks[f'raw{raw:03d}_{state}']
            union = masks['protected_nonroad_RGB'] | masks['original_road_RGB']
            assert np.array_equal(active, (warm['group_alpha_T'][7] > 0) & union)
            error = abs(c.astype(np.float64)-gt.astype(np.float64)).mean(0)
            ew = abs(w.astype(np.float64)-gt.astype(np.float64)).mean(0)
            ef = abs(f.astype(np.float64)-gt.astype(np.float64)).mean(0)
            cap = np.minimum(ew, ef) + 1e-6
            excess = error-cap
            selected = excess[active]
            relaxed_excess = selected-shared_z
            total_pixels += selected.size
            iy, ix = np.nonzero(active & (excess > 0))
            for y, x in zip(iy, ix):
                violations.append(dict(raw=raw, state=state, x=int(x), y=int(y),
                                       domain='protected_nonroad' if masks['protected_nonroad_RGB'][y, x] else 'originalroad',
                                       actual_native_error=float(error[y, x]), original_strict_cap=float(cap[y, x]),
                                       strict_excess=float(excess[y, x]), fallback_shared_z=shared_z,
                                       fallback_relaxed_cap_pass=bool(excess[y, x] <= shared_z)))
            masked_excess = np.where(active, excess, -np.inf)
            wy, wx = np.unravel_index(np.argmax(masked_excess), active.shape)
            pixels.append(dict(raw=raw, state=state, controlled_pixels=int(active.sum()),
                               strict_cap_violating_pixels=int((selected > 0).sum()),
                               strict_cap_max_excess=float(selected.max()),
                               strict_cap_pass=bool(np.all(selected <= 0)),
                               fallback_shared_z=shared_z,
                               fallback_relaxed_cap_violating_pixels=int((relaxed_excess > 0).sum()),
                               fallback_relaxed_cap_max_excess=float(relaxed_excess.max()),
                               fallback_relaxed_cap_pass=bool(np.all(relaxed_excess <= 0)),
                               worst_pixel_x=int(wx), worst_pixel_y=int(wy),
                               active_mask_exact_to_fixed_Cwarm_road_support=True))
    assert len(whole) == 16 and len(pixels) == 8 and total_pixels == registered['controlled_protection_pixels']
    invariants = csv_read(args.saved_audit / 'rendered_frozen_invariants.csv')
    assert all(r['byte_exact_to_Cwarm'] == 'True' for r in invariants if r['variant'] == name)
    old_gates = [r for r in csv_read(args.saved_audit / 'original94_gates.csv') if r['variant'] == name]
    original_photo_other = [r for r in old_gates if any(':' + field in r['name'] for field in ['a5_RGB', 'a5_upper_RGB', 'a6_RGB', 'a37_RGB'])]
    frozen_exact = saved['results'][name]['geometry_semantic_colour_byte_exact_to_Cwarm']
    scope = dict(complete16_protection_MAE_pass=all(r['passed'] for r in whole),
                 original_K6_pools_pass=saved['results'][name]['K_six_pools_pass'] == 6,
                 original_K106_bins_pass=saved['results'][name]['K106_pass'] == 106,
                 added_Kactual6_pools_pass=all(r['passed'] for r in extra_pools),
                 original_supported_vehicle_RGB_gates_pass=not any(r['passed'] == 'False' for r in original_photo_other),
                 frozen_render_geometry_and_semantic_RGB_byte_exact=bool(frozen_exact),
                 strict_main_native_pixel_caps_pass=all(r['strict_cap_pass'] for r in pixels),
                 fallback_relaxed_native_pixel_caps_pass=all(r['fallback_relaxed_cap_pass'] for r in pixels))
    strict_material = all(v for k, v in scope.items() if k != 'fallback_relaxed_native_pixel_caps_pass')
    result = dict(schema='INDEPENDENT_NATIVE_M4_MATERIAL_HARD_GATES_CPU_V1',
                  status='STRICT_NATIVE_MATERIAL_GATES_PASS' if strict_material else 'STRICT_NATIVE_MATERIAL_GATES_NOT_ALL_PASSED',
                  candidate_name=name, candidate_frames_root=str(folder.resolve()),
                  protocol_sha256=sha(args.protocol), solver_result_path=str(args.solver_result.resolve()),
                  solver_recovery_protocol_sha256=sha(args.recovery_protocol) if args.recovery_protocol else None,
                  finite_diagnostic_control_protocol_sha256=sha(args.control_protocol) if args.control_protocol else None,
                  original_registered_result_sha256=sha(args.registered_result) if args.registered_result else None,
                  solver_result_status=solve['status'], registered_fallback_used=fallback,
                  shared_pixel_slack=shared_z, scope_results=scope,
                  whole_protection_passed=sum(r['passed'] for r in whole), whole_protection_total=16,
                  extra_Kactual_pool_passed=sum(r['passed'] for r in extra_pools), extra_Kactual_pool_total=6,
                  strict_pixel_violating_pixels=sum(r['strict_cap_violating_pixels'] for r in pixels),
                  total_controlled_pixels=total_pixels, strict_native_material_gates_pass=strict_material,
                  main_LP_certified_and_native_gates_pass=bool(strict_material and not fallback and not args.control_protocol),
                  original94=saved['results'][name]['original94'],
                  original_vehicle_RGB_unsupported_gates=sum(r['passed'] == '' for r in original_photo_other),
                  pixel_numeric_policy='Float64 arithmetic on exact stored native float32 RGB, same as new sealed solver caps; old94 keeps original float32 pixel error then float64 fixed-pixel reduction.',
                  old_geometry_gates_preserved_and_separate_from_material_only_acceptance=True,
                  no_extra_evaluation_tolerance=True, original_threshold_changes=False,
                  positive_fallback_z_is_not_a_main_problem_pass=True,
                  automatic49=False, quality_accepted=False, human_verdict=None,
                  GPU_forwards=0, backwards=0, optimizer_updates=0, models_loaded=False)
    args.output.mkdir(parents=True)
    csv_write(args.output / 'complete16_protection_gates.csv', whole)
    csv_write(args.output / 'added_Kactual6_pools.csv', extra_pools)
    csv_write(args.output / 'exact_A_positive_pixel_caps.csv', pixels)
    csv_write(args.output / 'strict_pixel_violations.csv', violations)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
