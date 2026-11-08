"""No-solve K primal witness and exact fallback matrix row diagnosis."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[key] = '1'
import numpy as np
from scipy import sparse


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def same_sparse(a, b):
    a, b = a.tocsr(), b.tocsr()
    a.sort_indices(); b.sort_indices()
    return a.shape == b.shape and np.array_equal(a.indptr, b.indptr) and np.array_equal(a.indices, b.indices) and np.array_equal(a.data, b.data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-protocol', type=Path, required=True)
    parser.add_argument('--saved-fallback-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    protocol = read(args.recovery_protocol)
    source = Path(protocol['r2_source'])
    assert sha(source) == protocol['r2_source_sha256']
    spec = importlib.util.spec_from_file_location('immutable_m4_r2', source)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    inp = SimpleNamespace(**{k: Path(v) for k, v in protocol['input_paths'].items()})
    d = m.inputs(inp)
    problem = m.compile_problem(d, fallback=True)
    r2 = Path(protocol['r2_result']).parent
    original_primary = sparse.load_npz(r2 / 'compiled_primary_A_ub.npz')
    primary_vectors = m.load(r2 / 'compiled_primary_vectors.npz')
    saved_fallback = sparse.load_npz(args.saved_fallback_root / 'identical_fallback_A_ub.npz')
    saved_vectors = m.load(args.saved_fallback_root / 'identical_fallback_vectors.npz')
    assert same_sparse(problem['A'], saved_fallback), 'Source compile differs from saved fallback matrix'
    assert np.array_equal(problem['b'], saved_vectors['b_ub'])
    assert same_sparse(problem['A'][:, :-1], original_primary)
    assert np.array_equal(problem['b'], primary_vectors['b_ub'])
    assert np.array_equal(d['delta_k'], primary_vectors['delta_K'])
    nd, ns, npv = problem['nd'], problem['ns'], problem['npv']
    J = d['j0'] + d['SA'] @ d['delta_k']
    R = d['r0'] + d['PA'] @ d['delta_k']
    error = abs(R).mean(1)
    z_margin = 1e-6
    z = max(0.0, float(np.max(error-d['pixel_cap']+m.PIXEL_RESERVE))) + z_margin
    x = np.zeros(problem['nv'], np.float64)
    x[:nd] = d['delta_k'].ravel()
    x[nd:nd+ns] = abs(J).ravel()
    x[problem['photo_epigraph_start']:problem['photo_epigraph_start']+npv] = abs(R).ravel()
    x[problem['sz']] = z
    assert np.count_nonzero(x[problem['absolute_delta_start']:problem['absolute_delta_start']+nd]) == 0
    residual = np.asarray(problem['A'] @ x - problem['b'])
    blocks = [('seam_abs_positive', ns), ('seam_abs_negative', ns),
              ('photo_abs_positive', npv), ('photo_abs_negative', npv),
              ('K_colour_delta_abs_positive', nd), ('K_colour_delta_abs_negative', nd),
              ('seam118', len(d['seam_constraints'])), ('photo16', len(d['domains'])),
              ('per_pixel_cap_with_shared_z', len(d['pixel_cap']))]
    categories, slices = [], {}
    start = 0
    for name, count in blocks:
        slices[name] = slice(start, start+count)
        selected = residual[start:start+count]
        categories.append(dict(category=name, first_matrix_row=start, row_count=count,
                               positive_residual_count=int((selected > 0).sum()),
                               violation_gt1e9_count=int((selected > 1e-9).sum()),
                               max_signed_residual=float(selected.max(initial=-np.inf)),
                               max_violation=float(selected.max(initial=0))))
        start += count
    assert start == problem['A'].shape[0]
    expected_abs = [
        (d['SA']@d['delta_k']-abs(J)+d['j0']).ravel(),
        (-d['SA']@d['delta_k']-abs(J)-d['j0']).ravel(),
        (d['PA']@d['delta_k']-abs(R)+d['r0']).ravel(),
        (-d['PA']@d['delta_k']-abs(R)-d['r0']).ravel(),
        np.zeros(nd), np.zeros(nd)]
    algebra = []
    for (name, _), expected in zip(blocks[:6], expected_abs):
        algebra.append(dict(category=name, max_matrix_vs_direct_abs=float(abs(residual[slices[name]]-expected).max(initial=0))))
    seams = []
    for i, rec in enumerate(d['seam_constraints']):
        score = float(abs(J[rec['rows']]).mean())
        direct = score-rec['solver_target']
        matrix = float(residual[slices['seam118']][i])
        seams.append(dict(index=i, name=rec['name'], state=rec['state'], role=rec['role'],
                          rows=len(rec['rows']), K_operator_score=score,
                          original_target=rec['original_target'], reserved_solver_target=rec['solver_target'],
                          matrix_residual=matrix, direct_residual=direct, algebra_abs_difference=abs(matrix-direct),
                          original_gate_pass=score <= rec['original_target'], reserved_gate_pass=direct <= 1e-9))
    # Independent-pixel box relaxation: each pixel may choose each road-parent
    # RGB independently. It gives necessary lower bounds, not joint feasibility.
    lo = d['r0'] + d['PA'] @ (-d['c0'])
    hi = d['r0'] + d['PA'] @ (1-d['c0'])
    assert np.all(lo <= hi+1e-14)
    lower_pixel = np.maximum(np.maximum(lo, -hi), 0).mean(1)
    photos = []
    kroot = inp.k_palette.parent.parent / 'K_candidate_render_v1'
    for i, rec in enumerate(d['domains']):
        selected = rec['active_photo_rows']
        score = (float(error[selected].sum())+rec['constant_error_sum'])/rec['pixels']
        lower_bound = (float(lower_pixel[selected].sum())+rec['constant_error_sum'])/rec['pixels']
        direct = score-rec['solver_target']
        matrix = float(residual[slices['photo16']][i])
        frame = next(f for f in d['frames'] if f['raw'] == rec['raw'] and f['state'] == rec['state'])
        native = m.load(kroot / f"frames/{rec['raw']:03d}/{rec['state']}_float.npz")
        native_score = m.baseline_mae(m.chw(native['rgb']), frame['obs'], frame['masks'][rec['name']])
        photos.append(dict(index=i, raw=rec['raw'], state=rec['state'], domain=rec['name'],
                           full_pixels=rec['pixels'], positive_A_pixels=len(selected),
                           constant_pixels=rec['pixels']-len(selected), constant_error_sum=rec['constant_error_sum'],
                           latest21K_MAE=rec['latest21K_MAE'], Cwarm_MAE=rec['Cwarm_MAE'], C2000_MAE=rec['C2000_MAE'],
                           K_operator_MAE=score, K_native_original_precision_MAE=native_score,
                           original_target=rec['original_target'], reserved_solver_target=rec['solver_target'],
                           matrix_residual=matrix, direct_residual=direct, algebra_abs_difference=abs(matrix-direct),
                           K_original_gate_pass=score <= rec['original_target'], K_reserved_gate_pass=direct <= 1e-9,
                           independent_pixel_box_MAE_lower_bound=lower_bound,
                           lower_bound_exceeds_original_target=lower_bound > rec['original_target'],
                           lower_bound_exceeds_reserved_target=lower_bound > rec['solver_target'],
                           box_pixel_floor_exceeds_strict_pixel_cap_count=int((lower_pixel[selected] > d['pixel_cap'][selected]).sum()),
                           K_required_MAE_reduction_to_original=max(0.0, score-rec['original_target'])))
    direct_pixel = error-z-d['pixel_cap']+m.PIXEL_RESERVE
    algebra += [dict(category='seam118', max_matrix_vs_direct_abs=max(r['algebra_abs_difference'] for r in seams)),
                dict(category='photo16', max_matrix_vs_direct_abs=max(r['algebra_abs_difference'] for r in photos)),
                dict(category='per_pixel_cap_with_shared_z', max_matrix_vs_direct_abs=float(abs(residual[slices['per_pixel_cap_with_shared_z']]-direct_pixel).max(initial=0)))]
    bounds_violation = 0.0
    for value, (lower, upper) in zip(x, problem['bounds']):
        if lower is not None: bounds_violation = max(bounds_violation, lower-value)
        if upper is not None: bounds_violation = max(bounds_violation, value-upper)
    assert np.isfinite(x).all() and np.isfinite(residual).all()
    assert max(r['max_matrix_vs_direct_abs'] for r in algebra) < 1e-12, 'Matrix/source/direct formula inconsistency'
    summary = dict(schema='INDEPENDENT_K_COMPLETE_PRIMAL_WITNESS_MATRIX_DIAGNOSIS_CPU_V1',
                   status='COMPLETE_NO_SOLVE_K_WITNESS_DIAGNOSIS',
                   source_r2=str(source), source_r2_sha256=sha(source),
                   recovery_protocol_sha256=sha(args.recovery_protocol),
                   source_compile_equals_saved_r3_fallback_CSR_exact=True,
                   saved_r2_primary_matrix_equals_fallback_without_z_exact=True,
                   saved_r2_primary_b_and_deltaK_exact=True,
                   matrix_shape=list(problem['A'].shape), matrix_nnz=problem['A'].nnz,
                   witness_definition='deltaK; seam_epi=abs(J0+SA deltaK); photo_epi=abs(r0+PA deltaK); u=0; z=max(pixelerror-cap+8e-7,0)+1e-6',
                   shared_z=z, arbitrary_positive_witness_margin=z_margin,
                   witness_bound_max_violation=bounds_violation, categories=categories,
                   direct_algebra_checks=algebra,
                   seam_original_gate_failures=sum(not r['original_gate_pass'] for r in seams),
                   seam_reserved_gate_failures_gt1e9=sum(not r['reserved_gate_pass'] for r in seams),
                   photo_original_gate_failures=sum(not r['K_original_gate_pass'] for r in photos),
                   photo_reserved_gate_failures_gt1e9=sum(not r['K_reserved_gate_pass'] for r in photos),
                   independent_pixel_box_lower_bounds_exceed_original16=sum(r['lower_bound_exceeds_original_target'] for r in photos),
                   independent_pixel_box_lower_bounds_exceed_reserved16=sum(r['lower_bound_exceeds_reserved_target'] for r in photos),
                   interpretation='K witness may fail the added photo/reserve constraints. A witness failure is not an infeasibility proof. Per-pixel box lower bounds ignore shared-parent coupling and prove impossibility only if a lower bound exceeds a target.',
                   GPU_forwards=0, backwards=0, optimizer_updates=0, solver_calls=0,
                   original_inputs_modified=False, material_candidate_exported=False, automatic49=False)
    args.output.mkdir(parents=True)
    write_csv(args.output / 'matrix_row_categories.csv', categories)
    write_csv(args.output / 'seam118_K_witness.csv', seams)
    write_csv(args.output / 'photo16_K_witness_and_box_bounds.csv', photos)
    np.savez_compressed(args.output / 'K_complete_primal_witness.npz', primal_x=x, matrix_signed_residual=residual,
                        independent_pixel_box_error_lower_bound=lower_pixel)
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
