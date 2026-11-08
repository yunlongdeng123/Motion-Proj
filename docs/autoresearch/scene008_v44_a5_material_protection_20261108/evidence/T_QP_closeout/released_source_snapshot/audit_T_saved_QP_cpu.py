"""T的0solver保存结果复算：原矩阵、raw物理残差、盒端点修正及f32门。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'): os.environ[key] = '1'
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from scipy import sparse


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def exact_sparse(a, b):
    a, b = a.tocsc(), b.tocsc(); a.sort_indices(); b.sort_indices()
    return a.shape == b.shape and all(np.array_equal(x, y) for x, y in
        [(a.indptr, b.indptr), (a.indices, b.indices), (a.data, b.data)])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--source', type=Path, required=True)
    p.add_argument('--protocol', type=Path, required=True); p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True); a = p.parse_args(); assert not a.output.exists()
    c, result = read(a.protocol), read(a.run/'result.json')
    assert sha(a.source) == c['source_sha256'] == result['source_sha256']
    assert sha(a.protocol) == result['protocol_sha256']
    spec = importlib.util.spec_from_file_location('saved_T_solver', a.source)
    t = importlib.util.module_from_spec(spec); spec.loader.exec_module(t)
    d = t.load_anchored_inputs(c); problem = t.compile_qp(d)
    assert exact_sparse(problem['A'], sparse.load_npz(a.run/'A_original_unscaled.npz'))
    assert exact_sparse(problem['P'], sparse.load_npz(a.run/'P_lifted_residual_diagonal.npz'))
    with np.load(a.run/'QP_vectors_and_warm_start.npz', allow_pickle=False) as z:
        for key in ('q', 'l', 'u', 'row_scales', 'warm_x', 'warm_y'): assert np.array_equal(z[key], problem[key]), key
    with np.load(a.run/'OSQP_primal_dual_and_infeasibility_certificates.npz', allow_pickle=False) as z:
        x, y, primal_certificate = z['x'].copy(), z['y'].copy(), z['prim_inf_cert'].copy()
    out = dict(schema='T_SAVED_QP_FULL_CPU_AUDIT_V1', solver_calls=0, QP_calls=0, GPU_forwards=0,
        matrices_and_vectors_recompiled_exact=True, P_only_29898_lifted_residual_diagonal=problem['P'].nnz == 29898,
        no_BtransposeB=True, actual_earlier_QP_calls=result['actual_QP_calls'], solver_status=result.get('solver_info'),
        actual_native_anchor_audit=d['anchor_audit'], setup_calls=0,
        source_sha256=sha(a.source), protocol_sha256=sha(a.protocol), result_sha256=sha(a.run/'result.json'))
    if x.shape == (problem['nv'],) and np.isfinite(x).all():
        delta = x[:problem['nd']].reshape(-1, 3)
        out['raw_original_matrix'] = t.matrix_certificate(problem, x, y if np.isfinite(y).all() else None)
        out['raw_physical_colour_scientific_metrics'] = t.scientific_certificate(d, delta)
        raw = d['c0']+delta
        box = float(max(0., -raw.min(), raw.max()-1))
        out['raw_box_max_violation'] = box
        out['endpoint_canonicalization_permitted'] = box <= 1e-7
        if box <= 1e-7:
            projected = np.minimum(np.maximum(raw, 0), 1)
            colour = np.asarray(projected, np.float32); d32 = colour.astype(np.float64)-d['c0']
            out['raw_to_endpoint_change_count'] = int(np.count_nonzero(projected != raw))
            out['raw_to_endpoint_max_change'] = float(abs(projected-raw).max(initial=0))
            out['float32_original_scientific_metrics'] = t.scientific_certificate(d, d32)
            out['canonical_f32_witness_original_matrix'] = t.matrix_certificate(problem, t.canonical_witness(problem, d, d32))
            if result.get('candidate'):
                cp = Path(result['candidate']['path']); assert sha(cp) == result['candidate']['sha256']
                with np.load(cp, allow_pickle=False) as z:
                    full = z['full_child_RGB']; assert full.dtype == np.float32 and full.shape == (40392, 3)
                    assert np.array_equal(z['variable_child_ids'], d['ids'])
                    assert np.array_equal(full[d['ids']], colour)
                    fixed = np.ones(40392, bool); fixed[d['ids']] = False
                    assert full[fixed].tobytes() == d['full'][fixed].tobytes()
                assert out['float32_original_scientific_metrics']['passed']
                out['candidate_exactly_matches_declared_tinybox_then_float32'] = True
                out['fixed19667_child_RGB_byte_exact_Rchild'] = True
    out['original_matrix_primal_infeasibility_certificate'] = t.primal_infeasibility_check(problem, primal_certificate)
    out['quality_accepted'] = False; out['human_verdict'] = None; out['automatic49'] = False
    a.output.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(a.output), matrix_exact=True, solver_calls=0,
        raw_box=out.get('raw_box_max_violation'), f32_pass=out.get('float32_original_scientific_metrics', {}).get('passed'))))


if __name__ == '__main__': main()
