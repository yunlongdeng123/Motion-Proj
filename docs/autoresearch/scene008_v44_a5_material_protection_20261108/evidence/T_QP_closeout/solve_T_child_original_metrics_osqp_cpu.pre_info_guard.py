"""同S的child RGB MSE，只加入原118接缝/16完整保护指标与RGBbox；单次OSQP。"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for _k in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_k] = '1'
import argparse
import hashlib
import importlib.util
import json
import time
import warnings
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import osqp
from scipy import sparse

SETTINGS = dict(time_limit=600., max_iter=100000, eps_abs=1e-8, eps_rel=1e-8,
                polishing=True, warm_starting=True, verbose=True)
BOX_CANONICALIZATION_LIMIT = 1e-7


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def read(p): return json.loads(Path(p).read_text())
def write(p, d): Path(p).write_text(json.dumps(d, indent=2, allow_nan=False)+'\n')
def module(p, name):
    spec = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def load_anchored_inputs(c):
    for key in ('r2_source', 'adapter_source', 'registered_seam_helper'):
        assert sha(c[key]) == c[key+'_sha256'], key
    m = module(c['r2_source'], 'metric_qp_inputs')
    adapter = module(c['adapter_source'], 'metric_qp_child_adapter')
    helper = module(c['registered_seam_helper'], 'metric_qp_seam_masks')
    inp = SimpleNamespace(**{k: Path(v) for k, v in c['input_paths'].items()})
    d = adapter.adapt(m.inputs(inp), c['child_operator_root'], inp.k_palette)
    assert len(d['ids']) == c['variable_child_count'] == 20725
    assert hashlib.sha256(d['ids'].tobytes()).hexdigest() == c['variable_child_ids_sha256']
    palette_path = Path(c['anchor_palette'])
    palette = m.load(palette_path)
    full = palette['full_child_RGB']
    assert full.shape == (40392, 3) and full.dtype == np.float32
    assert np.isfinite(full).all() and full.min() >= 0 and full.max() <= 1
    assert np.array_equal(palette['variable_child_ids'], d['ids'])
    assert np.array_equal(palette['child_to_parent_ids'], d['child_to_parent_ids'])
    fixed = np.ones(40392, bool); fixed[d['ids']] = False
    assert full[fixed].tobytes() == d['full'][fixed].tobytes()
    anchor_root = Path(c['anchor_native_root'])
    receipt_path = anchor_root/'receipt.json'
    rp_path = anchor_root/'protocol.json'
    receipt, render_protocol = read(receipt_path), read(rp_path)
    assert receipt['actual_native_forwards'] == 8 and receipt['actual_backwards'] == 0
    for key in ('all_nonRGB_asset_arrays_exact', 'oldH2_geometry_8groups_identityRGB_exact',
                'source_and_candidate_unchanged', 'all_seam_support_RGB_byte_exact_K'):
        assert receipt[key] is True, key
    manifest_path = Path(render_protocol['candidate_manifest'])
    manifest = read(manifest_path); asset_path = manifest_path.parent/manifest['new_road']['file']
    assert sha(asset_path) == manifest['new_road']['sha256'] == render_protocol['asset_sha256']
    assert np.array_equal(m.load(asset_path)['render_colours_srgb'], full)
    source_paths = d['paths']+[palette_path, receipt_path, rp_path, manifest_path, asset_path,
                             Path(c['registered_seam_helper'])]
    original_palette = d['full'].copy()
    original_c0 = d['c0'].copy()
    d['full'] = full.copy(); d['c0'] = full[d['ids']].astype(np.float64)
    obs_frames = {}; anchor_frames = {}; K_frames = {}; anchors = []; native_records = []
    r0_parts = []
    for f in d['frames']:
        raw, state = f['raw'], f['state']
        p = anchor_root/f'frames/{raw:03d}/{state}_float.npz'
        z = m.load(p); rgb = m.chw(z['rgb'])
        kp = Path(c['reference_K_native_root'])/f'frames/{raw:03d}/{state}_float.npz'
        K_frames[(raw, state)] = m.chw(m.load(kp)['rgb'])
        assert rgb.dtype == np.float32 and rgb.shape == (3, 320, 480)
        obs = f['obs']; pixels = f['pixel_index']
        r0 = (rgb.astype(np.float64)-obs.astype(np.float64)).reshape(3, -1)[:, pixels].T
        # 实际native起点为精确存储图，绝不以旧anchor+A*delta冒充native起点。
        old_pred = f['rgb0'].reshape(3, -1)[:, pixels].T.astype(np.float64)+f['A']@(d['c0']-original_c0)
        actual = rgb.reshape(3, -1)[:, pixels].T.astype(np.float64)
        affinity = float(np.max(abs(old_pred-actual), initial=0))
        r0_parts.append(r0); obs_frames[raw] = obs; anchor_frames[(raw, state)] = rgb
        f['rgb0'] = rgb.copy()
        for v in d['domains']:
            if (v['raw'], v['state']) != (raw, state): continue
            mask = f['masks'][v['name']]
            active = np.zeros((320, 480), bool); active.ravel()[pixels] = True
            v['constant_error_sum'] = float(abs(rgb.astype(np.float64)-obs.astype(np.float64)).mean(0)[mask & ~active].sum())
            # 仅原科学门；不继承旧LP的reserve，也不增加逐pixel或整seam guard。
            v['solver_target'] = v['original_target']
        native_records.append(dict(raw=raw, state=state, native_anchor_file=str(p),
            old_affine_to_actual_anchor_max_abs=affinity, anchor_image_reused_byte_exact=True))
        source_paths += [p, kp]
    d['r0'] = np.concatenate(r0_parts)
    seam_meta = read(inp.seam_manifest)['seam_records']
    pairs = {raw: helper.seam_pairs(raw, Path(c['world_mask_root'])) for raw in (94, 105, 128, 137)}
    j0 = []
    for row in seam_meta:
        raw, state = row['raw'], row['state']
        p = next(v for v in pairs[raw] if (v['kind'], v['edge'], v['tangent_bin']) ==
                 (row['kind'], row['edge'], row['tangent_bin']))
        for key in ('minus_pixels', 'plus_pixels', 'minus_sha256', 'plus_sha256'):
            assert p[key] == row[key]
        residual = anchor_frames[(raw, state)]-obs_frames[raw]
        support = p['plus_mask'] | p['minus_mask']
        assert np.array_equal(anchor_frames[(raw, state)][:, support], K_frames[(raw, state)][:, support])
        # 原评估定义的float32两侧均值之差，保留符号；不从score反推。
        jump = residual[:, p['plus_mask']].mean(1)-residual[:, p['minus_mask']].mean(1)
        kr = K_frames[(raw, state)]-obs_frames[raw]
        kj = kr[:, p['plus_mask']].mean(1)-kr[:, p['minus_mask']].mean(1)
        assert np.array_equal(jump, kj)
        j0.append(jump.astype(np.float64))
        anchors.append(dict(operator_row=row['operator_row'], raw=raw, state=state, kind=row['kind'],
            edge=row['edge'], tangent_bin=row['tangent_bin'], signed_native_Rchild_residual=jump.tolist()))
    d['j0'] = np.stack(j0)
    assert d['r0'].shape == (9966, 3) and d['j0'].shape == (106, 3)
    for v in d['seam_constraints']:
        v['solver_target'] = v['original_target']
    d['paths'] = list(dict.fromkeys(source_paths))
    d['anchor_audit'] = dict(status='PASS_ACTUAL_RCHILD_NATIVE_ANCHOR_AND_SAME_FIXED_DATA',
        palette=str(palette_path), native_root=str(anchor_root), native_records=native_records,
        signed_j0_source='106 actual native float32 residual jumps; unchanged registered masks and signed orientation',
        actual_native_j0_equals_K_signed_byte_exact=True,
        seam_records=anchors, residual_pixels=9966, child_variables=20725,
        fixed_child_count=int(fixed.sum()), frozen_children_anchor_exact=True,
        geometry_opacity_actors_identity_frozen=True, setup_calls=0, QP_calls=0, GPU_forwards=0)
    return d


def compile_qp(d):
    nd, nr, ns = d['c0'].size, d['r0'].size, d['j0'].size
    sr, se, ss = nd, nd+nr, nd+2*nr
    nv = nd+2*nr+ns
    B = sparse.kron(d['PA'], sparse.eye(3), format='csr')
    S = sparse.kron(sparse.csr_matrix(d['SA']), sparse.eye(3), format='csr')
    blocks, lower, upper, labels = [], [], [], []

    def add(parts, l, u, name):
        n = len(l)
        blocks.append(sparse.hstack([parts.get(start, sparse.csr_matrix((n, width)))
            for start, width in ((0, nd), (sr, nr), (se, nr), (ss, ns))], format='csr'))
        lower.extend(np.asarray(l).ravel()); upper.extend(np.asarray(u).ravel())
        labels.append(dict(name=name, first_row=len(lower)-n, rows=n))

    I, E = sparse.eye(nd, format='csr'), sparse.eye(nr, format='csr')
    add({0: I}, -d['c0'].ravel(), 1-d['c0'].ravel(), 'RGB_box')
    add({0: -B, sr: E}, d['r0'].ravel(), d['r0'].ravel(), 'lifted_native_photo_residual')
    add({sr: E, se: -E}, np.full(nr, -np.inf), np.zeros(nr), 'photo_absolute_positive')
    add({sr: -E, se: -E}, np.full(nr, -np.inf), np.zeros(nr), 'photo_absolute_negative')
    J = sparse.eye(ns, format='csr')
    add({0: S, ss: -J}, np.full(ns, -np.inf), -d['j0'].ravel(), 'seam_absolute_positive')
    add({0: -S, ss: -J}, np.full(ns, -np.inf), d['j0'].ravel(), 'seam_absolute_negative')
    R = sparse.lil_matrix((118, ns))
    for i, v in enumerate(d['seam_constraints']):
        for row in v['rows']: R[i, row*3:row*3+3] = 1/(3*len(v['rows']))
    add({ss: R.tocsr()}, np.full(118, -np.inf),
        [v['original_target'] for v in d['seam_constraints']], 'original118_seam_metrics')
    D = sparse.lil_matrix((16, nr)); rhs = []
    for i, v in enumerate(d['domains']):
        for row in v['active_photo_rows']: D[i, row*3:row*3+3] = 1/(3*v['pixels'])
        rhs.append(v['original_target']-v['constant_error_sum']/v['pixels'])
    add({se: D.tocsr()}, np.full(16, -np.inf), rhs, 'original16_complete_photo_MAE')
    A = sparse.vstack(blocks, format='csc'); l, u = np.array(lower), np.array(upper)
    pdiag = np.zeros(nv); pdiag[sr:sr+nr] = 2/nr
    P = sparse.diags(pdiag, format='csc'); q = np.zeros(nv)
    scales = np.ones(A.shape[0])
    seam_rows = next(v for v in labels if v['name'] == 'original118_seam_metrics')['first_row']
    photo_rows = next(v for v in labels if v['name'] == 'original16_complete_photo_MAE')['first_row']
    for i, v in enumerate(d['seam_constraints']): scales[seam_rows+i] = 3*len(v['rows'])
    for i, v in enumerate(d['domains']): scales[photo_rows+i] = 3*v['pixels']
    solver_A = sparse.diags(scales, format='csc')@A
    solver_l, solver_u = l*scales, u*scales
    restored = sparse.diags(1/scales, format='csc')@solver_A-A
    err = float(np.max(abs(restored.data), initial=0))
    assert err <= 8*np.finfo(np.float64).eps
    warm = np.zeros(nv)
    warm[sr:sr+nr] = d['r0'].ravel(); warm[se:se+nr] = abs(d['r0']).ravel()
    warm[ss:ss+ns] = abs(d['j0']).ravel()
    assert P.nnz == nr and np.count_nonzero(pdiag[:sr]) == 0
    return dict(P=P, q=q, A=A, l=l, u=u, solver_A=solver_A, solver_l=solver_l, solver_u=solver_u,
        row_scales=scales, warm_x=warm, warm_y=np.zeros(A.shape[0]), row_blocks=labels,
        nd=nd, nr=nr, ns=ns, nv=nv, r_start=sr, e_start=se, s_start=ss,
        positive_row_scale_roundtrip_max_abs=err)


def matrix_certificate(problem, x, y=None):
    value = problem['A']@x
    low = np.maximum(problem['l']-value, 0); high = np.maximum(value-problem['u'], 0)
    violation = np.maximum(low, high)
    out = dict(full_original_matrix_max_violation=float(np.max(violation, initial=0)),
        per_block=[dict(**v, max_violation=float(np.max(violation[v['first_row']:v['first_row']+v['rows']], initial=0)))
                   for v in problem['row_blocks']], objective=float(.5*x@(problem['P']@x)))
    if y is not None:
        original_y = problem['row_scales']*y
        stationarity = problem['P']@x+problem['q']+problem['A'].T@original_y
        out['original_dual_stationarity_max_abs'] = float(np.max(abs(stationarity), initial=0))
    return out


def scientific_certificate(d, delta):
    r = d['r0']+d['PA']@delta; j = d['j0']+d['SA']@delta
    photo = abs(r).mean(1); seams, domains = [], []
    for v in d['seam_constraints']:
        value = float(abs(j[v['rows']]).mean())
        seams.append(dict(name=v['name'], state=v['state'], role=v['role'], value=value,
            original_target=v['original_target'], margin=v['original_target']-value,
            passed=bool(value <= v['original_target'])))
    for v in d['domains']:
        value = (float(photo[v['active_photo_rows']].sum())+v['constant_error_sum'])/v['pixels']
        domains.append({k: v[k] for k in ('raw', 'state', 'name', 'pixels', 'latest21K_MAE')} |
            dict(value=value, original_target=v['original_target'], margin=v['original_target']-value,
                 constant_error_sum=v['constant_error_sum'], passed=bool(value <= v['original_target'])))
    box = float(max(0., np.max(-d['c0']-delta, initial=0), np.max(d['c0']+delta-1, initial=0)))
    return dict(seams=seams, domains=domains, original118_pass=sum(v['passed'] for v in seams),
        original16_pass=sum(v['passed'] for v in domains), box_max_violation=box,
        equal_9966_pixel_channel_MSE=float(np.mean(r*r)),
        passed=bool(box == 0 and all(v['passed'] for v in seams+domains)))


def canonical_witness(problem, d, delta):
    x = np.zeros(problem['nv']); x[:problem['nd']] = delta.ravel()
    r = (d['r0']+d['PA']@delta).ravel(); j = (d['j0']+d['SA']@delta).ravel()
    x[problem['r_start']:problem['r_start']+problem['nr']] = r
    x[problem['e_start']:problem['e_start']+problem['nr']] = abs(r)
    x[problem['s_start']:problem['s_start']+problem['ns']] = abs(j)
    return x


def primal_infeasibility_check(problem, cert):
    v = np.asarray(cert, np.float64)
    if v.shape != (problem['A'].shape[0],) or not np.isfinite(v).all() or np.max(abs(v), initial=0) == 0:
        return dict(independently_validated=False, reason='No finite nonzero primal certificate')
    v = problem['row_scales']*v
    v /= np.max(abs(v)); pos, neg = np.maximum(v, 0), np.minimum(v, 0)
    infinite = bool(np.any((pos > 0)&~np.isfinite(problem['u'])) or np.any((neg < 0)&~np.isfinite(problem['l'])))
    bound = float(problem['u'][np.isfinite(problem['u'])]@pos[np.isfinite(problem['u'])]+
                  problem['l'][np.isfinite(problem['l'])]@neg[np.isfinite(problem['l'])])
    residual = float(np.max(abs(problem['A'].T@v), initial=0))
    return dict(normalized_original_ATv_max_abs=residual, support_bound=bound,
        invalid_infinite_support=infinite, independently_validated=bool(not infinite and residual <= 1e-8 and bound < -1e-8),
        scope='Only this fixed anchored20725childRGB box and original118+16 constraints; no global material or geometry impossibility claim')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['inspect', 'solve'], default='inspect')
    args = p.parse_args(); assert not args.output.exists(); args.output.mkdir(parents=True)
    c = read(args.protocol)
    assert c['status'] == 'SEALED_BEFORE_SINGLE_ORIGINAL_METRICS_QP'
    assert sha(__file__) == c['source_sha256'] and osqp.__version__ == '1.0.4'
    assert c['settings'] == SETTINGS and c['RGB_endpoint_canonicalization_limit'] == BOX_CANONICALIZATION_LIMIT
    d = load_anchored_inputs(c); problem = compile_qp(d)
    source_paths = d['paths']+[Path(__file__), args.protocol]
    before = {str(v): sha(v) for v in source_paths}
    warm_cert = matrix_certificate(problem, problem['warm_x'])
    anchor_cert = scientific_certificate(d, np.zeros_like(d['c0']))
    result = dict(schema='T_CHILD_ORIGINAL_METRICS_OSQP_RESULT_V1', status='QP_PREPARED_NOT_LAUNCHED',
        source_sha256=sha(__file__), protocol_sha256=sha(args.protocol), osqp_version=osqp.__version__,
        mode=args.mode, material_basis='child_RGB', variable_child_count=len(d['ids']), fixed_child_count=40392-len(d['ids']),
        protected_pixels=9966, residual_channels=problem['nr'], QP_variables=problem['nv'],
        constraint_shape=list(problem['A'].shape), constraint_nnz=problem['A'].nnz, P_nnz=problem['P'].nnz,
        P_only_lifted_r_diagonal=True, formed_B_transpose_B=False,
        strict_pixel_cap_constraints=0, entire_seam_RGB_guard_constraints=0,
        original118_seam_constraints=118, original16_full_domain_MAE_constraints=16,
        warm_start_matrix_certificate=warm_cert, warm_start_original_scientific_certificate=anchor_cert,
        warm_start_feasibility_claim=False, actual_setup_calls=0, actual_QP_calls=0,
        actual_LP_calls=0, actual_LS_calls=0, GPU_forwards=0, backwards=0,
        optimizer_updates=0, original_model_updates=0, automatic49=False, quality_accepted=False, human_verdict=None)
    write(args.output/'anchor_audit.json', d['anchor_audit'])
    sparse.save_npz(args.output/'P_lifted_residual_diagonal.npz', problem['P'])
    sparse.save_npz(args.output/'A_original_unscaled.npz', problem['A'])
    np.savez_compressed(args.output/'QP_vectors_and_warm_start.npz', q=problem['q'], l=problem['l'], u=problem['u'],
        row_scales=problem['row_scales'], warm_x=problem['warm_x'], warm_y=problem['warm_y'],
        variable_child_ids=d['ids'], anchor_child_RGB=d['c0'], r0_actual_native=d['r0'], j0_actual_native=d['j0'])
    write(args.output/'prepared.json', result)
    if args.mode == 'solve':
        lock = Path(c['single_call_lock'])
        fd = os.open(str(lock), os.O_CREAT|os.O_EXCL|os.O_WRONLY)
        with os.fdopen(fd, 'w') as f: json.dump(dict(source_sha256=sha(__file__), protocol_sha256=sha(args.protocol), started_unix=time.time()), f)
        solver = osqp.OSQP(algebra='builtin')
        setup_start = time.monotonic()
        solver.setup(P=problem['P'], q=problem['q'], A=problem['solver_A'], l=problem['solver_l'], u=problem['solver_u'], **SETTINGS)
        result['actual_setup_calls'] = 1
        result['setup_wall_seconds'] = time.monotonic()-setup_start
        solver.warm_start(x=problem['warm_x'], y=problem['warm_y'])
        solve_start = time.monotonic()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always'); sol = solver.solve(raise_error=False)
        result.update(actual_QP_calls=1, solve_wall_seconds=time.monotonic()-solve_start,
            solve_warnings=[str(v.message) for v in caught], solver_info={k: v for k, v in vars(sol.info).items()
                if isinstance(v, (str, bool, int, float)) and (not isinstance(v, float) or np.isfinite(v))},
            status='OSQP_NO_CERTIFIED_COLOUR_CANDIDATE', optimality_claim=False)
        np.savez_compressed(args.output/'OSQP_primal_dual_and_infeasibility_certificates.npz',
            x=np.asarray(sol.x), y=np.asarray(sol.y), prim_inf_cert=np.asarray(sol.prim_inf_cert), dual_inf_cert=np.asarray(sol.dual_inf_cert))
        result['original_matrix_primal_infeasibility_certificate_check'] = primal_infeasibility_check(problem, sol.prim_inf_cert)
        x, y = np.asarray(sol.x), np.asarray(sol.y)
        if x.shape == (problem['nv'],) and np.isfinite(x).all():
            result['raw_original_matrix_certificate'] = matrix_certificate(problem, x, y if np.isfinite(y).all() else None)
            delta = x[:problem['nd']].reshape(-1, 3); raw_colour = d['c0']+delta
            box_error = float(max(0., -raw_colour.min(), raw_colour.max()-1))
            result['raw_RGB_box_max_violation'] = box_error
            if box_error <= BOX_CANONICALIZATION_LIMIT:
                colour = np.minimum(np.maximum(raw_colour, 0), 1)
                change = abs(colour-raw_colour)
                selected = np.asarray(colour, np.float32)
                d32 = selected.astype(np.float64)-d['c0']
                cert64 = scientific_certificate(d, colour-d['c0']); cert32 = scientific_certificate(d, d32)
                witness = canonical_witness(problem, d, d32)
                witness_cert = matrix_certificate(problem, witness)
                result.update(RGB_endpoint_canonicalization=dict(limit=BOX_CANONICALIZATION_LIMIT,
                    raw_box_max_violation=box_error, changed_channel_values=int(np.count_nonzero(change)),
                    changed_child_rows=int(np.count_nonzero(np.any(change>0, axis=1))), max_correction=float(change.max(initial=0)),
                    no_blending_no_shrink_no_resolve=True),
                    float64_certificate=cert64, float32_certificate=cert32,
                    float64_certificate_pass=cert64['passed'], float32_certificate_pass=cert32['passed'],
                    canonical_float32_auxiliary_matrix_certificate=witness_cert,
                    original_photo_gates_pass=bool(cert32['original16_pass'] == 16),
                    original_seam_gates_pass=bool(cert32['original118_pass'] == 118))
                if cert32['passed']:
                    full = d['full'].copy(); full[d['ids']] = selected
                    fixed = np.ones(40392, bool); fixed[d['ids']] = False
                    assert full[fixed].tobytes() == d['full'][fixed].tobytes()
                    dest = args.output/'T_child_original_metrics_QP_RGB.npz'
                    np.savez_compressed(dest, full_child_RGB=full, variable_child_ids=d['ids'], variable_child_RGB=selected,
                        child_to_parent_ids=d['child_to_parent_ids'], delta_from_Rchild=d32)
                    result.update(status='QP_ORIGINAL_METRICS_CERTIFIED_PENDING_NATIVE8F', export_integrity_pass=True,
                        optimality_claim=bool(sol.info.status_val == 1),
                        candidate=dict(path=str(dest), sha256=sha(dest), full_child_rows=40392,
                            variable_child_rows=len(d['ids']), fixed_children_bitwise_unchanged_Rchild=True))
                else:
                    result['status'] = 'QP_POST_CANONICALIZATION_ORIGINAL_METRIC_FAILED_STOP'
            else:
                result['status'] = 'QP_RGB_BOX_NUMERICAL_ERROR_EXCEEDS_1E_MINUS7_STOP'
    assert all(sha(v) == before[str(v)] for v in source_paths)
    result.update(source_integrity='PASS_READ_INPUT_BYTES_UNCHANGED',
        input_files=[dict(path=str(v), sha256=before[str(v)]) for v in source_paths],
        scope='Same9966obsRGB MSE as S, actual Rchild native anchor, existing childRGB only. Only original118+16 quality constraints and RGBbox. Any original metric failure after tinybox canonicalization/f32 export stops adoption. Native8F and image review remain required; inherited geometry failures unchanged.')
    write(args.output/'result.json', result)
    print(json.dumps({k: result.get(k) for k in ('status', 'mode', 'actual_setup_calls', 'actual_QP_calls', 'constraint_shape', 'candidate')}), flush=True)


if __name__ == '__main__': main()
