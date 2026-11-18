"""One deterministic CPU-only matrix-free output-constrained H initialization projection."""
from pathlib import Path
import argparse, os, json, time
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
from scipy import ndimage as ndi
from scipy.sparse.linalg import LinearOperator, lsmr
from trellis_hf_seed import blur, output_from_raw, field_stats, sha

ROOT = Path(__file__).resolve().parents[1]


def blur_transpose(value, sigma=3.):
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius+1)
    k = np.exp(-.5*(x/sigma)**2)
    k /= k.sum()
    result = np.asarray(value, np.float64)
    for axis in (0, 1):  # transpose of forward axis1 then axis0
        v = np.moveaxis(result, axis, 0)
        N = len(v)
        padded = np.zeros((N+2*radius, *v.shape[1:]))
        for i, weight in enumerate(k):
            padded[i:i+N] += weight*v
        adj = padded[radius:radius+N].copy()
        adj[0] += padded[:radius].sum(0)
        adj[-1] += padded[radius+N:].sum(0)
        result = np.moveaxis(adj, 0, axis)
    return result


class OutputProjection:
    def __init__(self, mH, legal, columns, *, bound=.34, outside_weight=1000.):
        self.mH, self.legal, self.columns = np.asarray(mH), np.asarray(legal), np.asarray(columns)
        self.H, self.W = self.mH.shape
        self.scale = .5*bound
        self.weights = np.array([[(1-u)*(1-v), u*(1-v), (1-u)*v, u*v]
                                 for v in (.25, .75) for u in (.25, .75)])
        self.qM = self.sample(self.mH)
        self.qLegal = legal[:-1, :-1]&legal[:-1, 1:]&legal[1:, :-1]&legal[1:, 1:]
        self.node_weight = np.where(legal, 1., outside_weight)
        self.query_weight = np.where(self.qLegal, 1., outside_weight)[None]
        self.node_size = self.H*self.W
        self.query_size = 4*(self.H-1)*(self.W-1)
        self.calls = dict(matvec=0, rmatvec=0)

    def sample(self, value):
        a, b, c, d = value[:-1, :-1], value[:-1, 1:], value[1:, :-1], value[1:, 1:]
        return np.stack([w[0]*a+w[1]*b+w[2]*c+w[3]*d for w in self.weights])

    def sample_transpose(self, value):
        out = np.zeros((self.H, self.W))
        out[:-1, :-1] += np.sum(self.weights[:, 0, None, None]*value, axis=0)
        out[:-1, 1:] += np.sum(self.weights[:, 1, None, None]*value, axis=0)
        out[1:, :-1] += np.sum(self.weights[:, 2, None, None]*value, axis=0)
        out[1:, 1:] += np.sum(self.weights[:, 3, None, None]*value, axis=0)
        return out

    def forward(self, vector):
        self.calls['matvec'] += 1
        raw = np.zeros((self.H, self.W))
        raw[self.columns] = vector
        high = self.scale*(raw-blur(raw))
        node = self.node_weight*self.mH*high
        query = self.query_weight*self.qM*self.sample(high)
        return np.concatenate([node.ravel(), query.ravel()])

    def transpose(self, vector):
        self.calls['rmatvec'] += 1
        node = vector[:self.node_size].reshape(self.H, self.W)
        query = vector[self.node_size:].reshape(4, self.H-1, self.W-1)
        grad = self.node_weight*self.mH*node + self.sample_transpose(self.query_weight*self.qM*query)
        return (self.scale*(grad-blur_transpose(grad)))[self.columns]

    def operator(self):
        return LinearOperator((self.node_size+self.query_size, int(self.columns.sum())),
                              matvec=self.forward, rmatvec=self.transpose, dtype=np.float64)

    def rhs(self, effective_target):
        intrinsic = np.divide(effective_target, self.mH, out=np.zeros_like(effective_target), where=self.mH>0)
        query = self.qM*self.sample(intrinsic)
        query[:, ~self.qLegal] = 0
        return np.concatenate([(self.node_weight*effective_target).ravel(), (self.query_weight*query).ravel()])

    def exact_column_norm(self):
        radius = 9
        x = np.arange(-radius, radius+1)
        k = np.exp(-.5*(x/3.)**2); k /= k.sum()
        Gy = ndi.convolve1d(np.eye(self.H), k, axis=0, mode='nearest')
        Gx = ndi.convolve1d(np.eye(self.W), k, axis=0, mode='nearest')
        norms = []
        for y, x in zip(*np.nonzero(self.columns)):
            iy, ix = np.flatnonzero(Gy[:, y]), np.flatnonzero(Gx[:, x])
            high = -self.scale*Gy[iy, y, None]*Gx[ix, x][None, :]
            high[y-iy[0], x-ix[0]] += self.scale
            norm2 = np.sum((high*self.mH[np.ix_(iy, ix)]*self.node_weight[np.ix_(iy, ix)])**2)
            y0, y1 = max(int(iy[0])-1, 0), min(int(iy[-1]), self.H-2)
            x0, x1 = max(int(ix[0])-1, 0), min(int(ix[-1]), self.W-2)
            values = np.zeros((y1-y0+2, x1-x0+2))
            values[np.ix_(iy-y0, ix-x0)] = high
            sampled = self.sample(values)
            qweight = self.query_weight[:, y0:y1+1, x0:x1+1]
            qm = self.qM[:, y0:y1+1, x0:x1+1]
            norm2 += np.sum((sampled*qweight*qm)**2)
            norms.append(np.sqrt(norm2))
        return np.asarray(norms)


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU only')
    ap = argparse.ArgumentParser()
    ap.add_argument('--permission', type=Path, required=True)
    ap.add_argument('--target', type=Path, default=ROOT/'seed_r3_b034/H_seed.npz')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--maxiter', type=int, default=500)
    ap.add_argument('--precondition', action='store_true')
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError('preserve the one-shot projection')
    with np.load(args.permission) as z:
        p = {k: z[k] for k in z.files}
    with np.load(args.target) as z:
        target = z['desired_effective_H'].astype(np.float64)
        legal = z['prior_legal']
    mH, ids = p['mH_union'], p['world_row_grid']
    with np.load(ROOT/'extraction_r1/nodes_2cm.npz') as z:
        source_rows = z['source_world_rows'].ravel()
    yy, xx = np.where(np.isin(ids, source_rows))
    domain = np.zeros(ids.shape, bool)
    # Parameter halo9; exported crop has halo18, so all output blur support is represented.
    domain[max(int(yy.min())-9, 0):min(int(yy.max())+10, ids.shape[0]),
           max(int(xx.min())-9, 0):min(int(xx.max())+10, ids.shape[1])] = True
    columns = domain & p['common_latent_mask'] & p['geometry_support'] & ~p['guard']
    engine = OutputProjection(mH, legal, columns)
    operator = engine.operator()
    rng = np.random.default_rng(101)
    a, b = rng.normal(size=operator.shape[1]), rng.normal(size=operator.shape[0])
    lhs, rhs = np.dot(operator@a, b), np.dot(a, operator.rmatvec(b))
    adjoint_error = abs(lhs-rhs)/max(1., abs(lhs), abs(rhs))
    if adjoint_error > 1e-11:
        raise AssertionError('replicate padding / bilinear adjoint mismatch')
    start = time.monotonic()
    precondition = None
    column_norm_check = None
    solve_operator = operator
    if args.precondition:
        norms = engine.exact_column_norm()
        precondition = 1/np.sqrt(norms**2+1e-10)
        # Exact spot checks against the original matrix-free columns, independent of the stencil formula.
        errors = []
        for j in np.linspace(0, operator.shape[1]-1, 24, dtype=int):
            unit = np.zeros(operator.shape[1]); unit[j] = 1
            errors.append(abs(np.linalg.norm(operator@unit)-norms[j])/max(norms[j], 1e-30))
        column_norm_check = max(errors)
        if column_norm_check > 1e-12:
            raise AssertionError('exact column norm mismatch')
        n = operator.shape[1]
        solve_operator = LinearOperator((operator.shape[0]+n, n), dtype=np.float64,
            matvec=lambda z: np.concatenate([operator@(precondition*z), 1e-5*precondition*z]),
            rmatvec=lambda y: precondition*(operator.rmatvec(y[:operator.shape[0]])+1e-5*y[operator.shape[0]:]))
        print(json.dumps(dict(preconditioner='exact column norm', check_relative_max=column_norm_check,
                              seconds=time.monotonic()-start)), flush=True)
    raw = np.zeros((*ids.shape, 3))
    solver = []
    for channel in range(3):
        target_rhs = engine.rhs(target[..., channel])
        if precondition is not None:
            target_rhs = np.concatenate([target_rhs, np.zeros(operator.shape[1])])
        answer = lsmr(solve_operator, target_rhs, damp=0 if args.precondition else 1e-5,
                      atol=1e-8, btol=1e-8, conlim=1e10, maxiter=args.maxiter, show=False)
        raw[..., channel][columns] = answer[0] if precondition is None else precondition*answer[0]
        record = dict(channel=channel, stop_code=int(answer[1]), iterations=int(answer[2]),
                      residual_norm=float(answer[3]), normal_residual_norm=float(answer[4]), seconds=time.monotonic()-start)
        solver.append(record)
        print(json.dumps(record), flush=True)
    before = output_from_raw(raw, mH, bound=.34)
    uniform_scale = min(1., .8/max(float(np.abs(raw).max()), 1e-30),
                        .02*(1-1e-6)/max(float(np.abs(before).max()), 1e-30))
    seed = np.ascontiguousarray(np.arctanh(raw*uniform_scale), dtype=np.float32)
    replay_raw = np.tanh(seed.astype(np.float64))
    high = .17*(replay_raw-blur(replay_raw))
    actual = mH[..., None]*high
    inside_a, inside_t = actual[legal].ravel(), target[legal].ravel()
    norm_a, norm_t = np.linalg.norm(inside_a), np.linalg.norm(inside_t)
    cosine = float(np.dot(inside_a, inside_t)/max(norm_a*norm_t, 1e-30))
    energy_ratio = float(norm_a**2/max(norm_t**2, 1e-30))
    explained = float(1-np.sum((inside_a-inside_t)**2)/max(norm_t**2, 1e-30))
    outside = (~legal)&(mH>0)
    node_out = actual[outside]
    query = np.stack([engine.qM*engine.sample(high[..., c]) for c in range(3)], -1)
    q_out_mask = np.broadcast_to(~engine.qLegal, engine.qM.shape)&(engine.qM>0)
    q_out = query[q_out_mask]
    node_max = float(np.abs(node_out).max()) if node_out.size else 0.
    node_rms = float(np.sqrt(np.mean(node_out**2))) if node_out.size else 0.
    query_max = float(np.abs(q_out).max()) if q_out.size else 0.
    query_rms = float(np.sqrt(np.mean(q_out**2))) if q_out.size else 0.
    gates = dict(signal_energy_atleast_half=energy_ratio>=.5, target_explained_atleast_half=explained>=.5,
                 cosine_atleast09=cosine>=.9, outside_node_max=node_max<=1e-4, outside_node_RMS=node_rms<=1e-5,
                 outside_query_max=query_max<=1e-4, outside_query_RMS=query_rms<=1e-5)
    origin = p['crop_origin_yx']
    full = np.zeros((*tuple(p['global_chart_shape']), 3), np.float32)
    full[origin[0]:origin[0]+ids.shape[0], origin[1]:origin[1]+ids.shape[1]] = seed
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output/'H_seed.npz', H_seed_latent=full, H_seed_crop=seed, actual_effective_H=actual,
        unmodulated_H=high, desired_effective_H=target, prior_legal=legal, raw_parameter_domain=columns,
        mH_combined=mH, world_xyz=p['world_xyz'], common_material_valid=p['common_material_valid'],
        crop_origin_yx=origin, global_chart_shape=p['global_chart_shape'], crop_world_rows=ids)
    with np.load(args.output/'H_seed.npz') as saved:
        saved_actual = output_from_raw(np.tanh(saved['H_seed_crop'].astype(np.float64)), saved['mH_combined'], bound=.34)
        if not np.array_equal(saved_actual, actual):
            raise AssertionError('full serialized seed/operator restore mismatch')
    report = dict(schema='T2_HF_output_constrained_CPU_seed_projection.v1',
        status='CPU_READY_PENDING_GPU_VALIDATION' if all(gates.values()) else 'T2_EFFECTIVE_SEED_NOT_READY_DO_NOT_SCHEDULE',
        gates=gates, legal_nodes=int(legal.sum()), raw_unknown_nodes=int(columns.sum()), model_H_bound=.34, effective_HF_cap=.02,
        raw_cap=.8, uniform_scaling=uniform_scale, raw_abs_max=float(np.abs(replay_raw).max()),
        seed_latent_abs_max=float(np.abs(seed).max()), actual=field_stats(actual, legal), target=field_stats(target, legal),
        retained_signal_energy_ratio=energy_ratio, target_explained_fraction=explained, cosine=cosine,
        outside_node_max=node_max, outside_node_RMS=node_rms, outside_quarter_query_max=query_max, outside_quarter_query_RMS=query_rms,
        outside_scope='all crop output nodes and 2x2 bilinear quarter queries with any nonlegal corner; RMS restricted to nonzero common mH',
        adjoint_relative_error=adjoint_error, outside_constraint_weight=1000., solver=solver, solver_calls=engine.calls,
        maxiter=args.maxiter, exact_column_norm_precondition=args.precondition, column_norm_spot_check_relative_max=column_norm_check,
        objective_unchanged='same weighted output least squares +1e-10*||r||^2; original ridge is explicit augmented rows under right preconditioning',
        one_projection_only=True, photo_optimization=False, extra_training_loss=False, CPU_seconds=time.monotonic()-start,
        permission_path=str(args.permission), permission_sha256=sha(args.permission),
        parent_ledger_path=str(p['source_ledger_path']), parent_ledger_sha256=str(p['source_ledger_sha256']),
        common_base_buffer_sha256=str(p['common_base_buffer_sha256']), source_target_sha256=sha(args.target),
        serialized_seed_sha256=sha(args.output/'H_seed.npz'), exact_CPU_serialized_operator_restore=True,
        low_A0_A200_Adam_H_state_not_touched=True, same_loss_same_reader_same_trace_required=True,
        GPU_operations=0, new_model_inference=0, model_training_updates=0, GPU_entry_permitted=False)
    (args.output/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
