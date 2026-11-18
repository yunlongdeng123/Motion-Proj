"""CPU dual integration and save/reload checks of the same world H initialization; no renderer."""
from pathlib import Path
import argparse, os, sys, json, time, math
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from trellis_hf_seed import blur, sha

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'source_snapshot'))
from road_chart_sampling import StructuredRoadChart


def sample0(field, uv):
    """align_corners pixel-coordinate bilinear with true zero padding, including partial border cells."""
    shape = uv.shape[:-1]
    p = np.nan_to_num(np.asarray(uv).reshape(-1, 2), nan=-1e9)
    lo = np.floor(p).astype(np.int64); t = p-lo
    H, W, C = field.shape
    out = np.zeros((len(p), C))
    for dx, dy in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        x, y = lo[:, 0]+dx, lo[:, 1]+dy
        weight = (t[:, 0] if dx else 1-t[:, 0])*(t[:, 1] if dy else 1-t[:, 1])
        valid = (x>=0)&(x<W)&(y>=0)&(y<H)
        out[valid] += weight[valid, None]*field[y[valid], x[valid]]
    return out.reshape(*shape, C)


def stats(a, b):
    d = np.abs(a-b)
    rowmax = d.max(-1)
    return dict(RGB_RMSE=float(np.sqrt(np.mean((a-b)**2))), rowmax_p99=float(np.quantile(rowmax, .99)),
                abs_max=float(d.max()), p99_1code=bool(np.quantile(rowmax, .99)<=1/255), max_4code=bool(d.max()<=4/255))


def main():
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise RuntimeError('CPU-only')
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=Path, required=True)
    ap.add_argument('--geometry', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    start = time.monotonic()
    with np.load(a.seed) as z:
        d = {k:z[k] for k in z.files}
    with np.load(a.geometry) as z:
        geo = {k:z[k] for k in z.files}
    full_raw = np.tanh(d['H_seed_latent'].astype(np.float64))
    high = .17*(full_raw-blur(full_raw))
    y0, x0 = d['crop_origin_yx']; H, W = d['mH_combined'].shape
    gate = np.zeros(high.shape[:2])
    gate[y0:y0+H, x0:x0+W] = d['mH_combined']
    outside = np.ones(high.shape[:2], bool); outside[y0:y0+H, x0:x0+W] = False
    outside_high_max = float(np.abs(high[outside]).max())
    if outside_high_max != 0:
        raise AssertionError('crop halo omitted nonzero global model output')
    if not np.array_equal(geo['world_xyz'][y0:y0+H, x0:x0+W], d['world_xyz']):
        raise ValueError('full chart geometry differs from source permission crop')
    if not np.array_equal(geo['common_material_valid'][y0:y0+H, x0:x0+W], d['common_material_valid']):
        raise ValueError('global common material availability differs')
    chart = StructuredRoadChart(geo['world_xyz'], geo['world_from_uv'])
    valid = geo['common_material_valid'][..., None].astype(np.float64)

    def query(world):
        result = chart.query(world)
        vm = sample0(valid, result['uv'])[..., 0]*result['valid']
        h = sample0(high, result['uv'])
        m = sample0(gate[..., None], result['uv'])[..., 0]
        return vm[..., None]*m[..., None]*h, vm

    with np.load(ROOT/'source_snapshot/previous_541_geometry.npz') as z:
        center, covariance, world_rows = z['world_xyz'][:, :2], z['world_covariance_xy'], z['active_world_rows']
    chol = np.linalg.cholesky(covariance)
    results = {}; masses = {}
    for order in [15, 41]:
        roots, weight = np.polynomial.hermite.hermgauss(order)
        xx, yy = np.meshgrid(roots, roots); wx, wy = np.meshgrid(weight, weight)
        nodes = np.stack([xx.ravel(), yy.ravel()], -1)*np.sqrt(2); weights = (wx*wy).ravel()/np.pi
        color = np.zeros((len(center), 3)); mass = np.zeros(len(center))
        for begin in range(0, len(center), 24):
            end = min(begin+24, len(center))
            world = center[begin:end, None]+np.einsum('nij,qj->nqi', chol[begin:end], nodes)
            num, den = query(world)
            mass[begin:end] = np.sum(weights*den, axis=1)
            color[begin:end] = np.sum(weights[None, :, None]*num, axis=1)/np.maximum(mass[begin:end, None], 1e-12)
        results[order], masses[order] = color, mass
    point_num, point_den = query(center)
    point = point_num/np.maximum(point_den[:, None], 1e-12)
    subset = np.linspace(0, len(center)-1, 16, dtype=int)  # fixed indices, independent of errors or signal
    midpoint = {}
    for step in [.00125, .000625]:
        values = []
        for row in subset:
            R = 5*np.sqrt(np.linalg.eigvalsh(covariance[row]).max())
            N = int(np.ceil(2*R/step)); N += N%2
            axis = (np.arange(N)+.5-N/2)*(2*R/N)
            xx, yy = np.meshgrid(axis, axis); delta = np.stack([xx.ravel(), yy.ravel()], -1)
            weights = np.exp(-.5*np.einsum('ni,ij,nj->n', delta, np.linalg.inv(covariance[row]), delta))
            num, den = query(center[row]+delta)
            values.append((weights[:, None]*num).sum(0)/max(float(weights@den), 1e-12))
        midpoint[step] = np.asarray(values)
    a.output.mkdir(parents=True)
    np.savez_compressed(a.output/'reader_arrays.npz', active_world_rows=world_rows, GH15=results[15], GH41=results[41],
        common_mass_GH15=masses[15], common_mass_GH41=masses[41], point_reader=point,
        midpoint_subset_local_rows=subset, midpoint_1250=midpoint[.00125], midpoint_0625=midpoint[.000625])
    report = dict(schema='T2_HF_CPU_same_world_field_dual_reader_integral.v1',
        status='NUMERICAL_CHECK_ONLY_DOES_NOT_OVERRIDE_SIGNAL_GATE', fixed_previous_rows=541,
        GH15_vs_GH41=stats(results[15], results[41]), midpoint1250_vs0625_fixed16=stats(midpoint[.00125], midpoint[.000625]),
        GH41_vs_midpoint0625_fixed16=stats(results[41][subset], midpoint[.000625]),
        point_vs_average_expected_difference=stats(point, results[41]), global_source_availability_used=True,
        normalized_filter='sum(w*validmass*bilinear(common_mH)*bilinear(H))/sum(w*validmass); no premultiplied gate/H interpolation',
        mass_below09_rows=int((masses[41]<.9).sum()), global_model_H_outside_stored_crop_exact_zero=True,
        full_model_high_rebuild_vs_stored_crop_max=float(np.abs(high[y0:y0+H, x0:x0+W]-d['unmodulated_H']).max()),
        tail_probability_bound_midpoint_box=2*math.erfc(5/np.sqrt(2)),
        input_seed_sha256=sha(a.seed), global_geometry_sha256=sha(a.geometry),
        source_fixture_sha256=sha(ROOT/'source_snapshot/previous_541_geometry.npz'),
        generated_alpha_used=False, GPU_operations=0, model_inference_calls=0, optimizer_updates=0,
        CPU_seconds=time.monotonic()-start, GPU_entry_permitted=False)
    (a.output/'receipt.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
