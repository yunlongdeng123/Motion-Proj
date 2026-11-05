"""BUILD-only attribution of physical crops to composited Gaussian-center depth.

Weights use PARENT transmittance: T_parent * alpha_native * .5 * crop_fraction.
These are first-order pre-retirement contribution assignments, not actual
alpha*T differences after retirement, where downstream transmittance changes.
"""
import gc
import hashlib
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import torch

from layered import RUN, Layered, fixed_group_mask, core

LAYERS = ('near', 'same', 'far')
RETIREMENT = .5
DEPTH_BAND_M = .12


def moments(weight, difference):
    return dict(mass=float(weight.sum()), signed_sum_m=float((weight*difference).sum()),
                absolute_sum_m=float((weight*difference.abs()).sum()))


def complete(moment, total=None):
    mass = moment['mass']
    return dict(moment, weighted_signed_mean_m=moment['signed_sum_m']/mass if mass else None,
                weighted_absolute_mean_m=moment['absolute_sum_m']/mass if mass else None,
                fraction_of_total=mass/total if total else None)


def parent_hits(experiment, bundle, mask):
    """Use exactly Bundle.coverage's FP64 selected-ray prefix and cutoff."""
    trace, window = bundle.trace, bundle.window
    counts = (trace.offsets[1:]-trace.offsets[:-1]).long()
    pixels = trace.pixels.long()
    slots = torch.where(mask.flatten()[pixels] & (counts > 0))[0]
    counts = counts[slots]
    n = int(counts.sum())
    bounds = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
    ray = torch.arange(len(slots), device='cuda').repeat_interleave(counts, output_size=n)
    hit = trace.offsets[slots[ray]].long()+torch.arange(n, device='cuda')-bounds[ray]
    rows = window.ids[hit].long()
    # alphas(0,0) zeros proxy rows and applies the native 1/255 threshold.
    alpha = bundle.alphas(0., 0.)[hit].double()
    logs = torch.cat((alpha.new_zeros(1), torch.log1p(-alpha).cumsum(0)))
    before = (logs[:-1]-logs[bounds[ray]]).exp()
    active = before*(1-alpha) >= .0001
    native = rows < len(experiment.actor_lineage[trace.raw])
    parent_weight = before*alpha*active*native
    pixel = pixels[slots[ray]]
    measured = experiment.tr.raw(trace.raw)['depth'].flatten()[pixel].double()
    difference = trace.feature[rows, 0].double()/experiment.tr.scale-measured
    return hit, parent_weight, difference


def diagnose(experiment, raw):
    experiment.set_layer('near')
    trace = experiment.merged(raw, 'black', True)
    mask = fixed_group_mask(experiment, raw)
    first = core.Bundle(experiment, trace)
    hit, parent_weight, difference = parent_hits(experiment, first, mask)
    rendered = dict(near=difference < -DEPTH_BAND_M,
                    same=difference.abs() <= DEPTH_BAND_M,
                    far=difference > DEPTH_BAND_M)
    if not torch.isfinite(difference[parent_weight > 0]).all():
        raise RuntimeError(f'Nonfinite center-depth difference at BUILD raw{raw}')
    output = {}
    for layer in LAYERS:
        experiment.set_layer(layer)
        bundle = first if layer == 'near' else core.Bundle(experiment, trace)
        weight = parent_weight*RETIREMENT*bundle.fraction[hit].double()
        total = moments(weight, difference)
        bins = {name: moments(weight[keep], difference[keep]) for name, keep in rendered.items()}
        _, parts = bundle.coverage(0., 0., mask)
        expected = RETIREMENT*parts['native']
        error = abs(total['mass']-expected)
        assert error <= 1e-9+abs(expected)*1e-9, (raw, layer, total['mass'], expected)
        assert abs(sum(v['mass'] for v in bins.values())-total['mass']) <= 1e-9+total['mass']*1e-9
        output[layer] = dict(
            physical_layer=layer, raw=int(raw), measured_group_pixels=int(mask.sum()),
            first_order_pre_retirement_total=complete(total),
            rendered_center_depth_bins={name: complete(value, total['mass']) for name, value in bins.items()},
            parent_native_alpha_T=float(parent_weight.sum()),
            retired_nonzero_hit_count=int((weight > 0).sum()),
            half_parent_layer_native_coverage=expected, coverage_consistency_abs_error=error,
        )
        del bundle
    return output


def pooled(rows):
    def add(items):
        return {key:sum(item[key] for item in items) for key in ('mass', 'signed_sum_m', 'absolute_sum_m')}
    total = add([r['first_order_pre_retirement_total'] for r in rows])
    bins = {name:complete(add([r['rendered_center_depth_bins'][name] for r in rows]), total['mass'])
            for name in LAYERS}
    return dict(build_frames=len(rows), measured_group_pixels=sum(r['measured_group_pixels'] for r in rows),
                first_order_pre_retirement_total=complete(total), rendered_center_depth_bins=bins,
                coverage_consistency_max_abs_error=max(r['coverage_consistency_abs_error'] for r in rows))


def main():
    assert len(core.BUILD) == 40 and not set(core.BUILD).intersection(core.HELD)
    groups = []
    for rank in range(3):
        path = RUN/f'group_ledger_{rank}.npz'
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path, allow_pickle=False) as bank:
            ledger = {key:bank[key] for key in bank.files}
        experiment = Layered(ledger, 'near')
        try:
            rows = {layer:[] for layer in LAYERS}
            for raw in core.BUILD:
                frame = diagnose(experiment, raw)
                for layer in LAYERS:
                    rows[layer].append(frame[layer])
            groups.append(dict(rank=rank, ledger_sha256=digest,
                physical_layers={layer:dict(pooled=pooled(rows[layer]), rows=rows[layer]) for layer in LAYERS}))
            assert experiment.tr.exp.unchanged()
            assert hashlib.sha256(path.read_bytes()).hexdigest() == digest
            print('RETIRED_DEPTH_AUDIT', rank,
                  {layer:pooled(rows[layer])['rendered_center_depth_bins'] for layer in LAYERS}, flush=True)
        finally:
            del experiment
            gc.collect()
            torch.cuda.empty_cache()
    core.write(RUN/'retired_depth_audit.json', dict(
        status='EXECUTION_PASS', groups=groups, build=list(core.BUILD), held_processed=False,
        selection_performed=False, training_updates=0, retirement=RETIREMENT, proxy_gamma=0.,
        quantity='first-order pre-retirement contribution T_parent * alpha_native * .5 * physical_crop_fraction',
        interpretation='Intervention attribution, not actual post-retirement alpha*T or hidden-geometry truth',
        rendered_center_depth_bins=dict(near='center_z - measured_z < -.12m',
                                        same='abs(center_z - measured_z) <= .12m',
                                        far='center_z - measured_z > .12m'),
        parent_prefix='FP64 selected-ray sums, alpha>=1/255, skip terminal hit when T_after<1e-4',
        measurement_domain='Original fixed measured group mask; BUILD only; no new selection'))


if __name__ == '__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
