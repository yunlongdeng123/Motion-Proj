"""CPU-only comparison of preserved static parameters; no renderer or updates.

OBB membership is only a spatial diagnostic on GN11's raw128 centers. It does
not assign ownership, prove image contribution, or authorize point deletion.
"""
from pathlib import Path
import datetime, gc, json, math, os, time
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['OPENBLAS_NUM_THREADS'] = '2'
import torch
import numpy as np

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1')
OUT = ROOT / 'baseline_engineering_20261018'
TASK = ROOT / 'identity_completion_20261018'


def read(p):
    return json.loads(p.read_text())


def stats(values):
    values = values.detach().flatten().double()
    if not values.numel():
        return dict(count=0)
    assert torch.isfinite(values).all()
    quantiles = torch.quantile(values, torch.tensor([.5, .9, .99, .999], dtype=torch.float64))
    return dict(count=values.numel(), median=float(quantiles[0]), p90=float(quantiles[1]),
                p99=float(quantiles[2]), p999=float(quantiles[3]), maximum=float(values.max()))


def materialize(captured):
    return dict(xyz=captured[1].detach(), dc=captured[2].detach(), rest=captured[3].detach(),
        sigma=captured[4].detach().exp(), opacity=captured[6].detach().sigmoid(),
        t=captured[7].detach(), velocity=captured[9].detach(), T=float(captured[16]))


def digest(model, scale, selected):
    sigma = model['sigma'].amax(-1) / scale
    result = dict(point_count=len(sigma), sigma_max_axis_m=stats(sigma),
        sigma_threshold_counts={str(v):int((sigma > v).sum()) for v in [.1, .25, .5, 1., 2.]},
        SH_DC_coefficient_L2=stats(model['dc'].flatten(1).norm(dim=1)),
        SH_nonDC_coefficient_L2=stats(model['rest'].flatten(1).norm(dim=1)),
        opacity=stats(model['opacity']))
    for name, take in selected.items():
        if len(take) != len(sigma):
            continue
        result[name] = dict(selection='GN11 raw128 SHM centers within genuine OBB; diagnostic region only, no ownership claim',
            sigma_max_axis_m=stats(sigma[take]), SH_nonDC_coefficient_L2=stats(model['rest'][take].flatten(1).norm(dim=1)))
    return result


def main():
    start = time.monotonic()
    torch.set_num_threads(2)
    spec = read(TASK / 'spec5.json')
    paths = dict(GN11=Path(spec['background_checkpoint']),
        old3=ROOT / 'rigid_runs/scene0008_089_137/full30k/checkpoint_030000.pt',
        old5=TASK / 'train5/checkpoint_030000.pt')
    before = {k:dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for k,p in paths.items()}
    with np.load(TASK / 'prepared5/instances.npz', allow_pickle=False) as b:
        scale = float(np.linalg.norm(b['normalization'][:3,:3], axis=0).mean())
        row = b['frame_ids'].tolist().index('0008128')
        timestamp = float(b['timestamps'][row])
        poses, sizes = b['poses'][row].copy(), b['box_size'][row].copy()
    raw = torch.load(str(paths['GN11']), map_location='cpu', weights_only=False, mmap=True)
    assert raw[1] == 30000
    origin = materialize(raw[0])
    del raw
    a = 2 * math.pi / origin['T']
    xyz = origin['xyz'] + origin['velocity'] * torch.sin((timestamp - origin['t']) * a) / a
    selected = {}
    for index, actor in [(3,6), (4,37)]:
        pose = torch.from_numpy(poses[index]).to(xyz)
        half = torch.from_numpy(sizes[index]).to(xyz) / 2
        local = (xyz - pose[:3,3]) @ pose[:3,:3]
        selected['a%d_initial_OBB_region'%actor] = (local.abs() <= half).all(-1)
    del xyz
    output = dict(GN11=digest(origin, scale, selected))
    lineage = {}
    for name in ['old3','old5']:
        saved = torch.load(str(paths[name]), map_location='cpu', weights_only=False, mmap=True)
        assert saved['step'] == 30000
        model = materialize(saved['background'])
        exclusion = saved['background_exclusion']
        n_equal = len(model['xyz']) == len(origin['xyz'])
        fixed_row_contract = n_equal and exclusion.get('removed_points') == 0 and exclusion.get('criterion') == 'preserve_all_GN11_static_Gaussians_no_center_box_deletion'
        lineage[name] = dict(point_count_equal=n_equal, actual_background_exclusion=exclusion,
            fixed_row_alignment_supported_by_preserve_all_and_actual_fixed_topology_source=fixed_row_contract)
        output[name] = digest(model, scale, selected if fixed_row_contract else {})
        if fixed_row_contract:
            ratio = model['sigma'].amax(-1) / origin['sigma'].amax(-1).clamp_min(1e-30)
            drift = (model['xyz'] - origin['xyz']).norm(dim=1) / scale
            changes = dict(max_axis_sigma_ratio=stats(ratio), base_center_drift_m=stats(drift),
                sigma_growth_over2x=int((ratio > 2).sum()), sigma_growth_over10x=int((ratio > 10).sum()))
            for region, take in selected.items():
                changes[region] = dict(max_axis_sigma_ratio=stats(ratio[take]), base_center_drift_m=stats(drift[take]),
                    sigma_growth_over2x=int((ratio[take] > 2).sum()), sigma_growth_over10x=int((ratio[take] > 10).sum()))
            output[name]['same_row_changes_from_GN11'] = changes
        del model, saved
        gc.collect()
    assert not torch.cuda.is_initialized()
    assert before == {k:dict(path=str(p), size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns) for k,p in paths.items()}
    result = dict(status='COMPLETE_CPU_STATIC_PARAMETER_AUDIT', at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        checkpoint_bindings=before, world_scale=scale, static_parameters=output, row_alignment=lineage,
        GPU_used=False, optimizer_updates=0, original_checkpoints_unchanged=True,
        new_matched_GN11_or_R15_parameters_used=False, seconds=time.monotonic()-start,
        caveats=['large sigma alone is not an artifact proof, especially distant contracted Gaussians',
                 'OBB center membership is diagnostic, not owner labeling or rendering contribution',
                 'no counterfactual pruning render or causal quality conclusion from this CPU audit'], human_verdict=None)
    (OUT / 'static_growth_receipt.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['static_parameters','checkpoint_bindings']}))


if __name__ == '__main__':
    main()
