"""Zero-update, source-bound counterfactual retirement after frozen G40.

Import the exact r3 runtime supplied by --runtime-dir. No frozen source file,
training configuration, parent checkpoint or optimizer update is modified.
"""
from pathlib import Path
from types import SimpleNamespace
import argparse
import importlib.util
import json
import os
import sys
import time
import traceback
import numpy as np


ERROR_LIMIT = 1.0/255.0
MINIMUM_COVERAGE = 0.99


def acceptance_metrics(prediction, target, baseline_error, alpha):
    """The baseline is fixed for the entire ablation, including every split."""
    if prediction.shape != target.shape or prediction.shape != baseline_error.shape:
        raise ValueError('Fixed V RGB shapes differ')
    if prediction.ndim != 2 or prediction.shape[0] != 3 or alpha.shape != (prediction.shape[1],):
        raise ValueError('Expected RGB[3,V] and alpha[V]')
    finite = all(np.isfinite(v).all() for v in (prediction, target, baseline_error, alpha))
    if not finite:
        return dict(passed=False, finite=False, max_positive_absolute_error_increase=None, alpha95_fraction=None)
    error = np.abs(prediction-target)
    increment = error-baseline_error
    maximum = max(0., float(increment.max()))
    coverage = float((alpha >= 0.95).mean())
    index = int(increment.argmax())
    return dict(passed=maximum <= ERROR_LIMIT and coverage >= MINIMUM_COVERAGE, finite=True,
        max_positive_absolute_error_increase=maximum, mean_absolute_error_increase=float(increment.mean()),
        absolute_RGB_error_mean=float(error.mean()), alpha95_fraction=coverage,
        pixels_with_error_increase_gt_1over255=int((increment.max(0)>ERROR_LIMIT).sum()),
        worst_channel=index//prediction.shape[1], worst_fixed_V_pixel_index=index % prediction.shape[1])


def candidate_order(source_id, source_row_id, active, views, fraction, minimum_views=2, minimum_fraction=0.8):
    shape = source_id.shape
    if any(v.shape != shape for v in (source_row_id, active, views, fraction)):
        raise ValueError('Source-bound candidate evidence shapes differ')
    ids = np.flatnonzero(active & (views >= minimum_views) & (fraction >= minimum_fraction))
    return ids[np.lexsort((ids, source_row_id[ids], source_id[ids]))]


def cpu_self_check():
    target = np.zeros((3, 10), np.float32); baseline = np.full((3, 10), 0.1, np.float32)
    alpha = np.ones(10, np.float32)
    assert acceptance_metrics(np.full_like(target, .103), target, baseline, alpha)['passed']
    assert not acceptance_metrics(np.full_like(target, .106), target, baseline, alpha)['passed']
    # Large image changes may pass when actual GT error improves: source RGB is not locked.
    assert acceptance_metrics(np.full_like(target, .05), target, baseline, alpha)['passed']
    broken_alpha = alpha.copy(); broken_alpha[0] = .9
    assert not acceptance_metrics(np.full_like(target, .05), target, baseline, broken_alpha)['passed']
    ids = candidate_order(np.array([1, 0, 0]), np.array([0, 8, 3]), np.ones(3, bool),
                          np.array([2, 3, 1]), np.array([.9, .85, .99]))
    assert np.array_equal(ids, [1, 0])
    return dict(passed=True, fixed_baseline_prevents_increment_accumulation=True,
                RGB_change_allowed_if_GT_error_improves=True, alpha_coverage_checked=True,
                deterministic_source_order=True, GPU_used=False)


def import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cpu-self-check', action='store_true')
    p.add_argument('--runtime-dir', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--evidence', type=Path)
    p.add_argument('--init', type=Path)
    p.add_argument('--manifest', type=Path)
    p.add_argument('--source-manifest', type=Path)
    p.add_argument('--output', type=Path)
    p.add_argument('--teacher', type=Path)
    p.add_argument('--prior-contract', type=Path)
    p.add_argument('--context-plan', type=Path)
    p.add_argument('--path-map', action='append', default=[])
    p.add_argument('--path-map-file', type=Path)
    return p.parse_args()


def run(args):
    started = time.monotonic()
    for key in ('runtime_dir', 'checkpoint', 'evidence', 'init', 'manifest', 'source_manifest', 'prior_contract', 'output'):
        if getattr(args, key) is None: raise ValueError('--'+key.replace('_', '-')+' is required')
    if args.output.exists(): raise FileExistsError('Preserve existing outputs: choose a new ablation directory')
    args.output.mkdir(parents=True)
    source_hashes_before = {}
    sys.path.insert(0, str(args.runtime_dir.resolve()))
    core = import_file('joint_core', args.runtime_dir/'joint_core.py')
    gpu = import_file('_a20_frozen_r3_joint_gpu', args.runtime_dir/'joint_gpu.py')
    torch = gpu.torch
    if not torch.cuda.is_available(): raise RuntimeError('Actual native CUDA raster is required')
    source_paths = [args.runtime_dir/'joint_core.py', args.runtime_dir/'joint_gpu.py', args.checkpoint, args.evidence]
    for p in source_paths: source_hashes_before[str(p.resolve())] = core.sha(p)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if state['phase'] != 'geometry' or state['step'] != 40:
        raise ValueError('Expected an exact formal geometry step40 checkpoint')
    if state['runtime_sha256'] != core.sha(args.runtime_dir/'joint_gpu.py'):
        raise ValueError('--runtime-dir is not the exact source runtime bound by G40')
    if state.get('role') != 'formal': raise ValueError('Postprocess must start from formal G40')
    bound = [r for r in state.get('contribution_history', []) if r['step'] == 40]
    if len(bound) != 1 or bound[0]['evidence']['sha256'] != core.sha(args.evidence):
        raise ValueError('Step40 actual contribution evidence is not bound by the checkpoint')
    core.write(args.output/'training_contract.restored.json', state['config'])
    runtime_args = SimpleNamespace(**vars(args), mode='geometry_ablation',
        config=args.output/'training_contract.restored.json', resume=None, geometry_gate=None, asset=None)
    inp = core.load_inputs(runtime_args)
    rt = gpu.Runtime(runtime_args, inp)
    opt = rt.optimizer('geometry'); rt.restore(state, opt)
    actor_restored = gpu.tree_equal(gpu.cpu(rt.actor.state_dict()), state['actor'])
    Adam_restored = gpu.tree_equal(gpu.cpu(opt.state_dict()), state['optimizer'])
    rng_before = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                      numpy=np.random.get_state(), python=gpu.random.getstate())
    rng_restored = gpu.tree_equal(rng_before['torch'], state['torch_rng']) and gpu.tree_equal(rng_before['cuda'], state['cuda_rng'])
    rng_restored = rng_restored and gpu.tree_equal(rng_before['numpy'], state['numpy_rng']) and gpu.tree_equal(rng_before['python'], state['python_rng'])
    if not (actor_restored and Adam_restored and rng_restored):
        raise RuntimeError('Exact actor, Adam or RNG restoration failed')
    model_before = gpu.cpu(rt.actor.state_dict()); optimizer_before = gpu.cpu(opt.state_dict())
    parent_active = model_before['active'].numpy().copy()
    evidence = core.bank(args.evidence)
    for key in ('source_id', 'source_row_id', 'native_global_row_id'):
        if not np.array_equal(evidence[key], inp['init'][key]): raise ValueError('Evidence row mapping differs: '+key)
    if not np.array_equal(evidence['active'], parent_active): raise ValueError('Evidence active state differs from saved G40')
    r = state['config']['redundancy']
    ids = candidate_order(inp['init']['source_id'], inp['init']['source_row_id'], parent_active,
        evidence['novel_outside_view_count'], evidence['conflict_fraction'],
        r['novel_outside_view_minimum'], r['conflict_fraction_minimum'])
    candidates_path = args.output/'candidate_source_rows.npz'
    np.savez_compressed(candidates_path, candidate_unified_row_id=ids,
        source_id=inp['init']['source_id'][ids], source_row_id=inp['init']['source_row_id'][ids],
        native_global_row_id=inp['init']['native_global_row_id'][ids],
        real_V_support_raw=evidence['real_V_support_raw'][ids],
        normalized_support=evidence['support'][ids], normalized_conflict=evidence['conflict'][ids],
        conflict_fraction=evidence['conflict_fraction'][ids], novel_outside_view_count=evidence['novel_outside_view_count'][ids])
    core.write(args.output/'postprocess_contract.json', dict(
        schema='a20.zero_update_geometry_ablation.v1', parent_checkpoint=core.ref(args.checkpoint),
        parent_runtime={p.name: core.ref(p) for p in source_paths[:2]}, source_evidence=core.ref(args.evidence),
        script=core.ref(__file__), candidate_rows=len(ids), candidate_table=core.ref(candidates_path),
        candidate_rule='active; step40 normalized conflict fraction>=.8; outside raw>.001 in at least2 novel views; no real-V-nearzero or opacity requirement',
        sorting=['source_id', 'source_row_id', 'unified_row_id'],
        acceptance=dict(render='actual native isolated full-field 3DGS, BUILD86/110',
            absolute_RGB_error_increase_max=ERROR_LIMIT, isolated_alpha95_fraction_minimum=MINIMUM_COVERAGE,
            error_baseline='immutable pre-ablation prediction versus true RGB over complete fixed V; shared by every trial'),
        bisection='whole batch first; failed contiguous source-order batches split left-first until singleton',
        formal_optimizer_updates=40, postprocess_optimizer_updates=0, learning_rates_changed=False,
        per_camera_row_selection=False, heldout_frames_opened=[]))
    before_metrics = rt.evaluate('geometry_ablation_before')
    baseline = {}; before_arrays = {}
    with torch.no_grad():
        for raw in core.BUILD:
            out = rt.render(raw=raw, actor_only=True); domain = rt.domains[raw]['V']
            pred = gpu.cpu(out['rgb'][:, domain]).numpy(); target = gpu.cpu(rt.domains[raw]['rgb'][:, domain]).numpy()
            alpha = gpu.cpu(out['alpha'][domain]).numpy(); yx = gpu.cpu(torch.nonzero(domain)).numpy()
            baseline[raw] = dict(target=target, error=np.abs(pred-target), yx=yx)
            before_arrays[f'raw{raw:03d}_RGB'] = pred; before_arrays[f'raw{raw:03d}_alpha'] = alpha
            before_arrays[f'raw{raw:03d}_target_RGB'] = target; before_arrays[f'raw{raw:03d}_V_yx'] = yx
    np.savez_compressed(args.output/'fixed_V_baseline.npz', **before_arrays)
    accepted = []; rejected = []; trials = 0; trial_raster_start = rt.calls
    stack = [(0, len(ids), 0, None)] if len(ids) else []
    with (args.output/'ablation_trials.jsonl').open('w', encoding='utf-8') as trial_file:
        while stack:
            lo, hi, depth, parent_trial = stack.pop(); group = ids[lo:hi]
            chosen = torch.as_tensor(group, device='cuda', dtype=torch.long)
            if not bool(rt.actor.active[chosen].all()): raise RuntimeError('Candidate batches are not disjoint')
            rt.actor.active[chosen] = False
            per_view = []; passed = False
            try:
                with torch.no_grad():
                    for raw in core.BUILD:
                        out = rt.render(raw=raw, actor_only=True); domain = rt.domains[raw]['V']
                        prediction = gpu.cpu(out['rgb'][:, domain]).numpy(); alpha = gpu.cpu(out['alpha'][domain]).numpy()
                        b = baseline[raw]; metrics = acceptance_metrics(prediction, b['target'], b['error'], alpha)
                        index = metrics.get('worst_fixed_V_pixel_index')
                        if index is not None: metrics['worst_pixel_xy'] = b['yx'][index, ::-1].tolist()
                        per_view.append(dict(raw=raw, **metrics))
                passed = all(v['passed'] for v in per_view)
            finally:
                if not passed: rt.actor.active[chosen] = True
            trials += 1
            record = dict(trial=trials, parent_trial=parent_trial, depth=depth, candidate_range_half_open=[lo, hi],
                rows=len(group), accepted=passed, per_view=per_view, accepted_rows_total=len(accepted)+(len(group) if passed else 0),
                elapsed_seconds=time.monotonic()-started)
            trial_file.write(json.dumps(record, allow_nan=False)+'\n'); trial_file.flush()
            if passed: accepted.extend(group.tolist())
            elif hi-lo == 1: rejected.append(int(group[0]))
            else:
                middle = lo+(hi-lo)//2
                stack.append((middle, hi, depth+1, trials)); stack.append((lo, middle, depth+1, trials))
            core.write(args.output/'ablation_progress.json', dict(PID=os.getpid(), trials=trials, remaining_batches=len(stack),
                accepted_rows=len(accepted), rejected_rows=len(rejected), latest=record,
                formal_optimizer_updates=40, postprocess_optimizer_updates=0, native_calls=rt.calls))
            if trials <= 3 or trials % 25 == 0: print(json.dumps(record, allow_nan=False), flush=True)
    accepted = np.asarray(accepted, np.int64); rejected = np.asarray(rejected, np.int64)
    after_arrays = {}; final_V = []
    with torch.no_grad():
        for raw in core.BUILD:
            out = rt.render(raw=raw, actor_only=True); domain = rt.domains[raw]['V']; b = baseline[raw]
            prediction = gpu.cpu(out['rgb'][:, domain]).numpy(); alpha = gpu.cpu(out['alpha'][domain]).numpy()
            metrics = acceptance_metrics(prediction, b['target'], b['error'], alpha)
            final_V.append(dict(raw=raw, **metrics))
            after_arrays[f'raw{raw:03d}_RGB'] = prediction; after_arrays[f'raw{raw:03d}_alpha'] = alpha
    # If no candidate passes, the parent result is retained even if its gate was already below99%.
    if len(accepted) and not all(v['passed'] for v in final_V): raise RuntimeError('Final accepted active set violates the fixed V gate')
    np.savez_compressed(args.output/'fixed_V_after.npz', **after_arrays)
    decisions_path = args.output/'ablation_decisions.npz'
    np.savez_compressed(decisions_path, candidate_unified_row_id=ids, accepted_unified_row_id=accepted,
        rejected_unified_row_id=rejected, source_id=inp['init']['source_id'], source_row_id=inp['init']['source_row_id'],
        native_global_row_id=inp['init']['native_global_row_id'], before_active=parent_active, after_active=gpu.cpu(rt.actor.active).numpy())
    for j in accepted.tolist():
        rt.retirements.append(dict(step=40, role='zero_update_post_geometry_ablation', unified_row=j,
            source_id=int(inp['init']['source_id'][j]), source_row_id=int(inp['init']['source_row_id'][j]),
            native_global_row_id=int(inp['init']['native_global_row_id'][j]),
            reason='step40 multi-view outside conflict; whole-field true-V counterfactual batch accepted',
            real_V_support_raw=float(evidence['real_V_support_raw'][j]),
            conflict_fraction=float(evidence['conflict_fraction'][j]),
            novel_outside_views=int(evidence['novel_outside_view_count'][j])))
    after_metrics = rt.evaluate('geometry_ablation'); asset = rt.export_asset('geometry_ablation')
    rt.reload_check(asset, 'geometry_ablation_reload')
    unchanged = {k: gpu.tree_equal(gpu.cpu(v), model_before[k]) for k, v in rt.actor.state_dict().items() if k != 'active'}
    if not all(unchanged.values()): raise RuntimeError('A Gaussian parameter/buffer changed during zero-update ablation')
    Adam_unchanged = gpu.tree_equal(gpu.cpu(opt.state_dict()), optimizer_before)
    rng_after = dict(torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                     numpy=np.random.get_state(), python=gpu.random.getstate())
    RNG_unchanged = gpu.tree_equal(rng_before, rng_after)
    if not Adam_unchanged or not RNG_unchanged: raise RuntimeError('Adam or RNG changed during no-update postprocess')
    source_unchanged = {str(p.resolve()): core.sha(p) == source_hashes_before[str(p.resolve())] for p in source_paths}
    if not all(source_unchanged.values()): raise RuntimeError('A frozen source/checkpoint/evidence file changed')
    post_state = rt.snapshot('geometry', 40, opt, 'formal')
    post_state['postprocess'] = dict(schema='a20.zero_update_geometry_ablation.v1', script=core.ref(__file__),
        parent_checkpoint=core.ref(args.checkpoint), evidence=core.ref(args.evidence), decisions=core.ref(decisions_path),
        formal_optimizer_updates=40, postprocess_optimizer_updates=0, accepted_rows=len(accepted), rejected_rows=len(rejected),
        trials=trials, fixed_V_acceptance=final_V)
    checkpoint = args.output/'checkpoints/geometry_ablation_step0040.pt'; checkpoint.parent.mkdir(parents=True)
    temporary = checkpoint.with_suffix('.tmp'); torch.save(post_state, temporary); temporary.replace(checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    rt.restore(saved, opt)
    saved_restore_exact = gpu.tree_equal(gpu.cpu(rt.actor.state_dict()), saved['actor']) and gpu.tree_equal(gpu.cpu(opt.state_dict()), saved['optimizer'])
    if not saved_restore_exact: raise RuntimeError('Postprocess checkpoint exact reload failed')
    source_counts = {str(s): dict(candidates=int((inp['init']['source_id'][ids]==s).sum()),
        accepted=int((inp['init']['source_id'][accepted]==s).sum()), rejected=int((inp['init']['source_id'][rejected]==s).sum())) for s in (0, 1)}
    proof = dict(schema='a20.zero_update_geometry_ablation.proof.v1', completed=True,
        actual_native_CUDA=True, no_grad_counterfactual_raster=True, formal_optimizer_updates=40, postprocess_optimizer_updates=0,
        exact_G40_actor_restore=actor_restored, exact_G40_Adam_restore=Adam_restored, exact_G40_RNG_restore=rng_restored,
        all_Gaussian_values_unchanged_except_global_active=all(unchanged.values()), Adam_unchanged=Adam_unchanged,
        RNG_unchanged=RNG_unchanged, frozen_sources_unchanged=source_unchanged, saved_checkpoint_exact_reload=saved_restore_exact,
        candidate_rows=len(ids), accepted_rows=len(accepted), rejected_rows=len(rejected), per_source=source_counts,
        trials=trials, counterfactual_native_calls=2*trials, fixed_V_acceptance=final_V,
        source_evidence=core.ref(args.evidence), decisions=core.ref(decisions_path), actor=core.ref(asset), checkpoint=core.ref(checkpoint),
        vehicle_geometry_gate=after_metrics['vehicle_geometry_gate'], scene_handoff_gate=after_metrics['scene_handoff_gate'],
        original_strict_gate=after_metrics['gate'], grey_review='PENDING_ROOT_GREY_REVIEW', scene_qualified=False,
        per_camera_row_selection=False, heldout_frames_opened=[], frozen=rt.check_frozen(),
        native_calls=rt.calls, backward_calls=rt.backward_calls, feature_vjp_calls=rt.vjp_calls,
        elapsed_seconds=time.monotonic()-started, GPU_keep_on=True, power_operations=0)
    core.write(args.output/'proof.json', proof)
    core.write(args.output/'retirement_ledger.json', dict(rows=rt.retirements, postprocess=post_state['postprocess'], per_source=source_counts))
    core.write(args.output/'resource_ledger.json', dict(formal_optimizer_updates=40, postprocess_optimizer_updates=0,
        new_backward_calls=rt.backward_calls, new_feature_vjp_calls=rt.vjp_calls, new_native_calls=rt.calls,
        counterfactual_native_calls=2*trials, elapsed_seconds=time.monotonic()-started,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated(), GPU_keep_on=True, power_operations=0))
    print('GEOMETRY_ABLATION_COMPLETED', json.dumps(dict(candidates=len(ids), accepted=len(accepted), rejected=len(rejected), trials=trials)), flush=True)


if __name__ == '__main__':
    args = parse_args()
    if args.cpu_self_check:
        print(json.dumps(cpu_self_check(), indent=2))
    else:
        try: run(args)
        except Exception as exc:
            if args.output and args.output.exists():
                (args.output/'ablation_failure.json').write_text(json.dumps(dict(error=str(exc), traceback=traceback.format_exc(),
                    formal_optimizer_updates=40, postprocess_optimizer_updates=0, PID=os.getpid()), indent=2)+'\n', encoding='utf-8')
            raise
