"""Fixed all-qualified-teacher BUILD queue; original QUERY sources stay frozen.

Explicit --execute runs short new-input lifecycle qualification, fresh FIELD1200,
then complete-workspace NUMERIC<=2000FG from that FIELD's baked colors. No held
quality, optimizer search, extra arm, or production-scene change is performed.
"""
from pathlib import Path
import argparse
import copy
import gc
import hashlib
import json
import random
import sys
import time
import traceback
import numpy as np
from build_inputs import (B, ROOT, FRAMES, actual_frames, objective_classes,
                          install_paths, read, write, sha, canonical_snapshot)

RECIPE = dict(m=10, maxls=20, factr=1e7, pgtol=1e-8, maxFG=2000)


def scalar(x):
    if isinstance(x, dict):
        return {k: scalar(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [scalar(v) for v in x]
    if hasattr(x, 'detach'):
        a = x.detach().cpu()
        return a.item() if a.numel() == 1 else a.tolist()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def tree_error(a, b):
    """Compare full snapshots by value, without conflating physical high-water counters."""
    import torch
    if torch.is_tensor(a):
        if not torch.is_tensor(b) or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('snapshot tensor topology differs')
        return 0. if not a.numel() else float((a.detach().cpu().double() - b.detach().cpu().double()).abs().max())
    if isinstance(a, np.ndarray):
        if not isinstance(b, np.ndarray) or a.shape != b.shape or a.dtype != b.dtype:
            raise ValueError('snapshot array topology differs')
        if a.dtype.kind not in 'biuf':
            return 0. if np.array_equal(a, b) else float('inf')
        return 0. if not a.size else float(np.max(np.abs(a.astype(float) - b.astype(float))))
    if isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError('snapshot mapping differs')
        return max([tree_error(v, b[k]) for k, v in a.items() if k != 'physical'] or [0.])
    if isinstance(a, (tuple, list)):
        if len(a) != len(b):
            raise ValueError('snapshot sequence differs')
        return max([tree_error(x, y) for x, y in zip(a, b)] or [0.])
    return 0. if a == b else float('inf')


def set_seed():
    import torch
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)


def append(path, row):
    with Path(path).open('a') as f:
        f.write(json.dumps(scalar(row), allow_nan=False) + '\n')


def assert_binding(binding):
    changed = [p for p, value in binding.items() if sha(p) != value]
    if changed:
        raise RuntimeError('immutable BUILD source/input changed: ' + repr(changed))


def field_smoke(make_state, ctx, out):
    import torch
    out.mkdir(parents=True, exist_ok=False)
    t = time.monotonic()
    s = make_state('smoke', stage_counts=(2, 2))
    # B1 -> save/restore -> B2, followed by the actual C entrance and replay.
    s.step()
    s.save(out / 'B1.pt')
    saved_B = s.snapshot()
    s.step()
    expected_B = s.snapshot()
    s.load(out / 'B1.pt')
    immediate_B = tree_error(saved_B, s.snapshot())
    s.step()
    next_B = tree_error(expected_B, s.snapshot())
    s.step()
    s.save(out / 'C1.pt')
    saved_C = s.snapshot()
    s.step()
    expected_C = s.snapshot()
    s.load(out / 'C1.pt')
    immediate_C = tree_error(saved_C, s.snapshot())
    s.step()
    next_C = tree_error(expected_C, s.snapshot())
    if immediate_B != 0 or immediate_C != 0 or next_B > 1e-6 or next_C > 1e-6:
        raise RuntimeError('BUILD full optimizer/RNG restore or next-step qualification failed')
    s.model.save(out / 'field.pt', dict(data_role='BUILD_fit_smoke', never_formal_warm_start=True))
    loaded, _ = type(s.model).load(out / 'field.pt', device='cuda')
    with torch.no_grad():
        field_ci_error = float((s.reader.colors(s.model) - s.reader.colors(loaded)).abs().max())
    if field_ci_error != 0:
        raise RuntimeError('saved shared field query is not exact')
    result = dict(completed=True, data_role='BUILD_fit', physical=s.physical,
                  immediate_B_full_snapshot_max=immediate_B, next_B_full_snapshot_max=next_B,
                  immediate_C_full_snapshot_max=immediate_C, next_C_full_snapshot_max=next_C,
                  field_reload_query_max_abs=field_ci_error, formal_warm_start=False,
                  logical_steps=4, actual_updates=6, seconds=time.monotonic() - t)
    write(out / 'receipt.json', result)
    del loaded, s, expected_B, expected_C, saved_B, saved_C
    gc.collect()
    torch.cuda.empty_cache()
    return result


def field_formal(make_state, ctx, layout, binding, out):
    import torch
    from gs_color_oracle import BoundedGSColors
    out.mkdir(parents=True, exist_ok=False)
    t = time.monotonic()
    s = make_state('formal')
    while s.cursor < 1200:
        row = s.step()
        append(out / 'steps.jsonl', row)
        if s.cursor % 100 == 0:
            s.save(out / ('checkpoint_%04d.pt' % s.cursor))
            print('BUILD_FIELD', s.cursor, row['loss'], flush=True)
    s.save(out / 'final_checkpoint.pt')
    s.model.save(out / 'field.pt', dict(arm='G_BUILD_FIELD', data_role='BUILD_fit_all_qualified',
        authority='world field; original source/known/unknown remains prior provenance',
        source_identity=s.source_identity, cursor=s.cursor, physical=s.physical))
    loaded, metadata = type(s.model).load(out / 'field.pt', device='cuda')
    with torch.no_grad():
        ci = s.reader.colors(loaded).detach()
        query_error = float((ci - s.reader.colors(s.model)).abs().max())
    if query_error != 0:
        raise RuntimeError('formal field authority query differs')
    derived = BoundedGSColors(ctx.base_rgb, layout)
    positions = torch.as_tensor(np.flatnonzero(s.reader.eligible), device=ci.device)
    with torch.no_grad():
        derived.rgb.copy_(ci[positions])
    if not torch.equal(derived(), ci):
        raise RuntimeError('derived Boundedci differs from shared-field authority')
    derived.save(out / 'oracle_colors.pt')
    reloaded = BoundedGSColors.load(out / 'oracle_colors.pt', device=ci.device)
    if not torch.equal(reloaded(), ci):
        raise RuntimeError('derived Boundedci reload differs')
    world_scope = np.zeros(len(ctx.road['active']), bool)
    world_scope[layout.eligible_world_rows] = True
    asset = ctx.save_asset(ci, world_scope, out / 'asset', training_record=dict(
        arm='G_BUILD_FIELD', data_role='BUILD_fit_all_qualified', logical_steps=1200,
        authority=str(out / 'field.pt'), derived_ci=str(out / 'oracle_colors.pt'),
        source_identity=s.source_identity, geometry_coverage_frozen=True))
    (out / 'rendered').mkdir()
    predictions = {}
    native_before = ctx.native_calls
    with torch.no_grad():
        for raw in FRAMES:
            value = ctx.render_road(raw, ci)
            predictions[raw] = value['rgb']
            np.savez_compressed(out / 'rendered' / ('%03d_native.npz' % raw),
                road_rgb=value['rgb'].cpu().numpy(), road_alpha=value['alpha'].detach().cpu().numpy().squeeze(), K=ctx.K_masks[raw])
        f32, terms = s.objective.objective(predictions, 'detail')
        rgb, _ = s.objective.objective(predictions, 'rgb')
    assert_binding(binding)
    ctx.assert_frozen()
    result = dict(arm='G_BUILD_FIELD', completed=True, data_role='BUILD_fit_all_qualified',
        asset_dir=str(asset), authority=str(out / 'field.pt'), derived_ci_authority=str(out / 'oracle_colors.pt'),
        checkpoint=str(out / 'final_checkpoint.pt'), source_freeze=str(ROOT / 'source_freeze.json'),
        loss_snapshot=str(ROOT / 'common/common_loss_snapshot.json'), initial_ci=str(ROOT / 'common/initial_ci.npz'),
        source_identity=s.source_identity, physical=s.physical, logical_steps=s.cursor,
        final_postupdate_BUILD_DETAIL=float(f32), final_postupdate_BUILD_RGB=float(rgb), terms=scalar(terms),
        field_reload_query_max_abs=query_error, derived_Boundedci_equal=True,
        endpoint_readonly_nativeF=ctx.native_calls - native_before, seconds=time.monotonic() - t,
        gpu_lease=str(ROOT / 'gpu_lease.json'), fiveview_public_pending=True,
        no_held_validation_claim=True, no_production_promotion=True, quality_PASS=False)
    write(out / 'receipt.json', result)
    write(out / 'endpoint.json', result)
    seed = ci.detach().cpu().numpy()
    del s, loaded, derived, reloaded, ci, predictions
    gc.collect()
    torch.cuda.empty_cache()
    return seed, result


class BuildNumericRuntime:
    """New data/seed adapter for the qualified unchanged complete-workspace solver."""
    def __init__(self, ctx, frames, norm, seed, eligible, binding, out, role, maxFG):
        import torch
        self.torch = torch
        self.ctx, self.seed = ctx, torch.as_tensor(seed, device=ctx.base_rgb.device)
        self.eligible = np.asarray(eligible, bool).copy()
        self.pos = torch.as_tensor(np.flatnonzero(self.eligible), device=self.seed.device)
        self.x0 = seed[self.eligible].astype(np.float64)
        self.output, self.role, self.maxFG = Path(out), role, maxFG
        self.output.mkdir(parents=True, exist_ok=False)
        self.frame_alpha = {r: torch.as_tensor(frames[r]['alpha'], device=self.seed.device) for r in FRAMES}
        original, precise = objective_classes()
        self.original = original(frames, device='cuda')
        self.precise = precise(frames, device='cuda')
        self.original.restore(copy.deepcopy(norm))
        self.precise.restore(copy.deepcopy(norm))
        self.ledger = dict(FG=0, nativeF=0, nativeB=0, torchBackward=0, accepted=0, rows=[])
        self.binding = dict(binding)
        for p in (ROOT / 'field/formal_r1/field.pt', ROOT / 'field/formal_r1/oracle_colors.pt',
                  ROOT / 'field/formal_r1/asset/road.npz'):
            self.binding[str(p)] = sha(p)
        self.identity = dict(data_role='BUILD_fit_all_qualified', phase=role, precision='unchanged operators; scalar square means FP64',
            binding=self.binding, seed_role='fixed G_BUILD_FIELD final bakedci',
            scope_SHA256=hashlib.sha256(ctx.active_world_rows[self.eligible].astype('<i8').tobytes()).hexdigest())
        self.cache = {}
        write(self.output / 'source_binding.json', self.identity)

    def _record(self):
        write(self.output / 'physical_ledger.json', self.ledger)

    def predictions(self, x):
        torch = self.torch
        p = torch.as_tensor(x, device=self.seed.device, dtype=self.seed.dtype).detach().requires_grad_(True)
        colors = self.seed.index_copy(0, self.pos, p.reshape(-1, 3))
        predictions, alphas = {}, {}
        for raw in FRAMES:
            self.ctx.require_lease()
            value = self.ctx.render_road(raw, colors)
            self.ledger['nativeF'] += 1
            if not torch.equal(value['alpha'].detach().squeeze(), self.frame_alpha[raw]):
                raise RuntimeError('BUILD numeric alpha changed')
            predictions[raw], alphas[raw] = value['rgb'], value['alpha'].detach()
        return p, colors, predictions, alphas

    def qualify_precision(self):
        torch = self.torch
        p, _, predictions, _ = self.predictions(self.x0)
        f32, _ = self.original.objective(predictions, 'detail')
        f64, _ = self.precise.objective(predictions, 'detail')
        g32 = torch.autograd.grad(f32, p, retain_graph=True)[0]
        g64 = torch.autograd.grad(f64, p)[0]
        self.ledger['FG'] += 2
        self.ledger['nativeB'] += 4
        self.ledger['torchBackward'] += 2
        result = dict(function32=float(f32), function64=float(f64),
            function_relative=abs(float(f32) - float(f64)) / max(abs(float(f64)), 1e-12),
            gradient_L2_relative=float((g32.double() - g64.double()).norm()) / max(float(g64.double().norm()), 1e-12),
            gradient_max_abs=float((g32 - g64).abs().max()), same_native_prediction=True)
        result['passed'] = result['function_relative'] <= 1e-6 and result['gradient_L2_relative'] <= 1e-5
        self._record()
        write(self.output / 'precision_qualification.json', result)
        if not result['passed']:
            raise RuntimeError('fixed BUILD precision qualification failed')
        return result

    def evaluate(self, x):
        torch = self.torch
        if self.ledger['FG'] >= self.maxFG:
            raise RuntimeError('registered FG cap exhausted')
        torch.cuda.synchronize()
        t = time.monotonic()
        p, colors, predictions, alphas = self.predictions(x)
        f, _ = self.precise.objective(predictions, 'detail')
        g = torch.autograd.grad(f, p)[0]
        torch.cuda.synchronize()
        self.ledger['FG'] += 1
        self.ledger['nativeB'] += 2
        self.ledger['torchBackward'] += 1
        self.ledger['rows'].append(dict(FG=self.ledger['FG'], objective=float(f.detach()), seconds=time.monotonic() - t))
        self.cache = dict(x=np.asarray(x).copy(), colors=colors.detach().cpu().numpy(),
            predictions={r: v.detach().cpu().numpy() for r, v in predictions.items()},
            alpha={r: v.cpu().numpy() for r, v in alphas.items()})
        self._record()
        return float(f.detach()), g.detach().cpu().numpy().astype(np.float64), dict(data_role='BUILD_fit_all_qualified')

    def rng(self):
        torch = self.torch
        return dict(python=random.getstate(), numpy=copy.deepcopy(np.random.get_state()),
            torch=torch.get_rng_state().clone(), cuda=[v.clone() for v in torch.cuda.get_rng_state_all()])

    def restore_rng(self, saved):
        torch = self.torch
        random.setstate(saved['python'])
        np.random.set_state(copy.deepcopy(saved['numpy']))
        torch.set_rng_state(saved['torch'].clone())
        torch.cuda.set_rng_state_all([v.clone() for v in saved['cuda']])


def numeric_checkpoint(solver, runtime, path):
    import torch
    solver.save(path.with_suffix('.npz'))
    torch.save(runtime.rng(), path.with_suffix('.rng.pt'))
    return dict(workspace=str(path.with_suffix('.npz')), workspace_SHA256=sha(path.with_suffix('.npz')),
                RNG=str(path.with_suffix('.rng.pt')), RNG_SHA256=sha(path.with_suffix('.rng.pt')))


def numeric_smoke(ctx, frames, norm, seed, eligible, binding, out):
    import torch
    from checkpointed_lbfgsb import CheckpointedLBFGSB
    t = time.monotonic()
    runtime = BuildNumericRuntime(ctx, frames, norm, seed, eligible, binding, out, 'smoke', 20)
    precision = runtime.qualify_precision()
    solver = CheckpointedLBFGSB(runtime.x0, runtime.evaluate, identity=runtime.identity, **dict(RECIPE, maxFG=20))
    first = solver.next_accepted()
    if first is None:
        raise RuntimeError('BUILD numeric short smoke needs accepted NEW_X')
    runtime.ledger['accepted'] += 1
    cp = numeric_checkpoint(solver, runtime, out / 'accepted1')
    saved = solver.snapshot()
    saved_rng = runtime.rng()
    second = solver.next_accepted()
    if second is None:
        raise RuntimeError('BUILD numeric smoke needs next accepted point')
    runtime.ledger['accepted'] += 1
    expected = solver.snapshot()
    expected_pred = copy.deepcopy(runtime.cache['predictions'])
    solver.load(cp['workspace'])
    runtime.restore_rng(saved_rng)
    restore_equal = {k: bool(np.array_equal(saved['arrays'][k], solver.snapshot()['arrays'][k])) for k in solver.ARRAY_NAMES}
    replay = solver.next_accepted()
    runtime.ledger['accepted'] += 1
    ci_error = float(np.max(np.abs(solver.x - expected['arrays']['x'])))
    f_relative = abs(float(solver.f) - float(expected['arrays']['f'])) / max(abs(float(expected['arrays']['f'])), 1e-12)
    native_error = max(float(np.max(np.abs(runtime.cache['predictions'][r] - expected_pred[r]))) for r in FRAMES)
    if replay is None or not all(restore_equal.values()) or ci_error > 1e-6 or f_relative > 1e-6 or native_error > 1e-6:
        raise RuntimeError('BUILD numeric complete workspace restore/next accepted replay failed')
    runtime._record()
    result = dict(completed=True, data_role='BUILD_fit_all_qualified', precision=precision,
        immediate_all_Fortran_arrays_exact=restore_equal, next_ci_max_abs=ci_error,
        next_objective_relative=f_relative, next_native_RGB_max_abs=native_error,
        physical=runtime.ledger, seconds=time.monotonic() - t, formal_warm_start=False,
        original_crossprocess_workspace_lifecycle=str(B / 'numerical_completion/smoke_r1/receipt.json'))
    write(out / 'receipt.json', result)
    del runtime, solver, expected, expected_pred, saved
    gc.collect()
    torch.cuda.empty_cache()
    return result


def numeric_formal(ctx, layout, frames, norm, seed, eligible, binding, out):
    import torch
    from checkpointed_lbfgsb import CheckpointedLBFGSB
    from gs_color_oracle import BoundedGSColors
    set_seed()
    t = time.monotonic()
    runtime = BuildNumericRuntime(ctx, frames, norm, seed, eligible, binding, out, 'formal', 2000)
    solver = CheckpointedLBFGSB(runtime.x0, runtime.evaluate, identity=runtime.identity, **RECIPE)
    while not solver.done:
        row = solver.next_accepted()
        if row is None:
            break
        runtime.ledger['accepted'] += 1
        runtime._record()
        append(out / 'steps.jsonl', row)
        if solver.accepted % 50 == 0:
            cp = numeric_checkpoint(solver, runtime, out / ('checkpoint_%04d' % solver.accepted))
            write(out / 'latest_checkpoint.json', cp)
            print('BUILD_NUMERIC', solver.accepted, solver.fg_calls, float(solver.f), row['relative_FW_gap'], flush=True)
        if row['relative_FW_gap'] <= .01:
            solver.stop_at_training_gate()
    cp = numeric_checkpoint(solver, runtime, out / 'final_checkpoint')
    write(out / 'latest_checkpoint.json', cp)
    x = solver.last_accepted_x.reshape(runtime.x0.shape)
    derived = BoundedGSColors(runtime.seed, layout)
    with torch.no_grad():
        derived.rgb.copy_(torch.as_tensor(x, device=derived.rgb.device, dtype=derived.rgb.dtype))
    derived.save(out / 'oracle_colors.pt')
    loaded = BoundedGSColors.load(out / 'oracle_colors.pt', device=derived.rgb.device)
    if not torch.equal(derived(), loaded()):
        raise RuntimeError('BUILD numeric lastaccepted authority reload differs')
    full = np.zeros(len(ctx.road['active']), bool)
    full[layout.eligible_world_rows] = True
    asset = ctx.save_asset(loaded(), full, out / 'asset', training_record=dict(
        arm='G_BUILD_NUMERIC', data_role='BUILD_fit_all_qualified', capacity_control_not_production=True,
        source_identity=runtime.identity, FGcount=solver.fg_calls, accepted_steps=solver.accepted,
        authority=str(out / 'oracle_colors.pt'), export_point='last accepted NEW_X'))
    (out / 'rendered').mkdir()
    with torch.no_grad():
        predictions = {}
        for raw in FRAMES:
            value = ctx.render_road(raw, loaded())
            runtime.ledger['nativeF'] += 1
            predictions[raw] = value['rgb']
            np.savez_compressed(out / 'rendered' / ('%03d_native.npz' % raw),
                road_rgb=value['rgb'].cpu().numpy(), road_alpha=value['alpha'].detach().cpu().numpy().squeeze(), K=ctx.K_masks[raw])
        f32, terms = runtime.original.objective(predictions, 'detail')
        f64, _ = runtime.precise.objective(predictions, 'detail')
        rgb, _ = runtime.original.objective(predictions, 'rgb')
    gap = solver.gap(solver.last_accepted_x, solver.last_accepted_g)
    assert_binding(runtime.binding)
    ctx.assert_frozen()
    runtime._record()
    result = dict(arm='G_BUILD_NUMERIC', completed=True, data_role='BUILD_fit_all_qualified',
        asset_dir=str(asset), authority=str(out / 'oracle_colors.pt'), checkpoint=cp,
        source_identity=runtime.identity, source_freeze=str(ROOT / 'source_freeze.json'),
        physical=runtime.ledger, solver=RECIPE, final_postupdate_BUILD_RGB=float(rgb),
        final_postupdate_BUILD_DETAIL=float(f32), final_FP64_reduction_DETAIL=float(f64), terms=scalar(terms),
        extra=dict(accepted_steps=solver.accepted, FGcount=solver.fg_calls, last_training_FW_gap=gap,
            relative_training_FW_gap=gap / max(solver.last_accepted_f, 1e-3), termination=solver.reason,
            best_evaluated_training_objective=solver.best_evaluated_f,
            export_point='last accepted NEW_X; pending Fortran trial never used'),
        numeric_training_converged=gap / max(solver.last_accepted_f, 1e-3) <= .01,
        strict_numerical_negative_certificate=False, gpu_lease=str(ROOT / 'gpu_lease.json'),
        seconds=time.monotonic() - t, fiveview_public_pending=True, no_held_validation_claim=True,
        quality_PASS=False, no_production_promotion=True)
    write(out / 'endpoint.json', result)
    write(out / 'receipt.json', result)
    return result


def run():
    install_paths()
    import torch
    from native_capacity_inputs import make_context, field_and_reader
    from shared_training_state import SharedTrainingState
    from gs_color_oracle import ColorLayout
    common = ROOT / 'common'
    initial_receipt = read(common / 'receipt.json')
    if initial_receipt.get('completed') is not True or initial_receipt.get('ci0_exact_original') is not True:
        raise RuntimeError('actual exact common GPU initialization required')
    protocol = read(ROOT / 'protocol.json')
    if protocol['arms']['G_BUILD_FIELD']['total_steps'] != 1200 or protocol['arms']['G_BUILD_NUMERIC']['maxFG'] != 2000:
        raise ValueError('unregistered BUILD budget')
    for folder in ('field/formal_r1', 'numerical/formal_r1'):
        if (ROOT / folder).exists():
            raise FileExistsError('preserve existing BUILD formal; explicit full-state recovery only')
    binding = read(common / 'source_freeze.json')['binding']
    for path in (Path(__file__), ROOT / 'code/test_build_inputs.py',
                 B / 'numerical_completion/code/checkpointed_lbfgsb.py',
                 B / 'numerical_completion/code/fp64_reduction_loss.py'):
        binding[str(path)] = sha(path)
    assert_binding(binding)
    write(ROOT / 'source_freeze.json', dict(binding=binding, protocol_sha256=sha(ROOT / 'protocol.json'),
        data_role='BUILD_fit_all_qualified', original_sources_readonly=True))
    authorization = dict(gpu_authorized=True, formal_authorized=True,
        source='Root independently registered all-qualified teacher BUILD with short lifecycle and autonomous fresh queue',
        protocol=str(ROOT / 'protocol.json'), protocol_sha256=sha(ROOT / 'protocol.json'), keep_GPU_on=True,
        physical_field_budget=1200, numeric_maxFG=2000, old_protocols_unchanged=True)
    write(ROOT / 'execution_authorization.json', authorization)
    lease = ROOT / 'gpu_lease.json'
    write(lease, dict(gpu_authorized=True, active=True, owner='all_teacher_build', phase='short_lifecycle_then_registered_fresh_queue',
        authorization=str(ROOT / 'execution_authorization.json')))
    ctx = None
    started = time.monotonic()
    try:
        torch.set_num_threads(1)
        set_seed()
        ctx = make_context(lease, 'all_teacher_build')
        frames, data_binding = actual_frames()
        norm = canonical_snapshot(read(common / 'common_loss_snapshot.json'))
        initial_model, reader, field_binding = field_and_reader(ctx.base_rgb, 'cuda')
        with torch.no_grad():
            computed = reader.colors(initial_model)
        with np.load(common / 'initial_ci.npz', allow_pickle=False) as z:
            if not np.array_equal(computed.cpu().numpy(), z['ci0']) or not np.array_equal(ctx.active_world_rows, z['active_world_rows']):
                raise RuntimeError('same-run fresh GPU initialization is not exact')
        model_type = type(initial_model)
        layout = ColorLayout(ctx.active_world_rows, reader.eligible)
        del initial_model, computed
        BuildLoss, _ = objective_classes()
        identity = dict(data_role='BUILD_fit_all_qualified', source_freeze_SHA256=sha(ROOT / 'source_freeze.json'),
            scope_signature=layout.signature, initial_ci_SHA256=sha(common / 'initial_ci.npz'))
        def make_state(purpose, stage_counts=None):
            set_seed()
            model, metadata = model_type.load(common / 'initial_field.pt', device='cuda')
            objective = BuildLoss(frames, device='cuda')
            objective.restore(copy.deepcopy(norm))
            return SharedTrainingState(model, reader, ctx, objective, mode='detail', source_identity=identity,
                source_metadata=dict(field_binding=field_binding, initial_metadata=metadata), purpose=purpose,
                stage_counts=stage_counts)
        smoke = field_smoke(make_state, ctx, ROOT / 'field/smoke_r1')
        print('BUILD_FIELD_LIFECYCLE_PASS', json.dumps(smoke), flush=True)
        seed, field_result = field_formal(make_state, ctx, layout, binding, ROOT / 'field/formal_r1')
        print('BUILD_FIELD_COMPLETED', field_result['final_postupdate_BUILD_DETAIL'], flush=True)
        smoke_numeric = numeric_smoke(ctx, frames, norm, seed, reader.eligible, binding, ROOT / 'numerical/smoke_r1')
        print('BUILD_NUMERIC_LIFECYCLE_PASS', json.dumps(smoke_numeric), flush=True)
        numeric_result = numeric_formal(ctx, layout, frames, norm, seed, reader.eligible, binding, ROOT / 'numerical/formal_r1')
        result = dict(completed=True, task_id=protocol['task_id'], data_role='BUILD_fit_all_qualified',
            field=field_result, numerical=numeric_result, engineering=dict(field=smoke, numeric=smoke_numeric,
                initial_readonly=initial_receipt), actual_native_forward_calls=ctx.native_calls,
            nativeF_initialization_separate=initial_receipt['actual_nativeF'], seconds=time.monotonic() - started,
            original_QUERY_protocols_results_unchanged=True, no_validation_selection=True,
            production_scene_modified=False, quality_PASS=False, GPU_keep_on=True)
        assert_binding(binding)
        ctx.assert_frozen()
        write(ROOT / 'results.json', result)
        print('ALL_TEACHER_BUILD_COMPLETED', json.dumps(dict(field=field_result['final_postupdate_BUILD_DETAIL'],
            numeric=numeric_result['final_postupdate_BUILD_DETAIL'], seconds=result['seconds'])), flush=True)
    except Exception:
        write(ROOT / ('failed_%d.json' % int(time.time())), dict(error=traceback.format_exc(), seconds=time.monotonic() - started,
            actual_native_forward_calls=None if ctx is None else ctx.native_calls))
        raise
    finally:
        if ctx is not None:
            ctx.close()
        state = read(lease)
        state.update(active=False, ended_at=time.time())
        write(lease, state)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    if not args.execute:
        print(json.dumps(dict(default_GPU=False, output=str(ROOT), field_steps=1200, numeric_maxFG=2000,
            data_role='BUILD_fit_all_qualified', original_sources_readonly=True)))
    else:
        run()
