"""固定旧12cm中心采样教师的成熟L-BFGS-B控制；默认prepare，不自动启动GPU。"""
from pathlib import Path
import argparse
import copy
import hashlib
import importlib.util
import json
import random
import shutil
import sys
import time
import numpy as np

M = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_marking_solver_20261118_r1')
OLD = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1/check2/native_relaxed_r1')
CPU = Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
CASE = 'marking_w120'
MAX_EVALUATIONS, MAX_SECONDS = 500, 300


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_modules():
    folder = Path(__file__).parent
    spec = importlib.util.spec_from_file_location('fixed_ideal_capacity_runner', folder / 'old_capacity_runner.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    from ideal_capacity_metrics import DetailMetrics
    return old, DetailMetrics


def configuration():
    return dict(schema='road_marking_solver_control.v1', case=CASE,
                initial_asset='canonical oldS2, never activation/current candidates',
                camera_role='all-five supervised ideal capacity oracle, not generalization',
                frames=[86, 110, 134, 179, 198], active_color_rows=151096, scalar_variables=453288,
                pruning='none: all active ci retained including exact-zero Dcolor columns at c0',
                ci_box=[0, 1], D_color=37206, RGB_denominator=111618,
                target='same frozen 12cm center-sampled independent world teacher',
                fixed='geometry/normal/opacity/all scales/active/density/camera/other native branches',
                solver='SciPy1.13.1 Fortran L-BFGS-B setulb, reverse communication',
                maxcor=10, ftol=1e-12, gtol=1e-6, maxls=20,
                scaled_objective='halfSSE = original normalized f *111618; g scaled by same constant',
                max_actual_objective_gradient_evaluations=MAX_EVALUATIONS,
                max_seconds=MAX_SECONDS, automatic_extension=False, automatic_additional_case=False,
                reference_levels=[1/255, 2/255, 4/255], detail_standard='unchanged frozen metric/masks',
                negative_certificate='unavailable unless separately established conservative numeric bound; no solver status alone proves capacity limit',
                production_promoted=False, human_verdict=None)


class NativeFG:
    def __init__(self, op):
        self.op = op
        self.best = None
        self.last = None
        self.native_objective_gradient_calls = 0

    def __call__(self, x):
        torch = self.op.torch
        colors = torch.as_tensor(x.reshape(-1, 3), device=self.op.device, dtype=torch.float32)
        f, p, g = self.op.loss_gradient(colors)
        self.native_objective_gradient_calls += 1
        state = dict(x=colors.detach().clone(), f=f, p=p.detach().clone(), g=g.detach().clone(),
                     native_evaluation=self.native_objective_gradient_calls)
        self.last = state
        if self.best is None or f < self.best['f']:
            self.best = state
        denominator = self.op.data.denominator
        return f * denominator, g.detach().cpu().numpy().astype(np.float64).reshape(-1) * denominator

    def saved_cache(self):
        torch = self.op.torch
        def transfer(state):
            return None if state is None else {k: v.detach().cpu().clone() if torch.is_tensor(v) else v for k, v in state.items()}
        return dict(best=transfer(self.best), last=transfer(self.last), native_objective_gradient_calls=self.native_objective_gradient_calls)

    def restore_cache(self, saved):
        torch = self.op.torch
        def transfer(state):
            return None if state is None else {k: v.to(self.op.device) if torch.is_tensor(v) else v for k, v in state.items()}
        self.best, self.last = transfer(saved['best']), transfer(saved['last'])
        self.native_objective_gradient_calls = saved['native_objective_gradient_calls']


def capture(op, solver, fg, binding, elapsed):
    torch = op.torch
    return dict(schema='marking_lbfgsb_full_state.v1', solver=solver.snapshot(), native=fg.saved_cache(),
                binding=binding, elapsed=elapsed, torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(), numpy_rng=np.random.get_state(), python_rng=random.getstate())


def restore(op, saved, binding):
    from scipy_box_lbfgsb import ScipyBoxLBFGSB
    torch = op.torch
    if saved['binding'] != binding or saved['schema'] != 'marking_lbfgsb_full_state.v1':
        raise ValueError('恢复来源/方法/source binding改变')
    solver = ScipyBoxLBFGSB.restore(saved['solver'])
    fg = NativeFG(op)
    fg.restore_cache(saved['native'])
    torch.set_rng_state(saved['torch_rng'])
    torch.cuda.set_rng_state_all(saved['cuda_rng'])
    np.random.set_state(saved['numpy_rng'])
    random.setstate(saved['python_rng'])
    return solver, fg


def selected_state(fg):
    if fg.best is None:
        raise RuntimeError('没有真实native f+g评估，不能导出见证')
    return fg.best


def evaluate_and_export(op, fg, details, old, output, role):
    state = selected_state(fg)
    evaluation = old.evaluate(op, state, details, output / 'evaluation')
    full = op.full(state['x']).detach()
    asset = op.ctx.save_asset(full, op.data.eligible_world, output / 'asset', training_record=dict(
        role=role, solver='SciPy L-BFGS-B', actual_objective_gradient_evaluations=fg.native_objective_gradient_calls,
        maximum_objective_gradient_evaluations=500, maximum_seconds=300,
        only_road_field_modified='reference_srgb', virtual_road_only=True, K_protection_not_claimed=True))
    reload = op.ctx.validate_public_reload(full, asset, output / 'public_reload.json')
    op.ctx.assert_frozen()
    return evaluation, reload


def native_smoke(op, details, old, binding, output):
    from scipy_box_lbfgsb import ScipyBoxLBFGSB, drive
    torch = op.torch
    begin = time.monotonic()
    output.mkdir(parents=True, exist_ok=False)
    start = ScipyBoxLBFGSB(op.c0.detach().cpu().numpy())
    fg = NativeFG(op)
    drive(start, fg, max_evaluations=1, max_seconds=300)
    saved = capture(op, start, fg, binding, 0.)
    torch.save(saved, output / 'checkpoint_fg1.pt')
    loaded = torch.load(output / 'checkpoint_fg1.pt', map_location='cpu', weights_only=False)
    restored, restored_fg = restore(op, loaded, binding)
    full_differences = old.differences(loaded, capture(op, restored, restored_fg, binding, 0.))
    if full_differences:
        raise RuntimeError('native完整L-BFGS-B工作区/缓存/RNG恢复不精确: ' + str(full_differences))
    a = drive(restored, restored_fg, max_evaluations=2, max_seconds=300)
    next_state = capture(op, restored, restored_fg, binding, 0.)
    torch.save(next_state, output / 'checkpoint_next_fg2.pt')
    replay, replay_fg = restore(op, loaded, binding)
    b = drive(replay, replay_fg, max_evaluations=2, max_seconds=300)
    replay_state = capture(op, replay, replay_fg, binding, 0.)
    next_differences = old.differences(next_state, replay_state)
    x64 = float(np.abs(restored.x - replay.x).max())
    x32 = float(np.abs(restored.x.astype(np.float32) - replay.x.astype(np.float32)).max())
    g = float((restored_fg.last['g'] - replay_fg.last['g']).abs().max())
    p = float((restored_fg.last['p'] - replay_fg.last['p']).abs().max())
    evaluation, reload = evaluate_and_export(op, restored_fg, details, old, output, 'engineering_native_lifecycle_smoke')
    formal_entry = ScipyBoxLBFGSB(op.c0.detach().cpu().numpy())
    task = formal_entry.advance()
    entry_exact = task.startswith('FG') and np.array_equal(formal_entry.x, op.c0.detach().cpu().numpy().astype(np.float64).reshape(-1))
    write(output / 'formal_entry.json', dict(fresh_canonical_c0_exact=entry_exact, evaluations=0,
                                            task=task, uses_smoke_warmstart=False))
    passed = x32 <= 1e-6 and p <= 1e-6 and g <= 1e-6 and entry_exact
    result = dict(schema='marking_lbfgsb_native_lifecycle_smoke.v1', passed=passed, GPU_executed=True,
                  complete_restore_exact=True, complete_restore_differences=full_differences,
                  engineering_actual_objective_gradient_calls=3, formal_evaluations=0,
                  replay_next_proposal_float64_max=x64, replay_next_proposal_float32_max=x32,
                  replay_normalized_gradient_max=g, replay_RGB_max=p,
                  next_state_bitwise=not next_differences,
                  next_full_state_differences=next_differences,
                  full_workspace_restored=True, public_reload=reload,
                  formal_entry_fresh_c0=entry_exact, evaluation=evaluation['capacity_status'],
                  native_forward_calls_in_context=op.ctx.native_calls,
                  additional_public_clone_forward_calls=len(reload['rows']),
                  native_backward_calls=op.backward_calls,
                  elapsed_seconds=time.monotonic()-begin,
                  scope='all-active virtual road-only; no K/production claim')
    write(output / 'completed.json', result)
    if not passed:
        raise RuntimeError('native next-evaluation/entry replay超出预登记工程容差')
    return result


def solve(op, details, old, binding, output, smoke_receipt):
    from scipy_box_lbfgsb import ScipyBoxLBFGSB, drive
    torch = op.torch
    smoke = read(smoke_receipt)
    if not smoke['passed'] or not smoke['complete_restore_exact'] or not smoke['formal_entry_fresh_c0']:
        raise ValueError('没有通过的真实native生命周期smoke')
    output.mkdir(parents=True, exist_ok=False)
    solver = ScipyBoxLBFGSB(op.c0.detach().cpu().numpy())
    fg = NativeFG(op)
    def save_checkpoint(current, row):
        if current.evaluations == 1 or current.evaluations % 50 == 0:
            torch.save(capture(op, current, fg, binding, row['elapsed']), output / f'checkpoint_fg{current.evaluations:04d}.pt')
        if current.evaluations == 1 or current.evaluations % 25 == 0:
            print(json.dumps(dict(evaluation=current.evaluations, f=row['scaled_halfSSE']/111618,
                                  elapsed=row['elapsed'], accepted_iterations=current.accepted_iterations)), flush=True)
    run = drive(solver, fg, max_evaluations=MAX_EVALUATIONS, max_seconds=MAX_SECONDS,
                after_evaluation=save_checkpoint)
    assert fg.native_objective_gradient_calls == solver.evaluations <= MAX_EVALUATIONS
    torch.save(capture(op, solver, fg, binding, run['elapsed']), output / 'checkpoint_final.pt')
    write(output / 'trace.json', run['trace'])
    evaluation, reload = evaluate_and_export(op, fg, details, old, output, 'unique_formal_marking_solver_control')
    thresholds = []
    for code in (1, 2, 4):
        rgb = evaluation['certificate']['RGB_RMSE'] <= code/255
        witness = rgb and evaluation['detail']['passed']
        thresholds.append(dict(code_values=code, RGB_RMSE_tolerance=code/255, RGB_pass=rgb,
                               detail_pass=evaluation['detail']['passed'], feasible_witness=witness,
                               negative_certificate_allowed=False,
                               status='feasible_solution_witness' if witness else 'optimization_or_numeric_certificate_inconclusive'))
    previous = read(OLD / CASE / 'evaluation/evaluation.json')
    result = dict(schema='marking_lbfgsb_control_results.v1', configuration=configuration(), GPU_executed=True,
                  actual_objective_gradient_evaluations=solver.evaluations,
                  accepted_LBFGSB_iterations=solver.accepted_iterations, solver_elapsed_seconds=run['elapsed'],
                  time_overrun_seconds=max(0., run['elapsed']-MAX_SECONDS),
                  time_cap_semantics='stop admitting new callbacks by elapsed plus last callback duration; already running native callbacks are non-preemptive',
                  stop_reason=run['stop_reason'], cap_extra_call_attempted=False, automatic_extension=False,
                  best_feasible_evaluation=solver.best['evaluation'], best_trial_may_precede_final_task=True,
                  selected_final='best evaluated feasible ci; not necessarily last line-search proposal',
                  certificate=evaluation['certificate'], detail=evaluation['detail'], levels=thresholds,
                  original_spectral_solver=dict(path=str(OLD / CASE), normalized_f=previous['certificate']['f'],
                                                 RGB_RMSE=previous['certificate']['RGB_RMSE'], detail=previous['detail'],
                                                 budget='500 projected spectral iterations; not equal native-callback compute'),
                  public_reload=reload, original_fixed_fields_unchanged=True,
                  strict_negative_certificate=False, production_promoted=False, human_verdict=None,
                  native_forward_calls_in_context=op.ctx.native_calls,
                  additional_public_clone_forward_calls=len(reload['rows']), native_backward_calls=op.backward_calls,
                  callback_calls_exclude_preflight_and_postevaluation=True)
    write(output / 'completed.json', result)
    write(M / 'results.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('prepare', 'fixture', 'smoke', 'solve'), default='prepare')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--lease-file', type=Path, default=M / 'gpu_lease.json')
    parser.add_argument('--lease-owner', default='stage_qa')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--smoke-receipt', type=Path, default=M / 'smoke_r1/completed.json')
    args = parser.parse_args()
    if args.mode == 'fixture':
        from scipy_box_lbfgsb import cpu_fixture
        result = cpu_fixture()
        write(args.output or M / 'cpu/fixture.json', result)
        print(json.dumps(result)); return
    old, DetailMetrics = frozen_modules()
    data = old.Data(CPU / 'ideal_targets_r2', 'relaxed')
    data.check_binding()
    if args.mode == 'prepare':
        result = dict(configuration=configuration(), GPU_executed=False,
                      actual_rows=int(data.eligible.sum()), D_color=data.total, RGB_denominator=data.denominator,
                      workspace_bytes_estimate=(25*453288 + 11*100 + 80)*8)
        write(args.output or M / 'cpu/prepared.json', result)
        print(json.dumps(result)); return
    if not args.execute:
        raise SystemExit('native必须--execute并持有显式active单GPU租约')
    from road_capacity_native import assert_gpu_lease, load_context
    assert_gpu_lease(args.lease_file, args.lease_owner)
    contract_path = M / 'execution_contract.json'
    if not contract_path.exists():
        raise RuntimeError('缺少根登记execution_contract，禁止native执行')
    source_dir = Path(__file__).parent
    binding = dict(inputs=data.binding, execution_contract_sha256=digest(contract_path),
                   code={name: digest(source_dir/name) for name in ['run_marking_lbfgsb.py', 'scipy_box_lbfgsb.py',
                                                                  'old_capacity_runner.py', 'ideal_capacity_metrics.py', 'road_capacity_native.py']})
    output = args.output or M / ('smoke_r1' if args.mode == 'smoke' else 'formal_r1')
    preflight_root = output.with_name(output.name + '_preflight')
    preflight_root.mkdir(parents=True, exist_ok=False)
    write(preflight_root / 'binding.json', binding)
    (preflight_root / 'source_snapshots').mkdir()
    for name in binding['code']:
        shutil.copyfile(source_dir / name, preflight_root / 'source_snapshots' / name)
    ctx = load_context(lease_file=args.lease_file, lease_owner=args.lease_owner)
    try:
        op = old.NativeOperator(ctx, data)
        old.native_preflight(op, preflight_root / 'operator_preflight.json')
        op.set_case(CASE)
        details = DetailMetrics(data)
        registration = json.loads(json.dumps(details.registration()))
        if registration != read(OLD / 'detail_metric_registration.json'):
            raise ValueError('旧评价口径/细节mask发生变化')
        write(preflight_root / 'detail_registration.json', registration)
        result = (native_smoke(op, details, old, binding, output) if args.mode == 'smoke' else
                  solve(op, details, old, binding, output, args.smoke_receipt))
        data.check_binding(); ctx.assert_frozen()
        print(json.dumps(dict(output=str(output), completed=True,
                              formal_actual_FG=result.get('actual_objective_gradient_evaluations', 0))), flush=True)
    finally:
        ctx.close()


if __name__ == '__main__':
    main()
