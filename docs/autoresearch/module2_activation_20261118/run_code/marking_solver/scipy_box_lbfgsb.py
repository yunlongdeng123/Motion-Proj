"""SciPy1.13 L-BFGS-B完整状态接口与实际f+g硬预算；不导入Torch/native。"""
import copy
import time
import numpy as np
import scipy
from scipy.optimize import _lbfgsb


class EvaluationBudgetReached(RuntimeError):
    pass


class ScipyBoxLBFGSB:
    ARRAY_NAMES = ('x', 'lower', 'upper', 'nbd', 'f', 'g', 'wa', 'iwa',
                   'task', 'csave', 'lsave', 'isave', 'dsave')

    def __init__(self, x0, *, maxcor=10, ftol=1e-12, gtol=1e-6, maxls=20):
        if 'csave' not in _lbfgsb.setulb.__doc__ or 'iprint' not in _lbfgsb.setulb.__doc__:
            raise RuntimeError('要求与真实native runtime相同的SciPy1.13 setulb接口；拒绝静默切换版本')
        self.scipy_version = scipy.__version__
        self.maxcor = int(maxcor)
        self.ftol, self.gtol, self.maxls = float(ftol), float(gtol), int(maxls)
        self.factr = self.ftol / np.finfo(np.float64).eps
        self.x = np.asarray(x0, np.float64).reshape(-1).copy()
        if not np.isfinite(self.x).all() or not np.all((self.x >= 0) & (self.x <= 1)):
            raise ValueError('初值必须满足finite box[0,1]')
        n, m = len(self.x), self.maxcor
        self.lower, self.upper = np.zeros(n), np.ones(n)
        self.nbd = np.full(n, 2, np.int32)
        self.f, self.g = np.array(0., np.float64), np.zeros(n)
        self.wa = np.zeros(2*m*n + 5*n + 11*m*m + 8*m)
        self.iwa = np.zeros(3*n, np.int32)
        self.task, self.csave = np.array('START', dtype='S60'), np.zeros((), dtype='S60')
        self.lsave, self.isave = np.zeros(4, np.int32), np.zeros(44, np.int32)
        self.dsave = np.zeros(29)
        self.needs_evaluation = False
        self.evaluations = 0
        self.accepted_iterations = 0
        self.finished = False
        self.terminal_task = None
        self.best = None

    def task_text(self):
        return self.task.item().decode('ascii').strip(' \x00')

    def advance(self):
        if self.needs_evaluation:
            raise RuntimeError('当前FG请求未完成，禁止重入setulb')
        if self.finished:
            return self.terminal_task
        _lbfgsb.setulb(self.maxcor, self.x, self.lower, self.upper, self.nbd,
                       self.f, self.g, self.factr, self.gtol, self.wa, self.iwa,
                       self.task, -1, self.csave, self.lsave, self.isave,
                       self.dsave, self.maxls)
        task = self.task_text()
        if task.startswith('FG'):
            self.needs_evaluation = True
        elif task.startswith('NEW_X'):
            self.accepted_iterations += 1
        else:
            self.finished, self.terminal_task = True, task
        return task

    def provide(self, f, g):
        if not self.needs_evaluation:
            raise RuntimeError('没有待完成的FG请求')
        gradient = np.asarray(g, np.float64).reshape(-1)
        if gradient.shape != self.g.shape or not np.isfinite(f) or not np.isfinite(gradient).all():
            raise FloatingPointError('f/g形状或finite检查失败')
        if not np.all((self.x >= 0) & (self.x <= 1)):
            raise ValueError('Fortran提出box外请求')
        self.f[...] = float(f)
        self.g[:] = gradient
        self.evaluations += 1
        self.needs_evaluation = False
        if self.best is None or f < self.best['f']:
            self.best = dict(x=self.x.copy(), f=float(f), g=self.g.copy(), evaluation=self.evaluations)

    def snapshot(self):
        return dict(schema='scipy_setulb_complete_state.v1', scipy_version=self.scipy_version,
                    maxcor=self.maxcor, ftol=self.ftol, gtol=self.gtol, maxls=self.maxls,
                    arrays={k: getattr(self, k).copy() for k in self.ARRAY_NAMES},
                    needs_evaluation=self.needs_evaluation, evaluations=self.evaluations,
                    accepted_iterations=self.accepted_iterations, finished=self.finished,
                    terminal_task=self.terminal_task, best=copy.deepcopy(self.best))

    @classmethod
    def restore(cls, saved):
        if saved['schema'] != 'scipy_setulb_complete_state.v1' or saved['scipy_version'] != scipy.__version__:
            raise ValueError('保存状态schema/SciPy版本不同')
        obj = cls(saved['arrays']['x'], maxcor=saved['maxcor'], ftol=saved['ftol'],
                  gtol=saved['gtol'], maxls=saved['maxls'])
        for key in cls.ARRAY_NAMES:
            prior = getattr(obj, key)
            value = saved['arrays'][key]
            if prior.shape != value.shape or prior.dtype != value.dtype:
                raise ValueError('保存的工作区shape/dtype不符: ' + key)
            setattr(obj, key, value.copy())
        for key in ('needs_evaluation', 'evaluations', 'accepted_iterations', 'finished', 'terminal_task', 'best'):
            setattr(obj, key, copy.deepcopy(saved[key]))
        return obj


def drive(solver, objective_gradient, *, max_evaluations, max_seconds, elapsed_before=0., after_evaluation=None):
    start = time.monotonic()
    trace = []
    stop = None
    last_evaluation_seconds = 0.
    while not solver.finished:
        elapsed = elapsed_before + time.monotonic() - start
        if solver.needs_evaluation:
            if solver.evaluations >= max_evaluations:
                stop = 'actual_objective_gradient_evaluation_cap'
                break
            if elapsed + last_evaluation_seconds >= max_seconds:
                stop = 'time_cap'
                break
            at = time.monotonic()
            f, g = objective_gradient(solver.x)
            last_evaluation_seconds = time.monotonic() - at
            solver.provide(f, g)
            row = dict(evaluation=solver.evaluations, scaled_halfSSE=float(f),
                       task=solver.task_text(), accepted_iterations=solver.accepted_iterations,
                       elapsed=elapsed_before + time.monotonic() - start,
                       evaluation_seconds=last_evaluation_seconds,
                       best_evaluation=solver.best['evaluation'])
            trace.append(row)
            if after_evaluation is not None:
                after_evaluation(solver, row)
        else:
            if elapsed >= max_seconds:
                stop = 'time_cap'
                break
            solver.advance()
    if stop is None:
        stop = solver.terminal_task
    return dict(stop_reason=stop, actual_objective_gradient_evaluations=solver.evaluations,
                accepted_iterations=solver.accepted_iterations,
                elapsed=elapsed_before + time.monotonic() - start, trace=trace,
                best_evaluation=solver.best['evaluation'] if solver.best else None,
                evaluation_cap=max_evaluations, time_cap_seconds=max_seconds)


def same_state(a, b):
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_state(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(same_state(x, y) for x, y in zip(a, b))
    return a == b


def cpu_fixture():
    from scipy.optimize import minimize
    rng = np.random.RandomState(101)
    A = rng.normal(size=(19, 8))
    A[:, -1] = 0
    target = rng.normal(size=19)
    initial = np.full(8, .6)
    calls = []
    def fg(x):
        calls.append(x.copy())
        residual = A @ x - target
        return .5 * residual @ residual, A.T @ residual
    first = ScipyBoxLBFGSB(initial)
    cap = drive(first, fg, max_evaluations=3, max_seconds=300)
    assert len(calls) == first.evaluations == 3
    saved = first.snapshot()
    restored = ScipyBoxLBFGSB.restore(saved)
    assert same_state(saved, restored.snapshot())
    interrupted = drive(first, fg, max_evaluations=200, max_seconds=300)
    replay = drive(restored, fg, max_evaluations=200, max_seconds=300)
    assert same_state(first.snapshot(), restored.snapshot())
    direct = ScipyBoxLBFGSB(initial)
    uninterrupted = drive(direct, fg, max_evaluations=200, max_seconds=300)
    assert same_state(first.snapshot(), direct.snapshot())
    reference = minimize(fg, initial, method='L-BFGS-B', jac=True, bounds=[(0, 1)] * 8,
                         options=dict(maxcor=10, ftol=1e-12, gtol=1e-6, maxls=20, maxfun=200, maxiter=200))
    assert abs(direct.best['f'] - reference.fun) <= 1e-12
    assert np.max(np.abs(direct.best['x'] - reference.x)) <= 1e-12
    assert direct.best['x'][-1] == initial[-1]
    capped = ScipyBoxLBFGSB(initial)
    before = len(calls)
    deadline = drive(capped, fg, max_evaluations=500, max_seconds=0)
    assert len(calls) == before and capped.evaluations == 0
    assert all((0 <= direct.best['x']) & (direct.best['x'] <= 1))
    return dict(schema='scipy_box_solver_CPU_fixture.v1', passed=True, scipy_version=scipy.__version__,
                native_loaded=False, GPU_executed=False, objective='independent bounded linear least squares fixture',
                actual_callback_cap_verified=cap['actual_objective_gradient_evaluations'],
                complete_workspace_restore_exact=True, resumed_next_state_exact=True,
                uninterrupted_vs_resumed_exact=True, native_minimize_wrapper_objective_abs_difference=abs(direct.best['f']-reference.fun),
                zero_column_c0_frozen=True, expired_deadline_actual_evaluations=deadline['actual_objective_gradient_evaluations'],
                full_convergence_evaluations=uninterrupted['actual_objective_gradient_evaluations'],
                maxcor=10, ftol=1e-12, gtol=1e-6, maxls=20,
                formal_budget_unchanged=dict(max_actual_objective_gradient_evaluations=500, max_seconds=300))
