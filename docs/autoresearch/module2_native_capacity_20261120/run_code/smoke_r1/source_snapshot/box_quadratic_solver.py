"""Monotone projected gradient with fixed majorization backtracking.

All objective queries stay in [0,1]. No validation-tuned learning rate, Adam,
nonconvex image score, training controller, or automatic GPU invocation.
"""
import copy
import math
import random
import time
import numpy as np


def box_frank_wolfe_gap(x,g,*,gradient_linf_error=None,objective_abs_error=None,value=None):
    x=np.asarray(x,dtype=np.float64);g=np.asarray(g,dtype=np.float64)
    if x.shape!=g.shape or not np.isfinite(x).all()or not np.isfinite(g).all()or np.any((x<0)|(x>1)):raise ValueError('finite feasible RGB and matching gradient required')
    s=(g<0).astype(np.float64)
    gap=float(np.sum(g*(x-s),dtype=np.float64))
    out=dict(measured_box_FW_gap=gap,linear_minimizer_rule='s_j=1 if g_j<0 else0',
             conditional_on='fixed masks, alpha, geometry, native linear color operator and convex quadratic training objective',
             quality_PASS=False,strict_numerical_certificate=False,training_optimum_interval=None)
    if gradient_linf_error is not None and objective_abs_error is not None and value is not None:
        if min(gradient_linf_error,objective_abs_error)<0:raise ValueError('conservative error bounds must be nonnegative')
        gap_upper=gap+float(gradient_linf_error)*float(np.maximum(x,1-x).sum())
        out.update(strict_numerical_certificate=True,gap_with_gradient_error_bound=gap_upper,
                   training_optimum_interval=[float(value)-float(objective_abs_error)-gap_upper,float(value)+float(objective_abs_error)])
    return out


class ProjectedBoxQuadratic:
    """Evaluator(x, need_gradient)->(float, gradient-or-None, details).

    Defaults are engineering rules, not a formal experimental budget. Root must
    bind the registered constants, accepted-step cap and actual call accounting.
    """
    SCHEMA='fixed_road_projected_box_quadratic.v1'

    def __init__(self,x,evaluator,*,identity,L0=1.,backtrack_factor=2.,trial_decrease=2.,max_backtracks=50,comparison_atol=0.):
        self.x=np.asarray(x,dtype=np.float64).copy();self.evaluator=evaluator;self.identity=copy.deepcopy(identity)
        if not np.isfinite(self.x).all()or np.any((self.x<0)|(self.x>1)):raise ValueError('initial color must lie in box')
        if L0<=0 or backtrack_factor<=1 or trial_decrease<1 or max_backtracks<1 or comparison_atol<0:raise ValueError('invalid fixed backtracking recipe')
        self.recipe=dict(L0=float(L0),backtrack_factor=float(backtrack_factor),trial_decrease=float(trial_decrease),max_backtracks=int(max_backtracks),comparison_atol=float(comparison_atol))
        self.L=float(L0);self.iteration=0;self.trace=[];self.evaluations=0;self.gradient_evaluations=0
        self.value,self.gradient,self.details=self._evaluate(self.x,True)

    def _evaluate(self,x,need_gradient):
        if np.any((x<0)|(x>1)):raise ValueError('oracle query outside global box')
        value,g,details=self.evaluator(x.copy(),need_gradient)
        self.evaluations+=1;self.gradient_evaluations+=int(need_gradient)
        if not math.isfinite(value):raise ValueError('nonfinite objective')
        if need_gradient:
            g=np.asarray(g,dtype=np.float64)
            if g.shape!=x.shape or not np.isfinite(g).all():raise ValueError('objective gradient shape/nonfinite')
            g=g.copy()
        return float(value),g,copy.deepcopy(details)

    def step(self):
        start=time.monotonic();trial_L=self.L/self.recipe['trial_decrease'];accepted=False
        for backtrack in range(self.recipe['max_backtracks']):
            candidate=np.clip(self.x-self.gradient/trial_L,0,1);d=candidate-self.x
            f,_,details=self._evaluate(candidate,False)
            majorant=self.value+float(np.sum(self.gradient*d))+.5*trial_L*float(np.sum(d*d))
            if f<=majorant+self.recipe['comparison_atol']:
                accepted=True;break
            trial_L*=self.recipe['backtrack_factor']
        if not accepted:raise RuntimeError('majorization backtracking exhausted; preserve state, investigate numerical operator')
        old=self.value;self.x=candidate;self.L=trial_L
        self.value,self.gradient,self.details=self._evaluate(self.x,True)
        if abs(self.value-f)>self.recipe['comparison_atol']:
            raise RuntimeError('value-only and value+gradient objective disagree beyond declared numerical tolerance')
        self.iteration+=1
        record=dict(iteration=self.iteration,objective=self.value,previous_objective=old,L=self.L,
            backtracking_trials=backtrack+1,step_L2=float(np.linalg.norm(d)),seconds=time.monotonic()-start,
            value_evaluations=self.evaluations,gradient_evaluations=self.gradient_evaluations,
            **box_frank_wolfe_gap(self.x,self.gradient,value=self.value))
        # Wall time is observational output, not deterministic optimizer state.
        self.trace.append({k:copy.deepcopy(v)for k,v in record.items()if k!='seconds'})
        return copy.deepcopy(record)

    def snapshot(self):
        return dict(schema=self.SCHEMA,x=self.x.copy(),value=self.value,gradient=self.gradient.copy(),details=copy.deepcopy(self.details),
            L=self.L,iteration=self.iteration,trace=copy.deepcopy(self.trace),evaluations=self.evaluations,
            gradient_evaluations=self.gradient_evaluations,identity=copy.deepcopy(self.identity),recipe=copy.deepcopy(self.recipe))

    def restore(self,saved):
        if saved.get('schema')!=self.SCHEMA or saved['identity']!=self.identity or saved['recipe']!=self.recipe:raise ValueError('solver checkpoint identity/recipe differs')
        x=np.asarray(saved['x']);g=np.asarray(saved['gradient'])
        if x.shape!=self.x.shape or g.shape!=x.shape or not np.isfinite(x).all()or not np.isfinite(g).all()or np.any((x<0)|(x>1)):raise ValueError('invalid solver checkpoint color/gradient')
        if saved['iteration']<0 or len(saved['trace'])!=saved['iteration']or not math.isfinite(saved['value'])or saved['L']<=0:raise ValueError('invalid solver checkpoint trace/value/L')
        if [r['iteration']for r in saved['trace']]!=list(range(1,saved['iteration']+1)):raise ValueError('solver trace does not match accepted-step cursor')
        self.x=x.copy();self.gradient=g.copy();self.value=float(saved['value']);self.details=copy.deepcopy(saved['details'])
        for name in ('L','iteration','evaluations','gradient_evaluations'):setattr(self,name,copy.deepcopy(saved[name]))
        self.trace=copy.deepcopy(saved['trace'])


def capture_solver_checkpoint(solver,*,role,frames,device='cpu'):
    """Capture complete deterministic full-batch solver/RNG state, no update."""
    import torch
    out=dict(schema='native_box_oracle_full_checkpoint.v1',solver=copy.deepcopy(solver.snapshot()),role=role,
        sampler=dict(kind='fixed_full_batch',frames=list(map(int,frames)),cursor=solver.iteration),
        numpy_rng=copy.deepcopy(np.random.get_state()),python_rng=copy.deepcopy(random.getstate()),
        torch_rng=torch.get_rng_state().clone())
    if str(device).startswith('cuda'):out['cuda_rng']=[s.clone()for s in torch.cuda.get_rng_state_all()]
    return out


def restore_solver_checkpoint(saved,solver,*,role,frames,device='cpu'):
    import torch
    expected=dict(kind='fixed_full_batch',frames=list(map(int,frames)),cursor=saved['solver']['iteration'])
    if saved.get('schema')!='native_box_oracle_full_checkpoint.v1'or saved['role']!=role or saved['sampler']!=expected:
        raise ValueError('oracle checkpoint role/full-batch sampler differs')
    solver.restore(copy.deepcopy(saved['solver']))
    np.random.set_state(copy.deepcopy(saved['numpy_rng']));random.setstate(copy.deepcopy(saved['python_rng']));torch.set_rng_state(saved['torch_rng'].clone())
    if str(device).startswith('cuda'):
        if 'cuda_rng'not in saved:raise ValueError('missing complete CUDA RNG')
        torch.cuda.set_rng_state_all([x.clone()for x in saved['cuda_rng']])
