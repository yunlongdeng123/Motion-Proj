"""固定线性RGB算子的box-LS证书接口与小型CPU数值fixture；不是GS训练器。"""
from pathlib import Path
import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np


def bounded_ls_certificate(colors, prediction, target, weights, adjoint):
    """f=.5*weighted RGB SSE/(3*sum weights), g=exact adjoint(normalized residual).

    只适用于固定native coverage、独立ci∈[0,1]。共享B+shader clip不可套用。
    返回FW gap与最优值lower bound；真实native浮点算子误差尚待GPU验证。
    """
    c=np.asarray(colors,np.float64);P=np.asarray(prediction,np.float64);T=np.asarray(target,np.float64)
    w=np.asarray(weights,np.float64)
    if P.shape!=T.shape or P.shape[-1]!=3 or w.shape!=P.shape[:-1]:raise ValueError('RGB/weight shape')
    if not all(np.isfinite(v).all() for v in (c,P,T,w)) or np.any(w<0) or not w.sum():raise ValueError('finite/weight')
    if np.any(c<0) or np.any(c>1):raise ValueError('ci must be primal-feasible in [0,1]')
    denominator=3*float(w.sum());residual=P-T
    f=.5*float((residual**2*w[...,None]).sum())/denominator
    g=np.asarray(adjoint(residual*w[...,None]/denominator),np.float64)
    if g.shape!=c.shape or not np.isfinite(g).all():raise ValueError('adjoint gradient shape/finite')
    gap=float((g*c-np.minimum(g,0)).sum())
    if gap < -1e-12:raise ValueError('negative FW gap violates box feasibility or gradient convention')
    gap=max(0.,gap)
    kkt=np.where(c<=1e-10,np.maximum(-g,0),np.where(c>=1-1e-10,np.maximum(g,0),np.abs(g)))
    return dict(primal_objective=f,frank_wolfe_gap=gap,optimal_objective_lower_bound=max(0.,f-gap),
                RGB_RMSE=float(np.sqrt(2*f)),box_primal_feasible=True,KKT_gradient_violation_max=float(kkt.max()),
                certificate_scope='independent fixed-linear-coverage ci box only',numeric_native_operator_error_bound_verified=False)


def classify(certificate, objective_tolerance, detail_feasible=None, operator_validated=False, objective_error_allowance=None):
    if not operator_validated or objective_error_allowance is None:return 'native_operator_unvalidated'
    if certificate['primal_objective']<=objective_tolerance and detail_feasible is not False:
        return 'feasible_RGB_solution_found' if detail_feasible is None else 'feasible_RGB_and_detail_solution_found'
    if certificate['optimal_objective_lower_bound']-objective_error_allowance>objective_tolerance:
        return 'registered_RGB_tolerance_infeasible_under_this_scope'
    return 'optimization_or_detail_inconclusive'


def toy(output):
    from scipy.optimize import lsq_linear
    rng=np.random.default_rng(101)
    # 抽象非负混合矩阵，不是任何真实GS、表面投影或自渲染teacher。
    A=np.zeros((12,6),np.float64)
    for row in range(12):
        a=row%6;b=(a+1+(row//6))%6;A[row,a]=.8;A[row,b]=.2
    A*=.95
    alpha=A.sum(1);truth=np.array([.4,.35,.3])
    target=alpha[:,None]*truth
    c0=np.tile(np.array([.05,.15,.3,.6,.8,.95])[:,None],(1,3))
    scopes=dict(primary_guarded=np.array([False,True,True,True,True,False]),conditional_relaxed=np.ones(6,bool))
    results={};checks={}
    for name,scope in scopes.items():
        fixed=A[:,~scope]@c0[~scope];free=A[:,scope]
        solved=np.column_stack([lsq_linear(free,target[:,ch]-fixed[:,ch],bounds=(0,1),method='bvls',tol=1e-12).x for ch in range(3)])
        solved=np.clip(solved,0,1)
        before=bounded_ls_certificate(c0[scope],free@c0[scope]+fixed,target,np.ones(12),lambda r:free.T@r)
        after=bounded_ls_certificate(solved,free@solved+fixed,target,np.ones(12),lambda r:free.T@r)
        tolerance=.5*(1/255)**2
        optimal=after['primal_objective']
        checks[name+'_lower_bound_valid_at_initial']=before['optimal_objective_lower_bound']<=optimal+1e-12
        checks[name+'_optimality_gap_small']=after['frank_wolfe_gap']<1e-10
        checks[name+'_KKT_small']=after['KKT_gradient_violation_max']<1e-10
        base=A@c0
        checks[name+'_fixed_offset_identity']=np.max(np.abs(free@solved+fixed-(base+free@(solved-c0[scope]))))<1e-12
        results[name]=dict(initial=before,solved=after,decision=classify(after,tolerance,operator_validated=True,objective_error_allowance=0.),
                           initial_decision=classify(before,tolerance,operator_validated=True,objective_error_allowance=0.),toy_objective_tolerance=tolerance)
    x=rng.normal(size=(6,3));y=rng.normal(size=(12,3))
    dot_error=float(abs(np.sum((A@x)*y)-np.sum(x*(A.T@y))))
    checks['abstract_adjoint_dot_identity']=dot_error<1e-12
    checks['guard_floor_is_not_pure_capacity_failure']=results['primary_guarded']['decision'].startswith('registered_RGB_tolerance_infeasible')
    checks['relaxed_constant_can_fit']=results['conditional_relaxed']['decision']=='feasible_RGB_solution_found'
    checks['unoptimized_loss_not_capacity_failure']=results['conditional_relaxed']['initial_decision']=='optimization_or_detail_inconclusive'
    checks['unknown_native_operator_defaults_to_no_verdict']=classify(after,tolerance)=='native_operator_unvalidated'
    checks={key:bool(value) for key,value in checks.items()}
    report=dict(schema='bounded_RGB_certificate_CPU_toy.v1',status='pass' if all(checks.values()) else 'failed',checks=checks,
        results=results,adjoint_dot_error=dot_error,actual_scene_operator_used=False,native_GS_render_calls=0,
        actual_model_training_updates=0,toy_small_dense_box_LS_solves=6,GPU_executed=False,CUDA_imported=False,
        interpretation='Verifies certificate algebra/interface and constraint-floor decisions only; no real road capacity verdict.',
        pending=['native operator linearity and adjoint consistency','fixed primary b and relaxed operator','registered pixel/detail thresholds','actual bounded-LS feasible solution or converged/lower-bound evidence'])
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(status=report['status'],checks=checks,decisions={k:v['decision'] for k,v in results.items()},output=str(output))))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    toy(parser.parse_args().output)
