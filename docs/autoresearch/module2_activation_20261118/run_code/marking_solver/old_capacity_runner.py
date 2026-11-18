"""门2：固定原生算子上的独立ci box-LS；导入和prepare模式不加载Torch/native。"""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
import numpy as np

CPU=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_cpu_20261117_r1')
GPU=Path('/root/autodl-tmp/drivingeditor_repro/campaign/road_capacity_gpu_20261118_r1')
FRAMES=(86,110,134,179,198)


def read(path):return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def load(path,keys=None):
    with np.load(path,allow_pickle=False) as z:return {key:z[key] for key in (z.files if keys is None else keys)}
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Data:
    def __init__(self,targets,scope):
        self.root=Path(targets);self.protocol=read(CPU/'protocol_cpu_prepared.json')
        self.field=read(self.root/'field_config.json');self.cases=self.protocol['check2']['cases'];self.geometry={};self.masks={}
        self.scopes=load(self.root/'color_scopes.npz');self.scope=scope
        self.eligible_world=self.scopes['primary_world'] if scope=='primary' else self.scopes['conditional_relaxed_world']
        self.active_world=self.scopes['active_world_rows'];self.eligible=self.eligible_world[self.active_world]
        self.total=0;self.strata={}
        for raw in FRAMES:
            z=load(self.root/'geometry'/f'{raw:03d}.npz',('fixed_evaluation_domain','canonical_mesh_hit_valid','fixed_alpha','surface_reference_kind','surface_world_xyz','K','baseline_forward_rgb_srgb__not_teacher'))
            d=z['fixed_evaluation_domain'];exact=d&z['canonical_mesh_hit_valid'];color=exact&(z['fixed_alpha']>.9)
            self.geometry[raw]=z;self.masks[raw]=color;self.total+=int(color.sum())
            self.strata[raw]=dict(D_full=int(d.sum()),D_exact=int(exact.sum()),D_color=int(color.sum()))
        if self.total!=37206:raise ValueError('D_color必须为37206，禁止动态改分母')
        self.denominator=3*self.total
        frozen=read(self.root/'freeze.json')
        pins=frozen['source_files']+[frozen['field'],frozen['scopes']]+frozen['targets']+frozen['geometry']
        self.binding={str(Path(v['path'])):v['sha256'] for v in pins}
        self.binding[str(CPU/'protocol_cpu_prepared.json')]=digest(CPU/'protocol_cpu_prepared.json')
        self.binding[str(GPU/'execution_contract.json')]=digest(GPU/'execution_contract.json')

    def check_binding(self):
        changed=[path for path,sha in self.binding.items() if digest(path)!=sha]
        if changed:raise ValueError('冻结来源/目标改变: '+str(changed))

    def target(self,name):
        if name not in self.cases:raise ValueError('未登记case')
        return {raw:load(self.root/name/f'{raw:03d}.npz',('target_rgb_srgb','paint_fraction','detail_evaluation_mask','stress_only')) for raw in FRAMES}


class NativeOperator:
    def __init__(self,ctx,data):
        import torch
        self.torch=torch;self.ctx=ctx;self.data=data;self.device=ctx.base_rgb.device
        if not np.array_equal(ctx.world_row_ids,data.active_world):raise ValueError('native active行顺序改变')
        self.ids=torch.as_tensor(np.flatnonzero(data.eligible),device=self.device,dtype=torch.long)
        self.c0=ctx.base_rgb[self.ids].detach().clone();self.masks={r:torch.as_tensor(data.masks[r],device=self.device) for r in FRAMES}
        self.alpha_errors={};self.shapes={};self.forward_calls=0;self.backward_calls=0
        for raw in FRAMES:
            baseline=ctx.baseline(raw);actual=baseline['alpha'].squeeze()
            reference=torch.as_tensor(data.geometry[raw]['fixed_alpha'],device=self.device)
            self.alpha_errors[raw]=float((actual-reference).abs().max())
            if self.alpha_errors[raw]>1e-6:raise ValueError('canonical alpha与冻结reference不同')
        self.set_case('constant')

    def set_case(self,name):
        torch=self.torch;self.case=name;self.targets=self.data.target(name)
        parts=[]
        for raw in FRAMES:
            target=torch.as_tensor(self.targets[raw]['target_rgb_srgb'],device=self.device)
            alpha=torch.as_tensor(self.data.geometry[raw]['fixed_alpha'],device=self.device)
            parts.append((target*alpha[None])[:,self.masks[raw]].T)
        self.target=torch.cat(parts,dim=0).detach()

    def full(self,x):return self.ctx.base_rgb.index_copy(0,self.ids,x)

    def predict(self,x):
        torch=self.torch;parts=[]
        with torch.no_grad():
            for raw in FRAMES:
                out=self.ctx.render_road(raw,self.full(x));self.forward_calls+=1
                parts.append(out['rgb'][:,self.masks[raw]].T.detach())
        return torch.cat(parts,dim=0)

    def loss_gradient(self,x):
        torch=self.torch;leaf=x.detach().clone().requires_grad_(True);parts=[];offset=0;value=0.
        for raw in FRAMES:
            out=self.ctx.render_road(raw,self.full(leaf));self.forward_calls+=1
            p=out['rgb'][:,self.masks[raw]].T
            target=self.target[offset:offset+len(p)];offset+=len(p)
            loss=.5*(p.double()-target.double()).square().sum()/self.data.denominator
            value+=float(loss.detach());loss.backward();self.backward_calls+=1;parts.append(p.detach())
        g=leaf.grad.detach().clone()
        if not torch.isfinite(g).all() or not math.isfinite(value):raise FloatingPointError('native loss/gradient非finite')
        return value,torch.cat(parts,dim=0),g

    def adjoint(self,weights):
        torch=self.torch;leaf=self.c0.detach().clone().requires_grad_(True);offset=0
        for raw in FRAMES:
            out=self.ctx.render_road(raw,self.full(leaf));self.forward_calls+=1
            p=out['rgb'][:,self.masks[raw]].T;n=len(p)
            (p.double()*weights[offset:offset+n].double()).sum().backward();offset+=n;self.backward_calls+=1
        return leaf.grad.detach().clone()

    def certificate(self,x,f,g):
        torch=self.torch
        gap=float((g.double()*x.double()-torch.minimum(g.double(),torch.zeros_like(g,dtype=torch.float64))).sum())
        return dict(f=f,RGB_RMSE=math.sqrt(max(0.,2*f)),FW_gap_numeric_proxy=max(0.,gap),
            lower_numeric_proxy=max(0.,f-gap),L_safe_numeric_proxy=max(0.,f-gap-1e-5),
            conservative_objective_plus_boxgradient_error_bound=None,strict_numeric_bound_established=False,
            negative_certificate_allowed=False,certified_lower_bound=None,
            numeric_boundary='FP32 native forward/atomic adjoint; spot checks do not bound global objective+box-gradient error by1e-5')


def native_preflight(op,output):
    torch=op.torch;torch.manual_seed(101);torch.cuda.manual_seed_all(101);np.random.seed(101);random.seed(101)
    start=time.monotonic();n=len(op.c0);shape=op.c0.shape
    c1=.1+.8*torch.rand(shape,device=op.device);c2=.1+.8*torch.rand(shape,device=op.device);lam=.37
    p1=op.predict(c1);p2=op.predict(c2);pm=op.predict(lam*c1+(1-lam)*c2)
    linearity=float((pm-(lam*p1+(1-lam)*p2)).abs().max())
    center=torch.full_like(op.c0,.5);v=2*torch.rand(shape,device=op.device)-1;eps=.125
    av=(op.predict(center+eps*v)-op.predict(center-eps*v))/(2*eps)
    w=torch.randn_like(av);w=w/w.double().norm().to(w.dtype)
    atw=op.adjoint(w)
    left=float((av.double()*w.double()).sum());right=float((v.double()*atw.double()).sum())
    adjoint_abs=abs(left-right)
    normalizer=max(1e-12,float(av.double().norm()*w.double().norm()),float(v.double().norm()*atw.double().norm()))
    adjoint_relative=adjoint_abs/normalizer
    mass=op.adjoint(torch.ones_like(op.target))
    if float(mass.min())<0:raise ValueError('nonnegative native color operator assumption不成立')
    L=max(float(mass.max())/op.data.denominator*1.01,1e-12)
    baseline=op.predict(op.c0);reference=[]
    for raw in FRAMES:
        reference.append(op.data.geometry[raw]['baseline_forward_rgb_srgb__not_teacher'][:,op.data.masks[raw]].T)
    baseline_error=float((baseline-torch.as_tensor(np.concatenate(reference),device=op.device)).abs().max())
    op.fixed_b=op.predict(torch.zeros_like(op.c0))
    op.box_upper=op.predict(torch.ones_like(op.c0))
    result=dict(schema='native_fixed_color_operator_preflight.v1',linearity_max_RGB_error=linearity,
        adjoint_dot_left=left,adjoint_dot_right=right,adjoint_dot_absolute_error=adjoint_abs,
        adjoint_dot_norm_relative_error=adjoint_relative,baseline_RGB_max_error=baseline_error,alpha_max_errors=op.alpha_errors,
        eligible_rows=n,zero_Dcolor_columns=int((mass.abs().amax(1)==0).sum()),L_step_initial_numeric_bound=L,
        passed=linearity<=2e-6 and adjoint_relative<=1e-4 and baseline_error<=1e-6,
        spotcheck_thresholds=dict(linearity_abs=2e-6,adjoint_norm_relative=1e-4,baseline_abs=1e-6),
        strict_error_bound_established=False,negative_certificate_allowed=False,
        fixed_b_and_box_upper_actual_native=True,fixed_b_min=float(op.fixed_b.min()),box_upper_max=float(op.box_upper.max()),
        note='上述是实际接线spotchecks，不构成<=1e-5保守全局数值误差界',
        GPU_executed=True,elapsed_seconds=time.monotonic()-start,native_forward_calls=op.forward_calls,native_backward_calls=op.backward_calls)
    write(output,result)
    if not result['passed']:raise RuntimeError('native operator preflight failed')
    return result


def prepared(data,output):
    data.check_binding()
    write(output,dict(schema='native_ideal_capacity_prepared.v1',D_color=data.total,RGB_denominator=data.denominator,
        strata=data.strata,scope=data.scope,legal_candidate_rows=int(data.eligible.sum()),cases=data.cases,
        work_cap=data.protocol['check2']['work_cap'],GPU_executed=False,native_loaded=False,
        strict_negative_certificate='requires conservative numeric error bound; default unavailable, proxy only'))


def start_state(op,L):
    x=op.c0.detach().clone();f,p,g=op.loss_gradient(x)
    return dict(x=x,f=f,p=p,g=g,eta=1/L,eta_initial=1/L,iteration=0,elapsed=0.)


def solver_step(op,state,deadline=None):
    """投影谱梯度，沿可行方向作二次精确线搜索；不是学习率扫描。"""
    torch=op.torch;x=state['x'];g=state['g'];p=state['p'];f=state['f'];eta=state['eta']
    z=(x-eta*g).clamp(0,1);d=z-x
    if not torch.any(d):return state,dict(status='projected_step_zero',accepted=False)
    if deadline and time.monotonic()>=deadline:return state,dict(status='time_cap',accepted=False)
    pz=op.predict(z);ad=pz.double()-p.double();r=p.double()-op.target.double()
    curvature=float(ad.square().sum())/op.data.denominator
    derivative=float((r*ad).sum())/op.data.denominator
    t=min(1.,max(0.,-derivative/curvature)) if curvature>0 else 0.
    if t<=0:return state,dict(status='numeric_no_descent',accepted=False,derivative=derivative)
    retries=0
    while True:
        if deadline and time.monotonic()>=deadline:return state,dict(status='time_cap',accepted=False)
        xn=(x+t*d).clamp(0,1).detach();fn,pn,gn=op.loss_gradient(xn)
        if fn<=f+max(5e-11,1e-7*f) or retries>=6:break
        t*=.5;retries+=1
    if fn>f+max(5e-11,1e-7*f):return state,dict(status='numeric_descent_not_verified',accepted=False,retries=retries)
    s=(xn-x).double();y=(gn-g).double();sy=float((s*y).sum());ss=float(s.square().sum())
    en=ss/sy if sy>1e-30 else state['eta_initial']
    en=min(state['eta_initial']*1e6,max(state['eta_initial']*1e-6,en))
    new=dict(x=xn,f=fn,p=pn,g=gn,eta=en,eta_initial=state['eta_initial'],iteration=state['iteration']+1,elapsed=state['elapsed'])
    return new,dict(status='accepted',accepted=True,line_step=t,step_eta=eta,next_eta=en,retries=retries)


def snapshot(op,state,binding):
    torch=op.torch
    return dict(schema='fixed_native_boxLS_complete_state.v1',state={key:value.detach().cpu().clone() if torch.is_tensor(value) else value for key,value in state.items()},
        binding=binding,case=op.case,scope=op.data.scope,world_rows=op.data.active_world.copy(),eligible=op.data.eligible.copy(),
        torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())


def differences(a,b,path=''):
    import torch
    if torch.is_tensor(a):
        return [] if torch.equal(a.cpu(),b.cpu()) else [path]
    if isinstance(a,np.ndarray):return [] if np.array_equal(a,b) else [path]
    if isinstance(a,dict):
        if a.keys()!=b.keys():return [path+':keys']
        return [v for key in a for v in differences(a[key],b[key],path+'/'+str(key))]
    if isinstance(a,(tuple,list)):
        if len(a)!=len(b):return [path+':length']
        return [v for idx,(x,y) in enumerate(zip(a,b)) for v in differences(x,y,path+'/'+str(idx))]
    return [] if a==b else [path]


def restore(op,saved,binding):
    torch=op.torch
    if saved['binding']!=binding or saved['case']!=op.case or saved['scope']!=op.data.scope:raise ValueError('restore inputs/method/case changed')
    if not np.array_equal(saved['world_rows'],op.data.active_world) or not np.array_equal(saved['eligible'],op.data.eligible):raise ValueError('restore row scope changed')
    state={key:value.to(op.device) if torch.is_tensor(value) else value for key,value in saved['state'].items()}
    torch.set_rng_state(saved['torch_rng']);torch.cuda.set_rng_state_all(saved['cuda_rng']);np.random.set_state(saved['numpy_rng']);random.setstate(saved['python_rng'])
    for key in ('x','p','g'):
        if not torch.isfinite(state[key]).all():raise ValueError('restore nonfinite state')
    if not torch.all((state['x']>=0)&(state['x']<=1)):raise ValueError('restore box infeasible')
    return state


def all_images(op,x):
    torch=op.torch;images={};alpha_errors={}
    with torch.no_grad():
        for raw in FRAMES:
            out=op.ctx.render_road(raw,op.full(x));images[raw]=out['rgb'].detach().cpu().numpy()
            alpha_errors[raw]=float((out['alpha']-op.ctx.baseline(raw)['alpha']).abs().max())
    if any(v!=0 for v in alpha_errors.values()):raise RuntimeError('color改变了固定alpha')
    return images,alpha_errors


def evaluate(op,state,details,output,*,save_images=True):
    from ideal_capacity_metrics import rgb_metric
    from PIL import Image
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    images,alpha_errors=all_images(op,state['x']);rows=[]
    for raw in FRAMES:
        g=op.data.geometry[raw];D=g['fixed_evaluation_domain'];exact=D&g['canonical_mesh_hit_valid'];color=op.data.masks[raw];a=g['fixed_alpha']
        T=op.targets[raw]['target_rgb_srgb'];P=images[raw];soft=a[None]*T
        metrics={name:rgb_metric(P,soft,mask) for name,mask in [('D_full',D),('D_exact',exact),('D_color',color),('fallback',D&~exact)]}
        floor=float(.5*np.mean(np.maximum(T[:,D]-a[D][None],0).astype(np.float64)**2))
        rows.append(dict(raw=raw,soft_metrics=metrics,opaque_target=rgb_metric(P,T,D),opaque_coverage_range_floor=floor,
            gap=int((D&(a<.05)).sum()),thin=int((D&(a>=.05)&(a<=.9)).sum()),supported=int((D&(a>.9)).sum())))
        if save_images:
            np.savez_compressed(output/f'{raw:03d}.npz',road_only_rgb=P,fixed_alpha=a,target_rgb_srgb=T,D_color=color)
            Image.fromarray(np.round(np.clip(P.transpose(1,2,0),0,1)*255).astype(np.uint8)).save(output/f'{raw:03d}_road.png')
            Image.fromarray(np.round(np.clip(T.transpose(1,2,0),0,1)*255).astype(np.uint8)).save(output/f'{raw:03d}_independent_teacher.png')
    detail=details.evaluate(op.case,images);certificate=op.certificate(state['x'],state['f'],state['g'])
    violation=op.torch.maximum(op.fixed_b.double()-op.target.double(),op.target.double()-op.box_upper.double()).clamp_min(0)
    certificate['guarded_box_range_floor_numeric_proxy']=float(.5*violation.square().mean())
    rms_limit=op.data.protocol['check2']['error_envelope']['reference_RGB_RMSE']
    stress=op.case=='stress_highfreq';witness=not stress and certificate['RGB_RMSE']<=rms_limit and detail['passed']
    result=dict(case=op.case,scope=op.data.scope,certificate=certificate,detail=detail,frames=rows,alpha_actual_frozen_errors=alpha_errors,
        feasible_box=bool(((state['x']>=0)&(state['x']<=1)).all()),reference_RGB_RMSE=rms_limit,
        capacity_status='diagnostic_stress_only' if stress else 'feasible_solution_witness' if witness else 'optimization_or_numeric_certificate_inconclusive',
        negative_capacity_claim=False,human_verdict=None)
    write(output/'evaluation.json',result)
    return result


def run_smoke(op,L,details,output,binding):
    torch=op.torch;output=Path(output);output.mkdir(parents=True,exist_ok=False);op.set_case('constant')
    state=start_state(op,L);state,one=solver_step(op,state)
    torch.save(snapshot(op,state,binding),output/'checkpoint_step1.pt')
    saved=torch.load(output/'checkpoint_step1.pt',map_location='cpu',weights_only=False)
    restored=restore(op,saved,binding)
    restore_differences=differences(saved,snapshot(op,restored,binding));exact=not restore_differences
    if not exact:raise RuntimeError('smoke full state restore not exact')
    next_a,record_a=solver_step(op,restored)
    next_b,record_b=solver_step(op,restore(op,saved,binding))
    replay_errors={key:float((next_a[key]-next_b[key]).abs().max()) for key in ('x','p','g')}
    eta_difference=abs(next_a['eta']-next_b['eta'])
    evaluation=evaluate(op,next_a,details,output/'evaluation')
    full=op.full(next_a['x']).detach();K=op.ctx.audit_K(full,output/'joint_K.json')
    asset=op.ctx.save_asset(full,op.data.eligible_world,output/'asset',training_record=dict(role='engineering_lifecycle_smoke',formal_updates=0,extra_solver_updates=3))
    reload=op.ctx.validate_public_reload(full,asset,output/'public_reload.json')
    torch.save(snapshot(op,next_a,binding),output/'checkpoint_step2.pt')
    extra_updates=sum(int(r['accepted']) for r in (one,record_a,record_b))
    result=dict(schema='native_boxLS_lifecycle_smoke.v1',GPU_executed=True,extra_solver_updates=extra_updates,formal_solver_updates=0,
        restore_tensor_state_exact=exact,full_restore_differences=restore_differences,replay_tensor_maxima=replay_errors,replay_eta_difference=eta_difference,
        next_update_bitwise=all(v==0 for v in replay_errors.values()) and eta_difference==0,
        actions=[one,record_a,record_b],K_exact=K['all_exact'],public_reload=reload,
        next_case_entry='formal cases independently restart exact canonical c0; no smoke warm-start',
        optimizer_state='current c, residual prediction, gradient, spectral step, iteration, elapsed and all RNG',
        replay_pass=all(v<=1e-6 for v in replay_errors.values()),evaluation=evaluation['capacity_status'])
    write(output/'completed.json',result)
    if not result['replay_pass']:raise RuntimeError('smoke native next-update replay >1e-6')
    return result


def run_case(op,L,details,case,output,binding):
    torch=op.torch;output=Path(output);output.mkdir(parents=True,exist_ok=False);op.set_case(case)
    cap=op.data.protocol['check2']['work_cap'];maxiter=cap['max_iterations_per_case'];seconds=cap['max_seconds_per_case']
    begin=time.monotonic();deadline=begin+seconds;state=start_state(op,L);trace=[];last_duration=0.;stop='iteration_cap'
    torch.save(snapshot(op,state,binding),output/'checkpoint000000.pt')
    for iteration in range(1,maxiter+1):
        if time.monotonic()+last_duration>=deadline:stop='time_cap';break
        start=time.monotonic();state,record=solver_step(op,state,deadline);last_duration=time.monotonic()-start
        state['elapsed']=time.monotonic()-begin
        cert=op.certificate(state['x'],state['f'],state['g'])
        trace.append(dict(iteration=iteration,accepted_iteration=state['iteration'],elapsed=state['elapsed'],**record,**cert))
        if iteration==1 or iteration%25==0:print(json.dumps(dict(case=case,iteration=iteration,elapsed=state['elapsed'],f=state['f'],FW=cert['FW_gap_numeric_proxy'],status=record['status'])),flush=True)
        if iteration%50==0:torch.save(snapshot(op,state,binding),output/f'checkpoint{iteration:06d}.pt')
        if not record['accepted']:stop=record['status'];break
    state['elapsed']=time.monotonic()-begin
    torch.save(snapshot(op,state,binding),output/'checkpoint_final.pt');write(output/'trace.json',trace)
    evaluation=evaluate(op,state,details,output/'evaluation')
    full=op.full(state['x']).detach();K=op.ctx.audit_K(full,output/'joint_K.json') if op.data.scope=='primary' else dict(all_exact=None,not_evaluated_reason='conditional virtual road-only oracle')
    asset=op.ctx.save_asset(full,op.data.eligible_world,output/'asset',training_record=dict(role='ideal_capacity_oracle',case=case,scope=op.data.scope,solver_updates=state['iteration'],maximum_iterations=maxiter,maximum_seconds=seconds))
    reload=op.ctx.validate_public_reload(full,asset,output/'public_reload.json')
    op.ctx.assert_frozen()
    result=dict(completed=True,case=case,scope=op.data.scope,accepted_iterations=state['iteration'],solver_attempts=len(trace),
        solver_elapsed_seconds=state['elapsed'],work_cap=cap,stop_reason=stop,time_overrun_seconds=max(0.,state['elapsed']-seconds),
        total_case_elapsed_seconds=time.monotonic()-begin,postfit_evaluation_seconds=time.monotonic()-begin-state['elapsed'],
        evaluation_status=evaluation['capacity_status'],negative_certificate_allowed=False,numeric_bound_established=False,
        K_exact=K['all_exact'],public_reload_max_RGB=max(r['RGB_max'] for r in reload['rows']),
        public_reload_max_alpha=max(r['alpha_max'] for r in reload['rows']),GPU_executed=True,production_promoted=False,human_verdict=None)
    write(output/'completed.json',result);return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=('prepare','preflight','smoke','solve'),default='prepare')
    parser.add_argument('--targets',type=Path,default=CPU/'ideal_targets_r2')
    parser.add_argument('--scope',choices=('primary','relaxed'),default='primary')
    parser.add_argument('--output',type=Path,default=GPU/'check2/prepared.json')
    parser.add_argument('--lease-file',type=Path,default=GPU/'gpu_lease.json');parser.add_argument('--lease-owner',default='stage_qa')
    parser.add_argument('--cases',nargs='*');parser.add_argument('--smoke-receipt',type=Path)
    parser.add_argument('--execute',action='store_true');args=parser.parse_args()
    data=Data(args.targets,args.scope)
    if args.mode=='prepare':prepared(data,args.output);print(json.dumps(dict(prepared=str(args.output),GPU_executed=False)));return
    if not args.execute:raise SystemExit('native requires --execute and an explicit active matching GPU lease')
    sys.path.insert(0,str(Path(__file__).parent));from road_capacity_native import assert_gpu_lease,load_context
    assert_gpu_lease(args.lease_file,args.lease_owner)
    data.check_binding();args.output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(__file__,args.output/'runner_source.py')
    shutil.copy2(Path(__file__).with_name('road_capacity_native.py'),args.output/'native_helper_source.py')
    shutil.copy2(Path(__file__).with_name('ideal_capacity_metrics.py'),args.output/'metrics_source.py')
    write(args.output/'binding.json',dict(inputs=data.binding,runner_sha256=digest(__file__),helper_sha256=digest(Path(__file__).with_name('road_capacity_native.py'))))
    ctx=load_context(lease_file=args.lease_file,lease_owner=args.lease_owner)
    try:
        op=NativeOperator(ctx,data);result=native_preflight(op,args.output/'operator_preflight.json')
        from ideal_capacity_metrics import DetailMetrics
        details=DetailMetrics(data);write(args.output/'detail_metric_registration.json',details.registration())
        binding=dict(inputs=data.binding,runner_sha256=digest(__file__),helper_sha256=digest(Path(__file__).with_name('road_capacity_native.py')),
            metrics_sha256=digest(Path(__file__).with_name('ideal_capacity_metrics.py')))
        if args.mode=='smoke':result=run_smoke(op,result['L_step_initial_numeric_bound'],details,args.output/'lifecycle',binding)
        if args.mode=='solve':
            if args.smoke_receipt is None:raise ValueError('正式solver前必须提供已通过的真实lifecycle smoke receipt')
            smoke=read(args.smoke_receipt)
            if not smoke['restore_tensor_state_exact'] or not smoke['replay_pass'] or not smoke['K_exact']:raise ValueError('smoke incomplete')
            names=args.cases or data.cases
            if any(c not in data.cases for c in names) or len(names)!=len(set(names)):raise ValueError('case不在固定母协议或重复')
            results=[run_case(op,result['L_step_initial_numeric_bound'],details,name,args.output/name,binding) for name in names]
            result=dict(completed=True,cases=results,extra_case_automatically_started=False,negative_certificate_allowed=False)
            write(args.output/'completed.json',result)
        ctx.assert_frozen();print(json.dumps(result),flush=True)
    finally:ctx.close()


if __name__=='__main__':main()
