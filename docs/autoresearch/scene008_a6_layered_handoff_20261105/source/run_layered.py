"""Controlled near/same/far physical support experiment, BUILD-only selection."""
import gc
import hashlib
import numpy as np
import torch

from layered import RUN,Layered,fixed_group_mask as group_mask,core
from coupled import BUILD,HELD,RAWS,read,write,PRIMARY
from run_coupled import measure,aggregate,evaluate


def non_target_depth(e,raw,before,after,mask):
    d=e.tr.raw(raw)
    fixed=d['depth_valid']&~mask
    z0=before['depth_sum'][0]/before['alpha'][0].clamp_min(1e-5)/e.tr.scale
    z1=after['depth_sum'][0]/after['alpha'][0].clamp_min(1e-5)/e.tr.scale
    value=(z1-z0).abs()[fixed]
    return dict(count=len(value),sum=float(value.double().sum()),mean=float(value.double().mean()) if len(value) else None)


def fit(e,rank,layer):
    cache=[]
    for raw in BUILD:
        b=e.bundle(raw,'black');m=group_mask(e,raw)
        before=b.render(0.,0.);ref,_=b.coverage(0.,0.,m)
        cache.append((raw,b,m,before,ref))
    target=sum(float(ref[m].double().sum()) for _,_,m,_,ref in cache)
    def residual(gamma):
        return sum(float(b.coverage(PRIMARY,gamma,m)[0][m].double().sum()) for _,b,m,_,_ in cache)-target
    lo,hi=0.,.35;f0,f1=residual(lo),residual(hi)
    trace=[dict(gamma=lo,residual=f0),dict(gamma=hi,residual=f1)]
    if target<1e-7 or f0>=-1e-7:
        gamma=0.;reason='NO_LOST_GROUP_SUPPORT'
    elif f1<0:
        gamma=hi;reason='INSUFFICIENT_PROXY_CAPACITY'
    else:
        reason='BRACKETED'
        for _ in range(10):
            mid=(lo+hi)/2;v=residual(mid);trace.append(dict(gamma=mid,residual=v))
            if v<0:lo=mid
            else:hi=mid
        gamma=(lo+hi)/2
    records={a:[] for a in ('parent','retirement','coupled')}
    guards=[];non_target=[];signed=[]
    for raw,b,m,before,ref in cache:
        d=e.tr.raw(raw)
        frame=dict(raw=raw)
        for name,(s,g) in {'parent':(0.,0.),'retirement':(PRIMARY,0.),'coupled':(PRIMARY,gamma)}.items():
            out=b.render(s,g);cov,parts=b.coverage(s,g,m)
            records[name].append(measure(e,raw,out,m,cov,ref,before))
            z=out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-5)/e.tr.scale
            frame[name]=dict(count=int(m.sum()),signed_depth_sum=float((z-d['depth'])[m].double().sum()),contributions=parts)
            if name=='coupled':
                guards.append(dict(raw=raw,issues=e.tr.measure(raw,out)['violations']))
                non_target.append(dict(raw=raw,**non_target_depth(e,raw,before,out,m)))
        signed.append(frame)
    p={a:aggregate(rows) for a,rows in records.items()}
    issues=[]
    if reason!='BRACKETED':issues.append(reason)
    if p['coupled']['coverage_error']['mean'] is None or p['coupled']['coverage_error']['mean']>.01:issues.append('coverage')
    if p['coupled']['rgb']['mean'] is None or p['coupled']['rgb']['mean']>p['parent']['rgb']['mean']+1e-7:issues.append('parent_rgb')
    if p['coupled']['depth']['mean'] is None or p['coupled']['depth']['mean']>=p['parent']['depth']['mean']-1e-5:issues.append('parent_depth')
    for k in ('rgb','depth'):
        if p['coupled'][k]['mean'] is None or p['coupled'][k]['mean']>p['retirement'][k]['mean']+1e-7:issues.append('retirement_'+k)
    if any(r['issues'] for r in guards):issues.append('black_guards')
    if any(r['mean'] is not None and r['mean']>.005 for r in non_target):issues.append('non_target_depth')
    choice=dict(gamma=gamma,status='BUILD_PASS' if not issues else 'BUILD_REJECT',reason=reason,issues=issues,
                target_old_alphaT=target,removed_alphaT=-f0,metrics=p,guard_rows=guards,non_target_depth=non_target)
    write(RUN/f'{layer}_group{rank}_fit.json',dict(status='EXECUTION_PASS',choice=choice,
          search=trace,build=BUILD,held=HELD,signed_depth=signed,geometry=e.audit(),selected_asset='candidate' if not issues else 'zero operation'))
    print('LAYER_FIT',rank,layer,reason,gamma,issues,flush=True)
    del cache
    return choice


def post_check(e,rank,layer,result):
    pools=result['pools'];p=pools['held']['black']
    issues=list(result['issues'])
    if p['coupled']['group']['rgb']['mean']>p['parent']['group']['rgb']['mean']+1e-7:issues.append('held_parent_rgb')
    if p['coupled']['group']['depth']['mean']>=p['parent']['group']['depth']['mean']-1e-5:issues.append('held_parent_depth')
    gamma=result['selected_gamma'];rows=[]
    for raw in RAWS:
        mask=group_mask(e,raw)
        for state in ('black','keep','delete'):
            b=e.bundle(raw,state)
            before=b.render(0.,0.);after=b.render(PRIMARY,gamma)
            rows.append(dict(raw=raw,state=state,**non_target_depth(e,raw,before,after,mask)))
            del b
    if any(r['mean'] is not None and r['mean']>.005 for r in rows):issues.append('all49_non_target_depth')
    result.update(issues=issues,method_accepted=not issues,non_target_depth=rows,
        final_asset='candidate' if not issues else 'zero operation',normal_slab_layer=layer)
    write(RUN/f'{layer}_group{rank}_evaluation.json',result)
    print('LAYER_ACCEPT',rank,layer,result['method_accepted'],issues,flush=True)
    return result


def main():
    import run_coupled
    run_coupled.RUN=RUN
    run_coupled.group_mask=group_mask
    records=[]
    for rank in range(3):
        path=RUN/f'group_ledger_{rank}.npz'
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path,allow_pickle=False) as f:
            ledger={k:f[k] for k in f.files}
        e=Layered(ledger,'near')
        for layer in ('near','same','far'):
            e.set_layer(layer)
            choice=fit(e,rank,layer)
            result=evaluate(e,rank,{str(PRIMARY):choice},prefix=layer+'_')
            result=post_check(e,rank,layer,result)
            records.append(dict(group=rank,layer=layer,build_status=choice['status'],
                accepted=result['method_accepted'],issues=result['issues'],final_asset=result['final_asset'],
                removed_alphaT=choice['removed_alphaT'],ledger_sha256=digest))
        assert e.tr.exp.unchanged() and hashlib.sha256(path.read_bytes()).hexdigest()==digest
        del e;gc.collect();torch.cuda.empty_cache()
    write(RUN/'summary.json',dict(status='EXECUTION_PASS',records=records,training_updates=0,
          selected_by_held=False,selection_scope='BUILD; HELD veto only, diagnostic arms all reported',
          accepted=[r for r in records if r['accepted']],final_asset='zero operation' if not any(r['accepted'] for r in records) else 'requires simultaneous-group check'))


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
