"""Find a shared transfer strength by the actual group contribution budget."""
import gc
import numpy as np
import torch

from coupled import RUN, BUILD, LEVELS, PRIMARY, Coupled, group_mask, read, write
from run_coupled import evaluate,measure,avg


def fit_root(exp,rank):
    cache=[]
    for raw in BUILD:
        b=exp.bundle(raw,'black')
        m=group_mask(exp,raw)
        before=b.render(0.,0.)
        reference,_=b.coverage(0.,0.,m)
        cache.append((raw,b,m,before,reference))
    count=sum(int(m.sum()) for _,_,m,_,_ in cache)
    choices={}
    curves={}
    for retire in LEVELS:
        target=sum(float(ref[m].double().sum()) for _,_,m,_,ref in cache)

        def contribution(gamma):
            return sum(float(b.coverage(retire,gamma,m)[0][m].double().sum()) for _,b,m,_,_ in cache)

        low,high=0.,1.
        f0,f1=contribution(low)-target,contribution(high)-target
        values=[dict(gamma=low,residual=f0),dict(gamma=high,residual=f1)]
        if f0>=0:
            selected=0.;reason='NO_LOST_GROUP_COVERAGE'
        elif f1<0:
            selected=1.;reason='INSUFFICIENT_PROXY_CAPACITY'
        else:
            reason='BRACKETED'
            for _ in range(10):
                mid=(low+high)/2
                value=contribution(mid)-target
                values.append(dict(gamma=mid,residual=value))
                if value<0:
                    low=mid
                else:
                    high=mid
            selected=(low+high)/2
        rows=[];direct=[]
        for raw,b,m,before,reference in cache:
            for gamma,dest in ((selected,rows),(0.,direct)):
                out=b.render(retire,gamma)
                cover,_=b.coverage(retire,gamma,m)
                dest.append(measure(exp,raw,out,m,cover,reference,before))
        a={k:avg([r[k] for r in rows]) for k in rows[0]}
        d={k:avg([r[k] for r in direct]) for k in direct[0]}
        errors=[k for k in ('rgb','depth') if a[k]['mean'] is None or a[k]['mean']>d[k]['mean']+1e-7]
        if a['coverage_error']['mean'] is None or a['coverage_error']['mean']>.01:
            errors.append('coverage')
        if reason!='BRACKETED':
            errors.append(reason)
        choices[str(retire)]=dict(gamma=selected,status='BUILD_REJECT' if errors else 'BUILD_PASS',
            reason=reason,rgb_depth_failures=errors,coverage_error=a['coverage_error']['mean'],
            build_group_pixels=count,metrics=a,retirement_metrics=d)
        curves[str(retire)]=values
    write(RUN/f'root_group{rank}_fit.json',dict(status='PASS',choices=choices,curves=curves,
          source='Only BUILD old-plus-new alpha*T budget; bracketed scalar bisection',iterations=10))
    print('ROOT_FIT',rank,{k:(v['gamma'],v['status'],v['rgb_depth_failures']) for k,v in choices.items()},flush=True)
    del cache
    return choices


def main():
    n=read(RUN/'groups.json')['qualified_count']
    summaries=[]
    for rank in range(n):
        with np.load(RUN/f'group_ledger_{rank}.npz',allow_pickle=False) as f:
            ledger={k:f[k] for k in f.files}
        e=Coupled(ledger)
        choices=fit_root(e,rank)
        summaries.append(evaluate(e,rank,choices,prefix='root_'))
        assert e.tr.exp.unchanged()
        del e
        gc.collect()
        torch.cuda.empty_cache()
    write(RUN/'root_summary.json',dict(status='EXECUTION_PASS',groups=n,all_groups_tested=True,
          accepted_groups=[v['rank'] for v in summaries if v['method_accepted']],
          selection_used_heldout=False,training_updates=0))


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
