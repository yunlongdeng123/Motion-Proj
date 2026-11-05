"""Evaluate every prespecified transfer level with BUILD-frozen strengths."""
import gc
import numpy as np
import torch
from coupled import RUN,HELD,LEVELS,Coupled,group_mask,read,write
from run_coupled import measure,aggregate


def main():
    n=read(RUN/'groups.json')['qualified_count']
    records=[]
    for rank in range(n):
        with np.load(RUN/f'group_ledger_{rank}.npz',allow_pickle=False) as f:
            ledger={k:f[k] for k in f.files}
        fit=read(RUN/f'root_group{rank}_fit.json')
        e=Coupled(ledger)
        local=[]
        for raw in HELD:
            b=e.bundle(raw,'black');mask=group_mask(e,raw)
            before=b.render(0.,0.);reference,_=b.coverage(0.,0.,mask)
            frame=dict(raw=raw,levels={})
            for t in LEVELS:
                gamma=fit['choices'][str(t)]['gamma']
                arms={}
                for name,(s,g) in {'parent':(0.,0.),'retirement':(t,0.),'coupled':(t,gamma)}.items():
                    pkg=b.render(s,g)
                    coverage,_=b.coverage(s,g,mask)
                    arms[name]=dict(metrics=measure(e,raw,pkg,mask,coverage,reference,before),
                                    failures=e.tr.measure(raw,pkg)['violations'])
                frame['levels'][str(t)]=arms
            local.append(frame)
        pooled={}
        for t in LEVELS:
            p={name:aggregate([r['levels'][str(t)][name]['metrics'] for r in local])
               for name in ('parent','retirement','coupled')}
            errors=[]
            if fit['choices'][str(t)]['status']!='BUILD_PASS':
                errors.append('build')
            for k in ('rgb','depth'):
                if not p['coupled'][k]['count'] or p['coupled'][k]['mean']>=p['retirement'][k]['mean']-1e-7:
                    errors.append('held_'+k)
            if p['coupled']['coverage_error']['mean'] is None or p['coupled']['coverage_error']['mean']>.01:
                errors.append('held_coverage')
            if any(r['levels'][str(t)]['coupled']['failures'] for r in local):
                errors.append('held_black_scene')
            pooled[str(t)]=dict(metrics=p,screen_issues=errors,build_choice=fit['choices'][str(t)])
        records.append(dict(rank=rank,rows=local,levels=pooled))
        print('OPERATING',rank,{k:v['screen_issues'] for k,v in pooled.items()},flush=True)
        del e;gc.collect();torch.cuda.empty_cache()
    write(RUN/'operating_curve.json',dict(status='EXECUTION_PASS',all_prespecified_levels=LEVELS,
          heldout_used_for_selection=False,groups=records,
          limit='Secondary curve is a heldout diagnostic. Full both-state acceptance remains separate.'))


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
