"""Fixed coverage-based coupling followed by independent RGB/depth gates."""
import gc
import hashlib
from pathlib import Path

import numpy as np
import torch

from coupled import (RUN, RAWS, BUILD, HELD, LEVELS, GAMMAS, PRIMARY, Coupled, create_group,
                     group_mask, jsonable, write, read, save, local_metrics, seam, compare)
from depth_order import metric, pool_street, GUARDS


def avg(items):
    n=sum(v['count'] for v in items)
    return dict(count=n,mean=sum(v['sum'] for v in items)/n if n else None)


def aggregate(rows):
    return {key:avg([r[key] for r in rows]) for key in rows[0]}


def measure(exp,raw,pkg,mask,group_map,reference,baseline):
    d=exp.tr.raw(raw)
    z=pkg['depth_sum'][0]/pkg['alpha'][0].clamp_min(1e-5)/exp.tr.scale
    return dict(rgb=metric((pkg['rgb']-d['rgb']).abs().mean(0),mask),
                depth=metric((z-d['depth']).abs(),mask),
                coverage_error=metric((group_map-reference).abs(),mask),
                old_new_coverage=metric(group_map,mask),
                total_alpha=metric(pkg['alpha'][0],mask),
                known_rgb=metric((pkg['rgb']-d['rgb']).abs().mean(0),d['obs']),
                known_change=metric((pkg['rgb']-baseline['rgb']).abs().mean(0),d['obs']),
                outside_change=metric((pkg['rgb']-baseline['rgb']).abs().mean(0),exp.tr.protection[raw]['outside']))


def fit(exp,rank):
    records={t:{g:[] for g in GAMMAS} for t in LEVELS}
    parities=[]
    for raw in BUILD:
        b=exp.bundle(raw,'black')
        mask=group_mask(exp,raw)
        baseline=b.render(0.,0.)
        m=exp.tr.measure(raw,baseline)
        assert m==exp.prior['black'][raw-89],raw
        reference,_=b.coverage(0.,0.,mask)
        # The new-surface tag is accumulated in the same exact compose pass.
        q,parts=b.coverage(0.,.4,mask)
        out=b.render(0.,.4)
        proxy=float(out['extra'][2][mask].double().sum())
        assert abs(parts['proxy']-proxy)<2e-4,(raw,parts,proxy)
        for t in LEVELS:
            for g in GAMMAS:
                after=b.render(t,g)
                group,_=b.coverage(t,g,mask)
                records[t][g].append(measure(exp,raw,after,mask,group,reference,baseline))
    pooled={str(t):{str(g):aggregate(v) for g,v in values.items()} for t,values in records.items()}
    choices={}
    for t in LEVELS:
        candidates=pooled[str(t)]
        if candidates['0.0']['coverage_error']['count']==0:
            choices[str(t)]=dict(gamma=0.,status='NO_BUILD_MEASUREMENT')
            continue
        chosen=min(GAMMAS,key=lambda g:(candidates[str(g)]['coverage_error']['mean'],g))
        selected=candidates[str(chosen)]
        direct=candidates['0.0']
        errors=[]
        for key in ('rgb','depth'):
            if selected[key]['mean']>direct[key]['mean']+1e-7:
                errors.append(key)
        if selected['coverage_error']['mean']>.01:
            errors.append('coverage')
        choices[str(t)]=dict(gamma=chosen,status='ABSTAIN' if chosen==0 else ('BUILD_REJECT' if errors else 'BUILD_PASS'),
                             rgb_depth_failures=errors,coverage_error=selected['coverage_error']['mean'])
    write(RUN/f'group{rank}_fit.json',dict(status='PASS',selection='Minimum BUILD old-plus-new surface contribution error',
          build=BUILD,held=HELD,level=PRIMARY,grid=pooled,choices=choices))
    print('COUPLED_FIT',rank,choices,flush=True)
    return choices


def evaluate(exp,rank,choices,prefix=''):
    gamma=choices[str(PRIMARY)]['gamma']
    arms={'parent':(0.,0.),'retirement':(PRIMARY,0.),'proxy_only':(0.,gamma),'coupled':(PRIMARY,gamma)}
    root=RUN/f'{prefix}group{rank}_render'
    root.mkdir()
    data=[]
    for raw in RAWS:
        mask=group_mask(exp,raw)
        row=dict(raw=raw,held=raw in HELD,states={})
        for state in ('black','keep','delete'):
            b=exp.bundle(raw,state)
            before=b.render(0.,0.)
            reference,_=b.coverage(0.,0.,mask)
            row['states'][state]={}
            for name,(s,g) in arms.items():
                out=b.render(s,g)
                group,parts=b.coverage(s,g,mask)
                metrics=measure(exp,raw,out,mask,group,reference,before)
                local=local_metrics(exp,raw,state,before,out,mask)
                if state=='black':
                    m=exp.tr.measure(raw,out)
                    tests=m['violations']
                    known=m['metrics']
                else:
                    known,tests=compare(exp,raw,state,before,out,seam(exp,raw,state))
                row['states'][state][name]=dict(group=metrics,scene=known,failures=tests,local=local,
                                               contributions=parts)
                if raw in HELD:
                    save(root/f'{raw:03d}_{state}_{name}.png',out['rgb'])
            del b
        data.append(row)
    pools={}
    for scope in ('all','build','held'):
        members=[r for r in data if scope=='all' or r['held']==(scope=='held')]
        pools[scope]={}
        for state in ('black','keep','delete'):
            pools[scope][state]={}
            for arm in arms:
                rows=[r['states'][state][arm] for r in members]
                pools[scope][state][arm]=dict(group=aggregate([r['group'] for r in rows]),
                    scene=aggregate([r['scene'] for r in rows]),
                    failed_frames=sum(bool(r['failures']) for r in rows))
    held=pools['held']['black']
    issues=[]
    if choices[str(PRIMARY)]['status']!='BUILD_PASS':
        issues.append('build_gate')
    if bool(np.all(np.asarray(gamma)==0)):
        issues.append('no_handoff')
    for metric_name in ('rgb','depth'):
        a=held['coupled']['group'][metric_name]
        b=held['retirement']['group'][metric_name]
        if not a['count'] or a['mean']>=b['mean']-1e-7:
            issues.append('held_'+metric_name)
    if held['coupled']['group']['coverage_error']['mean'] is None or held['coupled']['group']['coverage_error']['mean']>.01:
        issues.append('held_coverage')
    for state in ('black','keep','delete'):
        if pools['all'][state]['coupled']['failed_frames']:
            issues.append('scene_'+state)
        if state!='black':
            m=pools['all'][state]['coupled']['scene']
            for k in ('road','seam'):
                if m[k+'_after']['mean']>m[k+'_before']['mean']+GUARDS['road_mean']:
                    issues.append(state+'_'+k)
    result=dict(status='EXECUTION_PASS',method_accepted=not issues,rank=rank,frames=49,selected_gamma=gamma,
                retirement=PRIMARY,build_choice=choices[str(PRIMARY)],issues=issues,
                held=HELD,pools=pools,rows=data,training_updates=0)
    write(RUN/f'{prefix}group{rank}_evaluation.json',result)
    print('COUPLED_EVAL',rank,gamma,issues,{a:held[a]['group'] for a in arms},flush=True)
    return result


def main():
    with np.load(RUN/'group_ledger_0.npz',allow_pickle=False) as f:
        pass
    counts=read(RUN/'groups.json')['qualified_count']
    outcomes=[]
    for rank in range(counts):
        with np.load(RUN/f'group_ledger_{rank}.npz',allow_pickle=False) as f:
            ledger={k:f[k] for k in f.files}
        e=Coupled(ledger)
        choices=fit(e,rank)
        outcomes.append(evaluate(e,rank,choices))
        assert e.tr.exp.unchanged()
        del e
        gc.collect()
        torch.cuda.empty_cache()
    write(RUN/'coupled_summary.json',dict(status='EXECUTION_PASS',groups=counts,
          accepted_groups=[o['rank'] for o in outcomes if o['method_accepted']],
          all_groups_tested=True,selection_used_heldout=False,training_updates=0))


if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():
        main()
