"""Paired actual-volume and measured-panel evaluation of the heuristic."""
import hashlib
import numpy as np
import torch

from rescue import RUN, VOLUME, RAWS, HELD, BUILD, OBS, Probe, RETIRE, read, write, save, metric
from positive import Positive
from street_metrics import compare
from depth_order import pool_street, GUARDS, mean

ARMS=('parent','cut','proxy_only','rescue')
STATES=('black','keep','delete')


def pool(items):
    n=sum(m['count'] for m in items)
    return dict(count=n,mean=sum(m['sum'] for m in items)/n if n else None)


def pair(exp,raw,state):
    old,direct,*_=exp.raster_pair(raw,state,False)
    if exp.proxy is None:
        return dict(parent=old,cut=direct,proxy_only=old,rescue=direct)
    proxy,rescue,*_=exp.raster_pair(raw,state,True)
    return dict(parent=old,cut=direct,proxy_only=proxy,rescue=rescue)


def panel_mask(exp, raw):
    d=exp.tr.raw(raw)
    if not hasattr(exp,'panel'):
        return torch.zeros_like(d['depth_valid'])
    with np.load(OBS/'frames'/f'{raw:03d}.npz',allow_pickle=False) as f:
        K=f['K']
        inv=np.linalg.inv(f['world_to_camera']@f['world_from_local'][0])
    panel=d['depth'].new_tensor(exp.panel)
    y,x=d['depth_valid'].nonzero(as_tuple=True)
    z=d['depth'][y,x]
    cam=torch.stack(((x+.5-K[0,2])*z/K[0,0],(y+.5-K[1,2])*z/K[1,1],z),1)
    inv=d['depth'].new_tensor(inv)
    local=cam@inv[:3,:3].T+inv[:3,3]
    keep=((local>=panel[0])&(local<=panel[1])).all(1)
    out=torch.zeros_like(d['depth_valid'])
    out[y[keep],x[keep]]=True
    return out


def local_metrics(exp,raw,state,before,after,panel):
    tr=exp.tr
    d=tr.raw(raw)
    delta=(before['rgb']-after['rgb']).abs().mean(0)
    fixed=exp.evidence(raw,state=='delete') if state!='black' else None
    masks=dict(a6=d['obs'],outside=tr.protection[raw]['outside'],panel=panel)
    if fixed is not None:
        masks.update(a37=torch.zeros_like(fixed['a37']) if state=='delete' else fixed['a37'],
                     road=fixed['road'],other=fixed['protect'])
    out={}
    for k, mask in masks.items():
        v=delta[mask]
        out[k]=dict(count=len(v),max=float(v.max()) if len(v) else 0.,
                    above_002=int((v>.02).sum()),above_005=int((v>.05).sum()))
    alpha=after['alpha'][0]
    depth=after['depth_sum'][0]/alpha.clamp_min(1e-5)/tr.scale
    return dict(changes=out,panel_rgb=metric((after['rgb']-d['rgb']).abs().mean(0),panel),
                panel_depth=metric((depth-d['depth']).abs(),panel),
                panel_coverage=metric(alpha,panel))


def seam(exp,raw,state):
    key=raw,state
    if key not in exp.seam_reference:
        saved=exp.gate
        exp.gate=exp.zero
        try:
            exp.seam_reference[key]=exp.street(raw,state=='delete',exp.weight,exp.extra)['extra'][2].clone()
        finally:
            exp.gate=saved
    return exp.seam_reference[key]


def run_probe(exp,label,suffix=''):
    path=RUN/f'{label}_ledger.npz'
    ledger_hash=hashlib.sha256(path.read_bytes()).hexdigest()
    with np.load(path,allow_pickle=False) as src:
        ledger={k:src[k] for k in src.files}
    assert not np.isin(ledger['source'],HELD).any() and not np.isin(ledger['build_witness_raw'],HELD).any()
    exp.load_proxy(ledger)
    root=RUN/f'{label}{suffix}_render'
    root.mkdir()
    volume=read(VOLUME/'volume_controls.json')['controls']['corridor_quarter']['rows']
    result=[]
    for raw in RAWS:
        p=panel_mask(exp,raw)
        frame=dict(raw=raw,held=raw in HELD,states={})
        for state in STATES:
            output=pair(exp,raw,state)
            before=output['parent']
            frame['states'][state]={}
            for name, pkg in output.items():
                extras=local_metrics(exp,raw,state,before,pkg,p)
                if state=='black':
                    row=exp.measure_volume(raw,pkg)
                    if name=='parent':
                        assert row==exp.prior['black'][raw-89]
                    if name=='cut' and label=='rescue':
                        assert row['metrics']==volume[raw-89]['metrics'], raw
                    result_row=dict(metrics=row['metrics'],failures=row['violations'],local=extras)
                else:
                    values,failed=compare(exp,raw,state,before,pkg,seam(exp,raw,state))
                    result_row=dict(metrics=values,failures=failed,local=extras)
                frame['states'][state][name]=result_row
                if label=='rescue' or raw in HELD:
                    save(root/f'{raw:03d}_{state}_{name}.png',pkg['rgb'])
        result.append(frame)
        if raw in (105,113,130,137):
            print('PAIR',label,raw,frame['states']['black']['cut']['failures'],
                  frame['states']['black']['rescue']['failures'],flush=True)
    stats={}
    for scope in ('all','build','held'):
        members=[row for row in result if scope=='all' or row['held']==(scope=='held')]
        stats[scope]={}
        for arm in ARMS:
            values={}
            for state in STATES:
                rows=[row['states'][state][arm] for row in members]
                m={k:pool([r['metrics'][k] for r in rows]) for k in rows[0]['metrics']}
                lp='lower_pvd' if state=='black' else 'lower_after'
                dn='isolated_lower_alpha' if state=='black' else 'isolated'
                m['reveal']=m[lp]['mean']/m[dn]['mean']
                m['frame_failures']=sum(bool(r['failures']) for r in rows)
                for k in ('panel_rgb','panel_depth','panel_coverage'):
                    m[k]=pool([r['local'][k] for r in rows])
                m['damage']={k:{'above_002':sum(r['local']['changes'][k]['above_002'] for r in rows),
                                'above_005':sum(r['local']['changes'][k]['above_005'] for r in rows),
                                'max':max(r['local']['changes'][k]['max'] for r in rows)}
                             for k in rows[0]['local']['changes']}
                if state!='black':
                    m['pooled_guard_failures']=[k for k in ('road','seam')
                                               if m[k+'_after']['mean']>m[k+'_before']['mean']+GUARDS['road_mean']]
                values[state]=m
            stats[scope][arm]=values
    assert exp.tr.exp.unchanged()
    data=dict(status='PASS',frames=49,states=STATES,arms=ARMS,held=HELD,proxy_rows=len(ledger['xyz']),
              retirement=exp.retire.cpu().tolist(),mode=label,ledger_sha256=ledger_hash,selection_variant=suffix,stats=stats,rows=result,training_updates=0)
    assert hashlib.sha256(path.read_bytes()).hexdigest()==ledger_hash
    write(RUN/f'{label}{suffix}_evaluation.json',data)
    print('EVALUATED',label,len(ledger['xyz']),{a:{s:stats['all'][a][s]['reveal'] for s in STATES} for a in ARMS},flush=True)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('experiment',choices=('rescue','positive'))
    p.add_argument('--suffix',default='')
    a=p.parse_args()
    torch.set_num_threads(1)
    with torch.no_grad():
        run_probe(Probe() if a.experiment=='rescue' else Positive(),a.experiment,a.suffix)
