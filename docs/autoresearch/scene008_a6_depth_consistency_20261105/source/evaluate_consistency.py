"""Fixed five-arm, full49 support-consistency evaluation; no strength search."""
import hashlib
import numpy as np
import torch
from consistency import RUN,Consistency,RAWS,BUILD,HELD,read,write,save
from depth_order import metric,GUARDS
from heuristic_evaluate import seam
from street_metrics import compare

ARMS=('parent','corridor','center','dual','center_matched')
STATES=('black','keep','delete')

def fields(e,ledger):
    assert np.array_equal(ledger['native_row_id'],e.tr.native_ids.cpu().numpy())
    assert np.array_equal(ledger['boxes'],e.boxes.cpu().numpy())
    return dict(parent=e.zero.clone(),corridor=e.zero+.5,
        center=e.zero.new_tensor(ledger['eligible_center'])*.5,
        dual=e.zero.new_tensor(ledger['eligible_dual'])*.5,
        center_matched=e.zero.new_tensor(ledger['eligible_center'])*float(ledger['match_coefficient']))

def pool(rows):
    n=sum(r['count'] for r in rows)
    return dict(count=n,mean=sum(r['sum'] for r in rows)/n if n else None)

def extras(e,raw,state,before,out):
    d=e.tr.raw(raw)
    masks={'known':d['obs'],'outside':e.tr.protection[raw]['outside']}
    if state!='black':
        ev=e.evidence(raw,state=='delete')
        masks.update(road=ev['road'],other=ev['protect'],a37=ev['a37'] if state=='keep' else torch.zeros_like(ev['a37']))
    delta=(out['rgb']-before['rgb']).abs().mean(0)
    union=torch.zeros_like(d['obs'])
    tails={}
    for name,m in masks.items():
        union|=m;v=delta[m]
        tails[name]=dict(count=len(v),sum=float(v.double().sum()),max=float(v.max()) if len(v) else 0.,
            above_002=int((v>.02).sum()),above_005=int((v>.05).sum()))
    mask=d['depth_valid']&~e.tr.evidence[raw]['mask']
    z0=before['depth_sum'][0]/before['alpha'][0].clamp_min(1e-5)/e.tr.scale
    z1=out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-5)/e.tr.scale
    change=metric((z1-z0).abs(),mask)
    issues=[]
    if change['count'] and change['sum']/change['count']>.005:issues.append('measured_parent_depth_change')
    if bool((delta[union]>.05).any()):issues.append('protected_rgb_tail')
    return dict(depth_change=change,measured_depth=metric((z1-d['depth']).abs(),mask),tails=tails),issues

def main():
    e=Consistency();ledger_path=RUN/'ledger.npz'
    digest=hashlib.sha256(ledger_path.read_bytes()).hexdigest()
    with np.load(ledger_path,allow_pickle=False) as f:ledger={k:f[k] for k in f.files}
    coefficients=fields(e,ledger)
    for name,strength in coefficients.items():
        np.savez_compressed(RUN/f'{name}_state.npz',native_row_id=ledger['native_row_id'],
            strength=strength.cpu().numpy(),boxes=ledger['boxes'],ledger_sha256=np.asarray(digest))
    outdir=RUN/'render';outdir.mkdir()
    rows=[]
    for raw in RAWS:
        row=dict(raw=raw,held=raw in HELD,states={})
        q=e.tr.evidence[raw]['mask'];den=e.tr.evidence[raw]['lower_alpha']
        for state in STATES:
            w=e.frame(raw,state);before=w.render(e.zero)
            assert not any(e.parity(before,w.trace.base()).values())
            row['states'][state]={}
            edge=seam(e,raw,state) if state!='black' else None
            for arm,strength in coefficients.items():
                out=w.render(strength)
                if state=='black':
                    measured=e.tr.measure(raw,out);scene=measured['metrics'];issues=list(measured['violations'])
                    if arm=='parent':assert measured==e.prior['black'][raw-89],raw
                else:scene,issues=compare(e,raw,state,before,out,edge)
                local,additional=extras(e,raw,state,before,out)
                issues=list(issues)+additional
                lower=out['extra'][1 if state=='black' else 3]
                row['states'][state][arm]=dict(scene=scene,issues=issues,local=local,
                    lower=metric(lower,q),isolated=metric(den,q),center_feature_maxabs_m=w.center_feature_maxabs_m)
                if state=='black' or raw in HELD:
                    save(outdir/f'{raw:03d}_{state}_{arm}.png',out['rgb'])
                    if state=='black':
                        save(outdir/f'{raw:03d}_{arm}_pvd.png',lower[None].expand(3,-1,-1))
                        gap=(den-lower).clamp_min(0)*q
                        save(outdir/f'{raw:03d}_{arm}_blocked.png',torch.stack((gap,.2*q,.08*q)))
            del w
        rows.append(row)
        if raw in (97,105,113,130,134,137):
            print('CONSISTENCY_FRAME',raw,{a:row['states']['black'][a]['issues'] for a in ARMS},flush=True)
    pools={}
    for scope in ('all','build','held'):
        members=[r for r in rows if scope=='all' or r['held']==(scope=='held')]
        pools[scope]={}
        for state in STATES:
            pools[scope][state]={}
            for arm in ARMS:
                items=[r['states'][state][arm] for r in members]
                lo,den=pool([r['lower'] for r in items]),pool([r['isolated'] for r in items])
                pools[scope][state][arm]=dict(reveal=lo['mean']/den['mean'],lower=lo,isolated=den,
                    failed_frames=sum(bool(r['issues']) for r in items),
                    inherited_failed_frames=sum(bool([v for v in r['issues'] if v not in ('measured_parent_depth_change','protected_rgb_tail')]) for r in items),
                    scene={k:pool([r['scene'][k] for r in items]) for k in items[0]['scene']},
                    measured_depth=pool([r['local']['measured_depth'] for r in items]),
                    parent_depth_change=pool([r['local']['depth_change'] for r in items]))
    outcomes={}
    for arm in ARMS[1:]:
        issues=[]
        for state in STATES:
            v=pools['all'][state][arm]
            if v['failed_frames']:issues.append(state+'_frame_guards')
            if state!='black':
                for k in ('road','seam'):
                    if v['scene'][k+'_after']['mean']>v['scene'][k+'_before']['mean']+GUARDS['road_mean']:issues.append(state+'_'+k+'_mean')
        for state in ('black','delete'):
            if pools['all'][state][arm]['reveal']<=pools['all'][state]['parent']['reveal']+1e-7:issues.append(state+'_no_lower_gain')
        outcomes[arm]=dict(quality_passed=not issues,issues=issues)
    primary=outcomes['dual'];applicable=str(ledger['applicability'])=='APPLICABLE'
    adopted=primary['quality_passed'] and applicable
    selected=coefficients['dual'] if adopted else e.zero
    np.savez_compressed(RUN/'selected_state.npz',native_row_id=ledger['native_row_id'],
        strength=selected.cpu().numpy(),boxes=ledger['boxes'],ledger_sha256=np.asarray(digest),
        adopted=np.bool_(adopted))
    assert e.tr.exp.unchanged() and hashlib.sha256(ledger_path.read_bytes()).hexdigest()==digest
    result=dict(status='EXECUTION_PASS',frames=49,states=STATES,arms=ARMS,held=HELD,
        parent_asset='scene008_a6_depth_order_20261105/delivery/support_accepted.npz',
        primary_candidate='dual',applicability=str(ledger['applicability']),adopted=adopted,
        selected_asset='dual' if adopted else 'parent / zero operation',ledger_sha256=digest,
        pools=pools,arm_outcomes=outcomes,rows=rows,training_updates=0,
        scope='No hidden RGB/depth truth on lower target; lower contribution is visibility, not reconstruction accuracy')
    write(RUN/'evaluation.json',result)
    print('CONSISTENCY_EVALUATED',str(ledger['applicability']),outcomes,adopted,
        {state:{a:pools['all'][state][a]['reveal'] for a in ARMS} for state in STATES},flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
