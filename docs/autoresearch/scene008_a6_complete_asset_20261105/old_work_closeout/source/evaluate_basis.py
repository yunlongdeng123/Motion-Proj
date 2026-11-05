"""Evaluate full49 P/R first; S/U can only run after the full R gate."""
import hashlib
import numpy as np
import torch
from full_basis import RUN,SOURCE,FullBasis,RAWS,BUILD,HELD,read,write,save
from consistency import selected_hits
from evaluate_consistency import extras,pool
from heuristic_evaluate import seam
from street_metrics import compare
from depth_order import metric,GUARDS

STATES=('black','keep','delete')

def coverage(e,raw,packet,audit):
    w=e.frame(raw,'black');t=w.trace;d=e.tr.raw(raw);q=e.tr.evidence[raw]['mask']
    h=selected_hits(w,q|d['depth_valid']);lineage=e.actor_lineage[raw]
    take=(h['rows']<len(lineage))&(h['weight']>0)
    rows,pixel,weight=(h[k][take] for k in ('rows','pixel','weight'))
    ref=e.reference(raw)
    front=q.flatten()[pixel]&(t.depth[rows]/e.tr.scale<ref.flatten()[pixel]-.12)
    measured=d['depth_valid'].flatten()[pixel]
    rewritten=torch.zeros(len(lineage),device='cuda',dtype=torch.bool)
    rewritten[torch.as_tensor(packet['replaced'],device='cuda')]=True
    failed=torch.zeros_like(rewritten)
    failed[torch.as_tensor([r['index'] for r in audit['numerical_errors']],device='cuda',dtype=torch.long)]=True
    candidate=torch.as_tensor(e.eligible,device='cuda')[lineage[rows]]
    result={}
    for name,mask in dict(all=torch.ones_like(front),candidate=candidate,
                          rewritten=rewritten[rows],errors=failed[rows],
                          unsplit=candidate&~rewritten[rows]&~failed[rows]).items():
        result[name]=dict(front_mass=float(weight[mask&front].sum()),measured_mass=float(weight[mask&measured].sum()))
    return result

def measure(e,raw,state,before,out):
    q=e.tr.evidence[raw]['mask'];den=e.tr.evidence[raw]['lower_alpha']
    if state=='black':
        measured=e.tr.measure(raw,out);scene=measured['metrics'];issues=list(measured['violations'])
    else:scene,issues=compare(e,raw,state,before,out,seam(e,raw,state))
    local,extra=extras(e,raw,state,before,out);issues=list(issues)+extra
    d=e.tr.raw(raw)
    alpha_delta=(out['alpha'][0]-before['alpha'][0]).abs()
    alpha={k:metric(alpha_delta,m) for k,m in dict(known=d['obs'],outside=e.tr.protection[raw]['outside']).items()}
    for k,v in alpha.items():
        if v['count'] and v['sum']/v['count']>.01:issues.append(k+'_parent_alpha')
    i=1 if state=='black' else 3
    lower=metric(out['extra'][i],q);isolated=metric(den,q)
    shift=metric((out['extra'][i]-before['extra'][i]).abs(),q)
    if isolated['sum'] and shift['sum']/isolated['sum']>.01:issues.append('target_pvd_basis_drift')
    return dict(scene=scene,issues=issues,local=local,alpha_change=alpha,lower=lower,isolated=isolated,
                absolute_lower_change=shift)

def summarize(rows,arms):
    pools={}
    for scope in ('all','build','held'):
        members=[r for r in rows if scope=='all' or r['held']==(scope=='held')];pools[scope]={}
        for state in STATES:
            pools[scope][state]={}
            for arm in arms:
                items=[r['states'][state][arm] for r in members]
                lo,den=pool([r['lower'] for r in items]),pool([r['isolated'] for r in items])
                pools[scope][state][arm]=dict(reveal=lo['mean']/den['mean'],lower=lo,isolated=den,
                    failed_frames=sum(bool(r['issues']) for r in items),
                    scene={k:pool([r['scene'][k] for r in items]) for k in items[0]['scene']},
                    parent_depth_change=pool([r['local']['depth_change'] for r in items]),
                    measured_depth=pool([r['local']['measured_depth'] for r in items]),
                    alpha_change={k:pool([r['alpha_change'][k] for r in items]) for k in ('known','outside')},
                    absolute_lower_change=pool([r['absolute_lower_change'] for r in items]))
    return pools

def main():
    e=FullBasis();rows=[];audits=[];roles=[];packet_hash={};moments_errors=0
    output=RUN/'render';output.mkdir()
    for raw in RAWS:
        path=e.packet_dir/f'{raw:03d}.npz'
        if path.exists():packet=e.load(raw);audit=read(e.packet_dir/f'{raw:03d}.json');cached=True
        else:packet,audit=e.prepare(raw);cached=False
        # Saved frame must reconstruct the same parent input at its true time.
        e.rewrite_actor(raw,packet)
        moments_errors+=len(audit['numerical_errors']);audits.append(audit)
        roles.append(dict(raw=raw,held=raw in HELD,coverage=coverage(e,raw,packet,audit)))
        packet_hash[str(raw)]=hashlib.sha256(path.read_bytes()).hexdigest()
        row=dict(raw=raw,held=raw in HELD,cached_smoke_basis=cached,states={})
        for state in STATES:
            parent=e.merged(raw,state,False).base();basis=e.render_basis(raw,state,packet)
            if state=='black':assert e.tr.measure(raw,parent)==e.prior['black'][raw-89]
            row['states'][state]={}
            for name,out in (('P',parent),('R',basis)):
                row['states'][state][name]=measure(e,raw,state,parent,out)
                if state=='black' or raw in HELD or raw==97:
                    save(output/f'{raw:03d}_{state}_{name}.png',out['rgb'])
                    if state=='black':
                        lower=out['extra'][1];den=e.tr.evidence[raw]['lower_alpha'];q=e.tr.evidence[raw]['mask']
                        save(output/f'{raw:03d}_{name}_pvd.png',lower[None].expand(3,-1,-1))
                        gap=(den-lower).clamp_min(0)*q
                        save(output/f'{raw:03d}_{name}_blocked.png',torch.stack((gap,.2*q,.08*q)))
            if state=='black':
                q=e.tr.evidence[raw]['mask']
                row['resolved_keep']=metric(basis['extra'][2],q)
                row['resolved_replace']=metric(basis['extra'][3],q)
        rows.append(row)
        print('BASIS_FRAME',raw,len(audit['numerical_errors']),
              {s:row['states'][s]['R']['issues'] for s in STATES},flush=True)
    pools=summarize(rows,('P','R'));issues=[]
    if moments_errors:issues.append('unresolved_moment_primitives')
    for state in STATES:
        if pools['all'][state]['R']['failed_frames']:issues.append(state+'_frame_guards')
        if state!='black':
            scene=pools['all'][state]['R']['scene']
            for k in ('road','seam'):
                if scene[k+'_after']['mean']>scene[k+'_before']['mean']+GUARDS['road_mean']:issues.append(state+'_'+k+'_pooled')
    buildrows=[r for r in rows if not r['held']]
    dose_keep=sum(r['resolved_keep']['sum'] for r in buildrows);dose_replace=sum(r['resolved_replace']['sum'] for r in buildrows)
    if dose_keep+dose_replace<=1e-9:issues.append('no_resolved_child_target_dose')
    write(RUN/'basis_evaluation.json',dict(status='EXECUTION_PASS',R_passed=not issues,issues=issues,rows=rows,pools=pools,
        frame_audits=audits,coverage=roles,packet_sha256=packet_hash,training_updates=0,
        build_role_dose=dict(keep=dose_keep,replace=dose_replace),source_scope='All1502 UID planes; unresolved/tiny components retained unchanged and reported'))
    if issues:
        write(RUN/'stage2.json',dict(status='NOT_RUN_R_GATE_FAILED',issues=issues,S_executed=False,U_executed=False))
        np.savez_compressed(RUN/'selected_state.npz',native_row_id=e.tr.native_ids.cpu().numpy(),use_basis=np.bool_(False),
                            retirement=np.float32(0),source=np.asarray('parent / identity'))
        assert e.tr.exp.unchanged()
        print('BASIS_GATE_FAILED',issues,pools['all']['black']['R']['reveal'],flush=True)
        return
    # A passed R is the only valid entry to S/U. Explicitly persist the exact
    # matched dose so this optional next stage can be replayed independently.
    matched=.5*dose_replace/(dose_keep+dose_replace)
    assert 0<=matched<=.5
    write(RUN/'stage2_entry.json',dict(status='READY',matched_coefficient=matched,source_packet_sha256=packet_hash))
    write(RUN/'stage2.json',dict(status='READY_NOT_RUN',S_executed=False,U_executed=False,
                              required_action='Execute stage2 using the saved entry before publishing a method result'))

if __name__=='__main__':
    torch.set_num_threads(1)
    with torch.no_grad():main()
