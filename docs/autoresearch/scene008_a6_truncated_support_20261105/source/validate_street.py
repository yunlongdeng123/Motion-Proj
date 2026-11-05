"""Run the accepted partial-support replacement in the separate street readout."""
import numpy as np
import torch

from retained_slabs import RUN, LATEST, Trial, RAWS, PILOTS, load, write, metric, save
from street_check import draw
from render_pvd import OBS


def street(trial,raw,rows,delete):
    previous=trial.split_cache[raw]
    trial.split_cache[raw]=rows
    try:
        mask=rows['mask'].new_zeros(len(rows['mask']))
        return draw(trial,raw,mask,delete)
    finally:
        trial.split_cache[raw]=previous


def pooled(rows):
    answer={}
    for state in ('keep','delete'):
        answer[state]={}
        for key in rows[0][state]:
            values=[row[state][key] for row in rows]
            count=sum(v['count'] for v in values)
            answer[state][key]=sum(v['sum'] for v in values)/count if count else None
        m=answer[state]
        m['before_fraction']=m['lower_before']/m['isolated']
        m['after_fraction']=m['lower_after']/m['isolated']
    return answer


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    trial=Trial()
    trial.load_parent()
    result=load(RUN/'result.json')
    previous=load(LATEST/'street.json')
    with np.load(RUN/'accepted.npz',allow_pickle=False) as data:
        assert np.array_equal(data['native_row_id'],trial.native_ids.cpu().numpy())
        selected=torch.as_tensor(data['selected'],device='cuda')
        slabs=int(data['slabs'])
    with np.load(RUN/'planes.npz',allow_pickle=False) as data:
        assert np.array_equal(data['native_row_id'],trial.native_ids.cpu().numpy())
        order=torch.as_tensor(data['order'],device='cuda')
        normal=torch.as_tensor(data['normal_local'],device='cuda')
        cut=torch.as_tensor(data['cut_local_m'],device='cuda')
    folder=RUN/'street'
    folder.mkdir()
    rows=[]
    for raw in RAWS:
        prepared={raw:trial.decomposed(raw,order,normal,cut,slabs)}
        after=trial.composite_rows(raw,selected,prepared)
        data=trial.raw(raw)
        with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as obs:
            a37=torch.as_tensor(obs['visible'][1],device='cuda')
            road=torch.as_tensor(obs['road'],device='cuda')
            revealed=torch.as_tensor(obs['road_edit']&(trial.fixed[raw-89,3]>0),device='cuda')
            road_ref=torch.as_tensor(obs['road_edit_rgb'],device='cuda',dtype=torch.float32).permute(2,0,1)
        if road_ref.max()>1:
            road_ref/=255
        row=dict(raw=raw)
        e=trial.evidence[raw]
        for state in ('keep','delete'):
            delete=state=='delete'
            old=street(trial,raw,trial.current[raw],delete)
            new=street(trial,raw,after,delete)
            error=lambda p,rgb:(p['rgb']-rgb).abs().mean(0)
            q=e['mask']
            previous_lower=metric(old['extra'][3],q)
            expected=previous['rows'][raw-89][state]['lower_after']
            assert previous_lower['count']==expected['count'] and abs(previous_lower['sum']-expected['sum'])<.001
            rd=road_ref if delete else data['rgb']
            mask=revealed if delete else road
            differ=error(old,new['rgb'])
            row[state]=dict(isolated=metric(e['lower_alpha'],q),lower_before=previous_lower,
                    lower_after=metric(new['extra'][3],q),
                    a6_before=metric(error(old,data['rgb']),data['obs']),
                    a6_after=metric(error(new,data['rgb']),data['obs']),
                    a37_delta=metric(differ,a37&~a37 if delete else a37),
                    road_before=metric(error(old,rd),mask),road_after=metric(error(new,rd),mask),
                    full_rgb_delta=metric(differ,torch.ones_like(q)))
            if raw in PILOTS:
                save(folder/f'{raw:03d}_{state}_previous.png',old['rgb'])
                save(folder/f'{raw:03d}_{state}_new.png',new['rgb'])
        rows.append(row)
    assert trial.exp.unchanged()
    out=dict(status='PASS',method=result['chosen'],rows=rows,pooled=pooled(rows),
             previous_street_replay=True,training_updates=0)
    write(RUN/'street.json',out)
    print(out['pooled'],flush=True)


if __name__=='__main__':
    main()
