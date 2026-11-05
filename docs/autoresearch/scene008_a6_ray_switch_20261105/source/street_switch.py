"""Replay the accepted ray support in both edit states on all 49 views."""
import numpy as np
import torch

from switch import RUN,Trial
from ray_support import (PARENT,RAWS,PILOTS,read,write,save,metric,
                         padded,private,select_rows,prepare_rasterization,merge_rows,road_kwargs)
from render_pvd import OBS


def tag(rows,index):
    feature=rows['features'].new_zeros((len(rows['features']),6))
    feature[:,:2]=rows['features'][:,:2]
    if index is not None:
        feature[:,index+2]=1
    return dict(rows,features=padded(dict(rows,features=feature)))


def native_state(trial,raw,delete,metadata,selection):
    n=trial.n
    camera=trial.raw(raw)['cam']
    ground=trial.exp.repair
    static=prepare_rasterization(camera,n.static,n.args)
    static['means2D']=torch.zeros_like(static['means3D'])
    ids=ground.ids
    static['opacities']=static['opacities'].index_copy(
        0,ids,static['opacities'][ids]*ground.gate_raw.sigmoid()[:,None])
    a6=tag(trial.latest_rows[raw],0)
    if metadata is not None:
        rows,parents,code=metadata
        hit=selection[parents]
        a6['features'][rows[hit],6:]=code[hit]
    merged=merge_rows(tag(static,None),a6)
    if not delete:
        dy=n.actor_kwargs(camera,delete_a37=False)
        a37=select_rows(dy,~n.keep_a6)
        merged=merge_rows(merged,tag(a37,1))
    channels=merged['shs'].shape[1] if merged['shs'] is not None else 1
    road=road_kwargs(camera,ground.road,n.args,channels,precomputed_color=merged['colors_precomp'] is not None)
    merged=merge_rows(merged,tag(road,2))
    pvd=select_rows(trial.patch(raw,static),trial.lower)
    merged=merge_rows(merged,tag(pvd,3))
    out=private(trial,raw,merged,background=n.bg,channels=4)
    env=n.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
    out['rgb']=out['rgb']+(1-out['alpha'])*env
    assert float((out['extra'].sum(0)-out['alpha'][0]).max())<4e-6
    assert not delete or out['extra'][1].abs().max().item()==0
    return out


def pool(rows):
    result={}
    for state in ('keep','delete'):
        result[state]={}
        for key in rows[0][state]:
            values=[row[state][key] for row in rows]
            count=sum(v['count'] for v in values)
            result[state][key]=sum(v['sum'] for v in values)/count if count else None
        m=result[state]
        m['before_fraction']=m['lower_before']/m['isolated']
        m['after_fraction']=m['lower_after']/m['isolated']
    return result


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    trial=Trial()
    trial.parents()
    previous=read(PARENT/'street.json')
    with np.load(RUN/'switch.npz',allow_pickle=False) as data:
        assert np.array_equal(data['native_row_id'],trial.native_ids.cpu().numpy())
        use=torch.as_tensor(data['selected'],device='cuda')
    dest=RUN/'street'
    dest.mkdir(exist_ok=False)
    frames=[]
    for raw in RAWS:
        c=trial.raw(raw)
        md=trial.code[raw]
        ev=trial.evidence[raw]
        with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as f:
            a37=torch.as_tensor(f['visible'][1],device='cuda')
            road=torch.as_tensor(f['road'],device='cuda')
            protected=torch.as_tensor(f['protect'],device='cuda')
            reveal=torch.as_tensor(f['road_edit']&(trial.fixed[raw-89,3]>0),device='cuda')
            ref=torch.as_tensor(f['road_edit_rgb'],device='cuda',dtype=torch.float32).permute(2,0,1)
        if ref.max()>1:
            ref/=255.
        row=dict(raw=raw)
        for state in ('keep','delete'):
            delete=state=='delete'
            old=native_state(trial,raw,delete,None,None)
            parent=trial.latest_rows[raw]
            trial.latest_rows[raw]=trial.switch_rows(raw,use)
            try:
                new=native_state(trial,raw,delete,md,use)
            finally:
                trial.latest_rows[raw]=parent
            q=ev['mask']
            delta=(old['rgb']-new['rgb']).abs().mean(0)
            old_lower=metric(old['extra'][3],q)
            ref_lower=previous['rows'][raw-89][state]['lower_after']
            assert old_lower['count']==ref_lower['count'] and abs(old_lower['sum']-ref_lower['sum'])<.001,(raw,state)
            ground=ref if delete else c['rgb']
            mask=reveal if delete else road
            rgb_err=lambda out,img:(out['rgb']-img).abs().mean(0)
            row[state]=dict(isolated=metric(ev['lower_alpha'],q),lower_before=old_lower,
                            lower_after=metric(new['extra'][3],q),
                            a6_before=metric(rgb_err(old,c['rgb']),c['obs']),
                            a6_after=metric(rgb_err(new,c['rgb']),c['obs']),
                            a37_delta=metric(delta,torch.zeros_like(a37) if delete else a37),
                            road_before=metric(rgb_err(old,ground),mask),
                            road_after=metric(rgb_err(new,ground),mask),
                            protected_rgb_delta=metric(delta,protected),
                            full_rgb_delta=metric(delta,torch.ones_like(q)))
            if raw in PILOTS:
                save(dest/f'{raw:03d}_{state}_previous.png',old['rgb'])
                save(dest/f'{raw:03d}_{state}_new.png',new['rgb'])
        frames.append(row)
    assert trial.exp.unchanged()
    write(RUN/'street.json',dict(status='PASS',rows=frames,pooled=pool(frames),training_updates=0,
                                 prior_street_replay_passed=True,share_across_edits=True))
    print(pool(frames),flush=True)


if __name__=='__main__':
    main()
