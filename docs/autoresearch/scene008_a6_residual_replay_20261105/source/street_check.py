"""Verify the accepted black-a6 handoff in the road scene with and without a37."""
import json
from pathlib import Path
import numpy as np
import torch

from replay_retirement import RUN, Trial, RAWS, PILOTS, write, metric, save
from editor_common import prepare_rasterization, merge_rows, road_kwargs, raster
from render_pvd import OBS


def tagged(rows, index):
    tags=rows['features'].new_zeros((len(rows['features']),4))
    if index is not None:
        tags[:,index]=1
    return dict(rows,features=torch.cat((rows['features'][:,:2],tags),1))


def draw(trial,raw,remove,delete):
    d=trial.raw(raw)
    n=trial.n
    c=d['cam']
    old=prepare_rasterization(c,n.static,n.args)
    old['means2D']=torch.zeros_like(old['means3D'])
    road=trial.exp.repair
    ids=road.ids
    old['opacities']=old['opacities'].index_copy(0,ids,old['opacities'][ids]*road.gate_raw.sigmoid()[:,None])
    a6=trial.split_cache[raw]
    a6=dict(a6,mask=a6['mask']&~remove)
    rows=merge_rows(tagged(old,None),tagged(a6,0))
    if not delete:
        dynamic=n.actor_kwargs(c,delete_a37=False)
        from fuse_a6 import select_rows
        a37=select_rows(dynamic,~n.keep_a6)
        rows=merge_rows(rows,tagged(a37,1))
    channels=rows['shs'].shape[1] if rows['shs'] is not None else 1
    road_rows=road_kwargs(c,road.road,n.args,channels,precomputed_color=rows['colors_precomp'] is not None)
    rows=merge_rows(rows,tagged(road_rows,2))
    from fuse_a6 import select_rows
    pvd=select_rows(trial.patch(raw,old),trial.lower)
    rows=merge_rows(rows,tagged(pvd,3))
    out=raster(c,n.args,n.static.active_sh_degree,rows,n.bg)
    env=n.env(c.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
    out['rgb']=out['rgb']+(1-out['alpha'])*env
    assert float((out['extra'].sum(0)-out['alpha'][0]).max())<4e-6
    assert not delete or out['extra'][1].abs().max().item()==0
    return out


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    trial=Trial()
    trial.start()
    with np.load(RUN/'shared_retirement.npz',allow_pickle=False) as data:
        assert np.array_equal(data['native_row_id'],trial.native_ids.cpu().numpy())
        remove=torch.as_tensor(data['extra_retired'],device='cuda')
    dest=RUN/'street'
    dest.mkdir(exist_ok=False)
    all_rows=[]
    for raw in RAWS:
        d=trial.raw(raw)
        with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as obs:
            a37=torch.as_tensor(obs['visible'][1],device='cuda')
            visible_road=torch.as_tensor(obs['road'],device='cuda')
            reveal_road=torch.as_tensor(obs['road_edit']&(trial.fixed[raw-89,3]>0),device='cuda')
            ref_road=torch.as_tensor(obs['road_edit_rgb'],device='cuda',dtype=torch.float32).permute(2,0,1)
        if ref_road.max()>1:
            ref_road/=255.
        e=trial.evidence[raw]
        rec=dict(raw=raw)
        for state in ('keep','delete'):
            is_delete=state=='delete'
            first=draw(trial,raw,trial.zero,is_delete)
            second=draw(trial,raw,remove,is_delete)
            diff=(first['rgb']-second['rgb']).abs().mean(0)
            selected=reveal_road if is_delete else visible_road
            target=ref_road if is_delete else d['rgb']
            rec[state]=dict(isolated_lower_alpha=metric(e['lower_alpha'],e['mask']),
                            lower_before=metric(first['extra'][3],e['mask']),
                            lower_after=metric(second['extra'][3],e['mask']),
                            a6_rgb_delta=metric(diff,d['obs']),a37_rgb_delta=metric(diff,a37) if not is_delete else None,
                            road_rgb_before=metric((first['rgb']-target).abs().mean(0),selected),
                            road_rgb_after=metric((second['rgb']-target).abs().mean(0),selected))
            if raw in PILOTS:
                save(dest/f'{raw:03d}_{state}_previous.png',first['rgb'])
                save(dest/f'{raw:03d}_{state}_retired.png',second['rgb'])
        all_rows.append(rec)
    assert trial.exp.unchanged()
    means={}
    for state in ('keep','delete'):
        means[state]={}
        for key in all_rows[0][state]:
            records=[r[state][key] for r in all_rows if r[state][key] is not None]
            if not records:
                continue
            count=sum(r['count'] for r in records)
            means[state][key]=sum(r['sum'] for r in records)/count if count else None
        base=means[state]['isolated_lower_alpha']
        means[state]['lower_fraction_before']=means[state]['lower_before']/base
        means[state]['lower_fraction_after']=means[state]['lower_after']/base
    write(RUN/'street.json',dict(status='PASS',means=means,rows=all_rows,training_updates=0,
                                  rule='Same native road and environment, same a6 row masks both states; delete only changes a37.'))
    print(json.dumps(means,indent=2))


if __name__=='__main__':
    main()
