"""Attach the existing fixed stage-D a6 PVD rows to the frozen native 10K + R3 scene."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent.parent
ROAD=ROOT.parent/'scene008_native_road1k_20261105_r1'
INSTANCES=ROOT.parent/'scene008_instances_20261105_r1'
PVD=INSTANCES/'gpu_formal_r1/d/checkpoint000000.pt'
sys.path[:0]=[str(ROAD/'source'),str(INSTANCES/'source')]
from repair37 import Experiment
from native_binding import OBS,REF
from editor_common import prepare_rasterization,road_kwargs,merge_rows,raster,save
from object_layer import ObjectLayer


def generated_layer():
    state=torch.load(PVD,map_location='cpu',weights_only=False)
    saved=state['pvd']
    layer=ObjectLayer(saved['anchor'],saved['normal'],saved['sigma'],saved['initial_rgb'],
                      saved['half_box']*2,generated=True)
    layer.register_buffer('pvd_source_rows',saved['pvd_source_rows'].clone())
    layer.load_state_dict(saved,strict=True)
    assert len(layer.anchor)==1477
    return layer.cuda().requires_grad_(False)


def append_tags(rows, values):
    rows['features']=torch.cat((rows['features'], values),1)
    return rows


def draw(exp, camera, pvd, pose, delete):
    native=exp.n
    static=prepare_rasterization(camera,native.static,native.args)
    static['means2D']=torch.zeros_like(static['means3D'])
    ids=exp.repair.ids
    static['opacities']=static['opacities'].index_copy(
        0,ids,static['opacities'][ids]*exp.repair.gate_raw.sigmoid()[:,None])
    dynamic=native.actor_kwargs(camera,delete_a37=delete)
    dtype=static['features'].dtype
    actors=native.keep_a6
    zero=torch.zeros_like(actors)
    tags=torch.stack((zero,actors,~actors,zero),1).to(dtype)
    append_tags(static,static['features'].new_zeros((len(static['features']),4)))
    merged=merge_rows(static,append_tags(dynamic,tags))
    channels=merged['shs'].shape[1] if merged['shs'] is not None else 1
    precomputed=merged['colors_precomp'] is not None
    road=road_kwargs(camera,exp.repair.road,native.args,channels,precomputed_color=precomputed)
    road_tag=road['features'].new_zeros((len(road['features']),4))
    road_tag[:,0]=1
    merged=merge_rows(merged,append_tags(road,road_tag))
    patch=pvd.raster_rows(pose,camera,native.args,channels,precomputed)
    patch_tag=patch['features'].new_zeros((len(patch['features']),4))
    patch_tag[:,3]=1
    merged=merge_rows(merged,append_tags(patch,patch_tag))
    out=raster(camera,native.args,native.static.active_sh_degree,merged,native.bg)
    env=native.env(camera.get_world_directions(False).permute(1,2,0)).permute(2,0,1)
    out['rgb']=out['rgb']+(1-out['alpha'])*env
    return out


def stat(value, mask):
    sample=value[mask]
    assert torch.isfinite(sample).all()
    return dict(count=sample.numel(),sum=float(sample.double().sum()))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    exp=Experiment()
    road=ROAD/'formal/checkpoint001000.pt'
    parent=torch.load(road,map_location='cpu',weights_only=False)
    assert parent['step']==1000 and parent['parent_identity']==exp.n.before
    exp.repair.load_state_dict(parent['repair'],strict=True)
    exp.repair.requires_grad_(False)
    pvd=generated_layer()
    dest=ROOT/'render'
    dest.mkdir(exist_ok=False)
    with np.load(REF,allow_pickle=False) as file:
        fixed=file['depth_m'].copy()
    rows=[]
    max_replay=0
    max_conservation=0.
    from PIL import Image
    parent_eval=ROAD/'evaluation/all49'
    for raw in range(89,138):
        camera=exp.camera(raw)
        with np.load(OBS/f'{raw:03d}.npz',allow_pickle=False) as obs:
            assert abs(float(obs['model_time'])-camera.timestamp)<1e-6
            assert np.allclose(obs['model_to_camera'].T,camera.world_view_transform.cpu().numpy(),atol=1e-6)
            pose=torch.as_tensor(obs['model_from_local'][0],device='cuda',dtype=torch.float32)
            a6, a37=torch.as_tensor(obs['visible'],device='cuda')
            target=torch.as_tensor(obs['rgb'],device='cuda',dtype=torch.float32).permute(2,0,1)
            ground=torch.as_tensor(obs['road_edit']&(fixed[raw-89,3]>0),device='cuda')
            ground_rgb=torch.as_tensor(obs['road_edit_rgb'],device='cuda',dtype=torch.float32).permute(2,0,1)
            known=torch.as_tensor(obs['depth_valid'][0],device='cuda')
            depth=torch.as_tensor(obs['depth_m'][0],device='cuda')
        if target.max()>1:
            target/=255
        if ground_rgb.max()>1:
            ground_rgb/=255
        reveal=torch.as_tensor(fixed[raw-89,1]>0,device='cuda')
        proposal=torch.as_tensor(fixed[raw-89,2]>0,device='cuda')
        record=dict(raw=raw)
        for edit in (False,True):
            old=exp.render(raw,keep=not edit)
            new=draw(exp,camera,pvd,pose,delete=edit)
            weights=new['extra']
            max_conservation=max(max_conservation,float((weights.sum(0)-new['alpha'][0]).max()))
            assert (weights>=0).all() and max_conservation<3e-6
            if edit:
                assert weights[2].abs().max().item()==0
            state='delete' if edit else 'keep'
            old_name=f'{raw:03d}_'+('repaired.png' if edit else 'noedit_road.png')
            old_rgb=np.asarray(Image.open(parent_eval/old_name))
            replay=np.uint8(np.round(old['rgb'].clamp(0,1).permute(1,2,0).cpu().numpy()*255))
            max_replay=max(max_replay,int(np.abs(replay.astype(np.int16)-old_rgb).max()))
            assert max_replay<=1
            error=lambda p, rgb: (p['rgb']-rgb).abs().mean(0)
            info={'a6_rgb':{'before':stat(error(old,target),a6),'pvd':stat(error(new,target),a6)},
                  'pvd_on_observed_a6':stat(weights[3],a6),
                  'pvd_on_observed_a37':stat(weights[3],a37),
                  'pvd_on_fixed_unknown_proxy':stat(weights[3],proposal),
                  'a6_total_on_reprojected':{'before':stat(old['extra'][1],reveal),
                                            'pvd':stat(weights[1]+weights[3],reveal)}}
            if edit:
                info['reobserved_road_rgb']={'before':stat(error(old,ground_rgb),ground),
                                            'pvd':stat(error(new,ground_rgb),ground)}
                z=lambda p: p['depth_sum'][0]/p['alpha'][0].clamp_min(1e-5)/exp.meta['meter_to_model']
                info['a6_measured_depth']={'before':stat((z(old)-depth).abs(),known),
                                          'pvd':stat((z(new)-depth).abs(),known)}
            else:
                info['a37_rgb']={'before':stat(error(old,target),a37),'pvd':stat(error(new,target),a37)}
            record[state]=info
            save(dest/f'{raw:03d}_{state}.png',new['rgb'])
            if raw in (105,113,130):
                save(dest/f'{raw:03d}_{state}_pvd_w.png',weights[3:4].expand(3,-1,-1))
        rows.append(record)
        if raw in (105,113,130,137):
            print(raw,record['delete']['pvd_on_fixed_unknown_proxy'],flush=True)
    assert exp.unchanged()
    summary=dict(status='PASS',rows=rows,pvd_rows=len(pvd.anchor),pvd_source=str(PVD),road=str(road),
                 baseline_replay_uint8_max=max_replay,contribution_over_alpha_max=max_conservation,
                 training_updates=0,stage_D_colour='inferred from a6 observations',
                 same_pvd_both_states=True,inputs_frozen=True)
    (dest/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':
    main()
