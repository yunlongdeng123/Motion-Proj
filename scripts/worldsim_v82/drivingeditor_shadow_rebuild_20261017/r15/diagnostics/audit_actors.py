"""对冻结 4k 模型做 a6/a37 分层、删除、阴影反事实及贡献审计，SGD=0。"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image

ROOT = Path('/root/autodl-tmp/drivingeditor_repro/campaign/nearby_actors_20261015_v1/shadow_rebuild_20261017_v3')
BASE = ROOT / 'backend_restore_r15'
sys.path.insert(0, str(ROOT / 'implementation_r15_train_v3/shadow_rebuild_v3'))
from shadow_rebuild.config import Config, LossWeights
from shadow_rebuild.data import Dataset
from shadow_rebuild.model import SceneModel
from shadow_rebuild.train import seed_from_state
from shadow_rebuild.raster import render_buffers, cuda_raster
from shadow_rebuild.losses import pixel_valid
from shadow_rebuild.math3d import linear_to_srgb

TARGETS = [6, 37]
ANCHORS = {6: [105, 113, 137], 37: [101, 121, 137]}


def write(path, value):
    path.write_text(json.dumps(value, indent=2))


def subset(g, mask):
    # 新几何对象必须丢弃父对象阴影缓存，保证删除同时移除 caster。
    return type(g)(**{key:value[mask] for key,value in g.items()})


def buffers(camera, g, count, sky):
    p = render_buffers(camera, g, count, 'cuda', 10, isolated=False)
    p['sky'] = sky
    p['canonical'] = p['appearance'] + (1 - p['alpha'])[...,None] * sky
    return p


def png(path, value, alpha=False):
    if value.ndim == 2:
        value = value[...,None].expand(-1,-1,3)
    value = value if alpha else linear_to_srgb(value)
    Image.fromarray((value.clamp(0,1)*255).round().byte().cpu().numpy()).save(path)


def mean(value, mask):
    return float(value[mask].mean()) if bool(mask.any()) else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    saved = torch.load(BASE / 'paused_checkpoint_004000_20261017/checkpoint.pt', map_location='cpu', weights_only=False)
    raw = saved['config'].copy()
    raw['weights'] = LossWeights(**raw['weights'])
    cfg = Config(**raw).validate()
    data = Dataset(BASE / 'data', 'cuda', True)
    model = SceneModel(seed_from_state(saved['state']), cfg).cuda()
    model.load_state_dict(saved['state'])
    model.set_stage('geometry')
    model.requires_grad_(False)
    cfg.shadows = False
    owner = model.field.owner.cpu().numpy()
    n = len(owner)
    evidence = np.zeros((n,4), np.float64)
    hits = np.zeros((n,3), np.uint16)
    records = []
    anchors = []
    start = time.monotonic()
    with torch.no_grad():
        for index in range(len(data.rows)):
            frame = data.frame(index)
            raw_frame = int(frame.frame_id[-3:])
            g, p = model.buffers(frame, len(data.actor_ids), 'cuda', isolated=False)
            full_off = model.train_image(p,g)
            static_g = subset(g, g['owner'] < 0)
            sp = buffers(frame.camera, static_g, len(data.actor_ids), p['sky'])
            static_off = model.train_image(sp,static_g)
            # 与competition_terms严格同口径：比较全静态/全动态孤立分支，
            # 不能用单个目标Actor深度替代全动态深度。
            dynamic = g['owner'] >= 0
            all_z = (g['xyz'] @ frame.camera.view[:3,:3].T + frame.camera.view[:3,3])[:,2:3]
            dynamic_moment = cuda_raster(frame.camera,g['xyz'][dynamic],g['cov'][dynamic],g['opacity'][dynamic],torch.cat((torch.ones_like(all_z),all_z),-1)[dynamic],10)
            dynamic_depth = dynamic_moment[...,1]/dynamic_moment[...,0].clamp_min(1e-6)
            branch_delta = (sp['depth']-dynamic_depth).abs()
            branch_match = torch.exp(-.5*(branch_delta/cfg.ownership_depth_tolerance_m).square())*(branch_delta<3*cfg.ownership_depth_tolerance_m)*(sp['alpha']>1e-4)*(dynamic_moment[...,0]>1e-4)
            valid = pixel_valid(frame)
            for actor in TARGETS:
                body = frame.masks[...,actor] & valid[...,actor]
                exterior = (~frame.masks[...,actor]) & valid[...,actor]
                deleted_g = subset(g, g['owner'] != actor)
                dp = buffers(frame.camera, deleted_g, len(data.actor_ids), p['sky'])
                delete_off = model.train_image(dp,deleted_g)
                selected = g['owner'] == actor
                z = (g['xyz'] @ frame.camera.view[:3,:3].T + frame.camera.view[:3,3])[:,2:3]
                channels = torch.cat((torch.ones_like(z),g['appearance'],z),-1)
                isolated = cuda_raster(frame.camera,g['xyz'][selected],g['cov'][selected],g['opacity'][selected],channels[selected],10)
                aa = isolated[...,0]
                actor_depth = isolated[...,4]/aa.clamp_min(1e-6)
                static_visible = (dp['alpha']-dp['visible_labels'].sum(-1)).clamp_min(0)
                target_delta = (sp['depth']-actor_depth).abs()
                delta, matched = branch_delta, branch_match
                record = dict(actor=actor, raw=raw_frame, split=frame.split, body_pixels=int(body.sum()),
                    isolated_alpha_inside_mean=mean(aa,body), isolated_alpha_exterior_mean=mean(aa,exterior),
                    isolated_alpha_exterior_mass=float(aa[exterior].sum()),
                    isolated_alpha_exterior_pixels_gt_005=int(((aa>.05)&exterior).sum()),
                    visible_actor_alpha_inside_mean=mean(p['visible_labels'][...,actor],body),
                    visible_actor_alpha_exterior_mass=float(p['visible_labels'][...,actor][exterior].sum()),
                    delete_static_visible_alpha_inside_mean=mean(static_visible,body),
                    delete_other_actor_visible_alpha_inside_mean=mean(dp['visible_labels'].sum(-1),body),
                    static_only_alpha_inside_mean=mean(sp['alpha'],body),
                    static_dynamic_depth_delta_inside_median=float(delta[body].median()) if bool(body.any()) else None,
                    isolated_target_static_depth_delta_inside_median=float(target_delta[body].median()) if bool(body.any()) else None,
                    gate_definition='exact full static isolated depth vs full dynamic isolated depth, with both alpha thresholds; same as competition_terms',
                    static_dynamic_competition_gate_inside_mean=mean(matched,body),
                    gate_rejected_fraction_inside=mean((delta>=3*cfg.ownership_depth_tolerance_m).float(),body),
                    known_exterior_definition='frozen pixel-valid complement; isolated exterior can include legitimate occluded support; not all classified as road')
                if raw_frame in ANCHORS[actor]:
                    folder = out/f'a{actor:02d}_raw{raw_frame}'
                    folder.mkdir(exist_ok=True)
                    cfg.shadows = True
                    full_on = model.train_image(p,g)
                    delete_on = model.train_image(dp,deleted_g)
                    cfg.shadows = False
                    record['delete_shadow_effect_MAE_inside_linear'] = mean((delete_on-delete_off).abs().mean(-1),body)
                    record['delete_shadow_effect_max_linear'] = float((delete_on-delete_off).abs().max())
                    images = dict(gt=frame.rgb, full=full_on, full_no_shadow=full_off,
                        delete=delete_on, delete_no_shadow=delete_off, static_only_no_shadow=static_off,
                        isolated_actor=isolated[...,1:4], other_actor_contribution=dp['dynamic_visible_rgb'],
                        static_visible_contribution=(delete_off-dp['dynamic_visible_rgb']-(1-dp['alpha'])[...,None]*p['sky']).clamp_min(0))
                    for label, value in images.items():
                        png(folder/(label+'.png'),value)
                    png(folder/'isolated_alpha.png',aa,True)
                    png(folder/'body_mask.png',body.float(),True)
                    png(folder/'competition_depth_gate.png',matched,True)
                    png(folder/'delete_rho.png',dp['rho_image'],True)
                    coords=body.nonzero()
                    crop=[0,0,frame.camera.width,frame.camera.height] if not len(coords) else [max(0,int(coords[:,1].min())-24),max(0,int(coords[:,0].min())-20),min(frame.camera.width,int(coords[:,1].max())+25),min(frame.camera.height,int(coords[:,0].max())+40)]
                    anchors.append(dict(actor=actor,raw=raw_frame,folder=folder.name,crop=crop))
                records.append(record)
                del deleted_g,dp,isolated

            # 用特征通道的精确梯度积累 alpha*T，不对模型参数求导或更新。
            if index in data.train_indices:
                union=(frame.masks&valid).any(-1)
                road=frame.road_certified & (frame.depth>0) & ~union
                domains=torch.stack((frame.masks[...,6]&valid[...,6],frame.masks[...,37]&valid[...,37],road,torch.ones_like(road)),-1).float()
                with torch.enable_grad():
                    feature=torch.ones(len(g['xyz']),4,device='cuda',requires_grad=True)
                    image=cuda_raster(frame.camera,g['xyz'],g['cov'],g['opacity'],feature,10)
                    contribution=torch.autograd.grad((image*domains).sum(),feature)[0].detach().cpu().numpy()
                rows=g['rows'].cpu().numpy()
                evidence[rows]+=contribution
                hits[rows]+=(contribution[:,:3]>.02).astype(np.uint16)
                del feature,image,contribution
            if (index+1)%7==0 or index==len(data.rows)-1:
                write(out/'progress.json',dict(frames=index+1,total=len(data.rows),seconds=time.monotonic()-start))
                print(json.dumps(dict(frames=index+1,seconds=time.monotonic()-start)),flush=True)

    np.savez_compressed(out/'contribution_4k.npz',owner=owner,evidence=evidence,frames=hits)
    history={}
    for label in ['before','after_split']:
        source=BASE/'gates/003000'/label
        a=np.load(source/'per_gaussian.npz')
        ev=a['evidence'];nf=a['frames'];ow=a['owner']
        history[label]={str(actor):dict(points=int((ow==actor).sum()),road_mass=float(ev[ow==actor,2].sum()),max_per_point_road_mass=float(ev[ow==actor,2].max()),max_road_support_frames=int(nf[ow==actor,2].max()),repeated_road_candidates=int(((ow==actor)&(nf[:,2]>=3)&(ev[:,2]>=.5)).sum()),mixed=int(a['mixed'][ow==actor].sum()),road_dominant=int(a['road_dominant'][ow==actor].sum())) for actor in TARGETS}
    pre=torch.load(BASE/'main/pre_owner_003000.pt',map_location='cpu',weights_only=False)
    alpha=pre['state']['field.opacity_logit'].sigmoid()
    history['pre_owner_3000_opacity']=dict(min=float(alpha.min()),max=float(alpha.max()),mean=float(alpha.mean()),reset_interval=cfg.opacity_reset_interval)
    current={}
    half=model.field.actor_box_half.cpu().numpy()
    xyz=model.field.xyz.cpu().numpy();seed=model.field.seed_xyz.cpu().numpy();scales=model.field.scales.cpu().numpy()
    for actor, column in [(6,0),(37,1)]:
        chosen=owner==actor
        sf=(owner<0)&(model.field.source_candidate_owner.cpu().numpy()==actor)
        delta=np.linalg.norm(xyz[chosen]-seed[chosen],axis=-1)
        current[str(actor)]=dict(points=int(chosen.sum()),initial_seed_points=int((data.seed['owner']==actor).sum()),
            known_body_total_alpha_mass=float(evidence[chosen,column].sum()),
            certified_road_mass=float(evidence[chosen,2].sum()),
            repeated_road_candidates=int((chosen&(hits[:,2]>=3)&(evidence[:,2]>=.5)).sum()),
            static_body_mass=float(evidence[owner<0,column].sum()),
            own_actor_body_mass=float(evidence[chosen,column].sum()),
            other_actor_body_mass=float(evidence[(owner>=0)&~chosen,column].sum()),
            static_points_with_source_candidate_actor=int(sf.sum()),
            static_candidate_body_mass=float(evidence[sf,column].sum()),
            xyz_displacement_p95_m=float(np.quantile(delta,.95)),xyz_displacement_max_m=float(delta.max()),
            max_scale_p95_m=float(np.quantile(scales[chosen].max(-1),.95)),
            centers_outside_box=int((np.abs(xyz[chosen])>half[actor]).any(-1).sum()),
            box_half_m=half[actor].tolist())
    result=dict(status='COMPLETE_TRAINING_REMAINS_PAUSED',checkpoint_step=4000,production_SGD_updates=0,
        frames=49,train_contribution_frames=len(data.train_indices),records=records,anchors=anchors,
        history_3k=history,current_4k=current,
        split_receipt=json.loads((BASE/'gates/003000/split/split.json').read_text()),
        owner_receipt={k:v for k,v in json.loads((BASE/'gates/003000/owner/owner.json').read_text()).items() if 'provenance' not in k},
        point_counts_by_owner={str(int(x)):int((owner==x).sum()) for x in np.unique(owner)},
        limits='No clean edited-road paired GT. Component counterfactuals identify rendering source; they do not establish unseen road truth or a successful repair.',seconds=time.monotonic()-start)
    write(out/'actors.json',result)
    print(json.dumps(dict(status=result['status'],current=current,history=history,seconds=result['seconds'])),flush=True)


if __name__=='__main__':
    main()
