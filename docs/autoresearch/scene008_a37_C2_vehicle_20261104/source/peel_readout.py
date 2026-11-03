"""冻结三维提案下的有界delete-only旧混合表示替换。

域只用于一次持久行选择与验收；renderer不读取2D mask，也不按像素贴图。
提案未知区不是GT。每批高纯度行保留独立现场，后续保护核验可回滚。
"""
import os
os.environ['PYTHONDONTWRITEBYTECODE']='1'
from pathlib import Path
import argparse,json,time
import numpy as np
import torch
from vehicle_runtime import VehicleExperiment,NAMES
from editor_common import raster,save,prepare_rasterization
from scipy.spatial import cKDTree

RUN=Path(__file__).resolve().parent.parent

def write(path,obj):
    p=Path(path);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n');os.replace(tmp,p)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='peel_r1');p.add_argument('--rounds',type=int,default=12)
    p.add_argument('--batch',type=int,default=2048);p.add_argument('--purity',type=float,default=.90)
    args=p.parse_args();dest=RUN/args.output;dest.mkdir(exist_ok=False)
    torch.set_num_threads(1);start=time.monotonic()
    cfg=dict(actor_subset='missing_lower',legacy_replacement='native',colour_source='observed_initial',road_policy='delete_only')
    e=VehicleExperiment(cfg);e.old.actor.requires_grad_(False)
    count=len(e.n.static.get_xyz);raws=list(range(89,138));fixed={};reference={};rows=[]
    # 独立已观测道路仅用于保护，0.15m近邻不授权删真实receiver。
    road_xyz=e.old.base.repair.road.xyz.detach().cpu().numpy()
    road_tree=cKDTree(road_xyz)
    scale=e.old.visibility.scale
    with torch.no_grad():
        saved=e.old.actor.active.clone();e.old.actor.active=torch.ones_like(saved)
        for raw in raws:
            c=e.old.base.camera(raw)
            kw=e.old.actor.kwargs(c,e.old.matrix(raw),e.n.args,e.n.static.get_features.shape[1],e.n.args.convert_SHs_python)
            e.tagged(kw,e.groups);alone=raster(c,e.n.args,e.n.static.active_sh_degree,kw,e.n.bg)
            view=e.old.view(raw);labels=torch.as_tensor(view['actor_labels'],device='cuda')
            unknown=torch.as_tensor(view['unknown_mask'],device='cuda',dtype=torch.bool)
            repair=torch.as_tensor(e.old.base.view(raw)['repair_mask'],device='cuda',dtype=torch.bool)
            amodal=alone['alpha'][0]>.1
            # 删除目标原来的像素必须属于可编辑域，不能被域外保护隐式锁死。
            allowed_edit = amodal | repair | (labels == 5)
            guard=(~allowed_edit)|((labels>0)&(labels<=4))
            domain=(alone['extra'][5]>.5)&(alone['alpha'][0]>.95)&unknown&~guard
            # a6真实OBB最近深度前的旧表达才可进入候选；不剥离车后合法背景。
            T=e.old.matrix(raw);half=torch.as_tensor(e.old.geometry['box_lwh_m']/2,device='cuda',dtype=T.dtype)
            signs=torch.tensor([[x,y,z] for x in (-1,1) for y in (-1,1) for z in (-1,1)],device='cuda',dtype=T.dtype)
            xyz=(signs*half)@T[:3,:3].T+T[:3,3]
            h=torch.cat((xyz,torch.ones_like(xyz[:,:1])),1)
            near_depth=float((h@c.world_view_transform[:,2:3]).min())
            fixed[raw]=dict(domain=domain,guard=guard,visible=labels==4,other=(labels>0)&(labels<4),near_depth=near_depth)
        e.old.actor.active=saved
        for raw in raws:
            out=e.render(raw,False);reference[raw]=out['rgb'].detach().cpu()
        for raw in (105,113,130):
            save(dest/f'initial_{raw}_delete.png',reference[raw].to('cuda'))
            save(dest/f'fixed_domain_{raw}.png',fixed[raw]['domain'][None].float().repeat(3,1,1))
    selected=torch.zeros(count,dtype=torch.bool,device='cuda')
    # Initial values and every accepted set are persisted independently.
    summaries=[]
    for iteration in range(args.rounds):
        if (RUN/'STOP_REQUESTED').exists():
            write(dest/'stop_receipt.json',dict(reason='user quota/resource stop',iteration=iteration));break
        fit=torch.zeros(count,device='cuda');guard_score=torch.zeros_like(fit)
        worst_guard=torch.zeros_like(fit);front=torch.zeros_like(selected);road_protected=torch.zeros_like(selected)
        adopted=0.;denom=0
        for raw in raws:
            if not fixed[raw]['domain'].any():continue
            marker=torch.zeros((count,2),device='cuda',requires_grad=True)
            out=e.render(raw,False,static_marker=marker)
            d,g=fixed[raw]['domain'],fixed[raw]['guard']
            value=out['static_marker'][0,d].sum()+out['static_marker'][1,g].sum()
            flow=torch.autograd.grad(value,marker,only_inputs=True)[0].detach()
            fit+=flow[:,0];guard_score+=flow[:,1];worst_guard=torch.maximum(worst_guard,flow[:,1])
            adopted+=float(out['component_alpha_T'][5,d].sum());denom+=int(d.sum())
            possible=(flow[:,0]>1e-5)&~selected
            with torch.no_grad():
                kw=prepare_rasterization(e.old.base.camera(raw),e.n.static,e.n.args)
                front|=possible&(kw['features'][:,0]<fixed[raw]['near_depth']-.01*scale)
                check=torch.nonzero(possible).flatten()
                if len(check):
                    distance=road_tree.query(kw['means3D'][check].detach().cpu().numpy(),workers=1)[0]
                    road_protected[check[torch.as_tensor(distance<.15*scale,device='cuda')]]=True
            del out,value,flow,marker
        purity=fit/(fit+guard_score+1e-9)
        candidate=(fit>1e-4)&(purity>=args.purity)&front&~road_protected&~selected
        # 某一帧guard的实际贡献上限同时约束，防止仅靠跨帧平均掩盖损伤。
        candidate &= worst_guard <= 1.0
        ids=torch.nonzero(candidate).flatten()
        if len(ids):ids=ids[torch.argsort(fit[ids],descending=True)[:args.batch]]
        before=adopted/max(denom,1)
        stat=dict(iteration=iteration,selected_before=int(selected.sum()),candidate_rows=int(candidate.sum()),
            batch_rows=len(ids),domain_pixels=denom,lower_mean_before=before,seconds=time.monotonic()-start,
            all_fit_mass=float(fit.sum()),eligible_fit_mass=float(fit[candidate].sum()),batch_fit_mass=float(fit[ids].sum()))
        if not len(ids):
            stat['stop']='no safe eligible rows';summaries.append(stat);print(json.dumps(stat),flush=True);break
        selected[ids]=True;e.ids=torch.nonzero(selected).flatten();e.keep=torch.zeros(len(e.ids),device='cuda')
        changed=dict(a6=[],other=[],guard=[]);after=0.;evaluation=[]
        with torch.no_grad():
            for raw in raws:
                out=e.render(raw,False);v=fixed[raw];ref=reference[raw].to('cuda')
                delta=(out['rgb']-ref).abs().mean(0)
                for name,key in [('a6','visible'),('other','other'),('guard','guard')]:
                    m=v[key];changed[name].append(float(delta[m].mean()) if m.any() else 0.)
                after+=float(out['component_alpha_T'][5,v['domain']].sum())
                if raw in (105,113,130):
                    folder=dest/f'round{iteration+1:02d}';folder.mkdir(exist_ok=True)
                    save(folder/f'{raw}_delete.png',out['rgb'])
                    save(folder/f'{raw}_lower_alpha.png',out['component_alpha_T'][5:6].repeat(3,1,1))
                evaluation.append(dict(raw=raw,a6_change=changed['a6'][-1],other_change=changed['other'][-1],guard_change=changed['guard'][-1]))
        stat.update(lower_mean_after=after/max(denom,1),selected_after=int(selected.sum()),
                    worst_a6_change=max(changed['a6']),worst_other_change=max(changed['other']),worst_guard_change=max(changed['guard']))
        # 这是逐批保护阈值，不是未知几何正确性的评分。
        accepted=(stat['worst_a6_change']<=.005 and stat['worst_other_change']<=.002 and stat['worst_guard_change']<=.002)
        stat['accepted_by_protection']=accepted
        np.savez_compressed(dest/f'rules_round{iteration+1:02d}.npz',ids=e.ids.cpu().numpy(),keep=np.zeros(len(e.ids),np.float32))
        write(dest/f'round{iteration+1:02d}_evaluation.json',evaluation)
        summaries.append(stat);write(dest/'progress.json',dict(rows=summaries,config=cfg,purity=args.purity,batch=args.batch))
        print(json.dumps(stat),flush=True)
        if not accepted:
            selected[ids]=False;e.ids=torch.nonzero(selected).flatten();e.keep=torch.zeros(len(e.ids),device='cuda')
            stat['rolled_back']=True;break
        if stat['lower_mean_after']>=.7:
            stat['stop']='readout improvement reached diagnostic cap';break
    np.savez_compressed(dest/'replacement_rules.npz',ids=e.ids.cpu().numpy(),keep=np.zeros(len(e.ids),np.float32))
    result=dict(config=cfg,rows=summaries,final_selected_rows=len(e.ids),seconds=time.monotonic()-start,updates=0,
        parent_unchanged=e.unchanged(),geometry_fixed=True,road_fixed=True,prior_fixed=True,
        new_appearance_prior=False,human_verdict=None,quality_accepted=False,
        warning='Native lower support is a frozen proposal domain, not hidden body GT. Contribution adoption is not completed geometry accuracy.')
    write(dest/'summary.json',result);print(json.dumps(dict(stage='COMPLETE',**{k:result[k] for k in ['final_selected_rows','seconds','parent_unchanged']})),flush=True)

if __name__=='__main__':main()
