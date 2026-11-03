"""车辆专用评价：真实可见RGB/测量深度、固定提案的未知域读出和保留域。"""
from pathlib import Path
import argparse,json,time,copy,shutil
import torch,numpy as np
from vehicle_runtime import VehicleExperiment,NAMES
from editor_common import save,raster
from baseline_audit import render_baseline

RUN=Path(__file__).resolve().parent.parent

def main():
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',required=True)
    p.add_argument('--road-protect-m',type=float);p.add_argument('--anchors',action='store_true');args=p.parse_args()
    dest=RUN/args.output;dest.mkdir(exist_ok=False);torch.set_num_threads(1);start=time.monotonic()
    shutil.copy2(Path(__file__),dest/'evaluation_source.py')
    ck=torch.load(args.checkpoint,map_location='cpu',weights_only=False);cfg=dict(ck['config'])
    if args.road_protect_m is not None:cfg['corridor_road_protect_m']=args.road_protect_m
    e=VehicleExperiment(cfg);e.old.actor.load_state_dict(ck['actor']);e.old.actor.requires_grad_(False)
    rows=[];totals={};raws=[105,113,130] if args.anchors else list(range(89,138))
    new_colour=e.old.actor.colour_logit.detach().clone()
    def add(method,metric,values):
        key=method+'/'+metric;acc=totals.setdefault(key,dict(sum=0.,count=0));acc['sum']+=float(values.sum());acc['count']+=int(values.numel())
    with torch.no_grad():
        for raw in raws:
            v=e.old.view(raw);visible=torch.as_tensor(v['visible_evidence_mask'],device='cuda',dtype=torch.bool)
            labels=torch.as_tensor(v['actor_labels'],device='cuda');other=(labels>0)&(labels<4)
            unknown=torch.as_tensor(v['unknown_mask'],device='cuda',dtype=torch.bool)
            camera=e.old.base.camera(raw)
            kw=e.old.actor.kwargs(camera,e.old.matrix(raw),e.n.args,e.n.static.get_features.shape[1],e.n.args.convert_SHs_python)
            e.tagged(kw,e.groups);alone=raster(camera,e.n.args,e.n.static.active_sh_degree,kw,e.n.bg)
            domain=(alone['extra'][5]>.1)&unknown
            new=e.render(raw,False);full=e.render(raw,True)
            baseline=render_baseline(e.old.base,raw,False,True)
            native=e.n.render(camera,dynamic=True)['rgb']
            e.old.actor.colour_logit.copy_(e.checkpoint['actor']['colour_logit'].to('cuda'))
            old=e.old.render(raw,False)
            e.old.actor.colour_logit.copy_(new_colour)
            truth=e.old.truth(raw);measurement=torch.as_tensor(v['measured_a6_depth_m'],device='cuda')
            valid=visible&torch.isfinite(measurement)&(measurement>0)
            rec=dict(raw=raw,a6_visible_pixels=int(visible.sum()),measured_depth_pixels=int(valid.sum()),fixed_lower_domain_pixels=int(domain.sum()),methods={})
            folder=dest/'frames'/f'{raw:03d}';folder.mkdir(parents=True)
            for name,out in [('r3',baseline),('oldC2',old),('new',new)]:
                rgb=out['rgb'].clamp(0,1);err=(rgb-truth).abs()[:,visible]
                depth=(out['depth_sum'][0]/out['alpha'][0].clamp_min(1e-8))/e.old.visibility.scale
                de=(depth[valid]-measurement[valid]).abs()
                rec['methods'][name]=dict(a6_visible_MAE=float(err.mean()),measured_depth_MAE_m=float(de.mean()) if de.numel() else None,
                    measured_depth_median_m=float(de.median()) if de.numel() else None,other_change_from_r3=float((rgb-baseline['rgb'].clamp(0,1)).abs()[:,other].mean()) if other.any() else 0.)
                add(name,'a6_visible_MAE',err);add(name,'measured_depth_MAE_m',de)
                save(folder/(name+'_delete.png'),rgb)
            rec['proposal_domain']=dict(actor_alpha_mean=float(new['actor_alpha'][0,domain].mean()) if domain.any() else None,
                generated_lower_alpha_mean=float(new['component_alpha_T'][5,domain].mean()) if domain.any() else None,
                static_alpha_mean=float(new['component_alpha_T'][0,domain].mean()) if domain.any() else None,
                road_alpha_mean=float(new['component_alpha_T'][1,domain].mean()) if domain.any() else None,
                oldC2_actor_alpha_mean=float(old['actor_alpha'][0,domain].mean()) if domain.any() else None)
            for key,val in [('new_actor',new['actor_alpha'][0,domain]),('new_lower',new['component_alpha_T'][5,domain]),('old_actor',old['actor_alpha'][0,domain])]:add(key,'fixed_lower_adoption',val)
            rec['noedit_native_maxabs']=float((full['rgb']-native).abs().max())
            save(folder/'new_lower_alpha.png',new['component_alpha_T'][5:6].repeat(3,1,1))
            save(folder/'native_full.png',native);save(folder/'gt_observed.png',truth)
            rows.append(rec)
            (dest/'progress.json').write_text(json.dumps(dict(last_raw=raw,frames=len(rows),seconds=time.monotonic()-start))+'\n')
            if raw in [105,113,130]:print(json.dumps(rec),flush=True)
    summary=dict(config=cfg,checkpoint=str(args.checkpoint),step=ck['step'],frames=len(rows),rows=rows,
        pooled={k:dict(mean=v['sum']/v['count'] if v['count'] else None,count=v['count']) for k,v in totals.items()},
        noedit_maxabs=max(r['noedit_native_maxabs'] for r in rows),parent_unchanged=e.unchanged(),seconds=time.monotonic()-start,
        geometry_fixed=True,road_fixed=True,unknown_geometry_GT=False,unknown_RGB_GT=False,
        depth_reference='Original actual a6 LiDAR zbuffer/point footprints intersect strict visible mask; observed-data alignment, not heldout amodal GT',
        noedit_policy='Native bypass by explicit edit-conditioned representation replacement; equality is by construction and not scientific reconstruction improvement',
        quality_accepted=False,human_verdict=None)
    (dest/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(stage='COMPLETE',pooled=summary['pooled'],seconds=summary['seconds'])),flush=True)

if __name__=='__main__':main()
